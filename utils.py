# Copyright(c) 2022 Liang Zhang 
# E-Mail: <zhangliang00@ruc.edu.cn>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import random
import logging
import argparse
import deepspeed
import numpy as np
from math import ceil
import torch.distributed as dist

from collections import OrderedDict, defaultdict

def obj_to_cuda(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    elif isinstance(obj, list):
        return [obj_to_cuda(t) for t in obj]
    elif isinstance(obj, tuple):
        return (obj_to_cuda(t) for t in obj)
    elif isinstance(obj, dict):
        return {key: obj_to_cuda(t) for key, t in obj.items()}
    else:
        return obj

def save_ckpt(args, model, optimizer, output_dir, epoch, logger):
    os.makedirs(os.path.join(output_dir, 'ckpts'), exist_ok=True)
    ckpt_path = os.path.join(output_dir, 'ckpts', f'checkpoint.{epoch}')
    logger.info(f'Saving checkpoint {ckpt_path}...')
    if args.deepspeed:
        ckpt_path = ckpt_path.rstrip('/')
        tag = ckpt_path.split('/')[-1]
        load_dir = '/'.join(ckpt_path.split('/')[:-1])   
        model.save_checkpoint(load_dir, tag)     
    else:
        checkpoint = OrderedDict()
        checkpoint['model'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)

def load_ckpt(args, ckpt_path, model, optimizer=None, logger=None, load_module_only=False):
    if logger is not None:
        logger.info(f'Loading model from {ckpt_path}')
    if args.deepspeed:
        ckpt_path = ckpt_path.rstrip('/')
        tag = ckpt_path.split('/')[-1]
        load_dir = '/'.join(ckpt_path.split('/')[:-1])
        model.load_checkpoint(load_dir, tag, load_module_only=load_module_only, 
                            load_module_strict=False,
                            load_optimizer_states=(not load_module_only),
                            load_lr_scheduler_states=(not load_module_only))
    else:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        if optimizer is not None and not load_module_only:
            if logger is not None:
                logger.info(f'Loading optimizer')        
            optimizer.load_state_dict(ckpt['optimizer'])

@torch.no_grad()
def retrieval_eval(score_matrix, txt_ids, img_ids, txt2img, img2txts, score_matrix_2=None, return_top_imgs=False):
    # image retrieval
    img2j = {i: j for j, i in enumerate(img_ids)}
    _, rank_txt = score_matrix.topk(min(10, score_matrix.size(1)), dim=1)
    txt2topimg = OrderedDict()
    topimgs = rank_txt[:, 0]
    for i, txt_id in enumerate(txt_ids):
        txt2topimg[txt_id] = img_ids[topimgs[i]]
        
    if score_matrix.size(1) < 10:
        print(f'WARNING: find {score_matrix.size(1)} candidate images, less than 10.')
    gt_img_j = torch.LongTensor([img2j[txt2img[txt_id]]
                                for txt_id in txt_ids],
                                ).to(rank_txt.device
                                    ).unsqueeze(1).expand_as(rank_txt)

    rank = (rank_txt == gt_img_j).nonzero()[:,1]
    if rank.numel():
        ir_r1 = (rank < 1).sum().item() / len(txt_ids)
        ir_r3 = (rank < 3).sum().item() / len(txt_ids)
        ir_r5 = (rank < 5).sum().item() / len(txt_ids)
        ir_r10 = (rank < 10).sum().item() / len(txt_ids)
    else:
        ir_r1, ir_r3, ir_r5, ir_r10 = 0, 0, 0, 0
 
    # text retrieval
    txt2i = {t: i for i, t in enumerate(txt_ids)}
    if score_matrix_2 is not None:
        score_matrix = score_matrix_2.t()

    _, rank_img = score_matrix.topk(min(10, score_matrix.size(0)), dim=0)
    if score_matrix.size(0) < 10:
        print(f'WARNING: find {score_matrix.size(0)} candidate txts, less than 10.')
    tr_r1, tr_r3, tr_r5, tr_r10 = 0, 0, 0, 0
    for j, img_id in enumerate(img_ids):
        gt_is = [txt2i[t] for t in img2txts[img_id]]
        ranks = [(rank_img[:, j] == i).nonzero() for i in gt_is]
        rank = min([10] + [r.item() for r in ranks if r.numel()])
        if rank < 1:
            tr_r1 += 1
        if rank < 3:
            tr_r3 += 1
        if rank < 5:
            tr_r5 += 1
        if rank < 10:
            tr_r10 += 1
    tr_r1 /= len(img_ids)
    tr_r3 /= len(img_ids)
    tr_r5 /= len(img_ids)
    tr_r10 /= len(img_ids)

    # tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    # ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    tr_mean = (tr_r1 + tr_r3 + tr_r5) / 3
    ir_mean = (ir_r1 + ir_r3 + ir_r5) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_log = {'qa2page_r1': ir_r1,
                'qa2page_r3': ir_r3,
                'qa2page_r5': ir_r5,
                'qa2page_r10': ir_r10,
                'qa2page_r_mean': ir_mean,
                'qa_nums': len(txt_ids),
                'page2qa_r1': tr_r1,
                'page2qa_r3': tr_r3,
                'page2qa_r5': tr_r5,
                'page2qa_r10': tr_r10,
                'page2qa_r_mean': tr_mean,
                'page_nums': len(img_ids),
                'r_mean': r_mean,
                }
    if return_top_imgs:
        return eval_log, txt2topimg
    else:
        return eval_log

def merge_recall(all_metrics):
    merged_metrics = defaultdict(int)
    all_qa_nums = sum([m['qa_nums'] for m in all_metrics])
    all_page_nums = sum([m['page_nums'] for m in all_metrics])
    merged_metrics['qa_nums'] = all_qa_nums
    merged_metrics['page_nums'] = all_page_nums

    for name in ['qa2page_r1', 'qa2page_r3', 'qa2page_r5', 'qa2page_r10', 'qa2page_r_mean']:
        merged_metrics[name] = sum([m[name]*m['page_nums'] for m in all_metrics]) / all_page_nums
        
    for name in ['page2qa_r1', 'page2qa_r3', 'page2qa_r5', 'page2qa_r10', 'page2qa_r_mean']:
        merged_metrics[name] = sum([m[name]*m['qa_nums'] for m in all_metrics]) / all_qa_nums
    
    merged_metrics['r_mean'] = (merged_metrics['qa2page_r_mean']+merged_metrics['page2qa_r_mean']) / 2

    return merged_metrics

def harmonic_mean(data):
    total = 0
    for i in data:
        if i == 0:
            return 0
        total += 1/i
    return len(data) / total

def boardcast_str(s, src_rank=0):
    object_list = [s]
    dist.broadcast_object_list(object_list=object_list, src=src_rank)
    return object_list[0]

def gather_list(list_to_gather):
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, list_to_gather)
    open_nest_results = []
    for result in results:
        open_nest_results.extend(result)
    return open_nest_results

def gather_tensor(tensor_to_gather):
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, tensor_to_gather)
    return results


def remove_repeat_sample(list_to_process, N_samples):
    samples_per_rank = ceil((N_samples-dist.get_rank())/dist.get_world_size())
    return list_to_process[:samples_per_rank]

def unique_index_and_value(dataids):
    unique_dataids = []
    unique_index = []
    hashset = set()
    for index, dataid in enumerate(dataids):
        if dataid not in hashset:
            hashset.add(dataid)
            unique_dataids.append(dataid)
            unique_index.append(index)
    dataids = unique_dataids
    return unique_dataids, unique_index


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger
