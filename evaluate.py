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
import json
import time
import torch
from torch.nn.functional import normalize as norm
import deepspeed
import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from math import ceil
# from torch.optim import Adam
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import set_seed, get_logger, obj_to_cuda, load_ckpt, \
    boardcast_str, gather_list, remove_repeat_sample, retrieval_eval, merge_recall, \
    unique_index_and_value
from parser import get_base_parser
from dataset.mqa_dataset import get_mqa_loader
from dataset.mqa_page_contrast import MQAContrastDataset, mqa_contrast_collate_fn
from models.mqa_model import MQAT5Model
from models.utils import pad_features
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import OrderedDict
from scripts.compute_metrics import compute_visual_answer_by_region_cls, compute_qa_score_by_region_cls


from nlgeval import NLGEval

PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]

nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)  # loads the models

def remove_punc(line):
    return ' '.join([w for w in line.rstrip().split(' ') \
        if w not in PUNCTUATIONS])

def compute_visual_answer_metics(pred_related_regions, gt_regions, all_regions):
    all_y_true = []
    all_y_pred = []
    instance_p = 0
    instance_r = 0
    instance_f1 = 0
    no_preds = 0
    for i, (instance_region_ids, instance_preds, instance_gts) in enumerate(zip(all_regions, pred_related_regions, gt_regions)):
        y_true = []
        y_pred = []
        if len(instance_preds) == 0:
            no_preds += 1
        for _id in instance_region_ids:
            if _id in instance_preds:
                y_pred.append(1)
            else:
                y_pred.append(0)
            if _id in instance_gts:
                y_true.append(1)
            else:
                y_true.append(0)
        instance_p += precision_score(y_true, y_pred, average='binary')
        instance_r += recall_score(y_true, y_pred, average='binary')
        instance_f1 += f1_score(y_true, y_pred, average='binary')
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    instance_p = instance_p / len(all_regions)
    instance_r = instance_r / len(all_regions)
    instance_f1 = instance_f1 / len(all_regions)
    all_p = precision_score(all_y_true, all_y_pred, average='binary')
    all_r = recall_score(all_y_true, all_y_pred, average='binary')
    all_f1 = f1_score(all_y_true, all_y_pred, average='binary')
    metrics = {
        'instance_precision': instance_p,
        'instance_recall': instance_r,
        'instance_f1': instance_f1,
        'all_precision': all_p,
        'all_recall': all_r,
        'all_f1': all_f1
    }
    if no_preds > 0:
        print('#########################################')
        print(f'{no_preds}/{len(all_regions)} instances has no predictions!!')
        print('#########################################')
    return metrics


def evaluate_page_contrast(args, model, page_contrast_dataset, logger, save_fn="temp.json"):
    model.eval()
    manuals = page_contrast_dataset.manuals
    all_metrics = []
    save_fn = ''.join(save_fn.split('.')[:-1])
    qa2topimgs = OrderedDict()
    for i, manual_name in tqdm(enumerate(manuals), total=len(manuals)):
        torch.cuda.empty_cache()
        page_contrast_dataset.set_manual(manual_name)
        
        sampler = torch.utils.data.DistributedSampler(
            page_contrast_dataset, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank(), 
            shuffle=False
        )
        sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=args.inf_batch_size,
            drop_last=False
        )
        dataloader = DataLoader(
            dataset=page_contrast_dataset,
            shuffle=False,
            batch_sampler=sampler,
            num_workers=args.n_workers,
            collate_fn=mqa_contrast_collate_fn
        )
        
        qaids = []
        dataids = []
        q_features = []
        page_features = []
        q_mask = []
        page_mask = []

        for j, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = obj_to_cuda(batch)
            qaids.extend(batch['qaids'])
            dataids.extend(batch['dataids'])
            with torch.no_grad():
                return_hidden = (args.page_contrast_type == 'tokenwise')
                try:
                    question_feature, context_feature = model.module.two_stream_encoding(return_hidden=return_hidden, **batch)
                except RuntimeError:
                    torch.cuda.empty_cache()
                    question_feature, context_feature = model.module.two_stream_encoding(return_hidden=return_hidden, **batch)
                if args.page_contrast_module_type is not None:
                    question_feature = model.module.page_contrast_module(question_feature)
                    context_feature = model.module.page_contrast_module(context_feature)
                question_feature = norm(question_feature, dim=-1)
                context_feature = norm(context_feature, dim=-1)
                q_features.append(question_feature.detach().cpu())
                page_features.append(context_feature.detach().cpu())
                q_mask.append(batch['question_attn_mask'])
                page_mask.append(batch['context_attn_mask'])

        
        qaids = gather_list(qaids)
        dataids = gather_list(dataids)
        q_features = gather_list(q_features)
        page_features = gather_list(page_features)
        q_mask = gather_list(q_mask)
        page_mask = gather_list(page_mask)

        qaids, unique_qa_index = unique_index_and_value(qaids)
        dataids, unique_page_index = unique_index_and_value(dataids)

        if dist.get_rank() == 0:
            if args.page_contrast_type == 'global':
                q_features = torch.cat(q_features, dim=0)
                page_features = torch.cat(page_features, dim=0)

                q_features = q_features[unique_qa_index]
                page_features = page_features[unique_page_index]

                sim_matrix = torch.matmul(q_features, page_features.t())
                metrics, qa2topimg = retrieval_eval(sim_matrix.float(), qaids, dataids, 
                    page_contrast_dataset.qaid2dataid, page_contrast_dataset.dataid2qaids, return_top_imgs=True)
            elif args.page_contrast_type == 'tokenwise':
                q_features = pad_features(q_features)
                page_features = pad_features(page_features)
                q_mask = pad_features(q_mask)
                page_mask = pad_features(page_mask)

                q_features = q_features[unique_qa_index]
                q_mask = q_mask[unique_qa_index]                
                page_features = page_features[unique_page_index]
                page_mask = page_mask[unique_page_index]

                # sim_matrix = torch.matmul(q_features, page_features.t())
                with torch.no_grad():
                    try:
                        sim_matrix_qc, sim_matrix_cq = model.module.similarity_score(q_features.cuda(), page_features.cuda(), q_mask.cuda(), page_mask.cuda())
                    except RuntimeError:
                        torch.cuda.empty_cache()
                        sim_matrix_qc, sim_matrix_cq = model.module.similarity_score(q_features, page_features, q_mask, page_mask)
                        sim_matrix_qc, sim_matrix_cq = sim_matrix_qc.float(), sim_matrix_cq.float()
                    metrics, qa2topimg = retrieval_eval(sim_matrix_qc, qaids, dataids, 
                        page_contrast_dataset.qaid2dataid, page_contrast_dataset.dataid2qaids, sim_matrix_cq, return_top_imgs=True)
                sim_matrix = sim_matrix_qc
                del sim_matrix_cq

            assert len(qa2topimgs.keys() & qa2topimg.keys()) == 0
            assert len(qa2topimg.keys() & set(qaids)) == len(qa2topimg.keys()) == len(qaids)
            qa2topimgs.update(qa2topimg)

            print(f'Manual: {manual_name}')
            for metric, score in metrics.items():
                print(f'{metric}: {score:.3f}')
            all_metrics.append(metrics)
            predict_dir = os.path.join(args.output_dir, 'predict', page_contrast_dataset.split, 'page_contrast', save_fn, manual_name)
            os.makedirs(predict_dir, exist_ok=True)
            with open(os.path.join(predict_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=1)
            with open(os.path.join(predict_dir, 'qaids.json'), 'w') as f:
                json.dump(qaids, f, indent=1)
            with open(os.path.join(predict_dir, 'dataids.json'), 'w') as f:
                json.dump(dataids, f, indent=1)
            with open(os.path.join(predict_dir, 'qaid2dataid.json'), 'w') as f:
                json.dump(page_contrast_dataset.qaid2dataid, f, indent=1)
            with open(os.path.join(predict_dir, 'dataid2qaids.json'), 'w') as f:
                json.dump(page_contrast_dataset.dataid2qaids, f, indent=1)
            np.save(os.path.join(predict_dir, 'score_matrix.npy'), sim_matrix.detach().cpu().numpy())

        else:
            metrics = defaultdict(int)

    if dist.get_rank() == 0:
        merged_metrics = merge_recall(all_metrics)
        predict_dir = os.path.join(args.output_dir, 'predict', page_contrast_dataset.split, 'page_contrast', save_fn)
        os.makedirs(predict_dir, exist_ok=True)
        path = os.path.join(predict_dir, 'all.json')
        with open(path, 'w') as f:
            json.dump(merged_metrics, f, indent=1)
        path = os.path.join(predict_dir, 'qa2topimgs.json')
        with open(path, 'w') as f:
            json.dump(qa2topimgs, f, indent=1)
        logger.info('Average page retrieval performance')
        for metric, score in merged_metrics.items():
            logger.info(f'{metric}: {score:.3f}')
        return merged_metrics
    else:
        return defaultdict(int)
        

def evaluate_visual_answer(args, model, val_loader, logger, sd_save_fn='temp.json', split='val'):
    model.eval()
    pred_related_regions = []
    gt_regions = []
    qa_ids = []
    all_regions = []

    for step, batch in tqdm(enumerate(val_loader), ncols=50, total=len(val_loader)):
        batch = obj_to_cuda(batch)
        with torch.no_grad():
            pred_related_region = model.module.visual_answer_inference(**batch)
        qa_ids.extend(batch['qa_ids'])
        pred_related_regions.extend(pred_related_region)
        gt_regions.extend(batch['related_regions'])
        all_regions.extend(list(batch['region_positions'][i].keys()) for i in range(len(batch['qa_ids'])))

    # Remove repeat samples
    N_samples = len(val_loader.dataset)
    samples_per_rank = ceil((N_samples-dist.get_rank())/dist.get_world_size())
    qa_ids = qa_ids[:samples_per_rank]
    pred_related_regions = pred_related_regions[:samples_per_rank]
    gt_regions = gt_regions[:samples_per_rank]
    all_regions = all_regions[:samples_per_rank]
    
    qa_ids_list = [None] * dist.get_world_size()
    dist.all_gather_object(qa_ids_list, qa_ids)
    pred_related_regions_list = [None] * dist.get_world_size()
    dist.all_gather_object(pred_related_regions_list, pred_related_regions)
    gt_regions_list = [None] * dist.get_world_size()
    dist.all_gather_object(gt_regions_list, gt_regions)
    all_regions_list = [None] * dist.get_world_size()
    dist.all_gather_object(all_regions_list, all_regions)

    if dist.get_rank() == 0:
        qa_ids, pred_related_regions, gt_regions, all_regions = [], [], [], []
        gt_region_types, pred_region_types, all_region_types = [], [], []
        for i in range(dist.get_world_size()):
            qa_ids.extend(qa_ids_list[i])
            pred_related_regions.extend(pred_related_regions_list[i])
            gt_regions.extend(gt_regions_list[i])
            all_regions.extend(all_regions_list[i])
        for rids in gt_regions:
            gt_region_types.append([val_loader.dataset.rid2cls[r] for r in rids if r in val_loader.dataset.rid2cls])
        for rids in pred_related_regions:
            pred_region_types.append([val_loader.dataset.rid2cls[r] for r in rids if r in val_loader.dataset.rid2cls])
        for rids in all_regions:
            all_region_types.append([val_loader.dataset.rid2cls[r] for r in rids if r in val_loader.dataset.rid2cls])

        predict_items = []
        for qa_id, pred_related_region, gt_region, all_region, gt_region_type, pred_region_type, all_region_type in zip(qa_ids, pred_related_regions, gt_regions, all_regions, gt_region_types, pred_region_types, all_region_types):
            predict_items.append({
                'image_id': qa_id,
                'pred_regions': pred_related_region,
                'pred_region_cls': pred_region_type,
                'gt_regions': gt_region,
                'gt_region_cls': gt_region_type,
                'all_regions': all_region,
                'all_region_cls': all_region_type
            })
        predict_dir = os.path.join(args.output_dir, 'predict', split, 'related_regions')
        os.makedirs(predict_dir, exist_ok=True)
        path = os.path.join(predict_dir, sd_save_fn)
        with open(path, 'w') as f:
            json.dump(predict_items, f, indent=1)
        
        metrics = compute_visual_answer_metics(pred_related_regions, gt_regions, all_regions)

        cls_metrics = compute_visual_answer_by_region_cls(predict_items, is_print=False)
        cls_metrics['All'] = metrics

        path = os.path.join(predict_dir, sd_save_fn.replace('.json', '_metrics.json'))
        with open(path, 'w') as f:
            json.dump(cls_metrics, f, indent=1)

        for metric, score in metrics.items():
            logger.info(f'{metric}: {score:.3f}')
        return metrics
    else:
        return defaultdict(int)

def evaluate_question_answer(args, model, val_loader, logger, save_fn='temp.json', split='val'):
    predictions = []
    questions = []
    gts = []
    gt_regions = []
    qa_ids = []
    predict_items = []
    image_paths = []
    model.eval()
    for step, batch in tqdm(enumerate(val_loader), ncols=50, total=len(val_loader)):
        batch = obj_to_cuda(batch)
        with torch.no_grad():
            if args.beam_size is None or args.beam_size <= 1:
                _, prediction = model.module.greedy_inference(**batch)
            else:
                _, prediction = model.module.beam_search(beam_size=args.beam_size, length_penalty=args.length_penalty, **batch)
        predictions.extend(prediction)
        qa_ids.extend(batch['qa_ids'])
        image_paths.extend(batch['image_paths'])
        questions.extend(model.module.tokenizer.batch_decode(batch['question_ids'], skip_special_tokens=True))
        gts.extend(model.module.tokenizer.batch_decode(batch['answer_ids'], skip_special_tokens=True))
        gt_regions.extend(batch['related_regions'])

    for qa_id, image_path, question, predict, gt, gt_region in zip(qa_ids, image_paths, questions, predictions, gts, gt_regions):
        predict_items.append({
            'image_id': qa_id,
            'image_path': image_path,
            'question': question,
            'caption': predict,
            'gt': gt,
            'gt_regions': gt_region,
            'gt_region_cls': [val_loader.dataset.rid2cls[r] for r in gt_region if r in val_loader.dataset.rid2cls]
        })

    # Remove repeat samples
    N_samples = len(val_loader.dataset)
    samples_per_rank = ceil((N_samples-dist.get_rank())/dist.get_world_size())
    predict_items = predict_items[:samples_per_rank]

    predict_list = [None] * dist.get_world_size()
    dist.all_gather_object(predict_list, predict_items)

    if dist.get_rank() == 0:
        candidates = []
        for predict in predict_list:
            candidates.extend(predict)
        # assert len(candidates) == N_samples
        try:
            candidates.sort(key=lambda x: int(x['image_id']))
        except:
            candidates.sort(key=lambda x: x['image_id'])

        predict_dir = os.path.join(args.output_dir, 'predict', split)
        os.makedirs(predict_dir, exist_ok=True)
        path = os.path.join(predict_dir, save_fn)
        
        with open(path, 'w') as f:
            json.dump(candidates, f, indent=1)


        all_predictions = [x['caption'] for x in candidates]
        all_answers = [x['gt'] for x in candidates]

        all_predictions = [remove_punc(sent).lower() for sent in all_predictions]

        all_answers = [[remove_punc(sent).lower() for sent in all_answers]]
        metrics = nlgeval.compute_metrics(all_answers, all_predictions)
        metrics_divide_by_cls = compute_qa_score_by_region_cls(candidates, is_print=False)
        metrics_divide_by_cls['All'] = metrics
        path = os.path.join(predict_dir, save_fn.replace('.json', '_metrics.json'))
        with open(path, 'w') as f:
            json.dump(metrics_divide_by_cls, f, indent=1)
        for metric, score in metrics.items():
            logger.info(f'{metric}: {score:.3f}')  
        return metrics      
    else:
        return defaultdict(int)    

def evaluate_ds(args, model, val_loader, logger, save_fn='temp.json', split='val'):
    logger.info(f'Evaluating on {split} split...')

    metrics = defaultdict(int)
    if args.page_contrast:
        dataset = MQAContrastDataset(args, args.root, model.module.tokenizer, split)
        recall_metrics = evaluate_page_contrast(args, model, dataset, logger, save_fn=save_fn)
        metrics.update(recall_metrics)
        torch.cuda.empty_cache()
    if args.use_retrieved_qa2dataid:
        dist.barrier()
        val_loader.dataset.set_use_retrieved_qa2dataid()
    if args.visual_answer:
        sd_metrics = evaluate_visual_answer(args, model, val_loader, logger, sd_save_fn=save_fn, split=split)
        metrics.update(sd_metrics)
        torch.cuda.empty_cache()
    if args.text_answer:
        qa_metrics = evaluate_question_answer(args, model, val_loader, logger, save_fn=save_fn, split=split)
        metrics.update(qa_metrics)
        torch.cuda.empty_cache()
    
    if dist.get_rank() == 0:
        for metric, score in metrics.items():
            logger.info(f'{metric}: {score:.3f}')
    return metrics

def main(args):
    set_seed(args.seed)
    torch.cuda.set_device(args.local_rank)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, 'log.txt'))
    logger.info(args)

    if args.deepspeed:
        deepspeed.init_distributed()

    
    nowtime  = None
    if not args.deepspeed or (args.deepspeed and dist.get_rank() == 0):
        nowtime = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        os.makedirs(os.path.join(args.output_dir, 'eval_opt', nowtime), exist_ok=True)
        with open(os.path.join(args.output_dir, 'eval_opt', nowtime, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=1, ensure_ascii=False)
        if args.deepspeed_config is not None:
            os.system(f'cp {args.deepspeed_config} {os.path.join(args.output_dir, "eval_opt", nowtime)}')
    if args.deepspeed:
        nowtime = boardcast_str(nowtime, src_rank=0)
        logger = get_logger(os.path.join(args.output_dir, 'eval_opt', nowtime, f'log.{dist.get_rank()}.txt'))
    else:
        logger = get_logger(os.path.join(args.output_dir, 'eval_opt', nowtime, f'log.txt'))
    logger.info(args)  

    model = MQAT5Model(args, pretrained_dir=args.pretrained_dir)
    split2loader = OrderedDict()
    
    if isinstance(args.eval_set, str):
        splits = [args.eval_set]
    else:
        splits = args.eval_set
    for split in splits:
        split2loader[split] = get_mqa_loader(args, root=args.root, tokenizer=model.tokenizer, batch_size=args.batch_size, split=split, num_workers=args.n_workers, eval_on_train=True)

    model.resize_token_embeddings()
    model.cuda()
    if args.deepspeed:
        model, _, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters()
        )

    if args.checkpoint:
        load_ckpt(args, args.checkpoint, model, logger=logger, load_module_only=True)
    
    for split, val_loader in split2loader.items():
        evaluate_ds(args, model, val_loader, logger, save_fn=args.save_fn, split=split)

if __name__ == '__main__':
    parser = get_base_parser()
    parser.add_argument('--save_fn', type=str, default='temp.json')
    parser.add_argument('--sd_save_fn', type=str, default='temp.json')
    args = parser.parse_args()
    if args.config is not None:
        args_dict = json.load(open(args.config, 'r', encoding='utf-8'))
        for key, value in args_dict.items():
            if key == 'local_rank':
                continue
            setattr(args, key, value)
    main(args)
