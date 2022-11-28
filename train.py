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
import deepspeed
import torch.distributed as dist

from tqdm import tqdm
from utils import set_seed, get_logger, obj_to_cuda, save_ckpt, load_ckpt, boardcast_str, harmonic_mean
from parser import get_base_parser
from dataset.mqa_dataset import get_mqa_loader
from models.mqa_model import MQAT5Model
from evaluate import evaluate_ds

def train(args, model, train_loader, val_loader, test_loader, optimizer, logger, val_metric='ROUGE_L'):
    logger.info('Start training')
    total_step = 0
    best_epoch = -1
    best_score = -1
    best_ckpt_path = None
    ckpt_dir = os.path.join(args.output_dir, 'ckpts')

    start_epoch = args.start_epoch
    if args.checkpoint:
        logger.info(f'Resume training from {args.checkpoint}')
        logger.info(f'Start epoch {start_epoch}')
        # assert args.start_epoch != 0
        load_ckpt(args, args.checkpoint, model, logger, load_module_only=args.load_module_only)
    
        metrics = evaluate_ds(args, model, val_loader, logger, save_fn=f'epoch_{start_epoch}.json', split='val')
        if isinstance(val_metric, str):
            val_score = metrics[val_metric]
        elif isinstance(val_metric, list):
            scores = [metrics[n] for n in val_metric]
            if args.val_metric_aggregate == 'harmonic_mean':
                val_score = harmonic_mean(scores)
            elif args.val_metric_aggregate == 'mean':
                val_score = sum(scores) / len(scores)

        if ((not args.deepspeed) or dist.get_rank()==0) and val_score > best_score:
            best_score = val_score
            best_epoch = start_epoch
            best_ckpt_path = os.path.join(ckpt_dir, f'checkpoint.{best_epoch}')
            logger.info(f'Epoch {best_epoch} get best {args.val_metric_aggregate} score {val_metric}: {best_score}')
            logger.info(f'Best checkpoint at {best_ckpt_path}')
    
    # use for schedule sampling
    total_step = (args.epoch - start_epoch) * len(train_loader)
    now_step = start_epoch * len(train_loader)
    
    for epoch in range(start_epoch, args.epoch):
        model.train()
        # model.eval() # for debug, DO NOT forget to remove
        model.module.roi_extractor.eval()
        # Set epoch must be called, otherwise the order of data in each epoch is the same
        train_loader.sampler.set_epoch(epoch)
        for step, batch in tqdm(enumerate(train_loader), ncols=50, total=len(train_loader)):
            batch = obj_to_cuda(batch)
            loss_dict = model(**batch, now_step=now_step, total_step=total_step)
            loss = loss_dict['loss']
            if args.deepspeed:
                model.backward(loss)
                model.step()    
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            now_step += 1
            if now_step % 100 == 0:
                if args.deepspeed:
                    logger.info(f'Epoch: {epoch+1}/{args.epoch}, step: {step}/{len(train_loader)}, loss: {float(loss.detach().cpu())}')
                else:
                    logger.info(f'Epoch: {epoch+1}/{args.epoch}, step: {step}/{len(train_loader)}, lr: {min(optimizer.get_lr())}-{max(optimizer.get_lr())}, loss: {float(loss.detach().cpu())}')
                for loss_name, loss_value in loss_dict.items():
                    if loss_name == 'loss':
                        continue
                    logger.info(f'{loss_name}: {float(loss_value.detach().cpu())}')
            if args.debug:
                break

        save_ckpt(args, model, None, args.output_dir, epoch=epoch+1, logger=logger)

        metrics = evaluate_ds(args, model, val_loader, logger, save_fn=f'epoch_{epoch+1}.json', split='val')

        if isinstance(val_metric, str):
            val_score = metrics[val_metric]
        elif isinstance(val_metric, list):
            scores = [metrics[n] for n in val_metric]
            val_score = harmonic_mean(scores)
        elif args.val_metric_aggregate == 'mean':
            val_score = sum(scores) / len(scores)

        if val_score > best_score and ((not args.deepspeed) or dist.get_rank()==0):
            best_score = val_score
            best_epoch = epoch+1
            best_ckpt_path = os.path.join(ckpt_dir, f'checkpoint.{best_epoch}')
            logger.info(f'Epoch {best_epoch} get best {args.val_metric_aggregate} score {val_metric}: {best_score}')
            logger.info(f'Best checkpoint at {best_ckpt_path}')
        # If the checkpoint of previous epoch did not perform best, remove it.
        if args.save_best_last and ((not args.deepspeed) or dist.get_rank()==0):
            previous_epoch = epoch
            previous_ckpt_path = os.path.join(ckpt_dir, f'checkpoint.{previous_epoch}')
            if previous_epoch > start_epoch and previous_ckpt_path != best_ckpt_path:
                logger.info(f'Remove {previous_ckpt_path} that does not preform best.')
                cmd = f'rm -r {previous_ckpt_path}'
                logger.info(f'Execute command: \n{cmd}')
                os.system(cmd)

    if (not args.deepspeed) or dist.get_rank() == 0:
        logger.info(f'Epoch {best_epoch} get best {args.val_metric_aggregate} mean score {val_metric}: {best_score}')
        logger.info(f'Load checkpoint {best_ckpt_path} to perform testing')
    best_ckpt_path = boardcast_str(best_ckpt_path, src_rank=0)
    load_ckpt(args, best_ckpt_path, model, logger=logger, load_module_only=True)
    del train_loader
    del val_loader
    del optimizer
    torch.cuda.empty_cache()
    if args.deepspeed:
        metrics = evaluate_ds(args, model, test_loader, logger, save_fn=f'epoch_{best_epoch}.json', split='test')
            
def get_parameter_group(args, model):
    '''Get optimize parameter group;
    '''
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    weight_decay = args.weight_decay

    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_param_tp], 'weight_decay': weight_decay},
        {'params': [p for _, p in no_decay_param_tp], 'weight_decay': 0.0},
    ]

    return optimizer_grouped_parameters

def main(args):
    set_seed(args.seed)
    torch.cuda.set_device(args.local_rank)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.deepspeed:
        deepspeed.init_distributed()

    nowtime = None
    # Saving arguments
    if not args.deepspeed or (args.deepspeed and dist.get_rank() == 0):
        nowtime = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        os.makedirs(os.path.join(args.output_dir, 'opt', nowtime), exist_ok=True)
        with open(os.path.join(args.output_dir, 'opt', nowtime, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=1, ensure_ascii=False)
        if args.deepspeed_config is not None:
            os.system(f'cp {args.deepspeed_config} {os.path.join(args.output_dir, "opt", nowtime)}')
    if args.deepspeed:
        nowtime = boardcast_str(nowtime, src_rank=0)
        logger = get_logger(os.path.join(args.output_dir, 'opt', nowtime, f'log.{dist.get_rank()}.txt'))
    else:
        logger = get_logger(os.path.join(args.output_dir, 'opt', nowtime, f'log.txt'))
    logger.info(json.dumps(vars(args), indent=2))

    model = MQAT5Model(args, pretrained_dir=args.pretrained_dir)
    train_loader = get_mqa_loader(args, root=args.root, tokenizer=model.tokenizer, batch_size=args.batch_size, split='train', num_workers=args.n_workers)
    val_loader = get_mqa_loader(args, root=args.root, tokenizer=model.tokenizer, batch_size=args.batch_size, split='val', num_workers=args.n_workers)
    test_loader = get_mqa_loader(args, root=args.root, tokenizer=model.tokenizer, batch_size=args.inf_batch_size, split='test', num_workers=args.n_workers)
    model.resize_token_embeddings()
    model.cuda()

    if args.deepspeed:
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=get_parameter_group(args, model)
        )
    train(args, model, train_loader, val_loader, test_loader, optimizer, logger, val_metric=args.val_metric)
    

if __name__ == '__main__':
    parser = get_base_parser()
    args = parser.parse_args()
    if args.config is not None:
        args_dict = json.load(open(args.config, 'r', encoding='utf-8'))
        for key, value in args_dict.items():
            if key == 'local_rank':
                continue
            setattr(args, key, value)
    if args.debug:
        args.output_dir = "expr/debug"

    main(args)