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
import cv2

import time
import json
import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from .utils import read_jsonl, pad_2d_mask, divide_box_grid
from transformers import T5TokenizerFast
import torch.distributed as dist

from .const import VRM_SEMANTIC_TOKENS, VRM_SEMANTIC_CLS2ID, VRM_SEMANTIC_CLS2TOKEN, \
    VRM_FINE_GRAIN_CLS
    
class MQADataset(Dataset):
    def __init__(self, args, root, tokenizer, split='train', task='qa'):
        super().__init__()
        self.args = args
        self.root = root
        self.tokenizer = tokenizer
        self.split = split
        self.task = task

        data_path = os.path.join(root, 'data', f'{split}.jsonl')
        self.data = read_jsonl(data_path)
        self.rid2cls = {}
        for data_dict in self.data:
            for region in data_dict['bounding_boxes']: 
                self.rid2cls[region['id']] = region['structure']
        self.qa_pairs, self.qaid2dataid = self.get_qa_pairs()
        
        self.set_const() 

        special_tokens = self.SEMANTIC_TOKENS.copy()
        if self.args.mask:
            special_tokens.append('<mask>')

        self.tokenizer.add_special_tokens({"additional_special_tokens":special_tokens})

        print('Special tokens:')
        print(self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES)

        self.subwords = list(self.tokenizer.get_vocab().keys())
        print(f'Total subwords: {len(self.subwords)}')

        
    def set_use_retrieved_qa2dataid(self):
        assert self.args.use_retrieved_qa2dataid
        with open(self.args.retrieved_qa2dataid[self.split], 'r') as f:
            retrieved_qa2dataid = json.load(f)
        retrieved_qa2dataid = {int(key): value for key, value in retrieved_qa2dataid.items()}
        print(f'Evaluate QA/sd with retrieved pages in {self.args.retrieved_qa2dataid[self.split]}')
        assert len(retrieved_qa2dataid.keys() & self.qaid2dataid.keys()) == len(self.qaid2dataid.keys()) == len(retrieved_qa2dataid.keys())
        self.qaid2dataid = retrieved_qa2dataid
        

    def set_const(self):
        self.SEMANTIC_TOKENS = VRM_SEMANTIC_TOKENS
        self.SEMANTIC_CLS2ID = VRM_SEMANTIC_CLS2ID
        self.SEMANTIC_CLS2TOKEN = VRM_SEMANTIC_CLS2TOKEN   
        self.FINE_GRAIN_CLS = VRM_FINE_GRAIN_CLS
        
    def get_qa_pairs(self):
        qaid = 0
        qa_pairs = []
        qaid2dataid = {}
        for dataid, item in enumerate(self.data):
            for qa_item in item['qa_data']:
                qa_pairs.append(qa_item)
                qaid2dataid[qaid] = dataid
                qaid += 1
        print(f'Total {len(qa_pairs)} qa pairs')
        return qa_pairs, qaid2dataid

    def convert_bbox(self, d):
        x1, y1, w, h = d['x'], d['y'], d['width'], d['height']
        x2, y2 = x1+w, y1+h
        return torch.tensor([x1,y1,x2,y2])
    
    def merge_bbox(self, box_list):
        x1, y1, x2, y2 = 1e9, 1e9, -1, -1
        for box in box_list:
            x1 = min(x1, box[0])
            y1 = min(y1, box[1])
            x2 = max(x2, box[2])
            y2 = max(y2, box[3])
        return torch.tensor([x1,y1,x2,y2])

    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        dataid = self.qaid2dataid[idx]
        # import pdb;pdb.set_trace()
        question = qa_pair['question']['text']
        answer = qa_pair['answer']['text']
        relevant_rids = qa_pair['answer']['relevant']
        question_dict = self.tokenizer(question, return_tensors='pt')
        question_ids, question_attn_mask = question_dict['input_ids'].squeeze(dim=0), question_dict['attention_mask'].squeeze(dim=0)
        question_segment_ids = torch.zeros_like(question_ids)
        question_segment_ids.fill_(self.SEMANTIC_CLS2ID['Question'])
        # T5 treat <pad> as start token
        # Default: No start token is inserted in position 0
        answer_all = self.tokenizer('<pad>'+answer, return_tensors='pt').input_ids.squeeze(dim=0)
        answer_ids = answer_all[:-1]
        answer_labels = answer_all[1:]
        answer_attn_mask = torch.tril(torch.ones((len(answer_ids), len(answer_ids)), dtype=question_attn_mask.dtype))
        region_positions = defaultdict(list)

        data_dict = self.data[dataid]
        image_path = os.path.join(self.root, data_dict['image_filename'])
        img = cv2.imread(image_path)
        retry = 0
        while img is None:
            time.sleep(1)
            img = cv2.imread(image_path)
            retry += 1
            if retry > 10:
                assert img is not None, f'Retrying to read {image_path} for 10 times but failed'
        img = torch.from_numpy(img)
        
        bboxes = []
        segment_ids = []
        region_ids = []
        related_region_labels = []
        tokens = []
        mlm_labels = [] # for whole word mlm

        for r, region in enumerate(data_dict['bounding_boxes']):
            is_related_region = int((region['id'] in relevant_rids))
            region_positions[region['id']].append(len(tokens))
            region_ids.append(region['id'])
            region_bbox = region['shape']
            semantic_class = region['structure']

            tokens.extend(self.tokenizer.encode(self.SEMANTIC_CLS2TOKEN[semantic_class], add_special_tokens=False))
            bboxes.append(self.convert_bbox(region_bbox))
            segment_ids.append(self.SEMANTIC_CLS2ID[semantic_class])
            related_region_labels.append(is_related_region)
            
            if 'ocr_info' in region:
                for ocr_region in region['ocr_info']:
                    ocr_word = ocr_region['word']
                    ocr_tokens = self.tokenizer.encode(ocr_word, add_special_tokens=False)
                    
                    n_tokens = len(ocr_tokens)
                    tokens.extend(ocr_tokens)
                    bboxes.extend([self.convert_bbox(ocr_region['bbox'])] * n_tokens)
                    segment_ids.extend([self.SEMANTIC_CLS2ID[semantic_class]] * n_tokens)
                    if self.args.va_type == 'tokenwise':
                        related_region_labels.extend([is_related_region] * n_tokens)
                    elif self.args.va_type == 'global':
                        related_region_labels.extend([-1] * n_tokens)
            # if len(tokens) < self.max_page_len:
            region_positions[region['id']].append(len(tokens))
            # else:
            #     region_positions[region['id']].append(self.max_page_len)

            assert len(tokens) == len(bboxes) == len(segment_ids) == len(related_region_labels), "length mismatch"
        if len(bboxes) == 0:
            import pdb;pdb.set_trace()

        context_ids = torch.tensor(tokens)
        context_attn_mask = torch.ones(len(context_ids), dtype=question_attn_mask.dtype)
        bboxes = torch.stack(bboxes, dim=0)
        segment_ids = torch.tensor(segment_ids)
        related_region_labels = torch.tensor(related_region_labels)
        mlm_labels = torch.tensor(mlm_labels)

        qa_ids = qa_pair['id']

        return {
            'qa_ids': qa_ids,
            'image_paths': image_path,
            'imgs': img,
            'question_ids': question_ids,
            'question_attn_mask': question_attn_mask,
            'question_segment_ids': question_segment_ids,
            'answer_ids': answer_ids,
            'answer_attn_mask': answer_attn_mask,
            'answer_labels': answer_labels,
            'context_ids': context_ids[:self.args.max_page_len],
            'context_attn_mask': context_attn_mask[:self.args.max_page_len],
            'bboxes': bboxes[:self.args.max_page_len],
            'segment_ids': segment_ids[:self.args.max_page_len],
            'related_region_labels': related_region_labels[:self.args.max_page_len],
            'region_positions': region_positions,
            'related_regions': relevant_rids,
            'mlm_labels': mlm_labels[:self.args.max_page_len]
        }

    def __len__(self):
        return len(self.qa_pairs)
    
    
def mqa_collate_fn(dict_list):
    batch_dict = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            batch_dict[key].append(value)
    
    batch_dict['question_ids'] = pad_sequence(batch_dict['question_ids'], batch_first=True, padding_value=0)
    batch_dict['question_attn_mask'] = pad_sequence(batch_dict['question_attn_mask'], batch_first=True, padding_value=0)
    batch_dict['question_segment_ids'] = pad_sequence(batch_dict['question_segment_ids'], batch_first=True, padding_value=0)
    batch_dict['answer_ids'] = pad_sequence(batch_dict['answer_ids'], batch_first=True, padding_value=0)
    batch_dict['answer_labels'] = pad_sequence(batch_dict['answer_labels'], batch_first=True, padding_value=0)
    batch_dict['answer_attn_mask'] = pad_2d_mask(batch_dict['answer_attn_mask'], padding_value=0)
    batch_dict['context_ids'] = pad_sequence(batch_dict['context_ids'], batch_first=True, padding_value=0)
    batch_dict['context_attn_mask'] = pad_sequence(batch_dict['context_attn_mask'], batch_first=True, padding_value=0)
    batch_dict['segment_ids'] = pad_sequence(batch_dict['segment_ids'], batch_first=True, padding_value=0)
    batch_dict['related_region_labels'] = pad_sequence(batch_dict['related_region_labels'], batch_first=True, padding_value=-1)

    return batch_dict

def get_mqa_loader(args, root, tokenizer, batch_size, split='train', num_workers=4, eval_on_train=False):
    dataset = MQADataset(args, root, tokenizer, split)
    sampler = None
    if hasattr(args, 'deepspeed') and args.deepspeed:
        if split == 'train' and not eval_on_train:
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
            )
        else:
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
            )
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=mqa_collate_fn,
        shuffle=(split=='train' and sampler is None and not eval_on_train),
        drop_last=(split=='train' and not eval_on_train)
    )
    return dataloader
