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

import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast
import torch.distributed as dist
from .utils import pad_2d_mask

from .mqa_dataset import MQADataset

import sys
sys.path.insert(0, '../')
from parser import get_base_parser

class MQAContrastDataset(MQADataset):
    def __init__(self, args, root, tokenizer, split='train'):
        super().__init__(args, root, tokenizer, split, task='retrieval')

        
        self.dataid2qaids = defaultdict(list)
        for qaid, dataid in self.qaid2dataid.items():
            self.dataid2qaids[dataid].append(qaid)

        self.manual2dataids = defaultdict(list)
        self.manual2qaids = defaultdict(list)
        for dataid, datum in enumerate(self.data):
            name = datum['image_filename'].split('/')[1]
            self.manual2dataids[name].append(dataid)
            self.manual2qaids[name].extend(self.dataid2qaids[dataid])
            
        self.manuals = list(self.manual2dataids.keys())
        
        print(f'Total {len(self.manuals)} manuals')
        self.now_manual = self.manuals[0]


    def get_page(self, dataid):
        data_dict = self.data[dataid]
        image_path = os.path.join(self.root, data_dict['image_filename'])
        img = cv2.imread(image_path)
        img = torch.from_numpy(img)        
        tokens, bboxes, segment_ids = [], [], []

        for region in data_dict['bounding_boxes']:
            region_bbox = region['shape']
            semantic_class = region['structure']
            tokens.extend(self.tokenizer.encode(self.SEMANTIC_CLS2TOKEN[semantic_class], add_special_tokens=False))
            bboxes.append(self.convert_bbox(region_bbox))
            segment_ids.append(self.SEMANTIC_CLS2ID[semantic_class])
            if 'ocr_info' in region:
                for ocr_region in region['ocr_info']:
                    ocr_word = ocr_region['word']
                    ocr_tokens = self.tokenizer.encode(ocr_word, add_special_tokens=False)
                    n_tokens = len(ocr_tokens)
                    tokens.extend(ocr_tokens)
                    bboxes.extend([self.convert_bbox(ocr_region['bbox'])] * n_tokens)
                    segment_ids.extend([self.SEMANTIC_CLS2ID[semantic_class]] * n_tokens)
            assert len(tokens) == len(bboxes) == len(segment_ids)
        if len(bboxes) == 0:
            import pdb;pdb.set_trace()
        context_ids = torch.tensor(tokens)
        context_attn_mask = torch.ones(len(context_ids), dtype=torch.int64)
        bboxes = torch.stack(bboxes, dim=0)
        segment_ids = torch.tensor(segment_ids)

        return img, context_ids, bboxes, segment_ids, context_attn_mask
    
    def __getitem__(self, idx):
        qaid = self.manual2qaids[self.now_manual][idx]
        dataid = self.qaid2dataid[qaid]
        qa_pair = self.qa_pairs[qaid]
        question = qa_pair['question']['text']
        question_dict = self.tokenizer(question, return_tensors='pt')
        question_ids, question_attn_mask = question_dict['input_ids'].squeeze(dim=0), question_dict['attention_mask'].squeeze(dim=0)   
        question_segment_ids = torch.zeros_like(question_ids)
        question_segment_ids.fill_(self.SEMANTIC_CLS2ID['Question'])      

        img, context_ids, bboxes, segment_ids, context_attn_masks = self.get_page(dataid)

        return {
            "qaids": qaid,
            "dataids": dataid,
            "imgs": img,
            "bboxes": bboxes,
            "question_ids": question_ids,
            "question_attn_mask": question_attn_mask,
            "question_segment_ids": question_segment_ids,
            "context_ids": context_ids,
            "context_attn_mask": context_attn_masks,
            "segment_ids": segment_ids
        }

    def set_manual(self, manual_name):
        self.now_manual = manual_name
        
    def __len__(self):
        return len(self.manual2qaids[self.now_manual])


def mqa_contrast_collate_fn(dict_list):
    batch_dict = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            batch_dict[key].append(value)
    
    batch_dict['question_ids'] = pad_sequence(batch_dict['question_ids'], batch_first=True, padding_value=0)
    batch_dict['question_attn_mask'] = pad_sequence(batch_dict['question_attn_mask'], batch_first=True, padding_value=0)
    batch_dict['question_segment_ids'] = pad_sequence(batch_dict['question_segment_ids'], batch_first=True, padding_value=0)
    batch_dict['context_ids'] = pad_sequence(batch_dict['context_ids'], batch_first=True, padding_value=0)
    batch_dict['context_attn_mask'] = pad_sequence(batch_dict['context_attn_mask'], batch_first=True, padding_value=0)
    batch_dict['segment_ids'] = pad_sequence(batch_dict['segment_ids'], batch_first=True, padding_value=0)

    return batch_dict