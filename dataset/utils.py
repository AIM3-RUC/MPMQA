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

import torch
import jsonlines

def read_jsonl(path):
    """Read jsonlines file into python list

    args:
        path - directory of the jsonlines file
    return:
        jsonlines file content in List
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            items.append(item)
    return items

def divide_box_grid(box, row_col):

    if isinstance(row_col, int):
        row_col = (row_col, row_col)
    row, col = row_col[0], row_col[1]
    with torch.no_grad():
        all_width = box[2] - box[0]
        all_height = box[3] - box[1]
        single_width = all_width / row
        single_height = all_height / col
        x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
    divided_boxes = []
    for i in range(row):
        for j in range(col):
            box_ij = torch.tensor([
                x1+i*single_width, y1+j*single_height,
                x1+(i+1)*single_width, y1+(j+1)*single_height
            ])
            divided_boxes.append(box_ij)
    return divided_boxes

def pad_2d_mask(mask_list, padding_value=0):
    """Perform padding, convert a list of 2-d masks into batch

    args:
        mask_list [N x N,] - list of 2-d masks
        padding_value - the value to occupy vacancy
    return: 
        batch_mask [B x Max_N x Max_N, ] - padded batch mask
    """
    max_len = float('-inf')
    batch_size = len(mask_list)
    for mask in mask_list:
        max_len = max(len(mask), max_len)
    batch_mask = torch.zeros((batch_size, max_len, max_len), dtype=mask_list[0].dtype)
    batch_mask.fill_(padding_value)
    for i, mask in enumerate(mask_list):
        n, n = mask.shape
        batch_mask[i, :n, :n] = mask
    return batch_mask

def get_sub_batch(batch, start_idx=0, end_idx=1):
    """Get part of a Dict batch, use for debugging

    args:
        batch - a dict that contains a batch of data
        start_idx - start id of the sub_batch
        end_idx - end id of the sub_batch
    return:
        sub_batch - dict contains the batch data with id in [start_idx, end_idx-1]

    """
    assert start_idx < end_idx

    sub_batch = {}
    for key, value in batch.items():
        sub_batch[key] = value[start_idx:end_idx]
    return sub_batch
