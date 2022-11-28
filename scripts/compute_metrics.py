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
import argparse
import json
from nlgeval import NLGEval, _strip
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score

PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]


VRM_SEMANTIC_CLS2ID = {
    "<pad>": 0,
    "Text": 1,
    "Title": 2,
    "Product Image": 3,
    "illustration": 4,
    "Table": 5,
    "graphic": 6,
    "Question": 7
}

def get_filter_list(items, class_filter):
    new_items = []
    for item in items:
        for region_cls in set(item['gt_region_cls']):
            if class_filter(VRM_SEMANTIC_CLS2ID[region_cls]):
                new_items.append(item)
                break
    return new_items

def get_multimodal_list(items):
    return get_filter_list(items, lambda x: x>=3)

def remove_punc(line):
    return ' '.join([w for w in line.rstrip().split(' ') \
        if w not in PUNCTUATIONS])

class NLGEvalNew(NLGEval):
    def compute_metrics(self, ref_list, hyp_list):
        ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
        assert len(refs) == len(hyps)

        ret_scores = {}
        instance_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, scs, m in zip(score, scores, method):
                        ret_scores[m] = sc
                        instance_scores[m] = scs
                else:
                    ret_scores[method] = score
                    instance_scores[method] = scores

        return ret_scores, instance_scores

print('Loading evaluator...')
nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)
print('Load evaluator finished')

def compute_visual_answer_metrics(pred_related_regions, gt_regions, all_regions, all_regions_cls, cls_split=None):
    all_y_true = []
    all_y_pred = []
    instance_p = 0
    instance_r = 0
    instance_f1 = 0
    no_preds = 0
    for i, (instance_region_ids, instance_preds, instance_gts, all_region_cls) in enumerate(zip(all_regions, pred_related_regions, gt_regions, all_regions_cls)):
        y_true = []
        y_pred = []
        if len(instance_preds) == 0:
            no_preds += 1
        for i, _id in enumerate(instance_region_ids):
            if cls_split is not None and all_region_cls[i] != cls_split:
                continue
            if _id in instance_preds:
                y_pred.append(1)
            else:
                y_pred.append(0)
            if _id in instance_gts:
                y_true.append(1)
            else:
                y_true.append(0)
        if len(y_pred) == 0:
            continue
        # instance_p += precision_score(y_true, y_pred, average='binary')
        # instance_r += recall_score(y_true, y_pred, average='binary')
        # instance_f1 += f1_score(y_true, y_pred, average='binary')
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    # instance_p = instance_p / len(all_regions)
    # instance_r = instance_r / len(all_regions)
    # instance_f1 = instance_f1 / len(all_regions)
    all_p = precision_score(all_y_true, all_y_pred, average='binary')
    all_r = recall_score(all_y_true, all_y_pred, average='binary')
    all_f1 = f1_score(all_y_true, all_y_pred, average='binary')
    metrics = {
        # 'instance_precision': instance_p,
        # 'instance_recall': instance_r,
        # 'instance_f1': instance_f1,
        'all_precision': all_p,
        'all_recall': all_r,
        'all_f1': all_f1
    }
    if no_preds > 0:
        print('#########################################')
        print(f'{no_preds}/{len(all_regions)} instances has no predictions!!')
        print('#########################################')
    return metrics

def compute_visual_answer_by_region_cls(items, is_print=True):
    pred_regions, gt_regions, all_regions, all_regions_cls = [], [], [], []
    for item in items:
        pred_regions.append(item['pred_regions'])
        gt_regions.append(item['gt_regions'])
        all_regions.append(item['all_regions'])
        all_regions_cls.append(item['all_region_cls'])
    cls2metrics = OrderedDict()
    for type in ['Text', 'Title', 'Product Image', 'illustration', 'Table', 'graphic']:
        print('Region predict of {}'.format(type))
        metrics = compute_visual_answer_metrics(pred_regions, gt_regions, all_regions, all_regions_cls, cls_split=type)
        cls2metrics[type] = metrics
        if is_print:
            for metric, score in metrics.items():
                print(f'{metric}: {score:.3f}')
    return cls2metrics

def compute_qa_score(items):
    all_predictions = [x['caption'] for x in items]
    all_answers = [x['gt'] for x in items]
    all_predictions = [remove_punc(sent).lower() for sent in all_predictions]
    all_answers = [[remove_punc(sent).lower() for sent in all_answers]]
    
    metrics = nlgeval.compute_metrics(all_answers, all_predictions)
    return metrics

def compute_qa_score_by_region_cls(items, is_print=True):
    cls2metrics = OrderedDict()
    for type in ['Text', 'Title', 'Product Image', 'illustration', 'Table', 'graphic']:
        type_id = VRM_SEMANTIC_CLS2ID[type]
        sub_items = get_filter_list(items, lambda x: x == type_id)
        metrics = compute_qa_score(sub_items)
        if is_print:
            print('Questions contains {}'.format(type))
            for metric, score in metrics.items():
                print(f'{metric}: {score:.3f}')
        cls2metrics[type] = metrics
    return cls2metrics
    
def compute_qa_score_instance(items):
    all_predictions = [x['caption'] for x in items]
    all_answers = [x['gt'] for x in items]
    all_predictions = [remove_punc(sent).lower() for sent in all_predictions]
    all_answers = [remove_punc(sent).lower() for sent in all_answers]

    metrics_list = []
    assert len(all_predictions) == len(all_answers)
    for prediction, answer in tqdm(zip(all_predictions, all_answers), ncols=50, total=len(all_predictions)):
        metrics_dict = nlgeval.compute_individual_metrics([answer], prediction)
        metrics_list.append(metrics_dict)
    metrics = {}
    
    N = len(metrics_list)
    
    for key in metrics_list[0].keys():
        metrics[key] = sum([metrics_list[i][key] for i in range(N)]) / N
        for metric, score in metrics.items():
            print(f'{metric}: {score:.3f}')

def item_process(input, output, task='qa'):
    items = json.load(open(input))
    if task == 'qa':
        metrics = compute_qa_score_by_region_cls(items)
    elif task == 'sd':
        metrics = compute_visual_answer_by_region_cls(items)
    else:
        raise NotImplementedError
    
    with open(output, 'w') as f:
        json.dump(metrics, f, indent=4)
    return metrics

def list_process(inputs, outputs):
    for input, output in zip(inputs, outputs):
        metrics = item_process(input, output)

