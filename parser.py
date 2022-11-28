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

import argparse
import deepspeed

def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='load args from json type config file. It will override the parser setting')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output')
    
    # Dist
    parser.add_argument('--local_rank', type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    
    # Model
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained/t5-base')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--load_module_only', action='store_true', default=False, help='only load model weights not optimizer')
    # Tasks
    ## QA
    parser.add_argument('--text_answer', type=bool, default=True, help='perform question answering')
    ## VA
    parser.add_argument('--visual_answer', action='store_true', default=False)
    parser.add_argument('--va_type', type=str, default='tokenwise', choices=['global', 'tokenwise'], help='predict related region at token/global level')
    parser.add_argument('--va_module_type', type=str, default='map', choices=['map', 'linear', 'mlp'])
    parser.add_argument('--va_label_smoothing', type=float, default=0.0, help='label smoothing value when performing bce loss')    
    parser.add_argument('--min_va',type=int, default=1, help='Minimum number of related regions predicted by the saliency detector')
   
    ## Retrieval
    parser.add_argument('--page_contrast', action='store_true', help='contrastively optimize question and page feature to be similar. loss: nce')
    parser.add_argument('--page_contrast_bidirection', action='store_true', help='whether to calculate nce loss bidirectionally')
    parser.add_argument('--page_contrast_type', default='global', choices=['global', 'tokenwise'], help='page contrast at global/local level')
    parser.add_argument('--page_contrast_t', type=float, default=0.01, help='nce temperature for page contrast')
    parser.add_argument('--page_contrast_module_type', type=str, default=None, choices=[None, 'linear', 'mlp'])

    # Data
    parser.add_argument('--root', type=str, default='data/VRManual', help='data root path')
    parser.add_argument('--mask', action='store_true', default=False, help='add <mask> token')
    
    # RoI feature extractor
    parser.add_argument('--roi_config', type=str, default='detector/VG-BUA.yaml')
    parser.add_argument('--roi_model', type=str, default='detector/pretrained/bua-d2-frcn-r101.pth')
    parser.add_argument('--roi_bua', type=bool, default=True)
    
    # Training
    parser.add_argument('--no_cross', action='store_true', help='if set, do not encode question and page jointly.')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode, only load the first batch')
    parser.add_argument('--start_epoch', type=int, default=0, help='set when resume training')
    parser.add_argument('--epoch', type=int, default=7)
    parser.add_argument('--save_best_last', type=bool, default=True, help='if set, only the best and last ckeckpoint will be saved')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--inf_batch_size', type=int, default=3)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Inference & Evaluate
    parser.add_argument('--use_retrieved_qa2dataid', default=False, help='whether to use top-1 retrieved page to perform QA/sd')
    parser.add_argument('--retrieved_qa2dataid', default=None, help='Use retrieved top-1 page to perform qa/sd. A dict of split2paths')
    parser.add_argument('--val_metric', type=str, default='ROUGE_L')
    parser.add_argument('--val_metric_aggregate', choices=['mean', 'harmonic_mean'], default='harmonic_mean')
    parser.add_argument('--eval_set', type=str, default='test')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--max_dec_len', type=int, default=20)
    parser.add_argument('--max_page_len', type=int, default=1024)

    return parser
