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

VRM_FINE_GRAIN_CLS = [
    "Product Image",
    "illustration",
    "graphic"
]

VRM_SEMANTIC_TOKENS = [
    '<Text>',
    '<Title>',
    '<Img>',
    '<illustration>',
    '<Table>',
    '<graphic>'
]

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

VRM_SEMANTIC_CLS2TOKEN = {
    "Text": '<Text>',
    "Title": '<Title>',
    "Product Image": '<Img>',
    'illustration': '<illustration>',
    'Table': '<Table>',
    'graphic': '<graphic>'
}