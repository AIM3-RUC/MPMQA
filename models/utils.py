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
import torch.nn as nn
import torch.distributed as dist

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, non_linear='relu', res=False):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_out)
        if non_linear == 'relu':
            self.activate = nn.ReLU()
        self.res = res
        if self.res:
            assert d_in == d_out
    
    def forward(self, x):
        res = x
        x = self.linear_1(x)
        x = self.activate(x)
        x = self.linear_2(x)
        if self.res:
            x = x + res
        return x

class NCELoss(nn.Module):
    def __init__(self, t=1.0, bidirectional=False):
        super().__init__()
        self.t = t
        self.bidirectional = bidirectional
    
    def get_loss(self, sim_matrix):
        sim_matrix = sim_matrix / self.t
        logpt = torch.nn.functional.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

    def forward(self, sim_matrix_ij, sim_matrix_ji=None):
        if self.bidirectional:
            if sim_matrix_ji is None:
                sim_matrix_ji = sim_matrix_ij.t()
            loss = (self.get_loss(sim_matrix_ij) + self.get_loss(sim_matrix_ji)) / 2
        else:
            loss = self.get_loss(sim_matrix_ij)

        return loss

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )


class AllGatherBatch(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(output, tensor)
        output_pad = pad_features(output)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        ctx.length = tensor.shape[1]
        return output_pad

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1), :ctx.length],
            None,
        )

def pad_features(tensors):
    shapes = [t.shape for t in tensors]
    total_batch = sum([s[0] for s in shapes])
    rank = dist.get_rank()
    dtype = tensors[rank].dtype
    device = tensors[rank].device
    requires_grad = tensors[rank].requires_grad
    padded_shape = [total_batch]
    for i in range(1, len(shapes[0])):
        padded_size_i = 0
        for s in shapes:
            padded_size_i = max(padded_size_i, s[i])
        padded_shape.append(padded_size_i)
    
    padded_tensor = torch.zeros(padded_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    b_start = 0
    for i, tensor in enumerate(tensors):
        padded_tensor[b_start:b_start+tensor.size(0), :tensor.size(1)] = tensor
        b_start += tensor.size(0)
    return padded_tensor