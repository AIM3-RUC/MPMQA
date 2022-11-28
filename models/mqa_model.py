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

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
import random


from detector.ROIFeatExtractor import ROIFeatExtractor
from models.utils import NCELoss, AllGather, AllGatherBatch, pad_features, MLP
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers.generation_beam_search import BeamSearchScorer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from torch.nn.functional import normalize as norm

class MQAT5Model(nn.Module):
    def __init__(self, args, pretrained_dir='t5-base'):
        super().__init__()
        self.args = args
        self.roi_extractor = ROIFeatExtractor(args.roi_config, args.roi_model, args.roi_bua)
        self.roi_extractor.eval()
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_dir)

        self.encoder = self.t5.get_encoder()
        self.decoder = self.t5.get_decoder()
        self.lm_head = self.t5.get_output_embeddings()

        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_dir)
        self.model_dim = self.t5.model_dim
        self.embed_token = self.t5.shared
        self.segment_embedding = nn.Embedding(11, self.model_dim, padding_idx=0)
        self.apperance_embedding = nn.Linear(2048, self.model_dim)
        self.location_embedding = nn.Linear(4, self.model_dim)

        if args.visual_answer:
            if args.va_module_type == 'map':
                self.saliency_detector = nn.Linear(self.model_dim, 2)
            elif args.va_module_type == 'linear':
                self.saliency_detector = nn.Sequential(
                    nn.Linear(self.model_dim, self.model_dim),
                    nn.Linear(self.model_dim, 2)
                )
            elif args.va_module_type == 'mlp':
                self.saliency_detector = nn.Sequential(
                    MLP(self.model_dim, self.model_dim, self.model_dim),
                    nn.Linear(self.model_dim, 2)
                )
            else:
                raise NotImplementedError

        if args.page_contrast:
            if args.page_contrast_module_type == 'linear':
                self.page_contrast_module = nn.Linear(self.model_dim, self.model_dim)
            elif args.page_contrast_module_type == 'mlp':
                self.page_contrast_module = MLP(self.model_dim, self.model_dim, self.model_dim, res=True)        
            elif args.page_contrast_module_type is None:
                pass
            else:
                raise NotImplementedError
            
        self.max_dec_len = args.max_dec_len
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.bce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean', label_smoothing=self.args.va_label_smoothing)
        self.nce_loss = NCELoss(t=args.page_contrast_t, bidirectional=args.page_contrast_bidirection)

    def resize_token_embeddings(self):
        self.t5.resize_token_embeddings(len(self.tokenizer))
    
    def norm_bboxes(self, bboxes):
        with torch.no_grad():
            x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            w = x_max - x_min
            h = y_max - y_min
            normed_bboxes = torch.stack([x_min / w, y_min / h, x_max / w, y_max / h], dim=1)
        return normed_bboxes
    
    def get_direction(self, region_boxes):
        box_centers = torch.stack([region_boxes[:,0]+region_boxes[:,2], 
                    region_boxes[:,1]+region_boxes[:,3]], dim=1) / 2
        relative = norm(box_centers.unsqueeze(dim=0) - box_centers.unsqueeze(dim=1), dim=2)
        angle_upper = torch.acos(relative[:,:,0])
        angle_bottom = angle_upper + 0.999 * torch.pi
        angle = torch.where(relative[:,:,1]>0, angle_upper, angle_bottom)
        direction_labels = (angle * 4 / torch.pi).long()
        direction_labels = direction_labels - (direction_labels.diag()+1).diag_embed()
        return direction_labels

    def combine_embedding_and_mask(self, question_embeddings, question_mask, context_embeddings, context_mask):
        batch_size = question_mask.size(0)
        question_lengths = question_mask.sum(dim=-1)
        context_lengths = context_mask.sum(dim=-1)
        total_lengths = question_lengths + context_lengths
        max_len = total_lengths.max()
        total_mask = torch.zeros((batch_size, max_len), 
                            dtype=question_mask.dtype, device=question_mask.device)
        total_embeddings = torch.zeros((batch_size, max_len, self.model_dim), 
                            dtype=question_embeddings.dtype, device=question_embeddings.device)
        for i in range(batch_size):
            q_length_i = question_lengths[i]
            c_length_i = context_lengths[i]
            total_embeddings[i, :q_length_i] = question_embeddings[i, :q_length_i]
            total_embeddings[i, q_length_i:q_length_i+c_length_i] = context_embeddings[i, :c_length_i]
            total_mask[i, :total_lengths[i]] = 1
        return total_embeddings, total_mask
    
    def divide_embedding(self, all_embeddings, question_mask, context_mask):
        batch_size = all_embeddings.size(0)
        question_lengths = question_mask.sum(dim=-1)
        context_lengths = context_mask.sum(dim=-1)
        q_max_len = question_lengths.max()
        c_max_len = context_lengths.max()
        question_embeddings = torch.zeros((batch_size, q_max_len, self.model_dim), 
                        dtype=all_embeddings.dtype, device=all_embeddings.device)
        context_embeddings = torch.zeros((batch_size, c_max_len, self.model_dim), 
                        dtype=all_embeddings.dtype, device=all_embeddings.device)
        for i in range(batch_size):
            q_length_i = question_lengths[i]
            c_length_i = context_lengths[i]        
            question_embeddings[i, :q_length_i] = all_embeddings[i, :q_length_i]
            context_embeddings[i, :c_length_i] = all_embeddings[i, q_length_i:q_length_i+c_length_i]

        return question_embeddings, context_embeddings

    def get_question_embedding(self, question_ids, question_segment_ids):
        question_embed = self.embed_token(question_ids)
        question_segment_embed = self.segment_embedding(question_segment_ids)
        question_embed = question_embed + question_segment_embed
        return question_embed
    
    def get_context_embedding(self, context_ids, imgs, bboxes, segment_ids):
        context_embed = self.embed_token(context_ids)
        segment_embed = self.segment_embedding(segment_ids) 
        context_embed += segment_embed
        # apperance_embed
        with torch.no_grad():
            roi_features = self.roi_extractor.float()(imgs, bboxes)
            roi_features = [f.type(self.apperance_embedding.weight.dtype) for f in roi_features]
        apperance_embed_list = [self.apperance_embedding(f) for f in roi_features]
        apperance_embed = torch.zeros_like(context_embed)
        for i, embed in enumerate(apperance_embed_list):
            apperance_embed[i, :len(embed)] = embed
        context_embed += apperance_embed
        # location_embed
        normed_bboxes = [self.norm_bboxes(bbox).type(self.location_embedding.weight.dtype) for bbox in bboxes]
        location_embed_list = [self.location_embedding(b) for b in normed_bboxes]
        location_embed = torch.zeros_like(context_embed)
        for i, embed in enumerate(location_embed_list):
            location_embed[i, :len(embed)] = embed
        context_embed += location_embed

        return context_embed                    

    def get_embeddings_and_mask(self, question_ids, context_ids, imgs, bboxes,
                        question_attn_mask, context_attn_mask,
                        segment_ids, question_segment_ids, **kwargs):
        question_embed = self.get_question_embedding(question_ids, question_segment_ids)
        context_embed = self.get_context_embedding(context_ids, imgs, bboxes, segment_ids)

        input_embeds, attn_mask = self.combine_embedding_and_mask(question_embed, question_attn_mask, context_embed, context_attn_mask)
        return input_embeds, attn_mask
    
    def context_hidden_weight(self, question_hidden, question_mask, context_hidden, context_mask, method, **kwargs):
        if method == 'hard':
            context_weights = kwargs['context_weights']
            context_hidden = context_hidden * context_weights
            all_hidden, attn_mask = self.combine_embedding_and_mask(question_hidden, question_mask, context_hidden, context_mask)
            return all_hidden, attn_mask
        else:
            raise NotImplementedError
    
    def beam_search(self, beam_size, question_ids, context_ids, imgs, bboxes,
                        question_attn_mask, context_attn_mask,
                        segment_ids, question_segment_ids, **kwargs):
        batch_size = question_ids.size(0)
        beam_scorer = BeamSearchScorer(batch_size, beam_size, device=question_ids.device, **kwargs)

        input_embeds, attn_mask = self.get_embeddings_and_mask(
            question_ids, context_ids, imgs, bboxes,
            question_attn_mask, context_attn_mask,
            segment_ids, question_segment_ids, **kwargs)
        encoder = self.t5.get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            output_attentions=False,
            output_hidden_states=False,
        )     
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
            beam_size, dim=0)
        attn_mask = attn_mask.repeat_interleave(
            beam_size, dim=0
        )
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=question_ids.dtype, device=question_ids.device)
        decoder_input_ids = decoder_input_ids.repeat_interleave(
            beam_size, dim=0
        )
        # import pdb;pdb.set_trace()
        outputs = self.t5.beam_search(
            encoder_outputs=encoder_outputs, attention_mask=attn_mask, input_ids=decoder_input_ids, beam_scorer=beam_scorer, max_length=self.max_dec_len
        )
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs, predictions

    def greedy_inference(self, question_ids, context_ids, imgs, bboxes,
                        question_attn_mask, context_attn_mask,
                        segment_ids, question_segment_ids, **kwargs):
        input_embeds, attn_mask = self.get_embeddings_and_mask(
            question_ids, context_ids, imgs, bboxes,
            question_attn_mask, context_attn_mask,
            segment_ids, question_segment_ids, **kwargs)
        
        batch_size = input_embeds.size(0)
        # <pad> as start token
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=question_ids.dtype, device=question_ids.device)

        out = self.t5(inputs_embeds=input_embeds, attention_mask=attn_mask, 
                    decoder_input_ids=decoder_input_ids, return_dict=True, use_cache=True)
        past_key_values = out.past_key_values
        encoder_outputs = (out.encoder_last_hidden_state,)
        outputs = []
        logits = out.logits
        outputs.append(logits.argmax(dim=-1))

        for i in range(self.max_dec_len-1):
            out = self.t5(encoder_outputs=encoder_outputs, attention_mask=attn_mask, past_key_values=past_key_values, 
                        decoder_input_ids=outputs[-1], use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits
            outputs.append(logits.argmax(dim=-1))
        outputs = torch.cat(outputs, dim=1)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs, predictions
    
    
    def mlm_inference(self, question_ids, context_ids, imgs, bboxes,
                question_attn_mask, context_attn_mask, 
                segment_ids, question_segment_ids, mlm_labels, **kwargs):
        input_embeds, attn_mask = self.get_embeddings_and_mask(
            question_ids, context_ids, imgs, bboxes,
            question_attn_mask, context_attn_mask,
            segment_ids, question_segment_ids, **kwargs)   
        encoder = self.t5.get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            output_attentions=False,
            output_hidden_states=True,
        )  
        encoder_last_hidden_states = encoder_outputs.last_hidden_state   
        _, context_hidden_state = self.divide_embedding(encoder_last_hidden_states, question_attn_mask, context_attn_mask)
        if self.t5.config.tie_word_embeddings:
            context_hidden_state = context_hidden_state * (self.model_dim**-0.5)
        probs = self.mlm_head(context_hidden_state).argmax(dim=-1)
        probs = probs.flatten()
        labels = mlm_labels.flatten()
        indices = torch.where(labels)[0]
        acc_num = (probs[indices] == labels[indices]).sum()
        total_num = len(indices)
        return float(acc_num), float(total_num)
    
    def visual_answer_inference(self, question_ids, context_ids, imgs, bboxes,
                    question_attn_mask, context_attn_mask,
                    segment_ids, question_segment_ids, region_positions, **kwargs):
        input_embeds, attn_mask = self.get_embeddings_and_mask(
            question_ids, context_ids, imgs, bboxes,
            question_attn_mask, context_attn_mask,
            segment_ids, question_segment_ids, **kwargs)
        encoder = self.t5.get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            output_attentions=False,
            output_hidden_states=True,
        )  
        encoder_last_hidden_states = encoder_outputs.last_hidden_state   
        question_hidden_state, context_hidden_state = self.divide_embedding(encoder_last_hidden_states, question_attn_mask, context_attn_mask)
        saliency_probs = self.saliency_detector(context_hidden_state).softmax(dim=-1)

        def aggregate_score(scores, method='mean'):
            if method == 'mean':
                return scores.mean()
            elif method == 'first':
                if len(scores) == 0:
                    return torch.tensor(0.0)
                else:
                    return scores[0]
        pred_related_regions = [[] for _ in range(len(saliency_probs))]
        for i, saliency_prob in enumerate(saliency_probs):
            region_score_list = []
            for region_id, region_position in region_positions[i].items():
                token_probs = saliency_prob[region_position[0]:region_position[1], 1]
                method = 'mean' if self.args.va_type=='tokenwise' else 'first'
                region_prob = aggregate_score(token_probs, method)
                region_score_list.append((region_id, region_prob))
            region_score_list.sort(key=lambda x: x[1], reverse=True)
            nums = 0
            for (region_id, score) in region_score_list:
                if score >= 0.5:
                    nums += 1
                    pred_related_regions[i].append(region_id)
                elif nums < self.args.min_va:
                    nums += 1
                    pred_related_regions[i].append(region_id)
                else:
                    break
        return pred_related_regions

    def get_global_indices(self, region_positions):
        batchsize = len(region_positions)
        global_indices = [[] for _ in range(batchsize)] 
        for i in range(batchsize):
            for region_id, region_position in region_positions[i].items():
                global_indices[i].append(region_position[0])
        return global_indices
    
    def cross_encoding(self, question_ids, context_ids, imgs, bboxes,
                question_attn_mask, context_attn_mask,
                segment_ids, question_segment_ids, **kwargs):
        input_embeds, attn_mask = self.get_embeddings_and_mask(
            question_ids, context_ids, imgs, bboxes,
            question_attn_mask, context_attn_mask,
            segment_ids, question_segment_ids, **kwargs)
        encoder = self.t5.get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        encoder_last_hidden_states = encoder_outputs.last_hidden_state
        return encoder_last_hidden_states, attn_mask
    
    def encoding_question(self, question_ids, question_attn_mask, question_segment_ids, return_hidden=False):
        question_embeddings = self.get_question_embedding(question_ids, question_segment_ids)
        encoder = self.t5.get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=question_embeddings,
            attention_mask=question_attn_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        encoder_last_hidden_states = encoder_outputs.last_hidden_state
        if return_hidden:
            return encoder_last_hidden_states
        else:
            return encoder_last_hidden_states[:, 0]
    
    def encoding_context(self, context_ids, context_attn_mask, imgs, bboxes, segment_ids, return_hidden=False):
        context_embeddings = self.get_context_embedding(context_ids, imgs, bboxes, segment_ids)
        encoder = self.t5.get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=context_embeddings,
            attention_mask=context_attn_mask,
            output_attentions=False,
            output_hidden_states=False,
        )        
        encoder_last_hidden_states = encoder_outputs.last_hidden_state
        if return_hidden:
            return encoder_last_hidden_states
        else:
            return encoder_last_hidden_states[:, 0]
    
    def two_stream_encoding(self, question_ids, question_attn_mask, question_segment_ids,
            context_ids, context_attn_mask, imgs, bboxes, segment_ids, return_hidden=False, **kwargs):
        question_features = self.encoding_question(question_ids, question_attn_mask, question_segment_ids, return_hidden)
        context_features = self.encoding_context(context_ids, context_attn_mask, imgs, bboxes, segment_ids, return_hidden)
        return question_features, context_features
    
    def similarity_score(self, question_hiddens, context_hiddens, question_attn_mask=None, context_attn_mask=None):
        assert len(question_hiddens.shape) == len(question_hiddens.shape) == 3
        assert len(question_attn_mask.shape) == len(context_attn_mask.shape) == 2
        assert len(question_attn_mask) == len(question_hiddens)
        assert len(context_attn_mask) == len(context_hiddens)
        
        b1, t1, d = question_hiddens.shape
        b2, t2, d = context_hiddens.shape
        score_matrix = torch.zeros((b1, b2), device=question_hiddens.device, dtype=question_hiddens.dtype, requires_grad=False)
        score_matrix_2 = torch.zeros((b2, b1), device=question_hiddens.device, dtype=question_hiddens.dtype, requires_grad=False)
        # token_score_matrix: b1 x b2 x t1 x t2
        token_score_matrix = torch.einsum('ind,jmd->ijnm', question_hiddens, context_hiddens)
        for i in range(b1):
            for j in range(b2):
                # t1 x t2
                token_score_matrix_ij = token_score_matrix[i, j]
                token_score_matrix_ij = token_score_matrix_ij[:question_attn_mask[i].sum(), :context_attn_mask[j].sum()]
                score = token_score_matrix_ij.max(dim=-1)[0].mean(dim=0)
                score_2 = token_score_matrix_ij.t().max(dim=-1)[0].mean(dim=0)
                score_matrix[i, j] = score_matrix[i, j].clone() + score.clone()
                score_matrix_2[j, i] = score_matrix_2[j, i].clone() + score_2.clone()

        score_matrix.requires_grad_(True)
        score_matrix_2.requires_grad_(True)
        return score_matrix, score_matrix_2

    def pad_features(self, tensors):
        # tensors: B x T x D
        shapes = [t.shape for t in tensors]
        total_batch = sum([s[0] for s in shapes])
        dtype = tensors[0].dtype
        device = tensors[0].device
        requires_grad = tensors[0].requires_grad
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
    
    def forward_page_contrast_global(self, question_hiddens, context_hiddens, compute_loss=True):
        question_features = question_hiddens[:, 0]
        context_features = context_hiddens[:, 0]
        if self.args.page_contrast_module_type is not None:
            question_features = self.page_contrast_module(question_features)
            context_features = self.page_contrast_module(context_features)
        question_features = norm(question_features, dim=-1)
        context_features = norm(context_features, dim=-1)
        question_features = AllGather.apply(question_features)
        context_features = AllGather.apply(context_features)
        dist.barrier()
        score_matrix = torch.matmul(question_features, context_features.t())
        if compute_loss:
            pc_loss = self.nce_loss(score_matrix)
            return pc_loss, score_matrix
        else:
            return score_matrix
    
    def forward_page_contrast_tokenwise(self, question_hiddens, context_hiddens, question_attn_mask, context_attn_mask, compute_loss=True):
        if self.args.page_contrast_module_type is not None:
            question_hiddens = self.page_contrast_module(question_hiddens)
            context_hiddens = self.page_contrast_module(context_hiddens)

        question_features = norm(question_hiddens, dim=-1)
        context_features = norm(context_hiddens, dim=-1)
        question_features = AllGatherBatch.apply(question_features)
        context_features = AllGatherBatch.apply(context_features)
        _question_attn_mask = AllGatherBatch.apply(question_attn_mask)
        _context_attn_mask = AllGatherBatch.apply(context_attn_mask)
        dist.barrier()
        score_matrix_qc, score_matrix_cq = self.similarity_score(question_features, context_features, _question_attn_mask, _context_attn_mask)
        if compute_loss:
            tpc_loss = self.nce_loss(score_matrix_qc, score_matrix_cq)
            return tpc_loss, score_matrix_qc, score_matrix_cq
        else:
            return score_matrix_qc, score_matrix_cq
    
    def forward_salient_detection(self, encoder_last_hidden_states, question_attn_mask, context_attn_mask, related_region_labels=None, compute_loss=True):
        _, context_hidden_state = self.divide_embedding(encoder_last_hidden_states, question_attn_mask, context_attn_mask)
        saliency_logits = self.saliency_detector(context_hidden_state)
        if compute_loss:
            saliency_logits_reshaped = saliency_logits.reshape(-1, saliency_logits.size(-1))
            sd_loss = self.bce_loss(saliency_logits_reshaped, related_region_labels.flatten())
            return sd_loss, saliency_logits
        else:
            return saliency_logits

    def forward_sep_qa(self, question_hiddens, context_hiddens, question_attn_mask, context_attn_mask, 
                        answer_ids, answer_attn_mask, answer_labels):
        encoder_last_hidden_states, attn_mask = \
            self.combine_embedding_and_mask(question_hiddens, question_attn_mask, context_hiddens, context_attn_mask)
        
        return self.forward_text_answer(encoder_last_hidden_states, attn_mask, answer_ids, answer_attn_mask, answer_labels)


    def forward_text_answer(self, encoder_last_hidden_states, attn_mask,
                        answer_ids, answer_attn_mask, answer_labels, 
                        question_attn_mask=None, context_attn_mask=None, related_region_labels=None, 
                        saliency_logits=None, region_positions=None, now_step=None, total_step=None):

        assert len(encoder_last_hidden_states) > 0
        
        decoder_outputs = self.decoder(encoder_hidden_states=encoder_last_hidden_states, encoder_attention_mask=attn_mask, 
                    input_ids=answer_ids, attention_mask=answer_attn_mask, return_dict=True)
        decoder_last_hidden_states = decoder_outputs.last_hidden_state

        if self.t5.config.tie_word_embeddings:
            decoder_last_hidden_states = decoder_last_hidden_states * (self.model_dim**-0.5)
        logits = self.lm_head(decoder_last_hidden_states)
        logits = logits.reshape(-1, logits.size(-1))
        labels = answer_labels.flatten()
        qa_loss = self.ce_loss(logits, labels)
        return qa_loss
    

    def forward(self, question_ids, answer_ids, context_ids, imgs, bboxes,
                question_attn_mask, answer_attn_mask, context_attn_mask,
                answer_labels, segment_ids, question_segment_ids, related_region_labels, **kwargs):
        loss_dict = {}

        # Cases of separatly encoding question and page
        if self.args.page_contrast:
            question_hiddens, context_hiddens = self.two_stream_encoding(question_ids, question_attn_mask, question_segment_ids,
                context_ids, context_attn_mask, imgs, bboxes, segment_ids, return_hidden=True)

        # Jointly encoding
        if not self.args.no_cross:
            encoder_last_hidden_states, attn_mask = self.cross_encoding(question_ids, context_ids, imgs, bboxes,
                                                            question_attn_mask, context_attn_mask,
                                                            segment_ids, question_segment_ids, **kwargs)

        if self.args.page_contrast:
            if self.args.page_contrast_type == 'global':
                pc_loss, _ = self.forward_page_contrast_global(question_hiddens, context_hiddens)
                loss_dict['loss_pc'] = pc_loss
            elif self.args.page_contrast_type == 'tokenwise':
                tpc_loss, _, _ = self.forward_page_contrast_tokenwise(question_hiddens, context_hiddens, question_attn_mask, context_attn_mask)
                loss_dict['loss_tpc'] = tpc_loss
            else:
                raise NotImplementedError

        # Calculate visual answering loss
        if self.args.visual_answer:
            va_loss, saliency_logits = self.forward_salient_detection(encoder_last_hidden_states, question_attn_mask, context_attn_mask, related_region_labels)
            loss_dict['loss_va'] = va_loss

        # Calculate question answer loss
        if self.args.text_answer:
            region_positions = kwargs.get('region_positions', None)
            now_step = kwargs.get('now_step', None)
            total_step = kwargs.get('total_step', None)
            if not self.args.visual_answer:
                saliency_logits = None
            qa_loss = self.forward_text_answer(encoder_last_hidden_states, attn_mask, answer_ids, answer_attn_mask, answer_labels, 
            question_attn_mask, context_attn_mask, related_region_labels, saliency_logits, region_positions, now_step, total_step)
            loss_dict['loss_qa'] = qa_loss                   

        # Calculate total loss
        loss = 0.0
        for _, loss_value in loss_dict.items():
            loss += loss_value
        loss_dict['loss'] = loss

        return loss_dict
