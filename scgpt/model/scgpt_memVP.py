from .model import TransformerModel
import torch.nn as nn
from torch import Tensor
import torch
from typing import Dict, Mapping, Optional, Tuple, Any, Union,List
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from transformers import AutoModel
def simple_rms_norm(hidden_states, eps=1e-6):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + eps)
class Adapter(nn.Module):
    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
    ):
        super().__init__()
        if hidden_dim > 0:
            self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
            self.fc2 = nn.Linear(hidden_dim, in_features, bias=False)
            self.hidden_dim = hidden_dim
            nn.init.zeros_(self.fc2.weight)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, vis_weight,vis_mask):
        # print(vis_mask.shape)
        with autocast():
            if vis_weight is not None:
                # print(f"x1 max: {x.max().item()}, min: {x.min().item()}")
                x = x @ (vis_weight[0] + vis_weight[1]).permute(0, 2, 1)
                x = self.dropout(F.silu(x))#attention_weight
                # x=x*vis_mask.unsqueeze(-2)
                # print(vis_weight[0])
                # print(f"x2 max: {x.max().item()}, min: {x.min().item()}")
                x = x @ (vis_weight[0] + vis_weight[2])
                # print(f"x3 max: {x.max().item()}, min: {x.min().item()}")
            # else:
            #     x = self.fc1(x)
            #     x = self.dropout(F.gelu(x))
            #     x = self.fc2(x)
        return x
class Projector(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=128,
            out_features=4096
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        with autocast():
            x = self.fc2(F.silu(self.fc1(x)))
        return x
from torch.nn import TransformerEncoderLayer

class TransformerEncoderLayerWithAdapter(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, d_hid, dropout, adapter_dim=8):
        super(TransformerEncoderLayerWithAdapter, self).__init__(d_model, nhead, d_hid, dropout, batch_first=True)
        self.adapter = Adapter(d_model, adapter_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None,text_feature=None,text_mask=None):
        output = super(TransformerEncoderLayerWithAdapter, self).forward(src, src_mask, src_key_padding_mask)
        # output_after_min = torch.min(output)
        # output_after_max = torch.max(output)
        # output_before_min = torch.min(simple_rms_norm(output))
        # output_before_max = torch.max(simple_rms_norm(output))
        output =output+ self.adapter(simple_rms_norm(output),text_feature,text_mask)*0.2
        # print(f"Before normalization: Min = {output_before_min.item():.4f}, Max = {output_before_max.item():.4f}")
        # print(f"After  normalization: Min = {output_after_min.item():.4f}, Max = {output_after_max.item():.4f}")
        return output

class MyTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)


    def forward(self, src, mask=None, src_key_padding_mask=None, text_feature=None, adapter_mask=None):
        output = src
        # print(mask)
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, text_feature=text_feature, text_mask=adapter_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output

class scgpt_memvp(TransformerModel):
    def __init__(
            self,
            ntoken: int,
            d_model: int,
            nhead: int,
            d_hid: int,
            nlayers: int,
            nlayers_cls: int = 3,
            n_cls: int = 1,
            vocab: Any = None,
            dropout: float = 0.5,
            pad_token: str = "<pad>",
            pad_value: int = 0,
            do_mvc: bool = False,
            do_dab: bool = False,
            use_batch_labels: bool = False,
            num_batch_labels: Optional[int] = None,
            domain_spec_batchnorm: Union[bool, str] = False,
            input_emb_style: str = "continuous",
            n_input_bins: Optional[int] = None,
            cell_emb_style: str = "cls",
            mvc_decoder_style: str = "inner product",
            ecs_threshold: float = 0.3,
            explicit_zero_prob: bool = False,
            use_fast_transformer: bool = False,
            fast_transformer_backend: str = "flash",
            pre_norm: bool = False,
            text_encoder_path:str ="microsoft/deberta-v3-base",
    ):
        super(scgpt_memvp, self).__init__( ntoken, d_model, nhead, d_hid, nlayers, nlayers_cls, n_cls, vocab, dropout,
                                           pad_token, pad_value,
                                           do_mvc, do_dab, use_batch_labels, num_batch_labels, domain_spec_batchnorm,
                                           input_emb_style, n_input_bins, cell_emb_style, mvc_decoder_style, ecs_threshold,
                                           explicit_zero_prob, use_fast_transformer, fast_transformer_backend, pre_norm )
        encoder_layers_with_adapter = TransformerEncoderLayerWithAdapter(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = MyTransformerEncoder(encoder_layers_with_adapter, nlayers)
        self.text_encoder=AutoModel.from_pretrained(text_encoder_path)
        self.text_encoder.requires_grad_(False)
        self.adapter_proj = Projector(in_features=self.text_encoder.config.hidden_size, out_features=d_hid)
        self.adapter_emb1 = nn.Parameter(torch.randn(1, 256, d_hid) * 0.02)
        self.adapter_emb2 = nn.Parameter(torch.zeros(1, 256, d_hid))

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        # print(mask)
        text_feature=self.text_encoder(text, attention_mask=mask).last_hidden_state
        # print(text_feature.last_hidden_state.shape)
        text_feature=self.adapter_proj(text_feature)
        expanded_mask = mask.unsqueeze(-1).expand_as(text_feature)
        text_feature =text_feature*expanded_mask
        text_feature=[text_feature*0.01,self.adapter_emb1,self.adapter_emb2]
        transformer_output = self._encode(
            src, values, src_key_padding_mask, batch_labels,text_feature=text_feature,mask=mask
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        output = {}
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if CCE:
            cell1 = cell_emb
            transformer_output2 = self._encode(
                src, values, src_key_padding_mask, batch_labels
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        return output
    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
        text_feature: Optional[List] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        self._check_batch_labels(batch_labels)

        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask,text_feature=text_feature,adapter_mask=mask
        )
        return output  # (batch, seq_len, embsize)
