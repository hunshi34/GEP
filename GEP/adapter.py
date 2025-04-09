import torch
from torch import nn
from torch.nn import functional as F
import GEP
from typing import Optional, Tuple
from torch.cuda.amp import autocast



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

    def forward(self, x, vis_weight):
        with autocast():
            if vis_weight is not None:
                x = x @ (vis_weight[0] + vis_weight[1]).permute(0, 2, 1)
                x = self.dropout(F.silu(x))#attention_weight
                attention_weight=x.detach().clone()
                x = x @ (vis_weight[0] + vis_weight[2])

            else:
                x = self.fc1(x)
                x = self.dropout(F.gelu(x))
                x = self.fc2(x)
        return (x,attention_weight)


def forward_llama(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                  vis_weight):
    if self.training and self.gradient_checkpointing:
        h = x + torch.utils.checkpoint.checkpoint(self.attention, self.attention_norm(x), start_pos, freqs_cis, mask)
        h_norm = self.ffn_norm(h)
        out_mid=self.adapter_mlp(h_norm*self.s1,vis_weight)
        out = h + torch.utils.checkpoint.checkpoint(self.feed_forward, h_norm) + out_mid[0] * self.s
    else:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out_mid = self.adapter_mlp(self.ffn_norm(h)*self.s1, vis_weight)
        out = h + self.drop_path(
            self.feed_forward((self.ffn_norm(h))) + out_mid[0] * self.s)
    return out,out_mid[1]





def set_Llama_Adapter(model, s=1, gradient_checkpointing=False,s1=1):
    for _ in model.children():
        if type(_) == GEP.model.TransformerBlock:
            _.adapter_mlp = Adapter(_.dim, hidden_dim=0)
            _.s = s
            _.s1= s1
            _.gradient_checkpointing = gradient_checkpointing
            bound_method = forward_llama.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Llama_Adapter(_, s, gradient_checkpointing=gradient_checkpointing,s1=s1)



