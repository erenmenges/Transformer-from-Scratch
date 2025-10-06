import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.scale = math.sqrt(d_model)     
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            (torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)


        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x *= self.scale
        x += self.pe[:, :x.size(1), :]
        return self.dropout(x)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, d_model: int, mask: torch.Tensor):
    scores = (Q @ K.transpose(-2,-1)) /math.sqrt(d_model)

    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(mask, float('-inf'))

    attn = math.softmax(scores, dim=1)
    out = attn @ V

    return (out, attn)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // n_heads
        self.scale = (self.d_k) ** (-0.5)

        self.W_Q = nn.Linear(d_model, n_heads * self.d_k, bias=True)
        self.W_K = nn.Linear(d_model, n_heads * self.d_k, bias=True)
        self.W_V = nn.Linear(d_model, n_heads * self.d_k, bias=True)

        self.W_O = nn.Linear(n_heads * self.d_k, d_model, bias=True)

    
    def forward(self, x_q, x_kv, mask=None):
        B, Lq, _ = x_q.shape
        B2, Lk, _ = x_kv.shape
        assert B == B2

        q = self.W_Q(x_q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)  
        k = self.W_K(x_kv).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2) 
        v = self.W_V(x_kv).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2) 

        out, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)  # out: (B,h,Lq,d_k)

        # 3) concat heads + output proj
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.n_heads * self.d_k)  # (B,Lq,d_model)
        y = self.W_O(out)  # (B,Lq,d_model)

        return y, attn


