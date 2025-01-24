import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, x_to_dim, x_from_dim, hidden_dim):  
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_Q = nn.Linear(x_to_dim, hidden_dim)
        self.W_K = nn.Linear(x_from_dim, hidden_dim)
        self.W_V = nn.Linear(x_from_dim, hidden_dim)
        
    def forward(self, x_to, x_from):
        Q = self.W_Q(x_to)
        K = self.W_K(x_from)
        V = self.W_V(x_from)
        similarity = torch.einsum("bqd,bkd->bqk", Q, K)
        attention_weights = F.softmax(similarity / math.sqrt(self.hidden_dim), dim=-1)
        output = torch.einsum("bqk,bkd->bqd", attention_weights, V)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, x_to_dim, x_from_dim, hidden_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dim_per_head = hidden_dim // n_heads
        self.all_attentions_blocks = nn.ModuleList([
            Attention(x_to_dim, x_from_dim, self.dim_per_head) for _ in range(self.n_heads)
        ])
        
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_to, x_from):
        result = []
        for i in range(self.n_heads):
            output = self.all_attentions_blocks[i](x_to, x_from)
            result.append(output)
        concatened = torch.cat(result, dim=-1)
        output = self.W_O(concatened)
        return output


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, x_dim, hidden_dim, n_heads):
        super().__init__(x_dim, x_dim, hidden_dim, n_heads)

    def forward(self, x):
        return super().forward(x, x)


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, x_to_dim, x_from_dim, hidden_dim, n_heads):
        super().__init__(x_to_dim, x_from_dim, hidden_dim, n_heads)

    def forward(self, x_to, x_from):
        return super().forward(x_to, x_from)