import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, d, epsilon=1e-5):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * ((x - mean) / torch.sqrt(variance + self.epsilon)) + self.beta


class FFN(nn.Sequential):
    def __init__(self, hidden_dim, dropout_rate=0.1, expansion_factor=2):
        super(FFN, self).__init__()
        self.add_module("linear1", nn.Linear(hidden_dim, hidden_dim * expansion_factor))
        self.add_module("activation", nn.LeakyReLU(negative_slope=0.1))
        self.add_module("dropout", nn.Dropout(dropout_rate))
        self.add_module("linear2", nn.Linear(hidden_dim * expansion_factor, hidden_dim))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, data_dim, hidden_dim, n_heads, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        from src.attention import MultiHeadSelfAttention  # local import to avoid circular issues
        self.attention = MultiHeadSelfAttention(data_dim, hidden_dim, n_heads)
        self.norm1 = LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, dropout_rate)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))
        ffn_ouput = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_ouput))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        device = x.device
        
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32, device=device) * 
                             (-math.log(10000.0) / hidden_dim))
        
        pe = torch.zeros(seq_len, hidden_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
        return x + pe


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len):
        super(LearnedPositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.positional_encoding = nn.Embedding(max_len, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        device = x.device
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        pe = self.positional_encoding(position_ids)
        return x + pe


class TransformerEncoder(nn.Module):
    def __init__(self, data_dim, hidden_dim, n_heads, dropout_rate=0.1, positional_encoding="sinusoidal", max_len=1000):
        super(TransformerEncoder, self).__init__()
        if positional_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim)
        elif positional_encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding(hidden_dim, max_len)
        else:
            raise ValueError("Invalid positional encoding type. Choose 'sinusoidal' or 'learned'.")

        from src.attention import MultiHeadSelfAttention  # local import
        self.attention = MultiHeadSelfAttention(data_dim, hidden_dim, n_heads)
        self.norm1 = LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, dropout_rate)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.positional_encoding(x)
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
