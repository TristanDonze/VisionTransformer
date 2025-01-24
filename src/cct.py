import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import TransformerEncoderBlock, LayerNorm

class ConvPatchEmbedding(nn.Module):
    def __init__(self, n_layers, kernel_size, hidden_dim):
        super().__init__()
        layers = []
        in_channels = 3
        for i in range(n_layers):
            out_channels = hidden_dim if i == (n_layers - 1) else hidden_dim // 2
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)
        return x


class SeqPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = self.attn_weights(x)
        weights = F.softmax(weights, dim=1)
        x = (weights * x).sum(dim=1)
        return x


class CCT(nn.Module):
    def __init__(
        self,
        n_conv_layers,
        kernel_size,
        n_transformer_layers,
        hidden_dim,
        n_heads,
        n_classes,
        dropout_rate=0.1
    ):
        super().__init__()
        self.tokenizer = ConvPatchEmbedding(n_conv_layers, kernel_size, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                data_dim=hidden_dim, 
                hidden_dim=hidden_dim, 
                n_heads=n_heads, 
                dropout_rate=dropout_rate
            )
            for _ in range(n_transformer_layers)
        ])

        self.seqpool = SeqPool(hidden_dim)
        self.mlp_head = nn.Sequential(
            LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        x = self.tokenizer(x)
        for block in self.blocks:
            x = block(x)
        x = self.seqpool(x)
        x = self.mlp_head(x)
        return x
