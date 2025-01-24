import torch
import torch.nn as nn
from src.layers import TransformerEncoder, LayerNorm

class ViT(nn.Module):
    def __init__(
        self,
        patch_size,
        hidden_dim,
        n_heads,
        n_layers,
        n_classes,
        dropout_rate=0.1,
        positional_encoding="sinusoidal",
        max_len=1000
    ):
        super(ViT, self).__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=3, 
            out_channels=hidden_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(
                data_dim=hidden_dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                positional_encoding=positional_encoding,
                max_len=max_len
            ) for _ in range(n_layers)
        ])

        self.mlp_head = nn.Sequential(
            LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        B, C, Hp, Wp = x.shape
        x = x.reshape(B, C, Hp*Wp).transpose(1, 2).contiguous()

        cls_token = self.cls_token.to(x.device).repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        cls_final = x[:, 0, :]
        out = self.mlp_head(cls_final)
        return out