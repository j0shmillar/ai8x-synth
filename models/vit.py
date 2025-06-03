import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=patch_size, padding=1)
        )

        H, W = img_size
        self.grid_size = (H // patch_size, W // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, d_model))
    
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)  # (B, d_model, H', W')
        H, W = x.shape[2], x.shape[3]
        n_patches = H * W
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1 + n_patches, d_model)

        if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1]:
            self.pos_embed = nn.Parameter(torch.zeros(1, x.shape[1], self.d_model).to(x.device))

        x = x + self.pos_embed

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=(28, 28), patch_size=7, in_channels=1, num_classes=10,
                 d_model=64, num_heads=4, num_layers=4, d_ff=128, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # CLS token

def create_model(**kwargs):
    return ViT(**kwargs)
