import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        # Reshape and linear projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)

        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        attn_output = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x

class PromptEncoder(nn.Module):
    def __init__(self, prompt_dim, num_prompts):
        super(PromptEncoder, self).__init__()
        self.prompt_embeddings = nn.Parameter(torch.zeros(num_prompts, prompt_dim))
        nn.init.normal_(self.prompt_embeddings, std=0.02)

    def forward(self, batch_size):
        # Return prompt embeddings repeated for each batch
        return self.prompt_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

class ImageEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super(ImageEncoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Convolutional layers to extract features
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, embed_dim, kernel_size=3, padding=1)
        # Max pooling to downsample
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, _, H, W = x.shape
        # Feature extraction with downsampling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        # Reshape to sequence of patches
        _, C, h, w = x.shape
        x = x.reshape(batch_size, C, -1).permute(0, 2, 1)  # (B, hw, C)

        return x

class ImageDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels):
        super(ImageDecoder, self).__init__()
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Convolutional layers
        self.conv1 = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_size):
        batch_size, seq_len, embed_dim = x.shape
        # Calculate the feature map dimensions after encoder downsampling
        h = input_size[0] // 8  # Downsampled 3 times in encoder
        w = input_size[1] // 8

        # Reshape sequence to 2D feature map
        x = x.permute(0, 2, 1).reshape(batch_size, embed_dim, h, w)
        # Upsampling and convolutions
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.upsample(x)
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.relu(self.conv3(x))
        x = self.conv_out(x)

        return x

class PromptIR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 embed_dim=512,
                 num_heads=8,
                 num_transformer_layers=6,
                 ff_dim=2048,
                 num_prompts=10,
                 dropout=0.1):
        super(PromptIR, self).__init__()

        # Image encoder
        self.encoder = ImageEncoder(in_channels, embed_dim)
        # Prompt encoder
        self.prompt_encoder = PromptEncoder(embed_dim, num_prompts)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_layers)
        ])
        # Image decoder
        self.decoder = ImageDecoder(embed_dim, out_channels)

    def forward(self, x):
        batch_size, _, H, W = x.shape
        input_size = (H, W)
        # Encode the input image
        image_features = self.encoder(x)
        # Get prompt embeddings
        prompt_embeddings = self.prompt_encoder(batch_size)
        # Concatenate prompt embeddings with image features
        features = torch.cat([prompt_embeddings, image_features], dim=1)
        # Add positional encoding
        features = self.pos_encoder(features)
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            features = transformer_block(features)
        # Separate prompt embeddings and image features
        image_features = features[:, prompt_embeddings.size(1):, :]
        # Decode to restored image
        output = self.decoder(image_features, input_size)
        return output

def build_model(config=None):
    if config is None:
        config = {
            'in_channels': 3,
            'out_channels': 3,
            'embed_dim': 512,
            'num_heads': 8,
            'num_transformer_layers': 6,
            'ff_dim': 2048,
            'num_prompts': 10,
            'dropout': 0.1
        }
    return PromptIR(
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('out_channels', 3),
        embed_dim=config.get('embed_dim', 512),
        num_heads=config.get('num_heads', 8),
        num_transformer_layers=config.get('num_transformer_layers', 6),
        ff_dim=config.get('ff_dim', 2048),
        num_prompts=config.get('num_prompts', 10),
        dropout=config.get('dropout', 0.1)
    )
