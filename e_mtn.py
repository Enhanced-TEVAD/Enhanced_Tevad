import torch
import torch.nn as nn
import torch.nn.functional as F
# from google.colab import files

import os
import numpy as np



class PDCBlock(nn.Module):
    """Pyramid Dilated Convolution block with multi-scale 1D convolutions."""
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (after fusion).
        """
        super(PDCBlock, self).__init__()
        # Three parallel convolutions with different dilation rates
        # Each produces out_channels feature maps.
        self.conv_d1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv_d3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        # After concatenation, fuse the channels back to out_channels (using kernel 1).
        self.fuse = nn.Conv1d(3*out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: Tensor of shape (batch, in_channels, seq_len)
        Returns: Tensor of shape (batch, out_channels, seq_len)
        """
        # Apply parallel dilated convolutions with ReLU activations
        out1 = F.relu(self.conv_d1(x))
        out2 = F.relu(self.conv_d2(x))
        out3 = F.relu(self.conv_d3(x))
        # Concatenate along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)  # shape: (batch, 3*out_channels, seq_len)
        # Fuse concatenated features back to desired output channels
        out = F.relu(self.fuse(out))  # shape: (batch, out_channels, seq_len)
        return out


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel-wise feature recalibration."""
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: Number of input/output channels.
            reduction: Squeeze reduction ratio (typically 16).
        """
        super(SEModule, self).__init__()
        # Squeeze: one FC layer to reduce channels, then ReLU
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        # Excitation: one FC layer to restore channels, then sigmoid
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        """
        x: Tensor of shape (batch, channels, seq_len)
        Returns: Tensor of same shape, recalibrated.
        """
        # Compute channel-wise global average: (batch, channels)
        b, c, t = x.size()
        # Squeeze: average over temporal dimension
        y = x.mean(dim=2)  # shape: (batch, channels)
        # Two FC layers with activation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))  # shape: (batch, channels), each in [0,1]
        # Reshape to (batch, channels, 1) and scale input
        y = y.view(b, c, 1)
        return x * y  # broadcasting over time dimension


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder layer (multi-head self-attention + feedforward)."""
    def __init__(self, channels, nhead=8, dim_feedforward=1024):
        """
        Args:
            channels: Number of input/output channels (d_model for transformer).
            nhead: Number of attention heads.
            dim_feedforward: Hidden size of the feed-forward network.
        """
        super(TransformerEncoderBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False  # we'll transpose manually
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        """
        x: Tensor of shape (batch, channels, seq_len)
        Returns: Tensor of same shape after self-attention.
        """
        # Permute to (seq_len, batch, channels) for the transformer
        x = x.permute(2, 0, 1)  # (seq_len, batch, channels)
        # Pass through the transformer encoder layer
        out = self.transformer(x)  # (seq_len, batch, channels)
        # Permute back to (batch, channels, seq_len)
        out = out.permute(1, 2, 0)
        return out




# visual_dir = "/content/visual_features"
# text_dir = "/content/text_features"

# # Print visual feature shapes
# print("=== Visual Feature Shapes ===")
# for fname in sorted(os.listdir(visual_dir)):
#     if fname.endswith(".npy"):
#         arr = np.load(os.path.join(visual_dir, fname))
#         print(f"{fname}: {arr.shape}")

# # Print text feature shapes
# print("\n=== Text Feature Shapes ===")
# for fname in sorted(os.listdir(text_dir)):
#     if fname.endswith(".npy"):
#         arr = np.load(os.path.join(text_dir, fname))
#         print(f"{fname}: {arr.shape}")


# class EMTN(nn.Module):
#     def __init__(self,
#                  vis_channels, txt_channels,
#                  visual_dir, text_dir,
#                  reduction=16, nhead=8, dim_feedforward=1024):
#         super().__init__()
#         self.visual_dir = visual_dir
#         self.text_dir   = text_dir

#         self.train_keys = [f"Train{i:03d}" for i in range(1,17)]
#         self.test_keys  = [f"Test{i:03d}"  for i in range(1,13)]

#         self.in_channels = vis_channels + txt_channels
#         mid1 = self.in_channels // 2
#         mid2 = self.in_channels - mid1

#         self.pdc         = PDCBlock(self.in_channels, mid1)
#         self.se          = SEModule(mid1, reduction=reduction)
#         self.conv_branch = nn.Conv1d(self.in_channels, mid2, kernel_size=3, padding=1)
#         self.transformer = TransformerEncoderBlock(mid2, nhead=nhead, dim_feedforward=dim_feedforward)

#     def load_and_preprocess(self, key):
#         vis = np.load(os.path.join(self.visual_dir, f"{key}_features.npy"))  # (T, C, Dv)
#         txt = np.load(os.path.join(self.text_dir,   f"{key}.npy"))          # (1, Dt) or (Dt,)
#         if txt.ndim == 1:
#             txt = txt.reshape(1, -1)

#         T, C, Dv = vis.shape
#         Dt = txt.shape[-1]

#         txt = np.tile(txt, (T, C, 1))  # (T, C, Dt)
#         fused = np.concatenate([vis, txt], axis=-1)  # (T, C, Dv+Dt)

#         fused = fused.transpose(2, 0, 1).reshape(Dv + Dt, T * C)  # (D, seq_len)
#         return torch.from_numpy(fused).float()

#     def forward(self, split: str):
#         keys = self.train_keys if split.lower() == "train" else self.test_keys

#         fused_seqs = [self.load_and_preprocess(key) for key in keys]
#         max_seq = max(seq.shape[1] for seq in fused_seqs)

#         padded = []
#         for seq in fused_seqs:
#             D, L = seq.shape
#             if L < max_seq:
#                 pad = torch.zeros(D, max_seq - L)
#                 seq = torch.cat([seq, pad], dim=1)
#             padded.append(seq.unsqueeze(0))  # (1, D, T)

#         x = torch.cat(padded, dim=0)  # (B, D, T)

#         f1 = self.se(self.pdc(x))                      # (B, mid1, T)
#         f2 = self.transformer(F.relu(self.conv_branch(x)))  # (B, mid2, T)

#         return torch.cat([f1, f2], dim=1) + x  # (B, D, T)

# # === USAGE ===
# visual_dir = "/content/visual_features"
# text_dir   = "/content/text_features"

# model = EMTN(
#     vis_channels=768,
#     txt_channels=768,
#     visual_dir=visual_dir,
#     text_dir=text_dir
# )

# train_out = model("train")
# print("Train output shape:", train_out.shape)  # (16, 1536, max_seq)

# test_out = model("test")
# print(" Test output shape:", test_out.shape)   # (12, 1536, max_seq)


# np.save("train_features.npy", train_out.detach().cpu().numpy())
# np.save("test_features.npy", test_out.detach().cpu().numpy())



# files.download("train_features.npy")
# files.download("test_features.npy")