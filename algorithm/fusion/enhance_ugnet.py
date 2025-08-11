# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import math
import torch.nn.init as init
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import torch.nn.functional as F

from .graph_algo import *

"""
Implementation of UGnet
Tcnblock: extract time feature
SpatialBlock: extract the spatial feature
"""

def TimeEmbedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def calculate_pearson_correlation(x: torch.Tensor):
    """
    Calculate Pearson correlation between feature channels across all batches and nodes.
    Args:
        x: Input tensor of shape (B, F, V, T)
    Returns:
        corr_matrix: Pearson correlation matrix of shape (F, F)
        correlated_pairs: List of tuples containing highly correlated feature indices
    """
    B, F, V, T = x.shape
    
    # Reshape and normalize
    x_flat = x.permute(0, 2, 1, 3).reshape(-1, F, T)  # (B*V, F, T)
    x_norm = (x_flat - x_flat.mean(dim=-1, keepdim=True)) / (x_flat.std(dim=-1, keepdim=True) + 1e-8)
    
    # Compute correlation matrix (average across all batches and nodes)
    corr_matrix = torch.matmul(x_norm, x_norm.transpose(1, 2)) / T  # (B*V, F, F)
    corr_matrix = corr_matrix.mean(dim=0)  # (F, F)
    
    # Find strongly correlated pairs (absolute correlation > 0.7)
    corr_pairs = []
    for i in range(F):
        for j in range(i+1, F):
            if abs(corr_matrix[i,j]) > 0.7:  # Threshold can be adjusted
                corr_pairs.append((i,j))
    
    return corr_matrix, corr_pairs


class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x: [b, c_in, time, n_nodes]
        # Lk: [3, n_nodes, n_nodes]
        if len(Lk.shape) == 2: # if supports_len == 1:
            Lk=Lk.unsqueeze(0)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta,
                            x_c) + self.b  # [b, c_out, time, n_nodes]
        return torch.relu(x_gc + x)

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, : -self.chomp_size]


class TcnBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation_size=1, droupout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(3, self.kernel_size), padding=(1, self.padding), dilation=(1, self.dilation_size))

        self.chomp = Chomp(self.padding)
        self.drop =  nn.Dropout(droupout)

        self.net = nn.Sequential(self.conv, self.chomp, self.drop)

        self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None


    def forward(self, x):
        # x: (B, C_in, V, T) -> (B, C_out, V, T)
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)

        return out + x_skip

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, config, kernel_size=3):
        """
        :param c_in: in channels
        :param c_out: out channels
        :param kernel_size:
        TCN convolution
            input: (B, c_in, V, T)
            output:(B, c_out, V, T)
        """
        super().__init__()
        self.tcn1 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.tcn2 = TcnBlock(c_out, c_out, kernel_size=kernel_size)
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1,1))
        self.t_conv = nn.Conv2d(config.d_h, c_out, (1,1))
        self.spatial = SpatialBlock(config.supports_len, c_out, c_out)

        self.norm = nn.LayerNorm([config.V, c_out])
    def forward(self, x, t, A_hat):
        # x: (B, c_in, V, T), return (B, c_out, V, T)

        h = self.tcn1(x)

        h += self.t_conv(t[:, :, None, None])

        h = self.tcn2(h)

        h = self.norm(h.transpose(1,3)).transpose(1,3) # (B, c_out, V, T)

        h = h.transpose(2,3) #(B, c_out, V, T)
        h = self.spatial(h, A_hat).transpose(2,3) # (B, c_out, V, T)
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        """
        :param c_in: in channels, out channels
        :param c_out:
        """
        super().__init__()
        self.res = ResidualBlock(c_in, c_out, config, kernel_size=3)

    def forward(self, x, t, supports):
        # x: (B, c_in, V, T), return (B, c_out, V, T)

        return self.res(x, t, supports)

class Downsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in,  kernel_size= (1,3), stride=(1,2), padding=(0,1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, supports):
        _ = t
        _ = supports
        return self.conv(x)


class  UpBlock(nn.Module):
    def __init__(self, c_in, c_out, config):
        super().__init__()
        self.res = ResidualBlock(c_in + c_out, c_out, config, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res(x, t, supports)

class Upsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_in, (1, 4), (1, 2), (0, 1))

    def forward(self, x, t, supports):
        _ = t
        _ = supports
        return  self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, c_in, config):
        super().__init__()
        self.res1 = ResidualBlock(c_in, c_in, config, kernel_size=3)
        self.res2 = ResidualBlock(c_in, c_in, config, kernel_size=3)

    def forward(self, x, t, supports):
        x = self.res1(x, t, supports)

        x = self.res2(x, t, supports)

        return x


class EnhanceUGnet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        d_h = self.d_h = config.d_h
        self.T_p = config.T_p
        self.T_h = config.T_h
        T = self.T_p + self.T_h
        self.F = config.F

        self.n_blocks = config.get('n_blocks', 2)

        # number of resolutions
        n_resolutions = len(config.channel_multipliers)

        # first half of U-Net = decreasing resolution
        down = []
        # number of channels
        out_channels = in_channels = self.d_h
        for i in range(n_resolutions):
            out_channels = in_channels * config.channel_multipliers[i]
            for _ in range(self.n_blocks):
                down.append(DownBlock(in_channels, out_channels, config))
                in_channels = out_channels

            # down sample at all resolution except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, config)

        # #### Second half of U-Net - increasing resolution
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(self.n_blocks):
                up.append(UpBlock(in_channels, out_channels, config))

            out_channels = in_channels // config.channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels, config))
            in_channels = out_channels
            # up sample at all resolution except last
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        self.x_proj = nn.Conv2d(self.F, self.d_h, (1,1))
        self.out = nn.Sequential(nn.Conv2d(self.d_h, self.F, (1,1)),
                                 nn.Linear(2 * T, T),)
        # for gcn
        a1 = asym_adj(config.A)
        a2 = asym_adj(np.transpose(config.A))
        self.a1 = torch.from_numpy(a1).to(config.device)
        self.a2 = torch.from_numpy(a2).to(config.device)
        config.supports_len = 2
        
        # for feature attehntion
        self.attention = nn.MultiheadAttention(self.F, num_heads=4, batch_first=True)
        
        # DWT and IDWT layers
        self.dwt = DWT1DForward(wave='db1', J=1, mode='symmetric')
        self.idwt = DWT1DInverse(wave='db1')
        
        self.weight_0 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.weight_2 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
    
    




    def forward(self, x: torch.Tensor, t: torch.Tensor, c):
        """
        :param x: x_t of current diffusion step, (B, F, V, T)
        :param t: diffsusion step
        :param c: condition information
            used information in c:
                x_masked: (B, F, V, T)
        :return:
        """

        x_masked, pos_w, pos_d = c  # x_masked: (B, F, V, T), pos_w: (B,T,1,1), pos_d: (B,T,1,1)
        corr_matrix, corr_pairs = calculate_pearson_correlation(x)
        print(f"Feature correlation matrix:\n{corr_matrix.detach().cpu().numpy()}")
        print(f"Highly correlated feature pairs: {corr_pairs}")
                
        # (DWT) for each node and feature channel
        B, feature_dim, V, T = x.shape
        x = x.permute(0, 3, 1, 2)  # Change to (B, T, F, V) for DWT

        # Pad x to make the length of T a power of 2
        padded_length = 2 ** (T - 1).bit_length()
        padding = padded_length - T
        padding_left = padding // 2
        padding_right = padding - padding_left
        x = F.pad(x, (0, 0, 0, 0, padding_left, padding_right), mode='constant')  # x:(B, padded_length, F, V)
        
        x_reshaped = x.permute(0, 3, 2, 1).contiguous().view(B * V, feature_dim, padded_length)          # x_reshaped：(B*V, F, padded_length)
        yl, yhs = self.dwt(x_reshaped)                                        # yl: (B*V, F, padded_length//2), yhs: [(B*V, F, padded_length//2)]                
        yl_reshape = yl.view(B, V, feature_dim, -1).permute(0, 2, 1, 3)  # yl: (B, F, V, padded_length//2)
        yhs_reshape = [yh.view(B, V, feature_dim, -1).permute(0, 2, 1, 3) for yh in yhs]  # yhs: [(B, F, V, padded_length//2)]
        
        # Step 2: Feature Fusion using Attention Mechanism
       
        # Process high-frequency components with data-driven fusion
        yhs_fused = []
        for yh in yhs_reshape:
            yh_fused = yh.clone()  # Start with original features
            
            # Apply fusion only to correlated pairs
            for feat1, feat2 in corr_pairs:
                # Create learnable weights for each correlated pair
                if not hasattr(self, f'weight_{feat1}_{feat2}'):
                    # Initialize weights if they don't exist
                    setattr(self, f'weight_{feat1}_{feat2}', 
                        nn.Parameter(torch.tensor(0.5, requires_grad=True)))
                
                weight = torch.sigmoid(getattr(self, f'weight_{feat1}_{feat2}'))
                
                # Apply weighted combination
                yh_fused[:, feat1, :, :] = weight * yh[:, feat1, :, :] + (1-weight) * yh[:, feat2, :, :]
                yh_fused[:, feat2, :, :] = (1-weight) * yh[:, feat1, :, :] + weight * yh[:, feat2, :, :]
            
            yhs_fused.append(yh_fused.view(B * V, feature_dim, padded_length // 2))

        '''
        # Step 2: To demonstrate our feature fusion approach using attention mechanisms, let's consider a 4-dimensional input tensor x. 
                  Through correlation analysis, we observe that dimensions 0 and 2 exhibit strong feature correlations, while dimensions 1 and 3 remain uncorrelated.
        
        # Process high-frequency components
        yhs_fused = []
        for yh in yhs_reshape:
            yh_fused = torch.zeros(B, feature_dim, V, padded_length // 2)
            
            # Use sigmoid to constrain weights to [0, 1] range
            weight_0 = torch.sigmoid(self.weight_0)
            weight_2 = torch.sigmoid(self.weight_2)
            
            for b in range(B):  # Iterate over batch dimension
                for v in range(V):  # Iterate over nodes dimension
                    # Weighted combination of feature 0 and 2
                    yh_fused[b, 0, v, :] = weight_0*yh[b, 0, v, :] + (1-weight_0)*yh[b, 2, v, :]
                    yh_fused[b, 2, v, :] = (1-weight_2)*yh[b, 0, v, :] + weight_2*yh[b, 2, v, :]
                    # Keep feature 1 and 3 unchanged
                    yh_fused[b, 1, v, :] = yh[b, 1, v, :]
                    yh_fused[b, 3, v, :] = yh[b, 3, v, :]
            
            yhs_fused.append(yh_fused)
        yhs_fused = [yh.view(B * V, feature_dim, padded_length // 2) for yh in yhs_fused]
        '''
        
        
        # Step 3: Inverse 3D-DWT (IDWT) to enhance feature representation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        yl_fused = yl.to(device)
        yhs_fused = [yh.to(device) for yh in yhs_fused]
        # combined = [yl] + yhs
        combined = [yl_fused] + yhs_fused
        out = self.idwt((combined[0], combined[1:]))

        # Remove padding
        padding_left = (padded_length - T) // 2
        padding_right = padded_length - T - padding_left
        feature_reconstructed = out[:, :, padding_left:padded_length-padding_right]
        enhanced_feature = feature_reconstructed.contiguous().reshape(B, feature_dim, V, T)   #enhanced_x:(B, 1, V, T)
        
        # spatio-temporal feature fusion
        x = torch.cat((x_masked, enhanced_feature), dim=3) # (B, F, V, 2 * T)
        
        # x = torch.cat((x, x_masked), dim=3) # (B, F, V, 2 * T)

        x = self.x_proj(x)

        t = TimeEmbedding(t, self.d_h)

        h = [x]

        supports = torch.stack([self.a1, self.a2])

        for m in self.down:
            x = m(x, t, supports)
            h.append(x)

        x = self.middle(x, t, supports)

        for m in self.up:
            if isinstance(m,  Upsample):
                x = m(x, t, supports)
            else:
                s =h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x,t, supports)

        e = self.out(x)
        return e

