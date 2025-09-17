import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv(nn.Module):
    """Standard Convolution Block: Conv2d + BatchNorm2d + SiLU Activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Sobel(nn.Module):
    """
    Fixed Sobel operator for gradient extraction.
    This module is not trainable.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels for x and y directions
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_kernel_x', sobel_kernel_x)
        self.register_buffer('sobel_kernel_y', sobel_kernel_y)
        
    def forward(self, x):
    
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        
        grad_x = F.conv2d(x_reshaped, self.sobel_kernel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_kernel_y, padding=1)
        
        # Calculate magnitude
        g_raw = torch.sqrt(grad_x**2 + grad_y**2)
        
        return g_raw.view(b, c, h, w)

# --- Core Modules from G²-YOLO ---

class GDS_M(nn.Module):
    """
    Gradient-Gated Dual-Stream Multi-scale Module (GDS-M).
    This module decouples features into content and structure streams and fuses them.
    
    Equation References:
    - Eq (1) G_raw: Implemented in the Sobel module.
    - Eq (2) F_structure: Implemented in the structure stream forward pass.
    - Eq (4) F_fused: Implemented at the end of the forward pass.
    """
    def __init__(self, c_in, c_out, dilation_rates=(1, 3, 5)):
        super().__init__()
        
        # --- Content Stream ---
        # A simple residual block as described in the paper
        self.content_stream = Conv(c_in, c_out, k=3, p=1)

        # --- Structure Stream ---
        self.sobel = Sobel()
        
        # Each dilated conv branch processes the single-channel raw gradient map
        # We assume the input to the structure stream is averaged over channels to produce G_raw
        num_dilated_branches = len(dilation_rates)
        c_mid_structure = c_out // 2 # Intermediate channels for structure stream
        
        self.dconvs = nn.ModuleList()
        for d in dilation_rates:
            self.dconvs.append(
                Conv(c_in, c_mid_structure // num_dilated_branches, k=3, p=d, d=d)
            )
        
        # 1x1 convolution to aggregate multi-scale structural features
        self.agg_conv = Conv(c_mid_structure, c_out, k=1)
        
        # --- Gating Mechanism ---
        # Lightweight gating network (N_gate)
        gate_channels = (c_out + c_out) # Concatenated features
        self.gate_net = nn.Sequential(
            Conv(gate_channels, gate_channels // 2, k=3, p=1),
            nn.ReLU(),
            nn.Conv2d(gate_channels // 2, c_out, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Content Stream
        f_content = self.content_stream(x)
        
        # Structure Stream
        g_raw = self.sobel(x)
        
        dilated_features = [dconv(g_raw) for dconv in self.dconvs]
        f_dilated_concat = torch.cat(dilated_features, dim=1)
        
        f_structure = self.agg_conv(f_dilated_concat)
        
        # Gradient-Gated Fusion
        f_concat = torch.cat((f_content, f_structure), dim=1)
        gate = self.sigmoid(self.gate_net(f_concat))
        
        f_fused = gate * f_content + (1 - gate) * f_structure
        
        return f_fused


class SAG(nn.Module):
    """
    Structural Alignment Gate (SAG).
    Aligns deep semantic features with shallow structural priors.
    
    Equation References:
    - Eq (7) V_q, V_k: Vector field projection.
    - Eq (8) A(h,w): Cosine similarity calculation.
    - Eq (9) X_guided: Final modulation.
    """
    def __init__(self, c_query, c_key, k_refine=3):
        super().__init__()
        self.sobel = Sobel()

        # Convolutions to project features into 2D vector fields (output channels = 2)
        self.conv_q = nn.Conv2d(c_query, 2, kernel_size=1, bias=False)
        self.conv_k = nn.Conv2d(c_key, 2, kernel_size=1, bias=False)
        
        # Small convolutional network to refine the alignment map
        padding = k_refine // 2
        self.refine_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=k_refine, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.eps = 1e-6

    def forward(self, x_query, x_key):
        # Project features into 2D vector fields (V_q, V_k)
        v_q = self.conv_q(x_query)
        
        grad_k = self.sobel(x_key)
        v_k = self.conv_k(grad_k)
        
        # Compute cosine similarity to get the spatial alignment map A
        # F.cosine_similarity computes along a given dimension (dim=1 for channels)
        alignment_map = F.cosine_similarity(v_q, v_k, dim=1, eps=self.eps).unsqueeze(1)
        
        # Refine and scale the map to produce the final gate G_align
        g_align = self.refine_gate(alignment_map)
        
        # Modulate the original semantic feature
        x_guided = x_query * g_align
        
        return x_guided
