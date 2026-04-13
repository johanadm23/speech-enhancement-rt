# %% [code]
# %% [code]
# %% [code]
# %% [code]

# # VoiceBank Speech Enhancement: Definition – UNet for Log-Mel Spectrogram Enhancement
# 
# **Project Goal**: Build a deep learning model to denoise speech using the VoiceBank-DEMAND dataset.
# 
# This notebook defines a UNet-style model for speech enhancement, with log-Mel spectrograms as input.
# 
# - Input:  (B, 1, M, T) - Noisy log-Mel
# - Output: (B, 1, M, T) - Cleaned log-Mel
# 
# **Notebook Overview**:
# - UNet model takes a log-Mel spectrogram of noisy speech as input.
# - Outputs a cleaned version with the same shape.
# - Learns a direct mapping from noisy to clean.
# 
# **Next Steps**:
# - Train the neural network model
# - Evaluate and visualize performance
# 
# **Author**: Jo  
# **Date**: Aug 2025
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model(model_type='unet', **kwargs):
    """Factory function to create models"""
    
    models = {
        'unet': UNet,
        'spectral_unet': SpectralUNet,
        'conv_tasnet': ConvTasNet
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)

class UNetBlock(nn.Module):
    """UNet block with optional attention and residual connections"""
    
    def __init__(self, in_channels, out_channels, use_attention=False, dropout=0.1):
        super(UNetBlock, self).__init__()
        
        self.use_attention = use_attention
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention mechanism
        if use_attention:
            self.attention = ChannelAttention(out_channels)
        
        # Residual connection adapter
        self.residual_adapter = None
        if in_channels != out_channels:
            self.residual_adapter = nn.Conv2d(in_channels, out_channels, 
                                            kernel_size=1, bias=False)
    
    def forward(self, x):
        identity = x
        
        out = self.block(x)
        
        if self.use_attention:
            out = self.attention(out)
        
        # Residual connection
        if self.residual_adapter is not None:
            identity = self.residual_adapter(identity)
        
        return out
        
class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature refinement"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class UNet(nn.Module):
    """UNet with attention mechanisms and skip connections"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], 
                 use_attention=True, dropout=0.1):
        super(UNet, self).__init__()
        
        self.features = features
        self.use_attention = use_attention
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        prev_channels = in_channels
        for feature in features:
            self.encoder.append(
                UNetBlock(prev_channels, feature, use_attention, dropout)
            )
            prev_channels = feature
        
        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2, 
                                  use_attention, dropout)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(
                UNetBlock(feature * 2, feature, use_attention, dropout)
            )
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for idx, (upconv, decoder_layer) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            skip_connection = skip_connections[idx]
            
            # Handle size mismatches
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', 
                                align_corners=False)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = decoder_layer(concat_skip)
        
        return self.final_conv(x)

def crop_to_match(tensor, target_tensor):
    _, _, h, w = tensor.size()
    _, _, ht, wt = target_tensor.size()
    dh, dw = (h - ht) // 2, (w - wt) // 2
    return tensor[:, :, dh:dh+ht, dw:dw+wt]


# Basic CNN model

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, input_size=(128,128)):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        # Dynamically compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            dummy_out = self.pool(F.relu(self.conv1(dummy)))
            dummy_out = self.pool(F.relu(self.conv2(dummy_out)))
            flat_size = dummy_out.view(1, -1).size(1)

        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




        


class SpectralUNet(nn.Module):
    """UNet specifically designed for spectrogram enhancement"""
    
    def __init__(self, in_channels=1, out_channels=1, n_fft=512):
        super(SpectralUNet, self).__init__()
        
        self.n_fft = n_fft
        
        # Frequency-aware convolutions
        self.freq_conv = nn.Conv2d(in_channels, 32, kernel_size=(7, 3), 
                                 padding=(3, 1))
        
        # Standard UNet architecture
        self.encoder1 = self._make_encoder_block(32, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        
        self.bottleneck = self._make_encoder_block(256, 512)
        
        self.decoder3 = self._make_decoder_block(512, 256)
        self.decoder2 = self._make_decoder_block(512, 128)  # 256 + 256 from skip
        self.decoder1 = self._make_decoder_block(256, 64)   # 128 + 128 from skip
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 + 64 from skip
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Tanh()  # Ensure output is in reasonable range
        )
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                   align_corners=True)
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Initial frequency-aware processing
        x1 = self.freq_conv(x)
        
        # Encoder
        x2 = self.encoder1(x1)
        p1 = self.pool(x2)
        
        x3 = self.encoder2(p1)
        p2 = self.pool(x3)
        
        x4 = self.encoder3(p2)
        p3 = self.pool(x4)
        
        # Bottleneck
        bottleneck = self.bottleneck(p3)
        
        # Decoder
        u3 = self.upsample(bottleneck)
        u3 = torch.cat([u3, x4], dim=1)
        d3 = self.decoder3(u3)
        
        u2 = self.upsample(d3)
        u2 = torch.cat([u2, x3], dim=1)
        d2 = self.decoder2(u2)
        
        u1 = self.upsample(d2)
        u1 = torch.cat([u1, x2], dim=1)
        d1 = self.decoder1(u1)
        
        # Final processing
        out = torch.cat([d1, x1], dim=1)
        return self.final_conv(out)

# convtasnet model
class TemporalBlock(nn.Module):
    """Temporal Block for TCN"""
    
    def __init__(self, in_channels, conv_channels, out_channels, 
                 kernel_size, dilation, padding, causal=False):
        super(TemporalBlock, self).__init__()
        
        self.causal = causal
        
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        
        # Depthwise conv
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, conv_channels, eps=1e-8)
        self.dsconv = nn.Conv1d(conv_channels, conv_channels, kernel_size,
                               stride=1, padding=padding, dilation=dilation,
                               groups=conv_channels)
        
        # Pointwise conv
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, conv_channels, eps=1e-8)
        self.pwconv = nn.Conv1d(conv_channels, out_channels, 1)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
                           if in_channels != out_channels else None
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, T]
        Returns:
            output: [B, out_channels, T]
        """
        residual = x
        
        # 1x1 conv
        x = self.conv1x1(x)
        
        # Depthwise conv
        x = self.prelu1(x)
        x = self.norm1(x)
        x = self.dsconv(x)
        
        # Causal truncation
        if self.causal:
            x = x[:, :, :-self.dsconv.dilation[0]]
        
        # Pointwise conv
        x = self.prelu2(x)
        x = self.norm2(x)
        x = self.pwconv(x)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        return x + residual

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for Conv-TasNet"""
    
    def __init__(self, n_basis, enc_dim, feature_dim, hidden_dim, 
                 layer, stack, causal=False):
        super(TemporalConvNet, self).__init__()
        
        # Normalization and projection
        self.layer_norm = nn.GroupNorm(1, n_basis, eps=1e-8)
        self.bottleneck_conv = nn.Conv1d(n_basis, enc_dim, 1)
        
        # TCN blocks
        self.tcn = nn.ModuleList()
        for s in range(stack):
            for i in range(layer):
                dilated = 2 ** i
                padding = (dilated * (3 - 1)) // 2 if not causal else dilated * (3 - 1)
                self.tcn.append(
                    TemporalBlock(enc_dim, hidden_dim, feature_dim, 
                                3, dilated, padding, causal)
                )
    
    def forward(self, mixture_w):
        """
        Args:
            mixture_w: [B, n_basis, T]
        Returns:
            output: [B, enc_dim, T]
        """
        output = self.layer_norm(mixture_w)
        output = self.bottleneck_conv(output)
        
        for tcn_block in self.tcn:
            output = tcn_block(output)
        
        return output


class ConvTasNet(nn.Module):
    """Conv-TasNet architecture adapted for spectrogram enhancement"""
    
    def __init__(self, n_src=1, n_basis=512, kernel_size=16, stride=8, 
                 enc_dim=512, feature_dim=128, hidden_dim=512, layer=8, 
                 stack=3, causal=False):
        super(ConvTasNet, self).__init__()
        
        # Hyper-parameters
        self.n_src = n_src
        
        # Encoder
        self.encoder = nn.Conv1d(1, n_basis, kernel_size, stride=stride, 
                                bias=False)
        
        # Separator
        self.separator = TemporalConvNet(n_basis, enc_dim, feature_dim, 
                                       hidden_dim, layer, stack, causal)
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(n_basis, 1, kernel_size, 
                                        stride=stride, bias=False)
        
        # Mask generator
        self.mask_generator = nn.Conv1d(enc_dim, n_basis * n_src, 1)
        
    def forward(self, mixture):
        """
        Args:
            mixture: [B, T] or [B, 1, T] - input mixture
        Returns:
            separated: [B, n_src, T] - separated sources
        """
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)  # [B, T] -> [B, 1, T]
        
        mixture_w = self.encoder(mixture)  # [B, n_basis, T_encoded]
        est_mask = self.separator(mixture_w)  # [B, enc_dim, T_encoded]
        est_mask = self.mask_generator(est_mask)  # [B, n_basis*n_src, T_encoded]
        
        # Reshape mask
        est_mask = est_mask.view(mixture.shape[0], self.n_src, 
                               mixture_w.shape[1], mixture_w.shape[2])
        
        # Apply mask
        separated_w = mixture_w.unsqueeze(1) * est_mask  # [B, n_src, n_basis, T_encoded]
        
        # Decode
        separated = []
        for i in range(self.n_src):
            separated.append(self.decoder(separated_w[:, i]))
        
        separated = torch.stack(separated, dim=1)  # [B, n_src, T]
        
        return separated.squeeze(1) if self.n_src == 1 else separated


