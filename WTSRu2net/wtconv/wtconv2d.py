import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .util import wavelet


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=1, bias=True,
                 wt_levels=2, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels, "in_channels must equal out_channels"

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.mid_channels = in_channels * 2
        self.stride = stride

        # ---- Fixed subband weights: shape [wt_levels, 1, 4, 1, 1] ----
        fixed = torch.tensor([
            [0.7, 0.05, 0.05, 0.2]
        ] * self.wt_levels, dtype=torch.float32)
        # reshape to (wt_levels, 1, 4, 1, 1) for proper broadcast on subbands
        fixed = fixed.view(self.wt_levels, 1, 4, 1, 1)
        self.register_buffer('subband_weight', fixed)

        # ---- Wavelet filters ----
        wt_filter, iwt_filter = wavelet.create_wavelet_filter(
            wt_type, self.mid_channels, self.mid_channels, torch.float
        )
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=True)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=True)

        # ---- Base and local convolutions ----
        self.base_conv = nn.Conv2d(
            in_channels, self.mid_channels,
            kernel_size, padding='same', stride=1,
            dilation=1, groups=in_channels, bias=bias
        )
        self.local_conv = nn.Conv2d(
            self.mid_channels, self.mid_channels,
            kernel_size=3, padding=1, bias=bias
        )
        self.base_scale = _ScaleModule([1, self.mid_channels, 1, 1])

        # ---- Wavelet pathway convolutions ----
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(
                self.mid_channels * 4,
                self.mid_channels * 4,
                kernel_size,
                padding='same',
                stride=1,
                bias=False
            ) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, self.mid_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

        # ---- Channel reducer ----
        self.channel_reduce = nn.Conv2d(
            self.mid_channels, out_channels, kernel_size=1, bias=bias
        )

        # ---- Optional stride ----
        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride) if stride > 1 else None

    def forward(self, x):
        # Base convolution and scaling
        x_doubled = self.base_scale(self.base_conv(x))

        # Prepare containers
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr = x_doubled

        # Wavelet decomposition levels
        for i in range(self.wt_levels):
            B, C, H, W = curr.shape
            shapes_in_levels.append((H, W))

            # pad if odd spatial dims
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                curr = F.pad(curr, (0, pad_w, 0, pad_h))

            # transform to subbands: shape [B, C, 4, H//2, W//2]
            subbands = wavelet.wavelet_transform(curr, self.wt_filter)

            # apply fixed subband weights: broadcast over channel dim
            # subband_weight[i]: [1,4,1,1] aligned to dim2
            w = self.subband_weight[i].unsqueeze(0)  # [1,1,4,1,1]
            subbands = subbands * w

            # split LL and HH subbands
            ll = subbands[:, :, 0, :, :]
            hh = subbands[:, :, 1:4, :, :]
            x_ll_in_levels.append(ll)
            x_h_in_levels.append(hh)

            # process all subbands together
            B, C, L, h, w_ = subbands.shape
            merged = subbands.reshape(B, C * 4, h, w_)
            merged = self.wavelet_scale[i](self.wavelet_convs[i](merged))
            subbands = merged.reshape(B, C, 4, h, w_)

            curr = ll

        # Reconstruction
        prev = torch.zeros_like(x_ll_in_levels[-1])
        for i in range(self.wt_levels - 1, -1, -1):
            ll = x_ll_in_levels[i]
            hh = x_h_in_levels[i]
            H, W = shapes_in_levels[i]

            # add residual low-pass
            ll = ll + prev
            stacked = torch.cat([ll.unsqueeze(2), hh], dim=2)
            prev = wavelet.inverse_wavelet_transform(stacked, self.iwt_filter)
            prev = prev[:, :, :H, :W]

        x_tag = prev

        # local feature fusion
        x_local = self.local_conv(x_doubled)
        x_doubled = x_doubled + x_local

        # combine and reduce channels
        x_out = x_doubled + x_tag
        x_out = self.channel_reduce(x_out)

        if self.do_stride:
            x_out = self.do_stride(x_out)
        return x_out


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return x * self.weight
