import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utils
# ------------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, g=1, act=nn.ReLU(inplace=True)):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = k // 2
            else:
                p = (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = act
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """Depthwise(k) + Pointwise(1x1)."""
    def __init__(self, in_c, out_c, k=3, s=1, p=None, act=nn.ReLU(inplace=True)):
        super().__init__()
        if p is None:
            if isinstance(k, int):
                p = k // 2
            else:
                p = (k[0] // 2, k[1] // 2)
        self.dw = ConvBNAct(in_c, in_c, k, s, p, g=in_c, act=act)
        self.pw = ConvBNAct(in_c, out_c, 1, 1, 0, act=act)
    def forward(self, x):
        return self.pw(self.dw(x))


# ------------------------------
# Dual-gate fusers
# ------------------------------
class SDGF(nn.Module):
    """Spatial Dual-Gate Fusion: fuse two spatial gates then apply."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable mix

    def forward(self, x, g1, g2):
        a = torch.clamp(self.alpha, 0, 1)
        g = a * g1 + (1 - a) * g2
        return x * g


class CDGF(nn.Module):
    """Channel Dual-Gate Fusion: fuse two channel gates then apply."""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj  = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.sig   = nn.Sigmoid()

    def forward(self, x, g1, g2):
        a = torch.clamp(self.alpha, 0, 1)
        g = a * g1 + (1 - a) * g2
        g = self.sig(self.proj(g))
        return x * g


# ------------------------------
# Attention modules
# ------------------------------
class ASA(nn.Module):
    """
    Adaptive Spatial Attention (dual path).
    Path-1: small kernel (default 3x3)
    Path-2: large kernel with dilation (default 7x7, dilation=2)
    """
    def __init__(self, k_small=3, k_large=7, dilation_large=2):
        super().__init__()
        p_small = k_small // 2
        p_large = dilation_large * (k_large // 2)

        self.conv_small = nn.Sequential(
            nn.Conv2d(2, 1, k_small, padding=p_small, bias=False),
            nn.Sigmoid()
        )
        self.conv_large = nn.Sequential(
            nn.Conv2d(2, 1, k_large, padding=p_large, dilation=dilation_large, bias=False),
            nn.Sigmoid()
        )
        self.fuse = SDGF()

    def forward(self, x):
        mean_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        s = torch.cat([mean_map, max_map], dim=1)  # B x 2 x H x W
        g1 = self.conv_small(s)
        g2 = self.conv_large(s)
        return self.fuse(x, g1, g2)


class ACA(nn.Module):
    """Channel attention with two pooling scales; unchanged (ringan & efektif)."""
    def __init__(self, channels, r=16):
        super().__init__()
        hidden = max(channels // r, 4)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.sig = nn.Sigmoid()
        self.fuse = CDGF(channels)

    def forward(self, x):
        g1 = self.sig(self.mlp1(self.pool1(x)))                     # B x C x 1 x 1
        p2 = self.pool2(x)
        g2 = self.sig(self.mlp2(F.adaptive_avg_pool2d(p2, 1)))      # reduce to 1x1
        return self.fuse(x, g1, g2)


# ------------------------------
# Core blocks (HAM/MAF)
# ------------------------------
class HAM(nn.Module):
    """
    Hybrid Attention Module (ringan).
    Perubahan: local depthwise conv kini bisa large-kernel (k_local).
    """
    def __init__(self, channels, k_local=3):
        super().__init__()
        self.local = DepthwiseSeparable(channels, channels, k=k_local, act=nn.SiLU(inplace=True))
        self.aca   = ACA(channels, r=8)
        self.asa   = ASA(k_small=3, k_large=7, dilation_large=2)  # ASA diperkuat (3x3 + 7x7 dilated)
        self.proj  = ConvBNAct(channels, channels, k=1, act=nn.Identity())

    def forward(self, x):
        y = self.local(x)
        y = self.aca(y)
        y = self.asa(y)
        y = self.proj(y)
        return x + y


class MAF(nn.Module):
    """Stack of HAMs; semua HAM dalam stack berbagi k_local yang sama."""
    def __init__(self, channels, depth=4, k_local=3):
        super().__init__()
        self.blocks = nn.Sequential(*[HAM(channels, k_local=k_local) for _ in range(depth)])
    def forward(self, x):
        return self.blocks(x)


# ------------------------------
# Encoder / Decoder pieces
# ------------------------------
class Down(nn.Module):
    """
    MaxPool -> 1x1 proj -> MAF (dengan k_local terkontrol) -> 1x1 proj
    """
    def __init__(self, in_c, out_c, maf_depth=1, k_local=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.proj_in  = ConvBNAct(in_c, out_c, k=1)
        self.maf      = MAF(out_c, depth=maf_depth, k_local=k_local)
        self.proj_out = ConvBNAct(out_c, out_c, k=1)
    def forward(self, x):
        x = self.pool(x)
        x = self.proj_in(x)
        x = self.maf(x)
        x = self.proj_out(x)
        return x


class BridgeAttention(nn.Module):
    """ACA -> ASA (ASA sudah versi large-kernel)."""
    def __init__(self, channels):
        super().__init__()
        self.aca = ACA(channels)
        self.asa = ASA(k_small=3, k_large=7, dilation_large=2)
    def forward(self, x):
        return self.asa(self.aca(x))


# ------------------------------
# Global Context Head (GCN-style, factorized)
# ------------------------------
class FactorizedLargeKernelContext(nn.Module):
    """
    Depthwise (1xK) + Depthwise (Kx1) + Pointwise(1x1)  → ringan.
    K besar default 11 (bisa 9/13 sesuai kebutuhan).
    """
    def __init__(self, channels, k=11, act=nn.SiLU(inplace=True)):
        super().__init__()
        pad = k // 2
        self.dw1 = nn.Conv2d(channels, channels, (1, k), padding=(0, pad), groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dw2 = nn.Conv2d(channels, channels, (k, 1), padding=(pad, 0), groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pw  = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act = act

    def forward(self, x):
        x = self.act(self.bn1(self.dw1(x)))
        x = self.act(self.bn2(self.dw2(x)))
        x = self.act(self.bn3(self.pw(x)))
        return x


# ------------------------------
# MAFU-Net (with Large-Kernel upgrades)
# ------------------------------
class MAFUNet(nn.Module):
    """
    Args:
        in_ch:   input channels
        out_ch:  output channels (1 untuk biner)
        base_c:  base width
        maf_depth: jumlah HAM per level
        lk_size: ukuran kernel besar untuk depthwise conv (5 atau 7 direkomendasikan)
        lk_stages: stage encoder yang memakai large-kernel pada local conv (1..5 untuk enc1..enc5)
        use_gcn: aktifkan Global Context Head factorized (1xK + Kx1 + 1x1)
        gcn_ks:  ukuran kernel besar pada context head (disarankan 11)
    """
    def __init__(self,
                 in_ch=3, out_ch=1, base_c=16, maf_depth=1,
                 lk_size=7, lk_stages=(3, 4, 5),
                 use_gcn=True, gcn_ks=11):
        super().__init__()
        C = base_c
        self.lk_size   = lk_size
        self.lk_stages = set(lk_stages)

        # Stem (tetap kernel kecil untuk detail)
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, C, 3),
            ConvBNAct(C, C, 3)
        )

        def k_for(stage_idx):
            return lk_size if stage_idx in self.lk_stages else 3

        # Encoder
        self.enc1 = Down(C,    2*C, maf_depth, k_local=k_for(1))   # 1/2
        self.enc2 = Down(2*C,  4*C, maf_depth, k_local=k_for(2))   # 1/4
        self.enc3 = Down(4*C,  8*C, maf_depth, k_local=k_for(3))   # 1/8
        self.enc4 = Down(8*C, 16*C, maf_depth, k_local=k_for(4))   # 1/16
        self.enc5 = Down(16*C,32*C, maf_depth, k_local=k_for(5))   # 1/32

        # Bridge attention
        self.bridge0 = BridgeAttention(C)
        self.bridge1 = BridgeAttention(2*C)
        self.bridge2 = BridgeAttention(4*C)
        self.bridge3 = BridgeAttention(8*C)
        self.bridge4 = BridgeAttention(16*C)
        self.bridge5 = BridgeAttention(32*C)

        # Lateral projections (align ke C)
        self.lat0 = ConvBNAct(C,     C, 1)
        self.lat1 = ConvBNAct(2*C,   C, 1)
        self.lat2 = ConvBNAct(4*C,   C, 1)
        self.lat3 = ConvBNAct(8*C,   C, 1)
        self.lat4 = ConvBNAct(16*C,  C, 1)
        self.lat5 = ConvBNAct(32*C,  C, 1)

        # Fusion head
        self.fuse = nn.Sequential(
            ConvBNAct(6*C, 2*C, 3),
            ConvBNAct(2*C,   C, 3)
        )

        # Global context (opsional)
        self.context = FactorizedLargeKernelContext(C, k=gcn_ks) if use_gcn else nn.Identity()

        # Prediction head
        self.head = nn.Conv2d(C, out_ch, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape

        # Stem + encoder
        x0 = self.stem(x)          # B x C x H x W
        s0 = self.bridge0(x0)

        x1 = self.enc1(x0)         # B x 2C x H/2
        s1 = self.bridge1(x1)

        x2 = self.enc2(x1)         # B x 4C x H/4
        s2 = self.bridge2(x2)

        x3 = self.enc3(x2)         # B x 8C x H/8
        s3 = self.bridge3(x3)

        x4 = self.enc4(x3)         # B x 16C x H/16
        s4 = self.bridge4(x4)

        x5 = self.enc5(x4)         # B x 32C x H/32
        s5 = self.bridge5(x5)

        # Align & upsample to HxW
        f0 = self.lat0(s0)
        f1 = F.interpolate(self.lat1(s1), size=(H, W), mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.lat2(s2), size=(H, W), mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.lat3(s3), size=(H, W), mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.lat4(s4), size=(H, W), mode='bilinear', align_corners=True)
        f5 = F.interpolate(self.lat5(s5), size=(H, W), mode='bilinear', align_corners=True)

        fused = torch.cat([f0, f1, f2, f3, f4, f5], dim=1)  # B x 6C x H x W
        fused = self.fuse(fused)                            # B x C x H x W

        # Global context (GCN-style)
        fused = self.context(fused)

        logits = self.head(fused)
        return logits


# Quick self-test
if __name__ == "__main__":
    model = MAFUNet(
        in_ch=3, out_ch=1, base_c=16, maf_depth=1,
        lk_size=7, lk_stages=(3, 4, 5),
        use_gcn=True, gcn_ks=11
    )
    x = torch.randn(1, 3, 288, 384)
    with torch.no_grad():
        y = model(x)
    print("Output:", y.shape)