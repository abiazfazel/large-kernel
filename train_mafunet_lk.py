import copy
import os, math, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from models.mafunet_lk import MAFUNet

# ------------------------------
# Utils
# ------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

import copy
...
def try_profile_flops(model, in_ch, H, W, device):
    try:
        from thop import profile
        # profile di deepcopy agar hook THOP tidak menempel ke model training
        m = copy.deepcopy(model).to(device).eval()
        x = torch.randn(1, in_ch, H, W, device=device)
        with torch.no_grad():
            flops, _ = profile(m, inputs=(x,), verbose=False)
        # buang copy supaya hook ikut hilang
        del m, x
        return flops / 1e9  # GFLOPs
    except Exception:
        return None

# ------------------------------
# Dataset
# ------------------------------
class PolypFolder(Dataset):
    def __init__(self, img_dir, mask_dir, size_hw=(288, 384), augment=False):
        super().__init__()
        self.img_paths = sorted([p for p in Path(img_dir).glob('*')
                                 if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}])
        self.mask_paths = [Path(mask_dir) / p.name for p in self.img_paths]
        self.size_hw = size_hw
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def _load_img(self, p):
        return Image.open(p).convert('RGB')

    def _load_mask(self, p):
        return Image.open(p).convert('L')

    def _to_tensor_pair(self, img, mask):
        H, W = self.size_hw
        img = TF.resize(img, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, [H, W], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img); mask = TF.vflip(mask)
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        if self.augment:
            b = 1.0 + random.uniform(-0.1, 0.1)
            c = 1.0 + random.uniform(-0.1, 0.1)
            img = torch.clamp((img - 0.5) * c + 0.5, 0, 1) * b
            img = torch.clamp(img, 0, 1)

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return img, mask

    def __getitem__(self, idx):
        img = self._load_img(self.img_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        return self._to_tensor_pair(img, mask)

# ------------------------------
# Losses & Metrics
# ------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2,3)) + self.eps
        den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.eps
        loss = 1 - (num / den)
        return loss.mean()

def confusion_counts(pred, gt):
    pred = pred.view(-1).float()
    gt   = gt.view(-1).float()
    tp = (pred * gt).sum()
    tn = ((1 - pred) * (1 - gt)).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    return tp, tn, fp, fn

def compute_metrics(logits, targets, thresh=0.5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > thresh).float()
        tp, tn, fp, fn = confusion_counts(preds, targets)
        eps = 1e-6
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        sen = tp / (tp + fn + eps)  # recall
        spe = tn / (tn + fp + eps)
        iou = tp / (tp + fp + fn + eps)
        dsc = (2 * tp) / (2 * tp + fp + fn + eps)
    return {'ACC': acc.item(), 'SEN': sen.item(), 'SPE': spe.item(),
            'mIoU': iou.item(), 'DSC': dsc.item()}

# ------------------------------
# EMA helper (float params only)
# ------------------------------
class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        # simpan HANYA parameter float (bukan buffer seperti num_batches_tracked)
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.dtype.is_floating_point:
                self.shadow[name] = p.detach().clone()
        self.device = device

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow and p.dtype.is_floating_point:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model):
        # salin hanya yang ada di shadow (float params)
        state_dict = model.state_dict()
        for name, val in self.shadow.items():
            if name in state_dict:
                state_dict[name].copy_(val)
        model.load_state_dict(state_dict, strict=False)


# ------------------------------
# Train / Val
# ------------------------------
def cosine_warmup_lambda(total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return fn

def validate(model, loader, device, threshold=0.5, use_amp=True):
    model.eval()
    meters = {'ACC':0.0,'SEN':0.0,'SPE':0.0,'mIoU':0.0,'DSC':0.0}
    n = 0
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            for imgs, masks in loader:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(imgs)
                m = compute_metrics(logits, masks, thresh=threshold)
                for k in meters: meters[k] += m[k]
                n += 1
    for k in meters: meters[k] /= max(1, n)
    return meters

def train_one_epoch(model, loader, optimizer, scaler, device, loss_cfg,
                    accum=1, use_amp=True, ema: EMA=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(imgs)
            loss = loss_cfg['alpha'] * loss_cfg['bce'](logits, masks) \
                 + (1 - loss_cfg['alpha']) * loss_cfg['dice'](logits, masks)
            loss = loss / accum

        scaler.scale(loss).backward()

        if (i + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        total_loss += loss.item()

    return total_loss / max(1, len(loader))

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset')
    parser.add_argument('--img_size', type=int, nargs=2, default=[288, 384])
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Model
    parser.add_argument('--base_c', type=int, default=16)
    parser.add_argument('--maf_depth', type=int, default=1)
    parser.add_argument('--lk_size', type=int, default=7)
    parser.add_argument('--lk_stages', type=str, default='3,4,5')
    parser.add_argument('--use_gcn', action='store_true')
    parser.add_argument('--gcn_ks', type=int, default=11)
    parser.add_argument('--no_flops', action='store_true', help='skip THOP FLOPs profiling')

    # Train tricks
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for BCE in BCE+Dice')
    parser.add_argument('--pos_weight', type=float, default=1.8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='runs/mafunet_lk')

    args = parser.parse_args()

    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = args.amp and (device == 'cuda')

    # Datasets
    H, W = args.img_size
    train_img = Path(args.data_root) / 'train' / 'images'
    train_msk = Path(args.data_root) / 'train' / 'masks'
    val_img   = Path(args.data_root) / 'val' / 'images'
    val_msk   = Path(args.data_root) / 'val' / 'masks'

    train_set = PolypFolder(train_img, train_msk, size_hw=(H, W), augment=True)
    val_set   = PolypFolder(val_img,   val_msk,   size_hw=(H, W), augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=max(1, args.batch_size//2), shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    def parse_stages(s): return tuple(int(x) for x in s.split(',') if x.strip())
    model = MAFUNet(in_ch=3, out_ch=1, base_c=args.base_c, maf_depth=args.maf_depth,
                    lk_size=args.lk_size, lk_stages=parse_stages(args.lk_stages),
                    use_gcn=args.use_gcn, gcn_ks=args.gcn_ks).to(device)

    # Params & FLOPs (optional)
    params_m = count_params_m(model)
    flops_g = None if args.no_flops else try_profile_flops(model, 3, H, W, device)



    # Optim & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(0.10 * total_steps)
    lr_lambda = cosine_warmup_lambda(total_steps, warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Loss
    pos_weight = torch.tensor([args.pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dice = DiceLoss()
    loss_cfg = {'alpha': args.alpha, 'bce': bce, 'dice': dice}

    # EMA
    ema = EMA(model, decay=0.999) if args.ema else None

    # AMP scaler (API baru)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Save dir
    os.makedirs(args.save_dir, exist_ok=True)
    best_dsc = -1.0

    print("====== TRAIN START ======")
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Params: {params_m:.3f} M | FLOPs: {('%.3f GFLOPs'%flops_g) if flops_g is not None else 'n/a'} @ {H}x{W}")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                   loss_cfg, accum=max(1, args.accum), use_amp=use_amp, ema=ema)
        scheduler.step()

        if epoch % args.val_every == 0:
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)

            val_metrics = validate(model, val_loader, device, threshold=args.threshold, use_amp=use_amp)

            if ema is not None and backup is not None:
                model.load_state_dict(backup, strict=True)

            dsc = val_metrics['DSC']
            is_best = dsc > best_dsc
            if is_best:
                best_dsc = dsc
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_dsc': best_dsc,
                    'args': vars(args)
                }, os.path.join(args.save_dir, 'best.pth'))

            print(f"[Ep {epoch:03d}] loss={avg_loss:.4f} | "
                  f"DSC={val_metrics['DSC']:.4f} mIoU={val_metrics['mIoU']:.4f} "
                  f"ACC={val_metrics['ACC']:.4f} SEN={val_metrics['SEN']:.4f} SPE={val_metrics['SPE']:.4f} "
                  f"| best_DSC={best_dsc:.4f}")

    print("====== TRAIN DONE ======")

if __name__ == '__main__':
    main()
