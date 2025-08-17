# eval_test_mafunet_lk.py
import os, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from models.mafunet_lk import MAFUNet

# ------------------------------
# Dataset (test)
# ------------------------------
class PolypFolderTest(Dataset):
    def __init__(self, img_dir, mask_dir, size_hw=(288,384)):
        super().__init__()
        self.img_paths = sorted([p for p in Path(img_dir).glob('*')
                                 if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}])
        self.mask_paths = [Path(mask_dir) / p.name for p in self.img_paths]
        self.size_hw = size_hw

    def __len__(self): return len(self.img_paths)

    def _load_img(self, p): return Image.open(p).convert('RGB')
    def _load_mask(self, p): return Image.open(p).convert('L')

    def __getitem__(self, idx):
        img = self._load_img(self.img_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])

        H, W = self.size_hw
        img_r = TF.resize(img, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        mask_r = TF.resize(mask, [H, W], interpolation=TF.InterpolationMode.NEAREST)

        x = TF.to_tensor(img_r)
        y = TF.to_tensor(mask_r)
        y = (y > 0.5).float()

        return x, y, str(self.img_paths[idx].name)

# ------------------------------
# Metrics
# ------------------------------
@torch.no_grad()
def confusion_counts(pred, gt):
    pred = pred.view(-1).float()
    gt   = gt.view(-1).float()
    tp = (pred * gt).sum()
    tn = ((1 - pred) * (1 - gt)).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    return tp, tn, fp, fn

@torch.no_grad()
def metrics_from_logits(logits, targets, thresh=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    tp, tn, fp, fn = confusion_counts(preds, targets)
    eps = 1e-6
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    sen = tp / (tp + fn + eps)          # recall / sensitivity
    spe = tn / (tn + fp + eps)          # specificity
    iou = tp / (tp + fp + fn + eps)     # IoU
    dsc = (2 * tp) / (2 * tp + fp + fn + eps)  # F1 / Dice
    return {'ACC': acc.item(), 'SEN': sen.item(), 'SPE': spe.item(),
            'mIoU': iou.item(), 'DSC': dsc.item()}

# ------------------------------
# TTA (optional)
# ------------------------------
@torch.no_grad()
def infer_with_tta(model, x, use_amp=True):
    # flips: none, H, V, HV -> average
    preds = []
    def fwd(inp):
        with torch.amp.autocast('cuda', enabled=use_amp):
            return torch.sigmoid(model(inp))

    p0 = fwd(x)
    preds.append(p0)

    xh = torch.flip(x, dims=[-1])
    pv = fwd(xh)
    preds.append(torch.flip(pv, dims=[-1]))

    xv = torch.flip(x, dims=[-2])
    ph = fwd(xv)
    preds.append(torch.flip(ph, dims=[-2]))

    xhv = torch.flip(x, dims=[-2,-1])
    phv = fwd(xhv)
    preds.append(torch.flip(phv, dims=[-2,-1]))

    return torch.stack(preds, dim=0).mean(dim=0)

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='dataset')
    ap.add_argument('--img_size', type=int, nargs=2, default=[288,384])
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--ckpt', type=str, default='runs/mafunet_lk/best.pth')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--base_c', type=int, default=16)
    ap.add_argument('--maf_depth', type=int, default=1)
    ap.add_argument('--lk_size', type=int, default=7)
    ap.add_argument('--lk_stages', type=str, default='3,4,5')
    ap.add_argument('--use_gcn', action='store_true')
    ap.add_argument('--gcn_ks', type=int, default=11)
    ap.add_argument('--save_csv', type=str, default='')  # path CSV per-image (opsional)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = args.amp and (device == 'cuda')

    # dataset
    H, W = args.img_size
    test_img = Path(args.data_root) / 'test' / 'images'
    test_msk = Path(args.data_root) / 'test' / 'masks'
    test_set = PolypFolderTest(test_img, test_msk, size_hw=(H, W))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # model (harus sama config-nya dengan saat training)
    def parse_stages(s): return tuple(int(x) for x in s.split(',') if x.strip())
    model = MAFUNet(in_ch=3, out_ch=1, base_c=args.base_c, maf_depth=args.maf_depth,
                    lk_size=args.lk_size, lk_stages=parse_stages(args.lk_stages),
                    use_gcn=args.use_gcn, gcn_ks=args.gcn_ks).to(device)

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # accumulators
    M = {'ACC':0.0,'SEN':0.0,'SPE':0.0,'mIoU':0.0,'DSC':0.0}
    n_batches = 0

    # optional CSV
    writer = None
    if args.save_csv:
        os.makedirs(Path(args.save_csv).parent, exist_ok=True)
        f = open(args.save_csv, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['filename','ACC','SEN','SPE','mIoU','DSC'])

    with torch.no_grad():
        for x, y, names in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if args.tta:
                probs = infer_with_tta(model, x, use_amp=use_amp)
            else:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    probs = torch.sigmoid(model(x))

            preds = (probs > args.threshold).float()
            tp, tn, fp, fn = confusion_counts(preds, y)
            eps = 1e-6
            acc = (tp + tn) / (tp + tn + fp + fn + eps)
            sen = tp / (tp + fn + eps)
            spe = tn / (tn + fp + eps)
            iou = tp / (tp + fp + fn + eps)
            dsc = (2 * tp) / (2 * tp + fp + fn + eps)

            batch_metrics = {'ACC':acc.item(),'SEN':sen.item(),'SPE':spe.item(),'mIoU':iou.item(),'DSC':dsc.item()}
            for k in M: M[k] += batch_metrics[k]
            n_batches += 1

            if writer is not None:
                # tulis per-image pakai nilai batch (agregat per-batch; jika ingin per-image detail, hitung per-sample)
                for nm in names:
                    writer.writerow([nm, batch_metrics['ACC'], batch_metrics['SEN'],
                                     batch_metrics['SPE'], batch_metrics['mIoU'], batch_metrics['DSC']])

    for k in M: M[k] /= max(1, n_batches)

    print("===== TEST METRICS =====")
    print(f"mIoU: {M['mIoU']:.4f}")
    print(f"F1/DSC: {M['DSC']:.4f}")
    print(f"ACC: {M['ACC']:.4f}")
    print(f"SEN: {M['SEN']:.4f}")
    print(f"SPE: {M['SPE']:.4f}")

    if writer is not None:
        f.close()
        print(f"=> per-image CSV saved to: {args.save_csv}")

if __name__ == '__main__':
    main()
