#!/usr/bin/env python3
# ============================================================
#  COMP3710 – Part 4.3.2: UNet on OASIS PNG slices (multi-class)
#  - CE + soft Dice loss
#  - per-class DSC + foreground-mean DSC
#  - qualitative overlays + best checkpoint
#  - CUDA (AMP), MPS, CPU friendly
# ============================================================

import os, time, glob, argparse, pathlib, random, re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms import InterpolationMode
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler   # AMP (new API, CUDA only)

# ----------------------------- utils -----------------------------
def find_dataset_root(explicit: Optional[str], marker="keras_png_slices_data") -> str:
    cand = []
    if explicit: cand.append(os.path.abspath(os.path.expanduser(explicit)))
    if os.environ.get("OASIS_ROOT"): cand.append(os.environ["OASIS_ROOT"])
    cand.append("/home/groups/comp3710")  # Rangpur shared
    here = pathlib.Path(__file__).resolve().parent
    cand += [str(here), str(pathlib.Path.cwd())]
    for c in cand:
        d = os.path.join(os.path.abspath(os.path.expanduser(c)), marker)
        if os.path.isdir(d): return os.path.abspath(os.path.expanduser(c))
    raise FileNotFoundError(
        f"Couldn't find '{marker}/'. Pass --data-root pointing to the PARENT "
        f"that contains '{marker}/'. Checked: {cand}"
    )

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def find_seg_dir(base: str, subset: str) -> str:
    names = [
        f"keras_png_slices_seg_{subset}",
        f"keras_png_slices_{subset}_seg",
        f"keras_png_slices_labels_{subset}",
        f"keras_png_slices_{subset}_labels",
        f"keras_png_slices_segmentation_{subset}",
    ]
    for name in names:
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return p
    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if os.path.isdir(full):
            low = entry.lower()
            if subset in low and ("seg" in low or "label" in low):
                return full
    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if os.path.isdir(full):
            low = entry.lower()
            if ("seg" in low or "label" in low):
                return full
    raise FileNotFoundError(f"No segmentation folder found in {base} for subset='{subset}'.")

# ----------------------------- dataset -----------------------------
class OasisSeg(Dataset):
    """
    Pairs <img, mask> from:
      <root>/keras_png_slices_data/keras_png_slices_{train,validate,test}
      <root>/keras_png_slices_data/<*seg*/ *label*>_{train,validate,test}
    Masks are PNG with integer labels {0..C-1} after remapping palette.
    """
    def __init__(self, parent_root: str, subset="train", size=128, scan_classes=True):
        assert subset in {"train","validate","test"}
        base = os.path.join(parent_root, "keras_png_slices_data")
        img_dir = os.path.join(base, f"keras_png_slices_{subset}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing image dir: {img_dir}")
        seg_dir = find_seg_dir(base, subset)

        self.size = size
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not self.imgs:
            raise FileNotFoundError(f"No PNGs in {img_dir}")

        mask_files = sorted(glob.glob(os.path.join(seg_dir, "*.png"))) or \
                     sorted(glob.glob(os.path.join(seg_dir, "*.PNG")))
        if not mask_files:
            raise FileNotFoundError(f"No PNGs in {seg_dir}")

        def canon(path: str) -> str:
            """
            Map both 'case_001_slice_0(.nii).png' and 'seg_001_slice_0(.nii).png'
            to the same key: '001_slice_0'. Also strip common mask suffixes.
            """
            n = os.path.basename(path).lower()
            n = re.sub(r'\.nii\.png$', '', n); n = re.sub(r'\.png$', '', n)
            n = re.sub(r'(_mask|_segmentation|_seg|_labels?|_label)$', '', n)
            n = re.sub(r'^(case_|seg_)', '', n)
            m = re.match(r'^(\d+)_slice_(\d+)$', n)
            return f"{m.group(1)}_slice_{m.group(2)}" if m else n

        # build lookup for masks by canonical key
        mask_by_key = {}
        for m in mask_files:
            key = canon(m)
            mask_by_key[key] = m

        # pair image→mask
        self.masks: List[str] = []
        missing = []
        for p in self.imgs:
            key = canon(p)
            q = mask_by_key.get(key)
            if q is None:
                alt = key.split('/')[-1]
                hits = [mask_by_key[k] for k in mask_by_key if k.endswith(alt)]
                q = hits[0] if hits else None
            if q is None:
                missing.append(os.path.basename(p))
            else:
                self.masks.append(q)

        if missing:
            print("\n[DEBUG] First 10 image basenames:", [os.path.basename(x) for x in self.imgs[:10]])
            print("[DEBUG] First 10 mask basenames:", [os.path.basename(x) for x in mask_files[:10]])
            print("[DEBUG] First 10 image keys:", [canon(x) for x in self.imgs[:10]])
            print("[DEBUG] First 10 mask  keys:", [canon(x) for x in mask_files[:10]])
            example = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Could not find matching masks for {len(missing)} image(s). "
                f"Examples: {example}\nLooked under: {seg_dir}\n"
                f"Tip: mask stems must match image stems (prefix 'case_' vs 'seg_' handled automatically)."
            )

        # ---- Build a global palette and mapping (fixes "256 classes" issue) ----
        # Scan up to ~800 masks to find unique intensities used as labels
        vals = set()
        step = max(1, len(self.masks)//800)
        for m in self.masks[::step]:
            arr = np.array(Image.open(m), dtype=np.uint8)
            vals.update(np.unique(arr).tolist())
        # common OASIS palettes: {0,255} or {0,63,127,191,255} etc.
        palette = sorted(int(v) for v in vals)
        if len(palette) > 16:
            # If something weird (too many unique values), quantise to 8 bins
            # to keep training sane.
            edges = np.linspace(0, 256, 9, endpoint=True)
            palette = [int(x) for x in np.linspace(0, 255, 8)]
        self.palette = palette
        self.lookup = {v:i for i,v in enumerate(self.palette)}
        self.num_classes = len(self.palette)
        print(f"Detected palette (mask intensities) → classes: {self.palette} -> C={self.num_classes}")

        self.subset = subset

    def __len__(self): return len(self.imgs)

    def _augment(self, img: Image.Image, msk: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            img = ImageOps.mirror(img); msk = ImageOps.mirror(msk)
        if random.random() < 0.5:
            img = ImageOps.flip(img); msk = ImageOps.flip(msk)
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR)
            msk = msk.rotate(angle, resample=Image.NEAREST)
        return img, msk

    def __getitem__(self, i):
        x = Image.open(self.imgs[i]).convert("L")
        y_img = Image.open(self.masks[i])  # integer intensities
        x = tv_resize(x, [self.size, self.size], InterpolationMode.BILINEAR)
        y_img = tv_resize(y_img, [self.size, self.size], InterpolationMode.NEAREST)
        if "train" in self.imgs[i]:
            x, y_img = self._augment(x, y_img)

        x = torch.from_numpy(np.array(x, dtype=np.float32)/255.0).unsqueeze(0)

        y_arr = np.array(y_img, dtype=np.uint8)
        # map intensities -> class ids using global lookup
        y = np.zeros_like(y_arr, dtype=np.int64)
        for val, idx in self.lookup.items():
            y[y_arr == val] = idx
        y = torch.from_numpy(y)

        return x, y

# ----------------------------- UNet -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, base=32):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*16))
        self.up1   = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.conv1 = DoubleConv(base*16, base*8)
        self.up2   = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv2 = DoubleConv(base*8, base*4)
        self.up3   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv3 = DoubleConv(base*4, base*2)
        self.up4   = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv4 = DoubleConv(base*2, base)
        self.outc  = nn.Conv2d(base, n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5); x = self.conv1(torch.cat([x, x4], dim=1))
        x  = self.up2(x);  x = self.conv2(torch.cat([x, x3], dim=1))
        x  = self.up3(x);  x = self.conv3(torch.cat([x, x2], dim=1))
        x  = self.up4(x);  x = self.conv4(torch.cat([x, x1], dim=1))
        return self.outc(x)  # logits (N,C,H,W)

# ----------------------------- losses / metrics -----------------------------
def soft_dice_per_class(logits, target, eps=1e-6):
    N, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.zeros((N, C, H, W), dtype=probs.dtype, device=probs.device)
    one_hot.scatter_(1, target.unsqueeze(1), 1.0)
    dims = (0,2,3)
    inter = (probs * one_hot).sum(dims)
    p_sum = probs.sum(dims)
    g_sum = one_hot.sum(dims)
    dice = (2*inter + eps) / (p_sum + g_sum + eps)
    return dice  # (C,)

class SegLoss(nn.Module):
    def __init__(self, ce_w=1.0, dice_w=1.0, class_weights=None):
        super().__init__()
        self.ce_w, self.dice_w = ce_w, dice_w
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
    def __call__(self, logits, target):
        ce = self.ce(logits, target)
        dice = soft_dice_per_class(logits, target).mean()
        return self.ce_w*ce + self.dice_w*(1.0 - dice), ce, dice

# ----------------------------- viz -----------------------------
@torch.no_grad()
def save_overlays(x, y, pred, out_path, n=12):
    x = x[:n].cpu(); y = y[:n].cpu(); pred = pred[:n].cpu()
    n = x.size(0)
    rows = []
    for i in range(n):
        img = x[i,0]
        gt  = y[i]
        pr  = pred[i]
        gt_rgb  = torch.stack([img, img, img], dim=0)
        pr_rgb  = torch.stack([img, img, img], dim=0)
        for c, ch in zip([1,2,3], [0,1,2]):
            gt_rgb[ch]  = torch.where(gt==c, torch.tensor(1.0), gt_rgb[ch])
            pr_rgb[ch]  = torch.where(pr==c, torch.tensor(1.0), pr_rgb[ch])
        row = torch.cat([gt_rgb, pr_rgb], dim=2)
        rows.append(row)
    grid = make_grid(rows, nrow=4, padding=4)
    save_image(grid, out_path)

# ----------------------------- train -----------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=None,
                    help="Parent folder that contains 'keras_png_slices_data/'")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP (CUDA)")
    ap.add_argument("--run", type=str, default=None)
    args = ap.parse_args(argv)

    # device / amp
    if torch.cuda.is_available():
        device = torch.device("cuda")
        AMP = not args.no_amp
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision("high")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps"); AMP = False
    else:
        device = torch.device("cpu"); AMP=False

    set_seed(args.seed)
    root = find_dataset_root(args.data_root)
    print(f"Using data root: {root}")
    print(f"Device: {device} | AMP: {'on' if AMP else 'off'}")

    # data
    train_ds = OasisSeg(root, "train", size=args.size)
    val_ds   = OasisSeg(root, "validate", size=args.size)
    n_classes = max(train_ds.num_classes, val_ds.num_classes)
    print(f"Detected classes: {n_classes} (palette {train_ds.palette})")

    batch = args.batch or (64 if device.type=="cuda" else 16)
    workers = args.workers if args.workers is not None else (
        int(os.environ.get("SLURM_CPUS_PER_TASK","4")) if device.type=="cuda" else 0
    )

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, pin_memory=(device.type=="cuda"),
                          drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=(device.type=="cuda"))

    # model / loss / opt
    model = UNet(in_ch=1, n_classes=n_classes, base=32).to(device)
    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(train_dl)
    sched = OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                       steps_per_epoch=steps, pct_start=0.1)
    scaler = GradScaler("cuda", enabled=(device.type=="cuda" and AMP))
    criterion = SegLoss(ce_w=1.0, dice_w=1.0)

    run_dir = args.run or time.strftime("unet_runs_oasis/%Y%m%d_%H%M%S")
    os.makedirs(run_dir, exist_ok=True)
    print("Saving to:", run_dir)

    # training
    best_fg_dsc = 0.0
    hist = {"tr_loss":[], "va_loss":[], "tr_dice":[], "va_dice":[], "va_dice_per_class":[]}

    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss=tr_dice_sum=0.0; seen=0
        t0=time.time()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type=="cuda" and AMP)):
                logits = model(x)
                loss, ce, dice = criterion(logits, y)
            if device.type=="cuda" and AMP:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
            bs = x.size(0); seen += bs
            tr_loss += float(loss.detach())*bs
            tr_dice_sum += float(dice.detach())*bs
        tr_loss/=seen; tr_dice = tr_dice_sum/seen

        # validation
        model.eval()
        va_loss=0.0; counts=0
        # MPS doesn't support float64 – use float32
        acc_dtype = torch.float32 if device.type=="mps" else torch.float32
        per_class_accum = torch.zeros(n_classes, dtype=acc_dtype, device=device)
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss, _, _ = criterion(logits, y)
                dice_c = soft_dice_per_class(logits, y)  # (C,)
                va_loss += float(loss)*x.size(0)
                per_class_accum += dice_c.to(acc_dtype)*x.size(0)
                counts += x.size(0)
            va_loss/=counts
            dpc = (per_class_accum / counts).detach().cpu().numpy()
            fg_mean = dpc[1:].mean() if n_classes>1 else dpc.mean()

        hist["tr_loss"].append(tr_loss);  hist["va_loss"].append(va_loss)
        hist["tr_dice"].append(tr_dice);  hist["va_dice"].append(fg_mean)
        hist["va_dice_per_class"].append(dpc.tolist())

        print(f"Epoch {ep:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
              f"val loss {va_loss:.4f} DSC per-class {np.round(dpc,3)} | FG mean {fg_mean:.3f} | "
              f"{len(train_ds)/max(1e-6,time.time()-t0):.0f} img/s")

        if fg_mean > best_fg_dsc:
            best_fg_dsc = fg_mean
            torch.save({"model": model.state_dict(),
                        "classes": n_classes,
                        "size": args.size,
                        "palette": train_ds.palette},
                       os.path.join(run_dir, "best.pt"))

        # qualitative overlays
        x0,y0 = next(iter(val_dl))
        x0,y0 = x0.to(device), y0.to(device)
        with torch.no_grad():
            pr0 = model(x0).argmax(1)
        save_overlays(x0, y0, pr0, os.path.join(run_dir, f"ep{ep:02d}_overlays.png"), n=12)

    # curves
    plt.figure(figsize=(7,5))
    ep_axis = np.arange(1, len(hist["tr_loss"])+1)
    plt.plot(ep_axis, hist["va_dice"], label="val FG mean DSC")
    plt.plot(ep_axis, hist["tr_dice"], label="train soft Dice (avg)", ls="--")
    plt.plot(ep_axis, hist["tr_loss"], label="train loss", ls=":")
    plt.plot(ep_axis, hist["va_loss"], label="val loss", ls="-.")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("metric/loss"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curves.png")); plt.close()

    final_dpc = np.array(hist["va_dice_per_class"][-1])
    print("\nFinal DSC per class:", np.round(final_dpc, 4))
    if n_classes>1:
        print("Foreground mean DSC:", final_dpc[1:].mean())
    print("Artifacts in:", run_dir)
    print("Done.")

if __name__ == "__main__":
    main()
