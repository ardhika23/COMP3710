# ============================================================
#  COMP3710 — Part 4.3.1  VAE on OASIS PNG slices (full marks)
#  - Loads preprocessed PNG slices (keras_png_slices_* folders)
#  - Trains a ConvVAE with KL warm-up + free-bits (stable)
#  - Saves reconstructions per epoch
#  - Saves random samples + latent traversal
#  - Saves 2D latent viz (UMAP if available, else PCA)
#  - Works on CUDA (Rangpur), MPS, or CPU
# ============================================================

import os, time, math, glob, argparse, random, pathlib
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ------------------------- Utils -------------------------
def find_dataset_root(explicit: Optional[str] = None,
                      marker: str = "keras_png_slices_data") -> str:
    """
    Returns the *parent* folder that directly contains `marker/`.
    Priority:
      1) --data-root arg (if provided) or OASIS_ROOT env
      2) '/home/groups/comp3710' (Rangpur shared)
      3) This script folder or current working dir
    """
    cands: List[str] = []
    if explicit: cands.append(os.path.abspath(os.path.expanduser(explicit)))
    if os.environ.get("OASIS_ROOT"): cands.append(os.environ["OASIS_ROOT"])
    cands.append("/home/groups/comp3710")
    here = pathlib.Path(__file__).resolve().parent
    cands += [str(here), str(pathlib.Path.cwd())]

    for c in cands:
        d = os.path.join(os.path.abspath(os.path.expanduser(c)), marker)
        if os.path.isdir(d):
            return os.path.abspath(os.path.expanduser(c))
    raise FileNotFoundError(
        f"Could not find '{marker}/'. Try --data-root /path/to/PARENT "
        f"(the parent that directly contains '{marker}/').\nChecked: {cands}"
    )

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ------------------------- Dataset -------------------------
class KerasPngSlices(Dataset):
    """
    Loads GREYSCALE PNGs from the OASIS 'keras_png_slices_*' folders.
    Inside <root>/keras_png_slices_data/ expect:
      - keras_png_slices_train/
      - keras_png_slices_validate/
      - keras_png_slices_test/
    """
    def __init__(self, parent_root: str, subset: str = "train",
                 size: int = 128):
        assert subset in {"train", "validate", "test", "all"}
        self.size = size
        base = os.path.join(parent_root, "keras_png_slices_data")
        sub_map = {
            "train":    ["keras_png_slices_train"],
            "validate": ["keras_png_slices_validate"],
            "test":     ["keras_png_slices_test"],
            "all":      ["keras_png_slices_train", "keras_png_slices_validate", "keras_png_slices_test"],
        }
        self.paths: List[str] = []
        for sub in sub_map[subset]:
            self.paths += sorted(glob.glob(os.path.join(base, sub, "*.png")))
        if not self.paths:
            raise RuntimeError(f"No PNG files found in {base} for subset='{subset}'")
        self.subset = subset

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("L").resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]
        x = torch.from_numpy(arr)[None, ...]             # (1,H,W)
        return x

# ------------------------- Model -------------------------
class ConvVAE(nn.Module):
    """
    Simple Conv VAE (128x128x1) + sane logvar init to avoid collapse.
    """
    def __init__(self, z_dim=64, img_ch=1):
        super().__init__()
        self.z_dim = z_dim
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(img_ch, 32, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),      # 32x32
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),     # 16x16
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),    # 8x8
            nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.enc_out = 256 * 8 * 8
        self.fc_mu = nn.Linear(self.enc_out, z_dim)
        self.fc_lv = nn.Linear(self.enc_out, z_dim)

        # Decoder
        self.fc_z = nn.Linear(z_dim, self.enc_out)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, img_ch, 4, 2, 1) # 128x128
        )

        # Logvar init (smaller variance initially)
        nn.init.zeros_(self.fc_lv.weight)
        nn.init.constant_(self.fc_lv.bias, -4.0)  # log(sigma^2) ≈ -4 → sigma≈0.135

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_z(z).view(z.size(0), 256, 8, 8)
        x_hat_logits = self.dec(h)
        return torch.sigmoid(x_hat_logits)   # [0,1]

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        x_hat = self.decode(z)
        return x_hat, mu, lv

# ------------------------- Loss (with free-bits + warm-up) -------------------------
def elbo_terms(x, x_hat, mu, logvar, recon_loss="bce",
               beta=1.0, free_bits=0.01, warmup=1.0):
    """
    recon: BCE or MSE, mean over pixels.
    KL: per-dimension with free-bits, then averaged; multiplied by beta*warmup.
    """
    if recon_loss == "bce":
        recon = F.binary_cross_entropy(x_hat.clamp(1e-6, 1-1e-6), x, reduction="mean")
    else:
        recon = F.mse_loss(x_hat, x, reduction="mean")

    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, z)
    kl_dim = torch.clamp(kl_dim, min=free_bits)               # free-bits
    kl = kl_dim.mean()

    elbo = recon + (beta * warmup) * kl
    return recon, kl, elbo

@torch.no_grad()
def evaluate(model, loader, device, amp_on, args):
    model.eval(); r_sum=k_sum=l_sum=n=0.0
    for xb in loader:
        xb = xb.to(device)
        with autocast(device_type=device.type, enabled=amp_on):
            xh, mu, lv = model(xb)
            r,k,l = elbo_terms(xb, xh, mu, lv, args.recon_loss,
                               beta=args.beta, free_bits=args.free_bits, warmup=1.0)
        bs = xb.size(0); n += bs
        r_sum += float(r) * bs; k_sum += float(k) * bs; l_sum += float(l) * bs
    return r_sum/n, k_sum/n, l_sum/n

def save_grid(tensor, path, nrow=8):
    tensor = tensor.clamp(0,1)
    save_image(tensor.cpu(), path, nrow=nrow)

def latent_traversal(model, device, z_dim=64, steps=9, var_dims=None, base=None):
    if var_dims is None: var_dims = (0,1)
    d1, d2 = var_dims
    grid = []
    with torch.no_grad():
        if base is None:
            base = torch.zeros(1, z_dim, device=device)
        vals = torch.linspace(-2, 2, steps, device=device)
        for a in vals:
            row = []
            for b in vals:
                z = base.clone()
                z[0, d1] = a; z[0, d2] = b
                img = model.decode(z)
                row.append(img)
            grid.append(torch.cat(row, dim=0))
    return torch.cat(grid, dim=0)

def plot_loss_curves(history, out_path):
    plt.figure(figsize=(7,5))
    ep = np.arange(1, len(history["train_elbo"])+1)
    plt.plot(ep, history["train_elbo"], label="train ELBO")
    plt.plot(ep, history["val_elbo"],   label="val ELBO")
    plt.plot(ep, history["train_rec"],  "--", label="train recon")
    plt.plot(ep, history["val_rec"],    "--", label="val recon")
    plt.plot(ep, history["train_kl"],   ":", label="train KL")
    plt.plot(ep, history["val_kl"],     ":", label="val KL")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def latent_umap_plot(all_mu, out_path):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, metric="euclidean", random_state=42)
        z2 = reducer.fit_transform(all_mu)
        title = "UMAP of latent μ"
    except Exception:
        from sklearn.decomposition import PCA
        z2 = PCA(n_components=2, random_state=42).fit_transform(all_mu)
        title = "PCA(2) of latent μ"
    plt.figure(figsize=(5,5))
    plt.scatter(z2[:,0], z2[:,1], s=3, alpha=0.6)
    plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()

# ------------------------- Main -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None,
        help="PARENT folder that contains 'keras_png_slices_data/'.")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--zdim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--free-bits", type=float, default=0.01, help="per-dim KL floor")
    parser.add_argument("--kl-warmup", type=int, default=10, help="epochs to ramp KL from 0→1")
    parser.add_argument("--recon-loss", type=str, default="bce", choices=["bce","mse"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run",  type=str, default=None, help="Run directory (auto if not set)")
    args = parser.parse_args(argv)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda"); amp_on=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision("high")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps"); amp_on=False  # AMP off on MPS (stability)
    else:
        device = torch.device("cpu"); amp_on=False

    root = find_dataset_root(args.data_root)
    print(f"Using data root: {root}")
    print(f"Device: {device.type}")

    # Batch/workers
    if args.batch is None:
        batch = 64 if device.type=="cuda" else 16
    else:
        batch = args.batch
    if args.workers is None:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) if device.type=="cuda" else 0
    else:
        workers = args.workers
    print(f"workers: {workers}")

    # Datasets/loaders
    ds_train = KerasPngSlices(root, "train", size=args.size)
    ds_val   = KerasPngSlices(root, "validate", size=args.size)
    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True,
                          num_workers=workers, pin_memory=(device.type=="cuda"))
    dl_val   = DataLoader(ds_val,   batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=(device.type=="cuda"))
    print(f"Train images: {len(ds_train)} | Val images: {len(ds_val)} | size: {args.size}x{args.size}")

    # Model/optim/sched
    seed_everything(args.seed)
    model = ConvVAE(z_dim=args.zdim, img_ch=1).to(device)
    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=(device.type=="cuda" and amp_on))

    # Run dir
    run_dir = args.run or time.strftime("vae_runs_oasis/%Y%m%d_%H%M%S")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving outputs to: {run_dir}")

    # Sanity forward
    xb = next(iter(dl_train)).to(device)
    with torch.no_grad(), autocast(device_type=device.type, enabled=amp_on):
        xh, mu, lv = model(xb)
    print("Forward OK | x̂:", tuple(xh.shape), "| μ:", tuple(mu.shape), "| logσ²:", tuple(lv.shape))

    # Training
    history = {"train_rec":[], "train_kl":[], "train_elbo":[],
               "val_rec":[],   "val_kl":[],   "val_elbo":[]}
    best_val = math.inf
    t0 = time.time()

    for ep in range(1, args.epochs+1):
        model.train()
        tr_r=tr_k=tr_l=0.0; ntr=0
        # KL warm-up factor
        warm = min(1.0, ep / max(1, args.kl_warmup))

        for xb in dl_train:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_on):
                xh, mu, lv = model(xb)
                r,k,l = elbo_terms(xb, xh, mu, lv,
                                   recon_loss=args.recon_loss,
                                   beta=args.beta, free_bits=args.free_bits,
                                   warmup=warm)

            if device.type=="cuda" and amp_on:
                scaler.scale(l).backward()
                scaler.step(opt); scaler.update()
            else:
                l.backward(); opt.step()

            bs = xb.size(0); ntr += bs
            tr_r += float(r) * bs; tr_k += float(k) * bs; tr_l += float(l) * bs

        tr_r/=ntr; tr_k/=ntr; tr_l/=ntr
        va_r, va_k, va_l = evaluate(model, dl_val, device, amp_on, args)
        history["train_rec"].append(tr_r); history["train_kl"].append(tr_k); history["train_elbo"].append(tr_l)
        history["val_rec"].append(va_r);   history["val_kl"].append(va_k);   history["val_elbo"].append(va_l)
        sched.step()

        print(f"Epoch {ep:02d}/{args.epochs} | "
              f"train: rec {tr_r:.4f} KL {tr_k:.4f} ELBO {tr_l:.4f} | "
              f"val: rec {va_r:.4f} KL {va_k:.4f} ELBO {va_l:.4f} | warm {warm:.2f}")

        # Save recon preview each epoch
        with torch.no_grad():
            val_batch = next(iter(dl_val)).to(device)[:16]
            recon, _, _ = model(val_batch)
            save_grid(val_batch, os.path.join(run_dir, f"ep{ep:02d}_input.png"), nrow=8)
            save_grid(recon,     os.path.join(run_dir, f"ep{ep:02d}_recon.png"), nrow=8)

        # Save best
        if va_l < best_val:
            best_val = va_l
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

    print(f"Training done in {time.time()-t0:.1f}s | best val ELBO: {best_val:.4f}")

    # -------- Post-training deliverables --------
    # 1) Loss curves
    plot_loss_curves(history, os.path.join(run_dir, "loss_curve.png"))

    # 2) Random samples (from N(0, I))
    with torch.no_grad():
        z = torch.randn(64, args.zdim, device=device)
        samples = model.decode(z)
        save_grid(samples, os.path.join(run_dir, "samples_grid.png"), nrow=8)

    # 3) Latent traversal on two highest-variance μ dimensions
    all_mu = []
    with torch.no_grad():
        for xb in dl_val:
            xb = xb.to(device)
            mu, _ = model.encode(xb)
            all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu, dim=0).numpy()
    var_order = np.argsort(all_mu.var(axis=0))[::-1]
    choose = (int(var_order[0]), int(var_order[1]))
    grid = latent_traversal(model, device, z_dim=args.zdim, steps=9, var_dims=choose)
    save_grid(grid, os.path.join(run_dir, f"latent_traversal_dims_{choose[0]}_{choose[1]}.png"), nrow=9)

    # 4) 2D manifold viz (UMAP or PCA)
    latent_umap_plot(all_mu, os.path.join(run_dir, "latent_umap.png"))

    print("✅ 4.3.1 artifacts saved in:", run_dir)

if __name__ == "__main__":
    main()
