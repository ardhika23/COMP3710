# train_cifar_fast.py — ResNet-18 from scratch (CIFAR-10), CUDA/MPS optimized #3.3
import os, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# ---------------- Args ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None,
                   help="PARENT folder that contains 'cifar-10-batches-py'. Default = this script folder.")
    p.add_argument("--batch", type=int, default=None, help="Default: 1024 (CUDA), 256 (MPS/CPU)")
    p.add_argument("--epochs", type=int, default=30, help="OneCycle (fast to target)")
    p.add_argument("--lr", type=float, default=None, help="max_lr for OneCycle (auto if not set)")
    p.add_argument("--opt", type=str, default="sgd", choices=["sgd","adamw"])
    p.add_argument("--target", type=float, default=0.93)
    p.add_argument("--workers", type=int, default=None, help="Default: SLURM_CPUS_PER_TASK (CUDA) or 0 (MPS/CPU)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable AMP")
    p.add_argument("--compile", action="store_true", help="torch.compile (CUDA only)")
    p.add_argument("--fast", action="store_true", help="lighter aug (crop+flip only)")
    return p.parse_args()

args = parse_args()

# ---------------- Device & seeds ----------------
if torch.cuda.is_available():
    device, device_type = "cuda", "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device, device_type = "mps", "mps"
else:
    device, device_type = "cpu", "cpu"

AMP_ENABLED = (not args.no_amp) and (device != "cpu")
AMP_DTYPE = torch.float16

if device == "cuda":
    # A100 speed knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

np.random.seed(args.seed); torch.manual_seed(args.seed)
print("Device:", device)

# ---------------- Data root ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.abspath(os.path.expanduser(args.data_root or script_dir))
cifar_dir = os.path.join(data_root, "cifar-10-batches-py")
if not os.path.isdir(cifar_dir):
    raise FileNotFoundError(
        f"Could not find folder: {cifar_dir}\n"
        "Put 'cifar-10-batches-py/' next to this script, "
        "or run with --data-root \"/path/to/parent\"."
    )
print("Using data from:", cifar_dir)

# ---------------- Batch / workers defaults ----------------
if args.batch is None:
    batch = 1024 if device == "cuda" else 256
else:
    batch = args.batch

if args.workers is None:
    if device == "cuda":
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    else:
        workers = 0
else:
    workers = args.workers

use_cuda = (device == "cuda")
pin = use_cuda
print(f"Batch={batch} | workers={workers} | AMP={'on' if AMP_ENABLED else 'off'}")

# ---------------- Augmentations ----------------
MEAN, STD = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
if args.fast:
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
else:
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

train_set = datasets.CIFAR10(data_root, train=True,  download=False, transform=train_tfms)
test_set  = datasets.CIFAR10(data_root, train=False, download=False, transform=test_tfms)
print(f"Train: {len(train_set)} | Test: {len(test_set)}")

# DataLoader: only enable prefetch/persistent when workers>0
dl_common = dict(num_workers=workers, pin_memory=use_cuda)
worker_opts = {}
if workers > 0:
    worker_opts.update(dict(persistent_workers=True, prefetch_factor=4))

train_loader = DataLoader(
    train_set, batch_size=batch, shuffle=True, drop_last=True,
    **dl_common, **worker_opts
)
test_loader  = DataLoader(
    test_set, batch_size=batch, shuffle=False,
    **dl_common, **worker_opts
)

# ---------------- ResNet-18 (from scratch, CIFAR variant) ----------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)

class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, num_classes)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = ResNet18_CIFAR().to(device)
if device == "cuda":
    model = model.to(memory_format=torch.channels_last)

if args.compile and device == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile: enabled")
    except Exception as e:
        print("torch.compile: skipped ->", e)

print("Model on:", next(model.parameters()).device)

# ---------------- Optim / Sched / Loss ----------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

if args.opt == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9,
                                weight_decay=5e-4, nesterov=True, foreach=True)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)

# Auto max_lr if not provided
if args.lr is None:
    if device == "cuda":
        max_lr = 0.4 * (batch / 1024)   # DAWNBench-ish scaling
    else:
        max_lr = 0.10 * (batch / 256)   # stable for MPS/CPU
else:
    max_lr = args.lr
print(f"max_lr = {max_lr:.3f}")

steps_per_epoch = len(train_loader)
scheduler = OneCycleLR(
    optimizer, max_lr=max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
    pct_start=0.1, div_factor=10.0, final_div_factor=100.0
)

# AMP scaler (CUDA only, new API)
if device == "cuda":
    scaler = GradScaler("cuda", enabled=AMP_ENABLED)
else:
    scaler = GradScaler(enabled=False)

# ---------------- Eval ----------------
@torch.no_grad()
def evaluate():
    model.eval(); n_ok = n_tot = 0
    with autocast(device_type=device_type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
        for xb,yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            if device == "cuda":
                xb = xb.to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            n_ok += (logits.argmax(1) == yb).sum().item()
            n_tot += xb.size(0)
    return n_ok/n_tot

# ---------------- Train ----------------
best_acc, t0 = 0.0, time.time()
for ep in range(1, args.epochs+1):
    model.train()
    ep_t0 = time.time()
    for xb,yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        if device == "cuda":
            xb = xb.to(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device_type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
            logits = model(xb)
            loss = criterion(logits, yb)

        if device == "cuda" and AMP_ENABLED:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()

        scheduler.step()

    ep_time = time.time() - ep_t0
    acc = evaluate(); best_acc = max(best_acc, acc)
    imgs_per_s = len(train_set) / max(ep_time, 1e-6)
    print(f"Epoch {ep:02d}/{args.epochs} | acc {acc:.4f} | best {best_acc:.4f} "
          f"| {imgs_per_s:.0f} img/s | lr {scheduler.get_last_lr()[0]:.3e}")

    if acc >= args.target:
        print(f"Reached target {args.target:.2%} at epoch {ep} in {time.time()-t0:.1f}s")
        break

print(f"Final acc: {evaluate():.4f} | total time: {time.time()-t0:.1f}s")
