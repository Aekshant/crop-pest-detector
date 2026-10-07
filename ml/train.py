#!/usr/bin/env python3
"""Fine-tune a small mobile backbone on ml/data/manifest.csv (task V3).

  python ml/train.py --out ml/runs/v3_lite0_cpu                     # CPU-sized defaults
  python ml/train.py --out ml/runs/v3_lite0_gpu --samples-per-epoch 0 --epochs-ft 15 \
      --unfreeze-blocks 0 --batch 128                                # rented GPU, full fine-tune

Two stages (transfer learning):
  1. head:      backbone frozen; features are computed once and cached, then
                only the classifier is trained (seconds, even on CPU).
  2. fine-tune: the last --unfreeze-blocks blocks + head train with
                augmentation (0 = whole network).

Sampling is balanced per class, and field photos (PlantDoc) get at least
--field-share of each class's samples, because they are ~3% of the data but
are what the app sees. After training: temperature is fitted on validation,
then the vision eval sets are predicted and scored with evals/score_vision.py.

Outputs in --out: model.pt (best checkpoint + metadata), train_log.jsonl,
train_log.txt, eval_preds.jsonl, eval_report.txt, eval_report.json.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ML = Path(__file__).resolve().parent
REPO = ML.parent
LABELS_V1 = ML / "labels_v1.json"
MANIFEST = ML / "data" / "manifest.csv"
EVAL_VISION = REPO / "evals" / "vision"
FIELD_SOURCES = {"plantdoc", "inaturalist", "commons"}


# ---------- data ----------

def load_labels():
    return [l["id"] for l in json.loads(LABELS_V1.read_text())["labels"]]


def split_rows(rows, val_frac_field, val_max, seed):
    """Train/val split. PlantVillage and IP102 keep their own (leaf-grouped /
    official) splits; PlantDoc has only train, so hold out a hash-stable slice."""
    train, val = [], []
    for r in rows:
        if r["source"] in FIELD_SOURCES:
            h = int(hashlib.sha256((r["sha256"] + "val").encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            (val if h < val_frac_field else train).append(r)
        elif r["split"] == "train":
            train.append(r)
        else:
            val.append(r)
    # Cap lab-photo validation per class so CPU runs stay quick; keep all field val.
    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for r in val:
        by_class[(r["label"], r["source"] in FIELD_SOURCES)].append(r)
    capped = []
    for (label, field), rs in sorted(by_class.items()):
        rng.shuffle(rs)
        capped += rs if field or not val_max else rs[:val_max]
    return train, capped


class ImageRows(Dataset):
    def __init__(self, paths, targets, tf):
        self.paths, self.targets, self.tf = paths, targets, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), self.targets[i]


def transforms(cfg, img):
    norm = T.Normalize(cfg["mean"], cfg["std"])
    # Eval matches the phone: resize, center crop to the model input.
    eval_tf = T.Compose([T.Resize(int(img * 256 / 224)), T.CenterCrop(img), T.ToTensor(), norm])
    train_tf = T.Compose([
        T.RandomResizedCrop(img, scale=(0.35, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(0.35, 0.35, 0.3, 0.04),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),  # phone focus misses
        T.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


def sample_weights(rows, field_share):
    """Equal total weight per class; within a class, field photos get at least
    field_share of it (when the class has any)."""
    groups = collections.Counter((r["label"], r["source"] in FIELD_SOURCES) for r in rows)
    labels = {r["label"] for r in rows}
    weights = []
    for r in rows:
        label, field = r["label"], r["source"] in FIELD_SOURCES
        n_field, n_lab = groups[(label, True)], groups[(label, False)]
        if n_field and n_lab:
            share = max(field_share, n_field / (n_field + n_lab))
            part = share if field else 1 - share
        else:
            part = 1.0
        weights.append(part / groups[(label, field)] / len(labels))
    return weights


# ---------- model ----------

def build_model(arch, n_classes):
    return timm.create_model(arch, pretrained=True, num_classes=n_classes)


def set_trainable(model, unfreeze_blocks):
    """Freeze all, then unfreeze the last N blocks + head (N=0: everything)."""
    if unfreeze_blocks == 0:
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    modules = [model.conv_head, model.bn2, model.classifier]
    if unfreeze_blocks > 0:
        modules += list(model.blocks[-unfreeze_blocks:])
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


def keep_frozen_bn_eval(model):
    # Frozen BatchNorm layers must keep their ImageNet running stats.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


@torch.no_grad()
def features(model, loader, device):
    model.eval()
    feats, ys = [], []
    for x, y in loader:
        feats.append(model.forward_head(model.forward_features(x.to(device)), pre_logits=True).cpu())
        ys.append(y)
    return torch.cat(feats), torch.cat(ys)


@torch.no_grad()
def logits_of(model, loader, device):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def macro_recall(pred, y, labels):
    present = sorted(set(y.tolist()))
    recall = {labels[c]: ((pred == c) & (y == c)).sum().item() / (y == c).sum().item() for c in present}
    return (sum(recall.values()) / len(recall) if recall else 0.0), recall


def metrics(logits, y, labels, field):
    """field: bool mask of validation rows that are field photos. Model selection
    uses the mean of lab and field macro recall, so a model can't win on lab
    photos alone."""
    pred = logits.argmax(1)
    mr, recall = macro_recall(pred, y, labels)
    field_mr, field_recall = macro_recall(pred[field], y[field], labels)
    lab_mr, _ = macro_recall(pred[~field], y[~field], labels)
    return {
        "acc": (pred == y).float().mean().item(),
        "macro_recall": mr,
        "lab_macro_recall": lab_mr,
        "field_macro_recall": field_mr,
        "select": (lab_mr + field_mr) / 2,
        "loss": F.cross_entropy(logits, y).item(),
        "recall": recall,
        "field_recall": field_recall,
    }


def choose_reject_threshold(probs, y, unknown, max_false_reject):
    """Highest reject threshold that turns away at most max_false_reject of
    in-scope field validation photos. Chosen on validation, never on the eval sets."""
    keep = y != unknown
    probs, top = probs[keep].max(1)
    best = 0.2
    for t in [x / 100 for x in range(20, 81, 5)]:
        if ((probs < t) | (top == unknown)).float().mean().item() <= max_false_reject:
            best = t
    return best


def fit_temperature(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return log_t.exp().item()


# ---------- main ----------

class Log:
    def __init__(self, out):
        self.jsonl = (out / "train_log.jsonl").open("a")
        self.txt = (out / "train_log.txt").open("a")

    def __call__(self, msg, **record):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.txt.write(line + "\n")
        self.txt.flush()
        if record:
            self.jsonl.write(json.dumps({"time": time.time(), **record}) + "\n")
            self.jsonl.flush()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", default="tf_efficientnet_lite0")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs-head", type=int, default=30)
    ap.add_argument("--epochs-ft", type=int, default=5)
    ap.add_argument("--samples-per-epoch", type=int, default=4000, help="fine-tune samples per epoch; 0 = one pass over train")
    ap.add_argument("--unfreeze-blocks", type=int, default=2, help="blocks to fine-tune; 0 = whole network")
    ap.add_argument("--lr-head", type=float, default=2e-3)
    ap.add_argument("--lr-ft", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--field-share", type=float, default=0.3)
    ap.add_argument("--val-frac-field", type=float, default=0.15)
    ap.add_argument("--val-max", type=int, default=120, help="max lab-photo val images per class (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-false-reject", type=float, default=0.2,
                    help="reject threshold is chosen so at most this share of field val photos is turned away")
    ap.add_argument("--accept", type=float, default=0.8, help="confident threshold")
    ap.add_argument("--finalize-only", action="store_true",
                    help="skip training; calibrate, save and evaluate from --out/best_state.pt")
    ap.add_argument("--limit-train", type=int, default=0, help="smoke test: use N random train and val images")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    log = Log(args.out)

    labels = load_labels()
    index = {l: i for i, l in enumerate(labels)}
    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()[:12]
    train_rows, val_rows = split_rows(rows, args.val_frac_field, args.val_max, args.seed)
    if args.limit_train:
        rng = random.Random(args.seed)
        train_rows = rng.sample(train_rows, min(args.limit_train, len(train_rows)))
        val_rows = rng.sample(val_rows, min(args.limit_train, len(val_rows)))
    log(f"device={device} arch={args.arch} train={len(train_rows)} val={len(val_rows)} manifest={manifest_sha}",
        event="start", args={k: str(v) for k, v in vars(args).items()}, device=device,
        train=len(train_rows), val=len(val_rows), manifest_sha=manifest_sha, commit=git_commit())
    counts = collections.Counter(r["label"] for r in train_rows)
    missing = [l for l in labels if counts[l] == 0]
    log("train per class: " + ", ".join(f"{l}={counts[l]}" for l in labels))
    if missing:
        log(f"WARNING: no training images for {missing}; the model cannot learn these classes yet")

    model = build_model(args.arch, len(labels)).to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    train_tf, eval_tf = transforms(cfg, args.img)

    def loader(rs, tf, shuffle=False, sampler=None):
        ds = ImageRows([ML / "data" / r["path"] for r in rs], [index[r["label"]] for r in rs], tf)
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, sampler=sampler,
                          num_workers=args.workers, pin_memory=device == "cuda", persistent_workers=False)

    val_loader = loader(val_rows, eval_tf)
    weights = sample_weights(train_rows, args.field_share)
    best = {"select": -1}
    val_field = torch.tensor([r["source"] in FIELD_SOURCES for r in val_rows])

    def save(stage, epoch, m):
        nonlocal best
        if m["select"] > best["select"]:
            best = {**m, "stage": stage, "epoch": epoch}
            torch.save(model.state_dict(), args.out / "best_state.pt")

    t0 = time.time()
    if args.finalize_only:
        # Finish an interrupted run from its saved best_state.pt.
        epochs = [json.loads(l) for l in (args.out / "train_log.jsonl").read_text().splitlines()]
        epochs = [e for e in epochs if e.get("event") == "epoch"]
        top = max(epochs, key=lambda e: e.get("select", e["macro_recall"]))
        best = {k: v for k, v in top.items() if k not in ("time", "event")}
        log(f"finalize only: using best_state.pt from {best['stage']} epoch {best['epoch']}", event="finalize")
    else:
        # ----- stage 1: head on cached features -----
        set_trainable(model, -1)
        log("stage 1: caching backbone features (no augmentation)")
        ftr, ytr = features(model, loader(train_rows, eval_tf), device)
        fval, yval = features(model, val_loader, device)
        log(f"features cached in {time.time() - t0:.0f}s: train {tuple(ftr.shape)}, val {tuple(fval.shape)}")
        head = model.classifier
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        sw = torch.tensor(weights, dtype=torch.float32)
        sw = sw / sw.mean()
        for epoch in range(1, args.epochs_head + 1):
            head.train()
            perm = torch.randperm(len(ftr))
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                loss = (F.cross_entropy(head(ftr[b].to(device)), ytr[b].to(device), reduction="none",
                                        label_smoothing=args.label_smoothing) * sw[b].to(device)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                m = metrics(head(fval.to(device)).cpu(), yval, labels, val_field)
            log(f"head epoch {epoch:2d}: val macro recall lab {m['lab_macro_recall']:.3f} "
                f"field {m['field_macro_recall']:.3f} | loss {m['loss']:.3f}",
                event="epoch", stage="head", epoch=epoch, **m)
            save("head", epoch, m)

        # ----- stage 2: fine-tune last blocks with augmentation -----
        model.load_state_dict(torch.load(args.out / "best_state.pt"))
        set_trainable(model, args.unfreeze_blocks)
        params = [p for p in model.parameters() if p.requires_grad]
        log(f"stage 2: fine-tuning {sum(p.numel() for p in params) / 1e6:.2f}M params "
            f"(unfreeze_blocks={args.unfreeze_blocks})")
        n = args.samples_per_epoch or len(train_rows)
        opt = torch.optim.AdamW(params, lr=args.lr_ft, weight_decay=args.weight_decay)
        total_steps = args.epochs_ft * math.ceil(n / args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr_ft, total_steps=max(total_steps, 1), pct_start=0.15)
        for epoch in range(1, args.epochs_ft + 1):
            t = time.time()
            model.train()
            keep_frozen_bn_eval(model)
            sampler = torch.utils.data.WeightedRandomSampler(weights, n, replacement=True)
            seen, run_loss = 0, 0.0
            for x, y in loader(train_rows, train_tf, sampler=sampler):
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                seen += len(y)
                run_loss += loss.item() * len(y)
            vl, vy = logits_of(model, val_loader, device)
            m = metrics(vl, vy, labels, val_field)
            log(f"ft epoch {epoch:2d}: train loss {run_loss / seen:.3f} | val macro recall "
                f"lab {m['lab_macro_recall']:.3f} field {m['field_macro_recall']:.3f} | loss {m['loss']:.3f} "
                f"| {time.time() - t:.0f}s",
                event="epoch", stage="ft", epoch=epoch, train_loss=run_loss / seen, **m)
            save("ft", epoch, m)

    # ----- calibrate, choose thresholds, save, evaluate -----
    state = args.out / "best_state.pt"
    if state.exists():
        model.load_state_dict(torch.load(state))
    else:  # --finalize-only on a run that already produced model.pt
        model.load_state_dict(torch.load(args.out / "model.pt")["state_dict"])
    vl, vy = logits_of(model, val_loader, device)
    # Calibrate on field photos: lab photos make the model look more certain
    # than it is in the field (V3 baseline: ECE 0.36 on the field eval set).
    use_field = val_field.sum() >= 50
    calib = val_field if use_field else torch.ones_like(val_field)
    temperature = fit_temperature(vl[calib], vy[calib])
    log(f"best: stage {best['stage']} epoch {best['epoch']} val macro recall lab "
        f"{best.get('lab_macro_recall', float('nan')):.3f} field {best.get('field_macro_recall', float('nan')):.3f}; "
        f"temperature {temperature:.3f} (fitted on {int(calib.sum())} {'field' if use_field else 'all'} val images)", event="best", best={k: v for k, v in best.items()}, temperature=temperature)
    reject = choose_reject_threshold(F.softmax(vl[val_field] / temperature, dim=1), vy[val_field],
                                     labels.index("unknown"), args.max_false_reject)
    thresholds = {"reject": reject, "accept": args.accept}
    log(f"reject threshold {reject:.2f} (rejects <= {args.max_false_reject:.0%} of in-scope field val photos); "
        f"confident >= {args.accept:.2f}", event="thresholds", **thresholds)
    (args.out / "thresholds.json").write_text(json.dumps({"temperature": temperature, **thresholds}, indent=2) + "\n")
    torch.save({
        "state_dict": model.state_dict(),
        "thresholds": thresholds,
        "arch": args.arch, "img": args.img, "mean": cfg["mean"], "std": cfg["std"],
        "labels": labels, "labels_version": 1, "temperature": temperature,
        "val": best, "manifest_sha": manifest_sha, "commit": git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }, args.out / "model.pt")
    state.unlink(missing_ok=True)

    eval_files = []
    for sub in ("field", "ood"):
        with (EVAL_VISION / sub / "labels.csv").open(newline="") as f:
            eval_files += [f"{sub}/{r['file']}" for r in csv.DictReader(f)]
    ds = ImageRows([EVAL_VISION / p for p in eval_files], [0] * len(eval_files), eval_tf)
    el, _ = logits_of(model, DataLoader(ds, batch_size=args.batch, num_workers=args.workers), device)
    probs = F.softmax(el / temperature, dim=1)
    with (args.out / "eval_preds.jsonl").open("w") as f:
        for file, p in zip(eval_files, probs):
            f.write(json.dumps({"file": file, "probs": {l: round(v, 5) for l, v in zip(labels, p.tolist())},
                                "quality": "ok"}) + "\n")
    report = subprocess.run([sys.executable, str(REPO / "evals" / "score_vision.py"),
                             str(args.out / "eval_preds.jsonl"), "--json", str(args.out / "eval_report.json"),
                             "--reject", str(reject), "--accept", str(args.accept)],
                            capture_output=True, text=True)
    (args.out / "eval_report.txt").write_text(report.stdout + report.stderr)
    log("eval report:\n" + report.stdout, event="eval", returncode=report.returncode)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
