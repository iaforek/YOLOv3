import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import KMeansAnchors, YOLOv3

# Path to VOC dataset
ROOT = "/mnt/scratch2/users/40464858/VOC_dataset/voc_yolo"
IMG, BATCH, EPOCHS, NUM_CLASSES = 416, 32, 100, 20
LEARNING_RATE = 1e-3

class YoloTxtDataset(Dataset):
    def __init__(self, root=ROOT, split="train", img_size=IMG):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.img_size = img_size
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def _label_path(self, img_path):
        base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        return os.path.join(self.lbl_dir, base)

    def _read_labels(self, path, W, H):
        if not os.path.exists(path) or os.path.getsize(path)==0:
            return np.zeros((0,5), np.float32)
        arr = np.loadtxt(path, ndmin=2, dtype=np.float32) # [cls, xc, yc, w, h] in 0..1

        # Basic label sanity check
        if np.isnan(arr).any():
            raise ValueError(f"NaN found in label file: {path}")
        if arr.shape[1] != 5:
            raise ValueError(f"Bad label shape in {path}: {arr.shape} (expected Nx5)")
        cls_ids = arr[:, 0]
        if (cls_ids < 0).any() or (cls_ids >= NUM_CLASSES).any():
            bad = cls_ids[(cls_ids < 0) | (cls_ids >= NUM_CLASSES)]
            raise ValueError(f"Class id(s) out of range in {path}: {bad}")

        cls, xc, yc, w, h = arr.T
        return np.stack([cls, xc*W, yc*H, w*W, h*H], axis=1).astype(np.float32)  # pixels

    def __getitem__(self, i):
        p = self.img_paths[i]
        img = Image.open(p).convert("RGB")
        width, height = img.size
        labels = self._read_labels(self._label_path(p), width, height)   # (N,5) [cls, x, y, w, h] in px

        # simple square resize
        img_size = self.img_size
        img = img.resize((img_size, img_size))
        sx, sy = img_size/width, img_size/height
        if labels.size:
            labels[:,1] *= sx
            labels[:,2] *= sy
            labels[:,3] *= sx
            labels[:,4] *= sy

        x = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
        t = torch.from_numpy(labels)  # (N,5)
        return x, t

def collate_fn(batch):
    imgs, targs = list(zip(*batch))
    imgs = torch.stack(imgs, 0)
    out = []
    for i, t in enumerate(targs):
        if t.numel():
            idx = torch.full((t.shape[0],1), i, dtype=t.dtype)
            out.append(torch.cat([idx, t], dim=1))  # (img_idx, cls, x, y, w, h)
    out = torch.cat(out, 0) if out else torch.zeros((0,6), dtype=torch.float32)
    return imgs, out

# --- collect GT widths/heights (pixels at 416) for KMeansAnchors
def collect_gt_boxes_np(root=ROOT, img_size=IMG, split="train"):
    lbl_dir = os.path.join(root, "labels", split)
    rows = []
    for txt in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
        if os.path.getsize(txt) == 0:
            continue
        arr = np.loadtxt(txt, ndmin=2, dtype=np.float32)  # [cls, xc, yc, w, h] in 0..1
        if arr.size == 0:
            continue
        w_px = arr[:, 3] * img_size
        h_px = arr[:, 4] * img_size
        zeros = np.zeros_like(w_px)
        rows.append(np.stack([zeros, zeros, w_px, h_px], axis=1))
    return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0,4), np.float32)

def anchors_pixels_to_grid(centers_px):
    a = torch.tensor(centers_px, dtype=torch.float32).clone()
    a[:3]  /= 8.0   # 52x52 head
    a[3:6] /= 16.0  # 26x26 head
    a[6:9] /= 32.0  # 13x13 head
    return a

# --- (very) simple per-head target & loss (starter) ---
def iou_wh(box_wh, anc_wh):
    b = box_wh[:,None,:]
    a = anc_wh[None,:,:]
    inter = torch.min(b, a).prod(-1)
    area_b = b[...,0]*b[...,1]
    area_a = a[...,0]*a[...,1]
    return inter / (area_b + area_a - inter + 1e-9)

def build_targets_dense(targets_px, batch_size, imgsz=IMG, strides=(8,16,32), anchors_per_scale=None, num_classes=NUM_CLASSES, device="cpu"):
    batch_size = int(batch_size)
    outs = []
    for s, anc in zip(strides, anchors_per_scale):
        height = width = imgsz // s
        obj  = torch.zeros((batch_size,height,width,anc.shape[0],1), device=device)
        cls  = torch.zeros((batch_size,height,width,anc.shape[0],num_classes), device=device)
        xywh = torch.zeros((batch_size,height,width,anc.shape[0],4), device=device)

        if targets_px.numel():
            t = targets_px.to(device)  # (M,6)
            b, c, x, y, w, h = t.T
            # Grid coordinates
            gx, gy = x/s, y/s
            gw, gh = w/s, h/s
            gi, gj = gx.long().clamp(0, width-1), gy.long().clamp(0, height-1)
            ious = iou_wh(torch.stack([gw,gh], dim=1), anc.to(device))
            best_a = ious.argmax(1)
            for n in range(t.shape[0]):
                bi = int(b[n].item())
                ai = int(best_a[n].item())
                ci = int(c[n].item())
                i = int(gi[n].item())
                j = int(gj[n].item())

                obj[bi,j,i,ai,0]  = 1.0
                cls[bi,j,i,ai,ci] = 1.0
                xywh[bi,j,i,ai,:] = torch.tensor([gx[n], gy[n], gw[n], gh[n]], device=device)

        outs.append(dict(obj=obj, cls=cls, xywh=xywh))
    return outs

def split_heads_to_grid(pred_flat, anchors=3, num_classes=NUM_CLASSES):
    batch_size = pred_flat.shape[0]
    classes = 5 + num_classes
    n_lg = 13 * 13 * anchors
    n_md = 26 * 26 * anchors
    lg = pred_flat[:, :n_lg, :].reshape(batch_size,13,13,anchors,classes)
    md = pred_flat[:, n_lg:n_lg+n_md, :].reshape(batch_size,26,26,anchors,classes)
    sm = pred_flat[:, n_lg+n_md:, :].reshape(batch_size,52,52,anchors,classes)
    return lg, md, sm

def yolo_head_loss(pred, tgt, lambda_obj=1.0, lambda_cls=1.0, lambda_box=5.0):
    obj_p = pred[...,0:1]
    xywh_p = pred[...,1:5]
    cls_p = pred[...,5:]
    obj_t = tgt["obj"]
    xywh_t = tgt["xywh"]
    cls_t = tgt["cls"]

    # Sanity checks to prevent CUDA device-side asserts
    assert torch.isfinite(obj_t).all(), "obj_t has NaN/Inf"
    assert torch.isfinite(cls_t).all(), "cls_t has NaN/Inf"
    mn_o, mx_o = float(obj_t.min()), float(obj_t.max())
    assert 0.0 <= mn_o <= 1.0 and 0.0 <= mx_o <= 1.0, f"obj_t outside [0,1]: {mn_o}..{mx_o}"
    if cls_t.numel():
        mn_c, mx_c = float(cls_t.min()), float(cls_t.max())
        assert 0.0 <= mn_c <= 1.0 and 0.0 <= mx_c <= 1.0, f"cls_t outside [0,1]: {mn_c}..{mx_c}"

    l_obj = F.binary_cross_entropy(obj_p, obj_t, reduction="mean")
    pos = obj_t.bool()
    if pos.any():
        l_cls = F.binary_cross_entropy(cls_p[pos.expand_as(cls_p)], cls_t[pos.expand_as(cls_t)], reduction="mean")
        l_box = F.smooth_l1_loss(xywh_p[pos.expand_as(xywh_p)], xywh_t[pos.expand_as(xywh_t)], reduction="mean")
    else:
        l_cls = torch.tensor(0.0, device=pred.device)
        l_box = torch.tensor(0.0, device=pred.device)
    return lambda_obj*l_obj + lambda_cls*l_cls + lambda_box*l_box

def train():
    torch.backends.cudnn.benchmark = True

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA build:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = YoloTxtDataset(root=ROOT, split="train", img_size=IMG)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    print("Collecting GT boxes for KMeans...")
    gt_boxes_np = collect_gt_boxes_np(ROOT, img_size=IMG, split="train")
    if gt_boxes_np.shape[0] == 0:
        raise RuntimeError("No GT boxes collected; check label files content.")

    centers_px = KMeansAnchors(gt_boxes_np, n_anchors=9).get_anchors()
    anchors_grid = anchors_pixels_to_grid(centers_px)

    model = YOLOv3(input_channels=3, anchors=anchors_grid, n_classes=NUM_CLASSES).to(device)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    anc_sm = anchors_grid[0:3].to(device)
    anc_md = anchors_grid[3:6].to(device)
    anc_lg = anchors_grid[6:9].to(device)

    print("Starting training...")
    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        total = 0.0
        for images, targets in dl:
            images = images.to(device)
            targets = targets.to(device)  # (M,6) [img_idx, cls, x, y, w, h] in px @ 416

            # Guard class ids per batch
            if targets.numel():
                classes = int(targets[:, 1].max().item())
                assert 0 <= classes < NUM_CLASSES, f"Class id {classes} out of range"

            tgts = build_targets_dense(targets,
                                       imgsz=IMG,
                                       batch_size=images.shape[0],
                                       strides=(8,16,32),
                                       anchors_per_scale=(anc_sm, anc_md, anc_lg),
                                       num_classes=NUM_CLASSES,
                                       device=device)

            preds = model(images)  # (B, 10647, 5+NUM_CLASSES)
            lg, md, sm = split_heads_to_grid(preds, anchors=3, num_classes=NUM_CLASSES)

            loss = yolo_head_loss(lg, tgts[2]) + yolo_head_loss(md, tgts[1]) + yolo_head_loss(sm, tgts[0])

            opt.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            opt.step()
            total += loss.item()

        # Print epoch summary
        dt = time.time() - start
        imgs_per_sec = len(ds) / dt
        print(f"Epoch {epoch+1}/{EPOCHS} loss={total/len(dl):.4f} time={dt:.1f}s speed={imgs_per_sec:.1f} img/s")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"ckpt_voc_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "yolov3_scratch_voc.pth")
    print("Saved → yolov3_scratch_voc.pth")

if __name__ == "__main__":
    train()
