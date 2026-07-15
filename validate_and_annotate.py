import cv2
import datetime
import numpy as np
import torch

from model import BoundingBoxPostProcessing, KMeansAnchors, YOLOv3
from train import collect_gt_boxes_np
from pathlib import Path

# --- COCO class names (80) ---
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

VOC_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
    "cow","diningtable","dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]

ROOT = "/mnt/scratch2/users/40464858/VOC_dataset/voc_yolo"
IMG = 416
NUM_CLASSES = 20

# --- helpers: split heads and scale by stride ---
def split_heads_to_grid(pred_flat, num_classes=NUM_CLASSES, A=3):
    """pred_flat: (B, total, 5+C) -> lg(13), md(26), sm(52) each (B,H,W,A,5+C)"""
    B = pred_flat.shape[0]
    C = 5 + num_classes
    n_lg = 13*13*A
    n_md = 26*26*A
    lg = pred_flat[:, :n_lg, :].reshape(B, 13,13, A, C)
    md = pred_flat[:, n_lg:n_lg+n_md, :].reshape(B, 26,26, A, C)
    sm = pred_flat[:, n_lg+n_md:, :].reshape(B, 52,52, A, C)
    return lg, md, sm

def heads_to_pixels(lg, md, sm, strides=(32,16,8)):
    """Multiply (cx,cy,w,h) by the stride for each head, then concat back to (B, total, 5+C)."""
    def scale_xywh(h, s):
        h = h.clone()
        h[..., 1:5] *= s  # fields: [obj, cx, cy, w, h, class...]
        return h
    lg_px = scale_xywh(lg, strides[0]).reshape(lg.shape[0], -1, lg.shape[-1])
    md_px = scale_xywh(md, strides[1]).reshape(md.shape[0], -1, md.shape[-1])
    sm_px = scale_xywh(sm, strides[2]).reshape(sm.shape[0], -1, sm.shape[-1])
    return torch.cat([lg_px, md_px, sm_px], dim=1)

# --- single-image inference ---
def infer_one(img_path, model, num_classes=NUM_CLASSES, conf_thres=0.25, iou_thres=0.5, out_path="pred_vis.jpg"):
    print(f"Running inference on {img_path} at {datetime.datetime.now()}")
    device = next(model.parameters()).device

    img0 = cv2.imread(img_path)
    assert img0 is not None, f"Image not found: {img_path}"

    # resize exactly like training
    img = cv2.resize(img0, (IMG, IMG), interpolation=cv2.INTER_LINEAR)

    # BGR->RGB safely (no negative strides)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image_batch = torch.from_numpy(img_rgb.transpose(2,0,1).copy()).float().unsqueeze(0) / 255.0
    input_image_batch = input_image_batch.to(device)

    model.eval()
    with torch.no_grad():
        all_candidate_predictions = model(input_image_batch)  # (1, total, 5+C)

    # split heads and scale to pixels
    lg, md, sm = split_heads_to_grid(all_candidate_predictions, num_classes=NUM_CLASSES, A=3)
    preds_px = heads_to_pixels(lg, md, sm, strides=(32,16,8))  # (1, total, 5+C) cxcywh in pixels

    # debug
    p = all_candidate_predictions[0]
    obj = p[:, 0]
    cls_prob, cls_idx = p[:, 5:].max(dim=1)
    conf = (obj * cls_prob)
    print("max obj:", float(obj.max()))
    print("max cls_prob:", float(cls_prob.max()))
    print("max conf (obj*cls):", float(conf.max()))

    # post-process
    print("Using conf_thres =", conf_thres)
    pp = BoundingBoxPostProcessing(preds_px, box_format="cxcywh")
    scores, boxes, classes = pp.filter_boxes(score_threshold=conf_thres)
    scores, boxes, classes = pp.non_max_suppression(scores, boxes, classes, iou_threshold=iou_thres)

    # debug
    print("After filter:",
          scores[0].numel(), "boxes  |  best conf:",
          (scores[0].max().item() if scores[0].numel() else 0.0))
    print("Top-5 objectness (pre-filter) snapshot:")
    with torch.no_grad():
        obj5 = all_candidate_predictions[0, :, 0]
        topk = torch.topk(obj5, k=min(5, obj5.numel())).values.cpu().numpy()
        print(topk)

    # draw on 416×416 image with class names
    vis = img.copy()
    if boxes[0].numel():
        for (x1,y1,x2,y2), s, c in zip(
            boxes[0].cpu().numpy(),
            scores[0].cpu().numpy(),
            classes[0].cpu().numpy()
        ):
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            cls_id = int(c)
            name = VOC_NAMES[cls_id] if 0 <= cls_id < len(VOC_NAMES) else str(cls_id)

            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"{name} {s:.2f}"
            cv2.putText(vis, label, (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    print(f"Running inference on a single image with YOLOv3... at {datetime.datetime.now()}")
    # --- rebuild anchors exactly like in training (or load the same ones you used) ---
    anchor_file = Path(ROOT) / f"anchors_{IMG}_train.npy"

    if anchor_file.exists():
        anchors_px = np.load(anchor_file)
        print(f"Loaded anchors from {anchor_file}")
    else:
        print("Calculating anchors...")

        gt_boxes_np = collect_gt_boxes_np(
            ROOT,
            img_size=IMG,
            split="train",
        )

        anchors_px = KMeansAnchors(
            gt_boxes_np,
            n_anchors=9,
        ).get_anchors()

        np.save(anchor_file, anchors_px)
        print(f"Saved anchors to {anchor_file}")

    print("Anchors (px):", anchors_px)
    anchors = torch.tensor(anchors_px, dtype=torch.float32)
    anchors[:3]  /= 8.0
    anchors[3:6] /= 16.0
    anchors[6:9] /= 32.0

    # --- build model & load weights ---
    model = YOLOv3(input_channels=3, anchors=anchors, n_classes=NUM_CLASSES)
    checkpoint = "yolov3_scratch_voc.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # --- run on one image ---
    # 000000000113.jpg shows two persons cutting a cake.
    infer_one("/mnt/scratch2/users/40464858/coco128/images/train2017/000000000113.jpg", model, num_classes=NUM_CLASSES,
              conf_thres=0.05, iou_thres=0.50, out_path="pred_vis_full_training.jpg")
