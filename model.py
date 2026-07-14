import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.cluster import KMeans
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import BasicBlock, ResNet

os.environ["OMP_NUM_THREADS"] = "4" # Windows MKL KMeans warning

class KMeansAnchors:
    def __init__(self, boxes: np.ndarray, n_anchors: int = 9, n_init: int = 30, max_iter: int = 1000):
        """
        boxes: np.ndarray of shape (N, 4) with columns [x, y, w, h] (pixels or normalized consistently)
        n_anchors: total number of anchors to learn (YOLOv3 default = 9)
        """
        self.boxes = boxes
        self.n_anchors = n_anchors
        self.km = KMeans(n_clusters=n_anchors, n_init=n_init, max_iter=max_iter, random_state=0)

    def get_wh(self) -> np.ndarray:
        """Return (N,2) array of widths and heights."""
        return self.boxes[:, 2:4]

    def get_anchors(self) -> np.ndarray:
        """Return (n_anchors, 2) = (w,h) sorted small -> large by area."""
        wh = self.get_wh()
        self.km.fit(wh)
        centers = self.km.cluster_centers_.astype(np.float32)  # (k,2)
        order = np.argsort(centers[:, 0] * centers[:, 1])      # by area
        return centers[order]

class Upsampler(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsampler, self).__init__()

        self.interpolate = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interpolate(
            x, 
            scale_factor=self.scale_factor,
            mode=self.mode)
        return x

class FeatureMapper(ResNet):
    def __init__(self, input_channels, n_anchors_per_scale, n_classes, pretrained=True):
        super(FeatureMapper, self).__init__(BasicBlock, [2, 2, 2, 2])

        if pretrained and input_channels == 3:
            self.load_state_dict(resnet18(weights=ResNet18_Weights.DEFAULT).state_dict())

        self.input_channels = input_channels
        self.n_anchors_per_scale = n_anchors_per_scale
        self.n_classes = n_classes
        self.output_channels = self.n_anchors_per_scale*(5 + self.n_classes)

        #init resnet18 weights
        if pretrained and self.input_channels == 3:
            self.load_state_dict(torchvision.models.resnet18(pretrained=True).state_dict())

        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False)

        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3),
            bias=False)

        sm_c, md_c, lg_c = 512, 256, 128

        self.lg_fmapper = nn.Conv2d(
            sm_c, self.output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False)

        self.md_fmapper = nn.Conv2d(
            sm_c+md_c, self.output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False)

        self.sm_fmapper = nn.Conv2d(
            sm_c+md_c+lg_c, self.output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False)

        self.upsampler = Upsampler(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        addon_1 = self.layer2(x)
        addon_2 = self.layer3(addon_1)

        fmap = self.layer4(addon_2)
        lg_scale_pred = self.lg_fmapper(fmap)

        fmap = self.upsampler(fmap)
        fmap = torch.cat((fmap, addon_2), dim=1)
        md_scale_pred = self.md_fmapper(fmap)

        fmap = self.upsampler(fmap)
        fmap = torch.cat((fmap, addon_1), dim=1)
        sm_scale_pred = self.sm_fmapper(fmap)

        return lg_scale_pred, md_scale_pred, sm_scale_pred

class BoundingBoxPredictor(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        anchors_per_scale = len(anchors)
        self.register_buffer("anchors", anchors.view(1, 1, 1, anchors_per_scale, 2).to(torch.float32))
        self.n_anchors_per_scale = anchors_per_scale

    def forward(self, feature_map):
        # (N, C, H, W) -> (N, H, W, C)
        feature_map = torch.permute(feature_map, (0, 2, 3, 1))
        N, H, W, _ = feature_map.shape

        # (N, H, W, C) -> (N, H, W, A, 5+C)
        feature_map = feature_map.reshape(N, H, W, self.n_anchors_per_scale, -1)

        obj_scores   = torch.sigmoid(feature_map[..., 0].unsqueeze(-1))   # (N,H,W,A,1)
        boxlocs      = feature_map[..., 1:5]                              # (N,H,W,A,4)
        class_scores = torch.sigmoid(feature_map[..., 5:])                # (N,H,W,A,C)

        # build (cx,cy) offsets on the same device/dtype as feature_map
        device = feature_map.device
        dtype  = feature_map.dtype
        yv = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)
        xv = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)
        idx = torch.stack((xv, yv), dim=-1).view(1, H, W, 1, 2)                          # (1,H,W,1,2)

        # decode
        box_xy = torch.sigmoid(boxlocs[..., :2]) + idx                                   # (N,H,W,A,2)
        box_wh = torch.exp(boxlocs[..., 2:4].clamp(min=-4.0, max=4.0)) * self.anchors    # (N,H,W,A,2)

        bboxes = torch.cat((box_xy, box_wh), dim=-1)                                     # (N,H,W,A,4)
        predictions = torch.cat((obj_scores, bboxes, class_scores), dim=-1)              # (N,H,W,A,1+4+C)
        predictions = predictions.view(N, -1, predictions.shape[-1])                     # (N, H*W*A, 1+4+C)
        return predictions

class YOLOv3(nn.Module):
    def __init__(self, input_channels, anchors, n_classes):
        super(YOLOv3, self).__init__()

        self.input_channels = input_channels
        self.anchors = anchors
        self.n_anchors_per_scale = len(self.anchors)//3
        self.n_classes = n_classes

        self.feature_mapper = FeatureMapper(self.input_channels, self.n_anchors_per_scale, self.n_classes)
        self.sm_box_predictor = BoundingBoxPredictor(self.anchors[:3])
        self.md_box_predictor = BoundingBoxPredictor(self.anchors[3:6])
        self.lg_box_predictor = BoundingBoxPredictor(self.anchors[6:9])

    def forward(self, x):
        lg_scale_pred, md_scale_pred, sm_scale_pred = self.feature_mapper(x)
        lg_scale_pred = self.lg_box_predictor(lg_scale_pred)
        md_scale_pred = self.md_box_predictor(md_scale_pred)
        sm_scale_pred = self.sm_box_predictor(sm_scale_pred)
        pred_bbox = torch.cat((lg_scale_pred, md_scale_pred, sm_scale_pred), dim=1)
        
        return pred_bbox

class BoundingBoxPostProcessing:
    def __init__(self, bboxes: torch.Tensor, *, box_format: str = "cxcywh"):
        assert box_format in ("cxcywh", "xyxy")
        self.bboxes = bboxes
        self.box_format = box_format

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack((cx - w/2, cy - h/2, cx + w/2, cy + h/2), dim=-1)

    def filter_boxes(self, score_threshold: float=0.5):
        N = self.bboxes.shape[0]
        obj = self.bboxes[..., 0]       # (N,M)
        boxes_raw = self.bboxes[..., 1:5]
        cls = self.bboxes[..., 5:]      # (N,M,C) — assume already sigmoid in your predictor

        boxes = self._cxcywh_to_xyxy(boxes_raw) if self.box_format=="cxcywh" else boxes_raw
        conf_per_class = cls * obj.unsqueeze(-1)            # (N,M,C)
        best_conf, best_cls = conf_per_class.max(dim=-1)    # (N,M), (N,M)

        out_scores, out_boxes, out_classes = [], [], []
        for b in range(N):
            mask = best_conf[b] >= score_threshold
            out_scores.append(best_conf[b][mask])
            out_boxes.append(boxes[b][mask])
            out_classes.append(best_cls[b][mask])
        return tuple(out_scores), tuple(out_boxes), tuple(out_classes)

    def non_max_suppression(self, scores: tuple, boxes: tuple, classes: tuple, iou_threshold: float=0.5, class_agnostic: bool=False):
        outs = ([],[],[])
        for s, bxs, cls in zip(scores, boxes, classes):
            if s.numel()==0:
                outs[0].append(s)
                outs[1].append(bxs) 
                outs[2].append(cls)
                continue
            bxs = bxs.to(dtype=torch.float32)
            s   = s.to(dtype=torch.float32)
            if class_agnostic:
                keep = torchvision.ops.nms(bxs, s, iou_threshold)
            else:
                keep = torchvision.ops.batched_nms(bxs, s, cls, iou_threshold)
            outs[0].append(s.index_select(0, keep))
            outs[1].append(bxs.index_select(0, keep))
            outs[2].append(cls.index_select(0, keep))
        return tuple(outs[0]), tuple(outs[1]), tuple(outs[2])
