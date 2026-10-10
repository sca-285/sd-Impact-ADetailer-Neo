"""DWPose: people, then 133 whole-body keypoints per person.

Two onnx graphs, the same ones the WebUI's ControlNet preprocessor uses:
YOLOX-L finds people at 640 px, RTMPose (dw-ll_ucoco_384) puts 133 points on
each one at 288x384 and returns them as SimCC vectors.

COCO-WholeBody order: 0-16 body, 17-22 feet, 23-90 face, 91-111 left hand,
112-132 right hand.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .findings import Person

DET_SIZE = (640, 640)          # h, w
POSE_SIZE = (288, 384)         # w, h
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def detect_people(net, bgr: np.ndarray, min_score: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """(boxes (N, 4) xyxy, scores (N,)) for class 0 (person)."""
    import cv2

    h, w = bgr.shape[:2]
    th, tw = DET_SIZE
    ratio = min(th / h, tw / w)
    resized = cv2.resize(bgr, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((th, tw, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    blob = padded.transpose(2, 0, 1)[None].astype(np.float32)

    out = net(blob)[0][0]
    grids, strides = [], []
    for stride in (8, 16, 32):
        gh, gw = th // stride, tw // stride
        xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((gh * gw, 1), stride))
    grid = np.concatenate(grids, 0)
    stride = np.concatenate(strides, 0)
    xy = (out[:, :2] + grid) * stride
    wh = np.exp(out[:, 2:4]) * stride
    scores = out[:, 4] * out[:, 5]          # objectness x person
    boxes = np.concatenate([xy - wh / 2.0, xy + wh / 2.0], axis=1) / ratio

    sel = scores > 0.1
    boxes, scores = boxes[sel], scores[sel]
    if not len(boxes):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    keep = _nms(boxes, scores, 0.45)
    boxes, scores = boxes[keep], scores[keep]
    sel = scores > min_score
    boxes, scores = boxes[sel], scores[sel]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes.astype(np.float32), scores.astype(np.float32)


def _warp(box: np.ndarray):
    """Centre, scale and the affine matrix that maps the padded person box
    onto the pose input."""
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * 1.25
    pw, ph = POSE_SIZE
    aspect = pw / ph
    if scale[0] > scale[1] * aspect:
        scale = np.array([scale[0], scale[0] / aspect], dtype=np.float32)
    else:
        scale = np.array([scale[1] * aspect, scale[1]], dtype=np.float32)
    sx = pw / scale[0]
    sy = ph / scale[1]
    mat = np.array(
        [[sx, 0.0, pw / 2.0 - center[0] * sx], [0.0, sy, ph / 2.0 - center[1] * sy]],
        dtype=np.float32,
    )
    return center, scale, mat


def estimate_pose(net, rgb: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints (N, 133, 2) in image pixels and scores (N, 133)."""
    import cv2

    pw, ph = POSE_SIZE
    crops, metas = [], []
    for box in boxes:
        center, scale, mat = _warp(box)
        crop = cv2.warpAffine(rgb, mat, (pw, ph), flags=cv2.INTER_LINEAR)
        crops.append(((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1))
        metas.append((center, scale))
    simcc_x, simcc_y = net(np.stack(crops, 0))[:2]
    x = simcc_x.argmax(axis=2).astype(np.float32) / 2.0
    y = simcc_y.argmax(axis=2).astype(np.float32) / 2.0
    scores = np.minimum(simcc_x.max(axis=2), simcc_y.max(axis=2))
    kps = np.stack([x, y], axis=-1)
    for i, (center, scale) in enumerate(metas):
        kps[i] = kps[i] / np.array([pw, ph], dtype=np.float32) * scale + center - scale / 2.0
    return kps, scores


def to_people(kps: np.ndarray, scores: np.ndarray, boxes: np.ndarray, box_scores: np.ndarray) -> list[Person]:
    people = []
    for i in range(len(kps)):
        pts = np.concatenate([kps[i], scores[i][:, None]], axis=1).astype(np.float32)
        x1, y1, x2, y2 = boxes[i]
        people.append(
            Person(
                bbox=(int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))),
                body=pts[0:17],
                feet=pts[17:23],
                face=pts[23:91],
                hands=[pts[91:112].copy(), pts[112:133].copy()],
                score=float(box_scores[i]),
            )
        )
    return people


def run(image: Image.Image, det_net, pose_net) -> list[Person]:
    rgb = np.asarray(image.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    boxes, box_scores = detect_people(det_net, bgr)
    whole = False
    if not len(boxes):
        # A close-up (one hand, a face) often has no "person" in it. Pose on the
        # whole frame still finds the hand; the body points then come back with
        # low scores and the body checks skip them.
        h, w = rgb.shape[:2]
        boxes = np.array([[0, 0, w, h]], dtype=np.float32)
        box_scores = np.array([0.0], dtype=np.float32)
        whole = True
    kps, scores = estimate_pose(pose_net, rgb, boxes)
    people = to_people(kps, scores, boxes, box_scores)
    if whole:
        people = [p for p in people if float(np.clip(p.body[:, 2], 0, 1).mean()) > 0.2
                  or any(h is not None and float(np.clip(h[:, 2], 0, 1).mean()) > 0.3 for h in p.hands)]
    return people
