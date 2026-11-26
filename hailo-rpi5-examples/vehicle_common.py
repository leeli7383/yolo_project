"""Shared COCO vehicle inference and drawing helpers for the two vehicle demos."""
import argparse
import os
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {1: (255, 180, 0), 2: (0, 220, 0), 3: (0, 180, 255),
          5: (255, 100, 255), 7: (255, 200, 80)}


def probability(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("阈值必须大于 0 且不超过 1")
    return value


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def default_hef_path():
    override = os.environ.get("YOLO_HEF_PATH")
    if override:
        return Path(override).expanduser()
    arch = os.environ.get("HAILO_ARCH", "hailo8l")
    local = ROOT / "resources/models" / arch / "yolov8s.hef"
    installed = Path("/usr/local/hailo/resources/models") / arch / "yolov8s.hef"
    return installed if not local.is_file() and installed.is_file() else local


def add_model_options(parser):
    parser.add_argument("--hef-path", type=Path, default=default_hef_path())
    parser.add_argument("--classes", nargs="+", choices=list(VEHICLES.values()),
                        default=list(VEHICLES.values()), help="需要检测的车辆类别")
    parser.add_argument("--iou", type=probability, default=0.45, help="NMS IoU 阈值")


def new_run_dir(parent):
    path = Path(parent).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True, exist_ok=False)
    return path


def timecode(seconds):
    ms = round(max(0, seconds) * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"


def letterbox(bgr, size):
    height, width = bgr.shape[:2]
    target_h, target_w = size
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h))
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, (new_w / width, new_h / height, left, top)


def decode_nms(output, original_shape, input_size, transform, threshold, classes):
    """Decode the explicitly requested Hailo TF NMS layout: [1, 80, 5, N]."""
    output = np.asarray(output)
    if output.ndim == 4 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 3 or output.shape[:2] != (80, 5):
        raise ValueError(f"需要 COCO 80 类 NMS 模型，收到输出形状 {output.shape}")
    height, width = original_shape[:2]
    input_h, input_w = input_size
    sx, sy, left, top = transform
    detections = []
    for class_id, name in VEHICLES.items():
        if name not in classes:
            continue
        for ymin, xmin, ymax, xmax, score in output[class_id].T:
            if not np.isfinite([ymin, xmin, ymax, xmax, score]).all() or score < threshold:
                continue
            x1 = int(np.clip((xmin * input_w - left) / sx, 0, width - 1))
            y1 = int(np.clip((ymin * input_h - top) / sy, 0, height - 1))
            x2 = int(np.clip((xmax * input_w - left) / sx, 0, width - 1))
            y2 = int(np.clip((ymax * input_h - top) / sy, 0, height - 1))
            if x2 > x1 and y2 > y1:
                detections.append(dict(class_id=class_id, label=name,
                                       confidence=float(score), box=[x1, y1, x2, y2]))
    return sorted(detections, key=lambda det: det["confidence"], reverse=True)


class VehicleDetector:
    def __init__(self, hef_path, threshold=0.3, iou=0.45, classes=None):
        self.hef_path = Path(hef_path)
        self.threshold, self.iou = threshold, iou
        self.classes = classes if classes is not None else list(VEHICLES.values())
        self.stack = ExitStack()

    def __enter__(self):
        if not self.hef_path.is_file():
            raise FileNotFoundError(f"模型不存在：{self.hef_path}，请按 README 准备 HEF 或使用 --hef-path")
        from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                    InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType)
        try:
            hef = HEF(str(self.hef_path))
            inputs, outputs = hef.get_input_vstream_infos(), hef.get_output_vstream_infos()
            if len(inputs) != 1 or len(outputs) != 1 or outputs[0].shape[:2] != (80, 5):
                raise ValueError("仅支持带 NMS 输出的 COCO 80 类检测 HEF（例如 yolov8s.hef）")
            self.input_name, self.output_name = inputs[0].name, outputs[0].name
            self.input_size = inputs[0].shape[:2]
            device = self.stack.enter_context(VDevice())
            config = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            network = device.configure(hef, config)[0]
            self.stack.enter_context(network.activate(network.create_params()))
            self.pipeline = self.stack.enter_context(InferVStreams(
                network, InputVStreamParams.make(network, format_type=FormatType.UINT8),
                OutputVStreamParams.make(network, format_type=FormatType.FLOAT32),
                tf_nms_format=True))
            # Set the threshold before NMS so lowering it can actually reveal more candidates.
            self.pipeline.set_nms_score_threshold(self.threshold)
            self.pipeline.set_nms_iou_threshold(self.iou)
            return self
        except BaseException:
            self.stack.close()
            raise

    def detect(self, bgr):
        image, transform = letterbox(bgr, self.input_size)
        result = self.pipeline.infer({self.input_name: image[np.newaxis]})
        return decode_nms(result[self.output_name], bgr.shape, self.input_size,
                          transform, self.threshold, self.classes)

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def annotate(frame, detections, caption):
    frame = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = COLORS[det["class_id"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det['label']} {det['confidence']:.3f}",
                    (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    # Separate caption band preserves the image and bounding-box coordinates.
    lines = caption.split("\n")
    frame = cv2.copyMakeBorder(frame, 0, 28 * len(lines) + 12, 0, 0,
                               cv2.BORDER_CONSTANT, value=(25, 25, 25))
    base = frame.shape[0] - 28 * len(lines)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, base + 28 * i + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return frame
