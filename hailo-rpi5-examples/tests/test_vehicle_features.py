"""CPU-only regression tests: coordinate restoration, thresholds and image reporting."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vehicle_common import decode_nms, letterbox, timecode
from vehicle_image_search import collect_images, write_report


def test_letterbox_restores_wide_image_and_filters_nonvehicles():
    original = np.zeros((360, 1280, 3), dtype=np.uint8)
    _, transform = letterbox(original, (640, 640))
    output = np.zeros((1, 80, 5, 3), dtype=np.float32)
    # Original rectangle (128, 36) to (1152, 324), with 230 px vertical padding.
    output[0, 2, :, 0] = [248 / 640, 0.1, 392 / 640, 0.9, 0.8]
    output[0, 2, :, 1] = [248 / 640, 0.1, 392 / 640, 0.9, 0.2]
    output[0, 0, :, 0] = output[0, 2, :, 0]  # person must be excluded
    results = decode_nms(output, original.shape, (640, 640), transform, 0.3, ["car"])
    assert len(results) == 1
    assert results[0]["label"] == "car"
    assert results[0]["box"] == pytest.approx([128, 36, 1152, 324], abs=1)
    assert decode_nms(output, original.shape, (640, 640), transform, 0.9, ["car"]) == []
    assert decode_nms(output, original.shape, (640, 640), transform, 0.3, ["bus"]) == []


def test_invalid_boxes_and_wrong_model_are_not_silent():
    output = np.zeros((1, 80, 5, 3), dtype=np.float32)
    output[0, 2, :, 0] = [0.5, 0.5, 0.2, 0.2, 0.9]  # inverted
    output[0, 2, :, 1] = [0.1, float("nan"), 0.5, 0.5, 0.9]
    assert decode_nms(output, (640, 640, 3), (640, 640), (1, 1, 0, 0), 0.3, ["car"]) == []
    with pytest.raises(ValueError, match="80"):
        decode_nms(np.zeros((1, 5, 100)), (640, 640, 3), (640, 640), (1, 1, 0, 0), 0.3, ["car"])


def test_timecode_carries_milliseconds():
    assert timecode(59.9996) == "00:01:00.000"
    assert timecode(3661.25) == "01:01:01.250"


def test_local_images_and_empty_detection_report(tmp_path, monkeypatch):
    image = tmp_path / "input.jpg"
    cv2.imwrite(str(image), np.zeros((20, 30, 3), dtype=np.uint8))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    def no_network(*args, **kwargs):
        pytest.fail("Local image mode must not use network")
    monkeypatch.setattr("vehicle_image_search.fetch", no_network)
    args = argparse.Namespace(images=[str(image)], hef_path=Path("model.hef"), iou=0.45, classes=["car"])
    sources = collect_images(args, run_dir)
    sources[0]["title"] = '<script>alert("x")</script>'
    write_report(run_dir, sources, [dict(original=sources[0]["file"],
                 annotated="annotated/test.jpg", threshold=0.5, detections=[])], args)
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "未检测到车辆" in report
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert len(json.loads((run_dir / "sources.json").read_text(encoding="utf-8"))["images"]) == 1
