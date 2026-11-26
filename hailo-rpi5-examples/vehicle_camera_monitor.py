"""Camera vehicle detection, annotated recording and frame-accurate appearance logs."""
import argparse
import csv
from datetime import datetime
import json
import os
import signal
import time

import cv2

from vehicle_common import (ROOT, VehicleDetector, add_model_options, annotate,
                            new_run_dir, positive, probability, timecode)


class Camera:
    def __init__(self, source, width, height, fps):
        self.source, self.width, self.height, self.fps = source, width, height, fps
        self.camera = None

    def __enter__(self):
        try:
            if self.source == "rpi":
                from picamera2 import Picamera2
                self.camera = Picamera2()
                # Picamera2 RGB888 arrays are BGR byte order, matching OpenCV.
                self.camera.configure(self.camera.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameRate": self.fps}))
                self.camera.start()
            else:
                source = int(self.source) if self.source.isdigit() else self.source
                self.camera = cv2.VideoCapture(source)
                if not self.camera.isOpened():
                    raise RuntimeError(f"无法打开摄像头：{self.source}")
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            return self
        except BaseException:
            self.close()
            raise

    def read(self):
        if self.source == "rpi":
            return self.camera.capture_array("main")
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("摄像头读取失败或已断开")
        return frame

    def close(self):
        if self.camera is not None:
            if self.source == "rpi":
                self.camera.close()
            else:
                self.camera.release()

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_options(parser)
    parser.add_argument("--input", default="rpi", help="rpi、USB 编号（如 0）或 /dev/video0")
    parser.add_argument("--threshold", type=probability, default=0.30)
    parser.add_argument("--width", type=positive, default=1280)
    parser.add_argument("--height", type=positive, default=720)
    parser.add_argument("--fps", type=positive, default=20)
    parser.add_argument("--duration", type=positive, help="录制秒数；省略则持续运行")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-dir", default=ROOT / "records/vehicles")
    args = parser.parse_args()
    show = not args.no_display and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    writer = None
    frames = 0
    run_dir = None
    try:
        with VehicleDetector(args.hef_path, args.threshold, args.iou, args.classes) as detector, \
                Camera(args.input, args.width, args.height, args.fps) as camera:
            run_dir = new_run_dir(args.output_dir)
            print(f"录像和时间记录：{run_dir}\n按 q 或 Ctrl+C 停止。", flush=True)
            with (run_dir / "detections.csv").open("w", newline="", encoding="utf-8") as df, \
                    (run_dir / "segments.csv").open("w", newline="", encoding="utf-8") as sf:
                detections_log, segments_log = csv.writer(df), csv.writer(sf)
                detections_log.writerow(["wall_time", "video_seconds", "video_time", "frame_index",
                                         "label", "confidence", "threshold", "x1", "y1", "x2", "y2"])
                segments_log.writerow(["start_seconds", "end_seconds_exclusive", "start_time", "end_time"])
                start = None
                active_start = None
                previous = None
                try:
                    while not stopped:
                        if start is not None and args.duration and time.monotonic() - start >= args.duration:
                            break
                        frame = camera.read()
                        captured = datetime.now().astimezone().isoformat(timespec="milliseconds")
                        detections = detector.detect(frame)
                        now = time.monotonic()
                        if start is None:
                            start = now
                        # Repeat the last processed frame across inference gaps to preserve playback time.
                        target = max(frames, int((now - start) * args.fps))
                        if writer is not None:
                            while frames < target:
                                writer.write(previous)
                                frames += 1
                        seconds = frames / args.fps
                        if detections and active_start is None:
                            active_start = seconds
                            print(f"车辆出现：录像 {timecode(seconds)}，实际 {captured}", flush=True)
                        elif not detections and active_start is not None:
                            segments_log.writerow([active_start, seconds, timecode(active_start), timecode(seconds)])
                            sf.flush()
                            active_start = None
                        caption = (f"Video {timecode(seconds)} | Vehicles {len(detections)} | threshold {args.threshold:.2f}\n"
                                   f"Captured {captured}\n"
                                   f"Current segment start: {timecode(active_start) if active_start is not None else '--'}")
                        annotated = annotate(frame, detections, caption)
                        if writer is None:
                            h, w = annotated.shape[:2]
                            writer = cv2.VideoWriter(str(run_dir / "vehicles.avi"),
                                                     cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))
                            if not writer.isOpened():
                                raise RuntimeError("无法创建 MJPG AVI 录像，请检查编码器和输出目录")
                        writer.write(annotated)
                        for det in detections:
                            detections_log.writerow([captured, f"{seconds:.3f}", timecode(seconds), frames,
                                                      det["label"], f"{det['confidence']:.6f}", args.threshold, *det["box"]])
                        df.flush()
                        frames += 1
                        previous = annotated
                        if show:
                            cv2.imshow("Vehicle Monitor - q to quit", annotated)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                            if cv2.getWindowProperty("Vehicle Monitor - q to quit", cv2.WND_PROP_VISIBLE) < 1:
                                break
                        delay = start + frames / args.fps - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                finally:
                    if active_start is not None:
                        end = frames / args.fps
                        segments_log.writerow([active_start, end, timecode(active_start), timecode(end)])
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if run_dir is not None:
            (run_dir / "session.json").write_text(json.dumps({
                "hef": str(args.hef_path), "threshold": args.threshold, "iou": args.iou,
                "classes": args.classes, "fps": args.fps, "frames": frames,
                "video_seconds": frames / args.fps,
                "note": "时间对应保存视频；推理间隙复制上一帧。片段按是否有车辆划分，不跟踪单车身份。"
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已保存 {frames} 帧：{run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"车辆监控失败：{error}") from error
