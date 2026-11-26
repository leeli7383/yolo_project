from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import atexit
import time
import csv
from datetime import datetime
import numpy as np
import cv2
import hailo
from collections import Counter

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class

class ScreenOnlyDetectionApp(GStreamerDetectionApp):
    def get_pipeline_string(self):
        pipeline = super().get_pipeline_string()
        pipeline = pipeline.replace("video-sink=autovideosink", "video-sink=fakesink")
        pipeline = pipeline.replace("video-sink=waylandsink", "video-sink=fakesink")
        pipeline = pipeline.replace("video-sink=xvimagesink", "video-sink=fakesink")
        return pipeline

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.timeline_dir = Path(__file__).resolve().parents[1] / "records"
        self.timeline_dir.mkdir(parents=True, exist_ok=True)
        self.timeline_file = self.timeline_dir / "activity_timeline.csv"
        self.last_timeline_second = -1
        self.video_dir = self.timeline_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_file = self.video_dir / f"record_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.avi"
        self.video_writer = None
        self.video_fps = 20.0
        self.video_frame_index = 0
        if not self.timeline_file.exists():
            with self.timeline_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "second_index", "activity", "person_count", "counts", "video_file"])
        self.new_variable = 42  # New variable example

    def new_function(self):  # New function example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    class_counts = Counter()
    valid_detections = []
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if confidence >= 0.30:
            class_counts[label] += 1
            valid_detections.append((label, confidence, bbox))
            string_to_print += (f"Detection: Label: {label} Confidence: {confidence:.2f}\\n")
            detection_count += 1
    activity_state = 1 if class_counts.get("person", 0) > 0 else 0

    if frame is not None:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        for label, confidence, bbox in valid_detections:
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        y = 30
        cv2.putText(
            frame,
            f"activity: {activity_state}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
        y += 32

        for name, num in class_counts.most_common(8):
            cv2.putText(
                frame,
                f"{name}: {num}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            y += 32
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if user_data.video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            user_data.video_writer = cv2.VideoWriter(
                str(user_data.video_file),
                fourcc,
                user_data.video_fps,
                (w, h)
            )

            if not user_data.video_writer.isOpened():
                print(f"Cannot open video writer: {user_data.video_file}", flush=True)
                return Gst.PadProbeReturn.REMOVE

        user_data.video_writer.write(frame)
        second_index = int(user_data.video_frame_index / user_data.video_fps)
        user_data.video_frame_index += 1
        if second_index != user_data.last_timeline_second:
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            person_count = class_counts.get("person", 0)
            counts_text = "|".join(f"{name}:{num}" for name, num in class_counts.items())
            with user_data.timeline_file.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    timestamp, second_index, activity_state, person_count, counts_text,
                    user_data.video_file.name,
                ])
            user_data.last_timeline_second = second_index
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK


def release_video_writer(user_data):
    if hasattr(user_data, "video_writer") and user_data.video_writer is not None:
        user_data.video_writer.release()
        print(f"Video saved to: {user_data.video_file}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ.setdefault("HAILO_ENV_FILE", env_path_str)
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    atexit.register(release_video_writer, user_data)
    app = ScreenOnlyDetectionApp(app_callback, user_data)
    user_data.video_fps = app.options_menu.frame_rate
    app.run()
