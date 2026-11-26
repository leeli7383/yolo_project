import argparse
import csv
from pathlib import Path
import cv2


def load_timeline(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["second_index"] = int(row["second_index"])
                row["activity"] = int(row["activity"])
                row["person_count"] = int(row["person_count"])
            except Exception:
                continue
            rows.append(row)
    return rows


def select_video(rows, video_name=None):
    if video_name:
        return video_name

    for row in reversed(rows):
        name = row.get("video_file", "").strip()
        if name:
            return name

    raise RuntimeError("No video_file found in timeline.")


def filter_rows_for_video(rows, video_name):
    return [r for r in rows if r.get("video_file", "").strip() == video_name]


def build_active_segments(rows):
    starts = []
    prev = 0
    for row in rows:
        cur = row["activity"]
        if cur == 1 and prev == 0:
            starts.append(row["second_index"])
        prev = cur
    return starts


def row_for_second(rows, second):
    best = None
    for row in rows:
        if row["second_index"] <= second:
            best = row
        else:
            break
    return best


def load_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 20.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No frames loaded from video.")

    return frames, fps


def second_to_index(second, fps, total_frames):
    idx = int(second * fps)
    if idx < 0:
        idx = 0
    if idx >= total_frames:
        idx = total_frames - 1
    return idx


def draw_info(frame, rows, current_second, segment_index, active_segments):
    frame = frame.copy()
    info = row_for_second(rows, current_second)

    if info:
        text = (
            f"time={current_second}s "
            f"activity={info['activity']} "
            f"person={info['person_count']} "
            f"counts={info.get('counts', '')}"
        )
    else:
        text = f"time={current_second}s"

    if active_segments:
        seg_text = f"segment={segment_index + 1}/{len(active_segments)}"
    else:
        seg_text = "segment=0/0"

    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        seg_text,
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    return frame


def show_frame(frames, rows, frame_index, fps, segment_index, active_segments):
    total_frames = len(frames)
    if frame_index < 0:
        frame_index = 0
    if frame_index >= total_frames:
        frame_index = total_frames - 1

    current_second = int(frame_index / fps)
    frame = draw_info(frames[frame_index], rows, current_second, segment_index, active_segments)
    cv2.imshow("Activity Replay", frame)
    return frame_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", default="records")
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    root = Path.cwd()
    records_dir = root / args.records_dir
    csv_path = records_dir / "activity_timeline.csv"
    video_dir = records_dir / "videos"

    rows = load_timeline(csv_path)
    video_name = select_video(rows, args.video)
    rows = filter_rows_for_video(rows, video_name)

    if not rows:
        raise RuntimeError("No timeline rows for video: " + video_name)

    active_segments = build_active_segments(rows)
    video_path = video_dir / video_name

    print("Timeline:", csv_path)
    print("Video:", video_path)
    print("Active segments:", active_segments)
    print("Loading video frames...")

    frames, fps = load_video_frames(video_path)
    total_frames = len(frames)

    print("Frames:", total_frames)
    print("FPS:", fps)
    print("")
    print("Controls:")
    print("  n: next activity segment")
    print("  p: previous activity segment")
    print("  a: backward 5 seconds")
    print("  d: forward 5 seconds")
    print("  space: pause/resume")
    print("  q: quit")

    segment_index = 0
    if active_segments:
        frame_index = second_to_index(active_segments[0], fps, total_frames)
    else:
        frame_index = 0

    paused = True

    while True:
        frame_index = show_frame(
            frames,
            rows,
            frame_index,
            fps,
            segment_index,
            active_segments,
        )

        key = cv2.waitKey(0 if paused else 30) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            paused = not paused

        elif key in (ord("n"), ord("N")):
            if active_segments:
                segment_index = (segment_index + 1) % len(active_segments)
                target_second = active_segments[segment_index]
                frame_index = second_to_index(target_second, fps, total_frames)
                print("jump next segment:", target_second)
                paused = True

        elif key in (ord("p"), ord("P")):
            if active_segments:
                segment_index = (segment_index - 1) % len(active_segments)
                target_second = active_segments[segment_index]
                frame_index = second_to_index(target_second, fps, total_frames)
                print("jump previous segment:", target_second)
                paused = True

        elif key in (ord("a"), ord("A")):
            current_second = int(frame_index / fps)
            target_second = max(0, current_second - 5)
            frame_index = second_to_index(target_second, fps, total_frames)
            print("jump backward:", target_second)
            paused = True

        elif key in (ord("d"), ord("D")):
            current_second = int(frame_index / fps)
            target_second = current_second + 5
            frame_index = second_to_index(target_second, fps, total_frames)
            print("jump forward:", target_second)
            paused = True

        else:
            if not paused:
                frame_index += 1
                if frame_index >= total_frames:
                    frame_index = total_frames - 1
                    paused = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
