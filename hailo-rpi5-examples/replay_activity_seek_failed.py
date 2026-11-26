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


def build_active_seconds(rows):
    return sorted({r["second_index"] for r in rows if r["activity"] == 1})




def build_active_segments(rows):
    starts = []
    prev_active = 0
    for row in rows:
        cur = row["activity"]
        if cur == 1 and prev_active == 0:
            starts.append(row["second_index"])
        prev_active = cur
    return starts

def find_next_active(active_seconds, current_second):
    for sec in active_seconds:
        if sec > current_second:
            return sec
    return active_seconds[0] if active_seconds else None


def find_prev_active(active_seconds, current_second):
    for sec in reversed(active_seconds):
        if sec < current_second:
            return sec
    return active_seconds[-1] if active_seconds else None


def row_for_second(rows, second):
    best = None
    for row in rows:
        if row["second_index"] <= second:
            best = row
        else:
            break
    return best


def seek_to_second(cap, second, fps):
    second = max(0, int(second))
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)




def show_second(cap, rows, second, fps, segment_index, active_segments):
    seek_to_second(cap, second, fps)
    ret, frame = cap.read()
    if not ret:
        print("failed to read frame at second:", second)
        return
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    current_second = int(current_frame / fps)
    frame = draw_info(frame, rows, current_second, segment_index, active_segments)
    cv2.imshow("Activity Replay", frame)
    cv2.waitKey(1)

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

    active_seconds = build_active_seconds(rows)
    active_starts = build_active_segments(rows)
    video_path = video_dir / video_name

    print("Timeline:", csv_path)
    print("Video:", video_path)
    print("Active seconds:", len(active_seconds))
    print("Active segments:", len(active_starts))
    print("")
    print("Controls:")
    print("  n: next activity=1")
    print("  p: previous activity=1")
    print("  a: backward 5 seconds")
    print("  d: forward 5 seconds")
    print("  space: pause/resume")
    print("  q: quit")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 20.0

    paused = False

    if active_starts:
        seek_to_second(cap, active_starts[0], fps)
        paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_second = int(current_frame / fps)
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

        cv2.putText(
            frame,
            text,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Activity Replay", frame)
        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("n"), ord("N")):
            target = find_next_active(active_starts, current_second)
            if target is not None:
                print("jump next activity segment:", target)
                seek_to_second(cap, target, fps)
                paused = False
        elif key in (ord("p"), ord("P")):
            target = find_prev_active(active_starts, current_second)
            if target is not None:
                print("jump previous activity segment:", target)
                seek_to_second(cap, target, fps)
                paused = False
        elif key in (ord("a"), ord("A")):
            target = max(0, current_second - 5)
            print("jump backward:", target)
            seek_to_second(cap, target, fps)
            paused = False
        elif key in (ord("d"), ord("D")):
            target = current_second + 5
            print("jump forward:", target)
            seek_to_second(cap, target, fps)
            paused = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
