import cv2
import numpy as np

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


HEF_PATH = "/usr/local/hailo/resources/models/hailo8l/yolov8s.hef"
VIDEO_PATH = "/usr/local/hailo/resources/videos/example.mp4"

CONF_THRES = 0.30

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = new_shape

    scale = min(dst_w / src_w, dst_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))

    pad_w = dst_w - resized_w
    pad_h = dst_h - resized_h

    pad_left = pad_w // 2
    pad_top = pad_h // 2

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return padded, scale, pad_left, pad_top

def parse_hailo_nms_output(nms_output, original_shape, input_shape, scale, pad_left, pad_top):
   

    detections = []

    orig_h, orig_w = original_shape[:2]
    input_h, input_w = input_shape[:2]

    
    if isinstance(nms_output, list):
        if len(nms_output) == 0:
            return detections

        
        if len(nms_output) == 1 and isinstance(nms_output[0], list):
            output = nms_output[0]
        else:
            output = nms_output
    else:
        output = nms_output

    
    if isinstance(output, np.ndarray):
        if output.ndim == 4:
            output = np.squeeze(output, axis=0)

        if output.ndim == 3 and output.shape[1] == 5:
            num_classes = output.shape[0]

            for class_id in range(num_classes):
                class_dets = output[class_id]

                for i in range(output.shape[2]):
                    ymin = float(class_dets[0, i])
                    xmin = float(class_dets[1, i])
                    ymax = float(class_dets[2, i])
                    xmax = float(class_dets[3, i])
                    score = float(class_dets[4, i])

                    if score < CONF_THRES:
                        continue

                    if max(xmin, ymin, xmax, ymax) <= 1.5:
                        xmin *= input_w
                        xmax *= input_w
                        ymin *= input_h
                        ymax *= input_h

                    x1 = (xmin - pad_left) / scale
                    y1 = (ymin - pad_top) / scale
                    x2 = (xmax - pad_left) / scale
                    y2 = (ymax - pad_top) / scale

                    x1 = int(np.clip(x1, 0, orig_w - 1))
                    y1 = int(np.clip(y1, 0, orig_h - 1))
                    x2 = int(np.clip(x2, 0, orig_w - 1))
                    y2 = int(np.clip(y2, 0, orig_h - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detections.append({
                        "class_id": class_id,
                        "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
                        "score": score,
                        "box": [x1, y1, x2, y2],
                    })

            return detections

    
    if isinstance(output, list):
        num_classes = len(output)

        for class_id in range(num_classes):
            class_dets = output[class_id]

            if class_dets is None:
                continue

            class_dets = np.array(class_dets, dtype=np.float32)

            if class_dets.size == 0:
                continue

            
            if class_dets.ndim == 1:
                if class_dets.shape[0] != 5:
                    continue
                class_dets = class_dets.reshape(1, 5)

            if class_dets.ndim != 2 or class_dets.shape[1] != 5:
                print(f"Skip class {class_id}, unexpected det shape: {class_dets.shape}")
                continue

            for det in class_dets:
                ymin = float(det[0])
                xmin = float(det[1])
                ymax = float(det[2])
                xmax = float(det[3])
                score = float(det[4])

                if score < CONF_THRES:
                    continue

                
                if max(xmin, ymin, xmax, ymax) <= 1.5:
                    xmin *= input_w
                    xmax *= input_w
                    ymin *= input_h
                    ymax *= input_h

                
                x1 = (xmin - pad_left) / scale
                y1 = (ymin - pad_top) / scale
                x2 = (xmax - pad_left) / scale
                y2 = (ymax - pad_top) / scale

                x1 = int(np.clip(x1, 0, orig_w - 1))
                y1 = int(np.clip(y1, 0, orig_h - 1))
                x2 = int(np.clip(x2, 0, orig_w - 1))
                y2 = int(np.clip(y2, 0, orig_h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append({
                    "class_id": class_id,
                    "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
                    "score": score,
                    "box": [x1, y1, x2, y2],
                })

        return detections

    raise ValueError(f"Unsupported NMS output type: {type(nms_output)}")


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["class_name"]} {det["score"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame


def main():
    hef = HEF(HEF_PATH)

    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()

    input_name = input_infos[0].name
    input_shape = input_infos[0].shape
    output_name = output_infos[0].name

    input_h, input_w, input_c = input_shape

    print("Input name:", input_name)
    print("Input shape:", input_shape)
    print("Output name:", output_name)
    print("Output shape:", output_infos[0].shape)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    params = VDevice.create_params()

    with VDevice(params) as target:
        configure_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )

        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make(
            network_group,
            format_type=FormatType.UINT8,
        )

        output_vstreams_params = OutputVStreamParams.make(
            network_group,
            format_type=FormatType.FLOAT32,
        )

        with network_group.activate(network_group_params):
            with InferVStreams(
                network_group,
                input_vstreams_params,
                output_vstreams_params,
            ) as infer_pipeline:

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    original_shape = frame.shape

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img, scale, pad_left, pad_top = letterbox(rgb, (input_w, input_h))

                    input_data = np.expand_dims(img, axis=0).astype(np.uint8)

                    results = infer_pipeline.infer({
                        input_name: input_data
                    })

                    nms_output = results[output_name]

                    detections = parse_hailo_nms_output(
                        nms_output=nms_output,
                        original_shape=original_shape,
                        input_shape=(input_h, input_w),
                        scale=scale,
                        pad_left=pad_left,
                        pad_top=pad_top,
                    )

                    frame = draw_detections(frame, detections)

                    cv2.imshow("Hailo YOLOv8 Python NMS", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()