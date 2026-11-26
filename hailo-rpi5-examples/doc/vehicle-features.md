# 车辆监控与图片识别

两个独立入口共用 `vehicle_common.py`，使用本机 Hailo 和现有 COCO YOLOv8s HEF。
不改变已有的人员监控程序。默认车辆类别为 bicycle、car、motorcycle、bus、truck。
先按[安装说明](../../README.md)准备 HailoRT、匹配的 HEF 和 Python 环境。CSI 摄像头入口需要系统 Picamera2。

在项目根目录执行：

```bash
cd hailo-rpi5-examples
source setup_env.sh
```

## 1. 摄像头车辆监控

```bash
python vehicle_camera_monitor.py
```

默认启用树莓派摄像头，1280×720、20 FPS，置信度阈值 0.30。
窗口显示车辆框、类别、置信度、录像时间、采集时间和当前有车片段的开始时间。
按 `q`、关闭窗口或 `Ctrl+C` 停止，录像会正常收尾。没有桌面显示环境时自动仅录像。

```bash
# 只检测机动车，调整阈值
python vehicle_camera_monitor.py --classes car motorcycle bus truck --threshold 0.45
# USB 摄像头
python vehicle_camera_monitor.py --input 0
# 无窗口录制 10 秒
python vehicle_camera_monitor.py --no-display --duration 10
```

每次运行独立保存到 `records/vehicles/日期时间/`：

- `vehicles.avi`：带框、带时间标注的 MJPG 录像，不含音频。
- `detections.csv`：每个检测框一行，记录实际采集时间、录像秒数、`HH:MM:SS.mmm`、零起始帧号、类别、置信度、阈值和像素坐标。
- `segments.csv`：画面从无车变为有车的开始时间，及重新无车时的结束时间；结束时间不包含在片段中。结束录制时仍有车的片段也会保存。
- `session.json`：模型、阈值、帧率、总帧数等运行参数。

录像时间由实际写入帧号 / FPS 计算；推理较慢时重复上一帧，避免录像加速后时间记录错位。
重复帧保留原采集时间和画面，不代表进行了新的推理，CSV 仅记录新检测结果。
车辆片段按“画面是否存在任意车辆”划分，没有跨帧单车身份跟踪；短暂漏检会分割片段。
实际时间含系统时区；录像时间可直接用于播放器定位。

## 2. 网络车辆图片识别（不使用摄像头）

```bash
python vehicle_image_search.py
```

默认通过 Wikimedia Commons API 搜索 `car traffic`，下载 3 张图片，分别按
0.25、0.50、0.75 的置信度阈值输出结果。

```bash
python vehicle_image_search.py --query "bus street" --count 5 --thresholds 0.2 0.4 0.6
python vehicle_image_search.py --query "truck highway" --classes car bus truck
# 使用已经下载的图片，避免重复搜索；支持 shell 通配符
python vehicle_image_search.py --images /path/to/images/*.jpg --thresholds 0.3 0.5
```

每次运行保存到 `records/vehicle_images/日期时间/`：

- `report.html`：用浏览器打开，同一张图片的不同阈值结果并列展示，点击查看大图。
- `originals/`：用于推理的图片（最长边最多 1600 像素）。
- `annotated/`：每张图片、每档阈值的框选结果。
- `detections.csv`、`results.json`：类别、置信度和检测框坐标；JSON 也保留零检测结果。
- `sources.json`：来源链接、作者、许可和下载失败记录。

框旁的数字是模型置信度，例如 `car 0.892`；阈值 0.50 表示保留置信度不低于 0.50 的框。
阈值更低通常检出更多目标，也可能增加误检。该工具用于目视检查，不提供有标注数据集上的准确率。
两项功能均可用 `--iou` 调整 NMS 重叠框抑制阈值，默认 0.45。
图片推理在 NMS 前设置最低比较阈值，再从同一结果筛选其他阈值。

程序不需要搜索 API 密钥。使用在线搜索才需要联网。两种图片输入方式都需要可用的 Hailo 设备，且均不打开摄像头。
如果下载部分失败会继续尝试其他搜索结果；最终数量不足时以实际成功数为准。
搜索结果与输入关键词有关，不保证每张图片都能检出车辆。

## 模型与故障定位

默认模型为 `resources/models/hailo8l/yolov8s.hef`。可使用 `--hef-path` 指定与
本机硬件匹配的其他 COCO 80 类、带 NMS 输出的 HEF。自定义类别或原始特征图输出模型不适用。
启动失败时检查摄像头/Hailo 是否被其他程序占用，以及是否已激活虚拟环境。
详细参数：`python vehicle_camera_monitor.py --help` 或 `python vehicle_image_search.py --help`。
