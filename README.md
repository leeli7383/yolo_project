# 树莓派 YOLO 识别与录像

项目在 Raspberry Pi 5 上使用 Hailo NPU 运行 YOLO。人员活动监控走 Hailo Apps 与 GStreamer。车辆入口直接调用 HailoRT，提供摄像头录像、出现时间记录和图片批量识别。源码入口位于 [hailo-rpi5-examples](hailo-rpi5-examples/)。Hailo 上游代码和示例沿用随仓库提供的 [MIT 许可](hailo-rpi5-examples/LICENSE)。

## 1. 硬件和系统

使用 Raspberry Pi 5、64 位 Raspberry Pi OS、Hailo-8L AI Kit/AI HAT+ 或与目标模型匹配的 Hailo-8 设备。摄像头路径可使用 CSI 摄像头或 USB 摄像头。要预览或回放，使用带 OpenCV GUI 的桌面会话。

若 Hailo 驱动和摄像头依赖尚未安装，在树莓派上执行：

```bash
sudo apt update
sudo apt install -y hailo-all python3-venv python3-picamera2
sudo reboot
```

重启后运行，确认识别到加速器和摄像头。仅处理文件时可略过摄像头检查。

```bash
dpkg --print-architecture
hailortcli fw-control identify
rpicam-hello --list-cameras
```

AI Kit 和 AI HAT+ 的安装及驱动核验见 [Raspberry Pi AI 软件文档](https://www.raspberrypi.com/documentation/computers/ai.html)。Hailo-8 与 Hailo-8L 的 HEF 按设备架构选用。

## 2. 安装项目依赖与 HEF

以下步骤从本 README 所在目录开始。项目保留 Hailo Apps **25.7.0** 的模块路径和安装流程。HailoRT、TAPPAS 与 Python binding 要相互匹配。设备已有可用的安装时沿用这一组合，具体前置条件见 [上游安装说明](hailo-rpi5-examples/doc/install-raspberry-pi5.md)和 [25.7.0 发布说明](https://github.com/hailo-ai/hailo-rpi5-examples/releases/tag/25.7.0)。

```bash
cd hailo-rpi5-examples
bash install.sh
source ./setup_env.sh
python -c "import cv2, numpy, hailo_platform; print(cv2.__version__, numpy.__version__)"
python -c "import gi, hailo; import hailo_apps.hailo_app_python"
```

`install.sh` 从 `config.yaml` 读取基础设施版本和虚拟环境目录，创建可继承系统包的 venv，安装依赖并生成本机 `.env`。如果只使用车辆入口，且设备已有能够导入 `hailo_platform` 的 HailoRT Python 环境，也可在包含系统包的独立 venv 中安装 `requirements.txt` 后直接运行车辆脚本。

如果项目的 `resources/models/<架构>/` 和系统的 `/usr/local/hailo/resources/models/<架构>/` 下都没有 `yolov8s.hef`，从 Hailo Model Zoo 下载。以下示例用于 Hailo-8L。Hailo-8 对应 `hailo8`。

```bash
export HAILO_ARCH=hailo8l
mkdir -p "resources/models/$HAILO_ARCH"
curl -fL "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/$HAILO_ARCH/yolov8s.hef" \
  -o "resources/models/$HAILO_ARCH/yolov8s.hef"
python inspect_hef.py
```

局部覆盖可用 `YOLO_HEF_PATH=/绝对路径/模型.hef` 或在 Python 命令中传 `--hef-path`。车辆入口要求 COCO 80 类、带 NMS 的模型，并检查期望的输出布局。通过 `inspect_hef.py` 查看流名、尺寸和格式。

## 3. 识别入口

以下命令在 `hailo-rpi5-examples` 下运行。

| 任务 | 启动命令 | 结果 |
| --- | --- | --- |
| 实时类别计数 | `bash run_counter.sh` | Hailo 管线预览、逐帧类别计数和 FPS |
| 人员活动录像 | `bash run_monitor_record.sh` | 带框 AVI 与每个录像秒首帧采样的活动日志 |
| 人员活动预览与录像 | `bash run_screen_counter.sh` | 与当前活动监控入口共用同一管线 |
| 人员活动回放 | `bash run_replay.sh` | 读取 `records/activity_timeline.csv` 和 `records/videos/` |
| 摄像头车辆监控 | `python vehicle_camera_monitor.py` | 标注录像、检测明细、车辆出现片段 |
| 本地车辆图片 | `python vehicle_image_search.py --images ./example.jpg` | 原图、标注图、阈值对照 HTML、CSV/JSON |
| 网络车辆图片 | `python vehicle_image_search.py --query "bus street" --count 3` | 从 Wikimedia Commons 获取图片，保留来源和许可信息 |

人员计数入口默认读取 CSI 摄像头。如果需要 USB 摄像头，可传 `--input usb` 或 `--input /dev/video0`。人员录像默认使用 20 FPS；录像秒数由已写入帧号与输出 FPS 计算。活动标记为 1 时，采样帧上至少检测到一人。该日志用于按秒回放定位。可在已有录像上执行：

```bash
bash run_replay.sh --video <现有录像文件名>
```

回放用空格暂停或继续，`n`/`p` 跳到后一个或前一个活动片段，`a`/`d` 后退或前进 5 秒，`q` 退出。该版本会将整段录像读入内存，长录像可直接用播放器打开。

车辆监控的现场验收可先录制一段 10 秒视频：

```bash
python vehicle_camera_monitor.py --input rpi --no-display --duration 10
python vehicle_image_search.py --images /path/to/image.jpg --thresholds 0.25 0.50 0.75
```

车辆支持 bicycle、car、motorcycle、bus、truck。这些入口及输出字段详见 [车辆功能说明](hailo-rpi5-examples/doc/vehicle-features.md)。

## 4. 运行流程和文件地图

**人员活动监控**。摄像头或文件 → GStreamer 帧输入 → Hailo 推理与检测元数据 → Python 回调过滤分数、计数和绘框 → MJPG 录像与按秒时间表 → 回放按活动片段跳转。

**车辆监控与图片识别**。图像 → BGR 转 RGB 并等比缩放补边 → HailoRT 推理及 NMS → 将归一化坐标还原到原图并筛选车辆类别 → 输出视频/表格或阈值对比报告。车辆录像的推理间隔用上一帧补齐，使帧号与播放时间保持一致。

| 文件或目录 | 职责 |
| --- | --- |
| `install.sh`、`config.yaml`、`requirements.txt` | 安装、运行库版本、环境和 Python 依赖 |
| `setup_env.sh`、`run_*.sh` | 激活环境、定位模型及选择入口 |
| `hailo_python_installation.sh`、`download_resources.sh` | 上游 wheel 与示例资源辅助脚本 |
| `basic_pipelines/detection.py`、`detection_simple.py` | 上游检测参考管线 |
| `basic_pipelines/detection_counter.py` | 对象统计和预览回调 |
| `basic_pipelines/detection_counter_screen.py` | 计数绘框、人员活动日志与录像 |
| `replay_activity.py` | 按人员活动时间回放 |
| `vehicle_common.py` | 模型加载、letterbox、NMS 输出解码与绘制 |
| `vehicle_camera_monitor.py`、`vehicle_image_search.py` | 车辆录像及图片识别两个入口 |
| `inspect_hef.py`、`hailo_yolov8_python_nms.py` | 模型检查、早期原生 HailoRT 实验 |
| `tests/`、`run_tests.sh` | 车辆离线测试及上游设备集成测试 |
| `community_projects/`、`doc/`、`local_resources/` | 上游扩展示例、参考文档及静态资源 |
| `*_stable.py`、`*_before_*.py`、其他回放旧文件 | 调试期间的历史版本 |

程序生成的录像和时间表在运行目录的 `records/`，本机环境和模型资源由安装步骤生成。

## 5. 验证与排查

```bash
python -m pytest -q tests/test_vehicle_features.py
```

离线测试覆盖 NMS 坐标反变换、车辆类别筛选、时间码和 HTML 报告。树莓派上可用 `bash run_tests.sh` 执行上游硬件测试，需摄像头和相关测试模型。

| 现象 | 检查路径 |
| --- | --- |
| `ModuleNotFoundError: hailo_platform` | 核对系统 HailoRT 和 Python binding 版本及环境激活状态 |
| `No module named gi` / `hailo` | 检查 apt 的 GStreamer/PyGObject、TAPPAS bindings 及 venv 的系统包继承 |
| 无法加载 HEF | 核对 `hailortcli fw-control identify`、模型架构、配置路径和 HEF 输出 |
| CSI 摄像头打不开 | 用 `rpicam-hello --list-cameras` 检查硬件，并关闭占用摄像头的应用 |
| 回放找不到视频 | 确认 CSV 中的 `video_file` 与 `records/videos/` 文件名相同 |
| SSH 无法显示窗口 | 车辆录像可加 `--no-display`，回放使用树莓派桌面会话 |

项目交接与本轮验证记录见 [PROGRESS.md](PROGRESS.md)。
