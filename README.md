# rk_studio

RK3588 上的多路摄像头预览、录制、RTSP 推流、手部关键点和目标检测应用。基于 Qt5 + GStreamer + RKNN，利用 RGA 硬件加速图像预处理。

## 快速开始

```bash
# 1. 安装依赖
sudo apt-get install -y \
  cmake g++ pkg-config \
  qtbase5-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-good1.0-dev libgstreamer-allocators1.0-0 \
  libgstrtspserver-1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libopencv-dev

# 2. 安装项目自带的 RKNN Toolkit2 2.3.0 runtime
sudo install -m 0644 third_party/rknn/runtime/aarch64/librknnrt.so /usr/lib/librknnrt.so
sudo ldconfig

# 3. 创建配置文件（按你的硬件修改）
cp config/board.example.toml config/board.toml
cp config/profile.example.toml config/profile.toml

# 4. 构建
cmake -S . -B build
cmake --build build -j$(nproc)

# 5. 运行
cd build && ./rk_studio
```

## 板端额外依赖

- `librknnrt.so` — RKNN 推理运行时；仓库内置 RKNN Toolkit2 2.3.0 的 aarch64 版本，路径为 `third_party/rknn/runtime/aarch64/librknnrt.so`
- `librga.so` — RGA 2D 硬件加速（可选，无则自动回退 CPU）
- RK3588 的 GStreamer / MPP / V4L2 运行环境

## 新板子初始化

LubanCat-5 V2 + IMX415 摄像头的快速跑通流程：

1. 在 `/boot/uEnv/uEnv.txt` 中启用需要的 IMX415 overlay；4 路运行时使用 `cam0` 到 `cam3`：

```text
dtoverlay=/dtb/overlay/rk3588-lubancat-5-cam0-imx415-1920x1080-60fps-overlay.dtbo
dtoverlay=/dtb/overlay/rk3588-lubancat-5-cam1-imx415-1920x1080-60fps-overlay.dtbo
dtoverlay=/dtb/overlay/rk3588-lubancat-5-cam2-imx415-1920x1080-60fps-overlay.dtbo
dtoverlay=/dtb/overlay/rk3588-lubancat-5-cam3-imx415-1920x1080-60fps-overlay.dtbo
```

2. 重启后确认摄像头被识别：

```bash
dmesg | grep -i imx415
```

3. 安装 RKNN runtime，替换系统旧版 `librknnrt.so`：

```bash
sudo install -m 0644 third_party/rknn/runtime/aarch64/librknnrt.so /usr/lib/librknnrt.so
sudo ldconfig
nm -D /usr/lib/librknnrt.so | grep rknn_mem_sync
```

4. 使用默认示例配置即可启动 4 路摄像头。当前模板使用已验证的 ISP mainpath 节点：

```text
cam0 -> /dev/video55
cam1 -> /dev/video64
cam2 -> /dev/video73
cam3 -> /dev/video82
```

## 目录结构

```text
rk_studio/
├── config/
│   ├── board.example.toml      # 板级硬件配置模板
│   └── profile.example.toml    # 会话/布局配置模板
├── models/
│   ├── hand_detector.rknn      # 手掌检测模型
│   ├── hand_landmarks.rknn     # 手部关键点模型
│   └── yolo11n_rk3588_int8.rknn # YOLO11n 目标检测模型
├── include/
│   ├── rk_studio/              # 应用层头文件
│   └── mediapipe/              # Mediapipe 推理管线头文件
├── src/                        # 源码
├── third_party/                # RKNN API / toml++ 头文件
└── CMakeLists.txt
```

## 配置说明

### board.toml

板级硬件配置，定义摄像头设备、音频设备、RTSP 参数等。

```toml
[camera.cam0]
record_device = "/dev/video55"   # V4L2 设备路径
input_format = "NV12"            # 输入格式
io_mode = "dmabuf"               # I/O 模式（dmabuf 启用 RGA 加速）
record_width = 1920
record_height = 1080
preview_width = 640
preview_height = 360
fps = 30
bitrate = 8000000                # 录制码率 (bps)

[rtsp]
port = 8554
codec = "h265"
bitrate = 4000000                # RTSP 推流码率 (bps)
mounts = ["cam", "cam0"]         # 要注册的 RTSP 路径；cam 为拼接流，cam0/cam1 为单路流
```

RTSP 根据 `mounts` 注册拼接流和单路摄像头流。`cam` 是拼接流，其他项需要匹配 `[camera.<id>]`。以板子 IP `10.31.2.28` 为例：

```text
rtsp://10.31.2.28:8554/cam     # 4 路拼接画面
rtsp://10.31.2.28:8554/cam0    # cam0 单路
rtsp://10.31.2.28:8554/cam1    # cam1 单路
rtsp://10.31.2.28:8554/cam2    # cam2 单路
rtsp://10.31.2.28:8554/cam3    # cam3 单路
```

单路 RTSP 从 camera 的 `record_width` / `record_height` 取全幅画面，再缩放到 `preview_width` / `preview_height`，避免低分辨率 V4L2 协商触发驱动裁剪；H.265 单路输出会把宽高向上对齐到 16 的倍数以兼容硬件编码/客户端解码；拼接流使用预览尺寸；录制仍使用 `record_width` / `record_height`。

### profile.toml

会话配置，定义预览/录制的摄像头选择、输出目录、UI 布局等。

### Mediapipe 模型

模型文件自动从 `models/` 目录加载，无需手动配置路径。如需自定义路径，在 `board.toml` 中添加：

```toml
[mediapipe]
detector_model = "/custom/path/hand_detector.rknn"
landmark_model = "/custom/path/hand_landmarks.rknn"
```

### YOLO 模型

YOLO 模型同样会从 `models/` 目录自动加载，默认优先使用 `yolo11n_rk3588_int8.rknn`，本地识别链路默认按 5fps 提交帧。如需自定义路径或阈值，在 `board.toml` 中添加：

```toml
[yolo]
model = "/custom/path/yolo11n_rk3588_int8.rknn"
fps = 5
confidence_threshold = 0.25
nms_threshold = 0.45
max_detections = 50
```

`profile.toml` 中用 `selected_mediapipe_camera` 和 `selected_yolo_camera` 分别选择 Mediapipe/YOLO 使用的摄像头。两条识别链路各自从对应摄像头的 selfpath 取 640x360 低分辨率流，避免互相抢同一路 selfpath。

## 功能

- **多路预览** — 最多 4 路摄像头实时预览（2x2 网格）
- **多路录制** — H.265 硬编码录制，支持同步录音
- **RTSP 推流** — 多路合成马赛克画面推流
- **手部关键点** — 基于 RKNN 的实时手部检测和 21 点关键点追踪
- **YOLO 目标检测** — 基于 RKNN 的 YOLO11n 目标识别，默认 5fps
- **RGA 硬件加速** — NV12 到 RGB 的色彩转换由 RGA 硬件完成，自动回退 CPU

## 使用方式

1. 启动程序，配置自动加载
2. 点击「启动预览」— 多路摄像头画面
3. 点击「启动 Mediapipe」— cam0 叠加手部关键点
4. 点击「启动 YOLO」— cam1 运行 5fps 目标检测
5. 点击「启动录制」— 开始录像 + 录音
6. 点击「启动 RTSP」— 推流到 `rtsp://<ip>:8554/cam`

预览、录制、RTSP 三种模式互斥。Mediapipe/YOLO 需要在录制前开启；进入录制后功能组合会冻结，只允许停止录制。

## 输出文件

录制会话输出到 `output_dir`（默认 `./records`）：

```text
records/rk_studio-20260416-143000/
├── cam0.mkv             # 各路视频
├── cam1.mkv
├── mic0.mkv             # 音频
├── session.meta.json    # 会话元信息
├── session.sync.json    # 多路同步分析
├── studio.events.jsonl  # 遥测事件流
├── mediapipe.hand.jsonl # Mediapipe 推理结果
└── yolo.objects.jsonl   # YOLO 物体检测结果
```
