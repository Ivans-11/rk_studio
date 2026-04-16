# rk_studio

RK3588 上的多路摄像头预览、录制、RTSP 推流和手部关键点检测应用。基于 Qt5 + GStreamer + RKNN，利用 RGA 硬件加速图像预处理。

## 快速开始

```bash
# 1. 安装依赖
sudo apt-get install -y \
  cmake g++ pkg-config \
  qtbase5-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-good1.0-dev libgstreamer-allocators1.0-0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libopencv-dev

# 2. 创建配置文件（按你的硬件修改）
cp config/board.example.toml config/board.toml
cp config/profile.example.toml config/profile.toml

# 3. 构建
cmake -S . -B build
cmake --build build -j$(nproc)

# 4. 运行
cd build && ./rk_studio
```

## 板端额外依赖

- `librknnrt.so` — RKNN 推理运行时
- `librga.so` — RGA 2D 硬件加速（可选，无则自动回退 CPU）
- RK3588 的 GStreamer / MPP / V4L2 运行环境

## 目录结构

```text
rk_studio/
├── config/
│   ├── board.example.toml      # 板级硬件配置模板
│   └── profile.example.toml    # 会话/布局配置模板
├── models/
│   ├── hand_detector.rknn      # 手掌检测模型
│   └── hand_landmarks.rknn     # 手部关键点模型
├── include/
│   ├── rk_studio/              # 应用层头文件
│   └── mediapipe/              # AI 推理管线头文件
├── src/                        # 源码
├── third_party/                # RKNN API / toml++ 头文件
└── CMakeLists.txt
```

## 配置说明

### board.toml

板级硬件配置，定义摄像头设备、音频设备、RTSP 参数等。

```toml
[camera.cam0]
record_device = "/dev/video44"   # V4L2 设备路径
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
```

### profile.toml

会话配置，定义预览/录制的摄像头选择、输出目录、UI 布局等。

### AI 模型

模型文件自动从 `models/` 目录加载，无需手动配置路径。如需自定义路径，在 `board.toml` 中添加：

```toml
[ai]
detector_model = "/custom/path/hand_detector.rknn"
landmark_model = "/custom/path/hand_landmarks.rknn"
```

## 功能

- **多路预览** — 最多 4 路摄像头实时预览（2x2 网格）
- **多路录制** — H.265 硬编码录制，支持同步录音
- **RTSP 推流** — 多路合成马赛克画面推流
- **手部关键点** — 基于 RKNN 的实时手部检测和 21 点关键点追踪
- **RGA 硬件加速** — NV12 到 RGB 的色彩转换由 RGA 硬件完成，自动回退 CPU

## 使用方式

1. 启动程序，配置自动加载
2. 点击「启动预览」— 多路摄像头画面
3. 点击「启动 Mediapipe」— cam0 叠加手部关键点
4. 点击「启动录制」— 开始录像 + 录音
5. 点击「启动 RTSP」— 推流到 `rtsp://<ip>:8554/cam`

预览、录制、RTSP 三种模式互斥。

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
└── ai.hand.jsonl        # AI 推理结果
```
