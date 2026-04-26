# rk_studio

RK3588 上的 4 路摄像头预览、录制、RTSP 推流、Mediapipe 手部关键点、YOLO 目标检测和 Zenoh 结果发布应用。项目面向 LubanCat-5 V2 + IMX415，使用 Qt5、GStreamer、RKNN 和 RGA。

## 快速部署

### 1. 开启摄像头 overlay

在板子的 `/boot/uEnv/uEnv.txt` 中启用 IMX415 overlay，4 路运行使用 `cam0` 到 `cam3`：

```text
dtoverlay=/dtb/overlay/rk3588-lubancat-5-cam0-imx415-1920x1080-60fps-overlay.dtbo
dtoverlay=/dtb/overlay/rk3588-lubancat-5-cam1-imx415-1920x1080-60fps-overlay.dtbo
dtoverlay=/dtb/overlay/rk3588-lubancat-5-cam2-imx415-1920x1080-60fps-overlay.dtbo
dtoverlay=/dtb/overlay/rk3588-lubancat-5-cam3-imx415-1920x1080-60fps-overlay.dtbo
```

重启后检查：

```bash
dmesg | grep -i imx415
```

### 2. 同步项目

从 Mac 同步到板子时建议用 `tar`，避免带上 `.DS_Store` / `._*`：

```bash
cd /Users/aksea/Project/Linux/RK3588
tar --exclude='.git' --exclude='build' --exclude='records' \
    --exclude='.DS_Store' --exclude='._*' --exclude='__MACOSX' \
    -czf /tmp/rk_studio.tar.gz rk_studio

scp /tmp/rk_studio.tar.gz cat@<board-ip>:/tmp/
ssh cat@<board-ip> 'rm -rf /home/cat/rk_studio && cd /home/cat && tar -xzf /tmp/rk_studio.tar.gz'
```

### 3. 安装依赖

```bash
sudo apt-get install -y \
  cmake g++ pkg-config \
  qtbase5-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-good1.0-dev libgstreamer-allocators1.0-0 \
  libgstrtspserver-1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libopencv-dev

cd /home/cat/rk_studio
./scripts/install_board_deps.sh
```

`install_board_deps.sh` 会安装项目自带的：

- RKNN Toolkit2 2.3.0 runtime: `third_party/rknn/runtime/aarch64/librknnrt.so`
- zenoh-c 1.9.0 arm64 deb: `third_party/zenohc/debian/arm64/`

验证：

```bash
nm -D /usr/lib/librknnrt.so | grep rknn_mem_sync
test -f /usr/lib/cmake/zenohc/zenohcConfig.cmake
```

### 4. 配置、构建、运行

```bash
cd /home/cat/rk_studio
cp config/board.example.toml config/board.toml
cp config/profile.example.toml config/profile.toml

cmake -S . -B build
cmake --build build -j$(nproc)

./build/rk_studio
```

默认 camera 节点：

```text
cam0 -> /dev/video55
cam1 -> /dev/video64
cam2 -> /dev/video73
cam3 -> /dev/video82
```

如果新板子 video 编号不同，只需要改 `config/board.toml` 里的 `[camera.<id>].record_device`。

## 配置

### board.toml

常改字段：

```toml
[rtsp]
mounts = ["cam", "cam0", "cam1", "cam2", "cam3"]

[zenoh]
mode = "peer"
connect = []
listen = []
key_prefix = "rk_studio"

[yolo]
model = "../models/yolo11n_rk3588_int8.rknn"
fps = 5
confidence_threshold = 0.25
nms_threshold = 0.45
max_detections = 50

[camera.cam0]
record_device = "/dev/video55"
preview_width = 640
preview_height = 360
fps = 30
```

RTSP mount 规则：

```text
rtsp://<board-ip>:8554/cam   # 4 路拼接流
rtsp://<board-ip>:8554/cam0  # cam0 单路
rtsp://<board-ip>:8554/cam1  # cam1 单路
rtsp://<board-ip>:8554/cam2  # cam2 单路
rtsp://<board-ip>:8554/cam3  # cam3 单路
```

单路 RTSP 直接向 V4L2 请求 `preview_width` / `preview_height` 尺寸；H.265 会把宽高向上对齐到 16 的倍数。帧率通过 GStreamer `videorate drop-only=true` 限制。

Zenoh 发布模型结果：

```text
rk_studio/mediapipe/<camera_id>/hands
rk_studio/yolo/<camera_id>/objects
```

### profile.toml

```toml
[session]
preview_cameras = ["cam0", "cam1", "cam2", "cam3"]
prefix = "rk_studio"
audio_source = ""
selected_mediapipe_camera = "cam0"
selected_yolo_camera = "cam1"
```

Mediapipe 和 YOLO 可以选择不同摄像头。两者不能使用同一个识别摄像头。

## 使用规则

- `启动预览`：显示 4 路预览画面。
- `启动 Mediapipe`：开启手部关键点推理；如果预览已开启，在对应画面叠加结果。
- `启动 YOLO`：开启目标检测；如果预览已开启，在对应画面叠加检测框。
- `启动 Zenoh`：至少开启一个 Mediapipe/YOLO 后才能启动，发布当前模型结果。
- `启动录制`：录制前可以先开启 Mediapipe/YOLO/Zenoh；录制开始后功能组合冻结，只允许停止录制。
- `启动 RTSP`：按 `board.toml` 的 `[rtsp].mounts` 注册推流地址。

预览、录制、RTSP 三种模式互斥，它们占用 mainpath。Mediapipe/YOLO 是独立 selfpath 推理链路，不要求先开启预览，可以和 RTSP 同时运行。Zenoh 只依赖至少一个模型开启。YOLO 只显示、记录和发布置信度大于 0.7 的目标。

## 输出

录制会话输出到 `records/`：

```text
records/rk_studio-YYYYMMDD-HHMMSS/
├── cam0.mkv
├── cam1.mkv
├── mic0.mkv
├── session.meta.json
├── session.sync.json
├── studio.events.jsonl
├── mediapipe.hand.jsonl
└── yolo.objects.jsonl
```

模型结果 jsonl 只保留核心字段；需要更多字段时再扩展。

## 上板测试清单

接入新功能后建议按这个顺序测：

1. `./build/rk_studio` 能启动并自动加载配置。
2. 只开预览：4 路画面正常，帧率限制生效。
3. 只开 Mediapipe：不开预览也能产生日志；开预览后只叠加关键点，不影响底层预览画面。
4. 只开 YOLO：确认使用当前模型和 COCO 类别名。
5. 同时开 Mediapipe + YOLO：两个摄像头分别叠加，UI 不闪烁、不抢画面。
6. 开 Zenoh：订阅端能收到 `rk_studio/mediapipe/...` 或 `rk_studio/yolo/...`。
7. 开 RTSP + 模型 + Zenoh：RTSP 画面和 Zenoh 模型结果同时正常。
8. 先开模型和 Zenoh，再开始录制：录制期间按钮状态冻结，停止后生成 jsonl。

## 目录

```text
rk_studio/
├── config/                     # board/profile 配置模板
├── models/                     # RKNN 模型
├── include/                    # 头文件
├── src/                        # 源码
├── scripts/
│   └── install_board_deps.sh   # 安装板端 RKNN / Zenoh runtime 依赖
├── third_party/
│   ├── rknn/                   # RKNN API 和 runtime
│   ├── zenohc/                 # zenoh-c 1.9.0 arm64 deb
│   └── tomlplusplus/           # TOML 解析头文件
└── CMakeLists.txt
```
