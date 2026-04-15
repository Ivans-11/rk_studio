# rk_studio

`rk_studio` 是面向 RK3588 的 Qt5 + GStreamer + RKNN 本地应用，用来统一 4 路预览、多路录像、单路录音和单路手部关键点叠加。

当前工程只做应用层编排和外围配置，媒体、录制和 AI 的底层能力复用自同级目录里的 `rk_recorder` 和 `mediapipe`。

## 目录说明

- `include/rk_studio/domain`：配置、状态、会话类型
- `include/rk_studio/media_core`：媒体引擎和相机管线接口
- `include/rk_studio/ai_core`：异步 AI 接口与结果类型
- `include/rk_studio/ui`：Qt Widgets 视图层
- `config/board.toml`：板级硬件配置
- `config/profile.toml`：会话和布局配置

## 依赖

当前 `CMakeLists.txt` 直接依赖这些开发包：

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake g++ pkg-config \
  qtbase5-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libopencv-dev
```

另外还需要板端已安装：

- `librknnrt.so`
- `rknn_api.h`
- RK3588 对应的 GStreamer / MPP / V4L2 运行环境
- 手部模型文件：
  - `/home/cat/mediapipe/hand_detector.rknn`
  - `/home/cat/mediapipe/hand_landmarks.rknn`

如果你的模型不在上述位置，请先改 `config/board.toml` 里的路径。

## 构建

在当前工作区直接构建：

```bash
cd /Users/aksea/Project/Linux/RK3588/rk_studio
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

如果板端已经装好 RGA，并且你希望启用 RGA 预处理路径：

```bash
cmake -S . -B build -DRK_STUDIO_ENABLE_RGA=ON
cmake --build build -j"$(nproc)"
```

## 运行

```bash
./build/rk_studio
```

程序启动后默认会读取：

- `config/board.toml`
- `config/profile.toml`

需要注意的是，当前 `MainWindow` 里默认配置路径仍是写死的绝对路径。只要你保持工程仍位于本 workspace 的这个目录下，就可以直接运行；如果把工程移动到别处，需要同步调整源码中的默认路径。

## 默认配置

当前默认配置按 4 路 MIPI 写好：

- `cam0`  -> `/dev/video44`
- `cam1`  -> `/dev/video53`
- `cam2`  -> `/dev/video62`
- `cam3`  -> `/dev/video71`

四路都默认使用 `NV12`、`dmabuf`、`1920x1080@30`，预览尺寸为 `640x360`。`profile.toml` 默认会把这 4 路都放进 2x2 预览网格，并默认选中 `cam0` 作为 AI 路。

## 使用方式

1. 启动程序后先点“加载配置”。
2. 点“启动预览”打开四宫格。
3. 点任意 tile，把该路切成 AI 叠加模式。
4. 点“开始录制”进入多路录像 + 单路录音。
5. 点“停止录制”结束当前会话并回到预览状态。
6. 点“全部停止”结束预览、录像和 AI。

## 输出文件

会话输出目录由 `session.output_dir` 和 `session.prefix` 决定，默认会生成类似下面的目录：

```text
/home/cat/records/rk_studio-YYYYmmdd-HHMMSS/
```

目录内默认包含：

- `cam*.mkv`
- `mic0.wav`
- `session.meta.json`
- `session.sidecar.jsonl`
- `session.sync.json`
- `ai.hand.jsonl`
- `studio.events.jsonl`

## 当前边界

- 首版只做手部关键点，不做全身骨骼
- 同一时刻只跑 1 路 AI
- 预览优先 `ximagesink`，失败后回退 `glimagesink`
- 当前默认面向 4 路 MIPI / NV12 场景
- `preview_device` 和 `ai_device` 已接入配置，但当前默认仍是按统一采集管线组织
