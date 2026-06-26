# Funyi

[English](README.md)

Funyi 是一个基于 `Qwen/Qwen3-ASR-1.7B` 的本地语音转文字和实时字幕应用。它包含本地实时 ASR WebSocket 服务和 Tauri 桌面字幕客户端，面向本地单用户使用，不是公网多用户服务。

## 演示

https://github.com/user-attachments/assets/cda710b8-5a05-4bd0-9e9f-5d2c9bc1de68

## 主要能力

- 本地实时语音识别服务：通过 `WS /ws/asr` 接收 16 kHz mono `pcm_s16le` 音频流。
- 桌面实时字幕：Windows/macOS 上捕获系统音频或麦克风音频并显示字幕。
- 本地文件转写：通过 `/api/transcriptions` 和 `/api/transcriptions/stream` 处理本地音视频文件。
- 时间戳对齐：默认使用 forced aligner 修正稳定字幕的时间边界。
- 可选翻译：HY-MT 翻译是服务层旁路，消费源语言转写事件；禁用翻译不会影响 ASR 路径。

## 环境要求

- Python 3.11+，推荐 Python 3.12。
- `uv`。
- 后端：Windows/Linux/WSL + NVIDIA CUDA，macOS 14+ Apple Silicon，或显式启用的 CPU fallback（`FUNYI_ALLOW_CPU=1`，很慢，不适合实时）。
- 桌面端：Node.js、Corepack `pnpm`、Rust/Cargo，以及对应平台的原生构建工具。
- 模型下载权限，或已经准备好的本地模型目录。

原生桌面构建还需要：

- Windows：Visual Studio Build Tools 2022。
- macOS：Xcode Command Line Tools。

## 模型

| 角色 | 默认模型 | 覆盖方式 |
|---|---|---|
| ASR | `Qwen/Qwen3-ASR-1.7B` | `FUNYI_ASR_MODEL` / `--model` |
| 时间戳 | `Qwen/Qwen3-ForcedAligner-0.6B` | `FUNYI_TIMESTAMP_MODEL` / `--timestamp-model` |
| 翻译 | `tencent/Hy-MT2-1.8B` | `FUNYI_TRANSLATION_MODEL` / `--translation-model` |

设置 `FUNYI_TRANSLATION_MODEL=` 可以禁用翻译。修改模型环境变量后需要重启后端。Apple Silicon MLX 模型 id 见 `docs/macos_mlx.md`。

## 快速开始

在仓库根目录安装依赖：

```bash
uv sync --python 3.12 --frozen
```

启动后端。首次运行或模型缓存为空时，先允许下载模型。

Linux、WSL 或 macOS：

```bash
FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh
./scripts/start_backend.sh
```

Windows PowerShell：

```powershell
$env:FUNYI_ALLOW_DOWNLOADS = "1"
.\scripts\start_backend.ps1
.\scripts\start_backend.ps1
```

检查后端：

```bash
curl http://127.0.0.1:8000/healthz
```

启动桌面客户端：

```bash
cd desktop
corepack pnpm install
corepack pnpm run dev
```

连接到：

```text
ws://127.0.0.1:8000/ws/asr
```

## 常用运行方式

| 命令 | 作用 |
|---|---|
| `FUNYI_ALLOW_DOWNLOADS=1 ./scripts/start_backend.sh` | 启动完整后端，并允许下载模型。 |
| `./scripts/start_backend.sh` | 使用缓存或本地模型启动完整后端。 |
| `FUNYI_TRANSLATION_MODEL= ./scripts/start_backend.sh` | 启动 ASR + 时间戳，不启用翻译。 |
| `FUNYI_PORT=8001 ./scripts/start_backend.sh` | 改用其他端口启动。 |
| `./scripts/start_backend.sh --no-vad` | 关闭 VAD 语音门控。 |
| `FUNYI_ALLOW_CPU=1 ./scripts/start_backend.sh` | CPU fallback；很慢，不适合实时。 |

Windows PowerShell 使用 `.\scripts\start_backend.ps1`，并在运行脚本前用 `$env:NAME = "value"` 设置环境变量。

Windows 的 `start_backend.ps1` 会在未设置时默认写入 `TORCHDYNAMO_DISABLE=1` 和 `TORCH_COMPILE_DISABLE=1`，因此默认 HY-MT 翻译路径不需要单独配置 TorchInductor/Triton。无需设置 `TORCH_COMPILE`。

也可以使用 Makefile 简写：

| 命令 | 作用 |
|---|---|
| `make backend-download` | 启动完整后端，并允许下载模型。 |
| `make backend` | 使用缓存或本地模型启动完整后端。 |
| `make backend-asr` | 启动 ASR + 时间戳，不启用翻译。 |
| `make desktop-install` | 安装桌面端依赖。 |
| `make desktop` | 启动桌面客户端。 |
| `make desktop-check` | 运行桌面端 lint、format、typecheck 和测试。 |

## 桌面端

桌面端位于 `desktop/`，会把系统音频或麦克风音频发送到 `/ws/asr`，并通过 `/api/transcriptions` 执行本地文件转写。

平台支持：

- Windows：WASAPI loopback 捕获系统输出，WASAPI input 捕获麦克风。
- macOS：ScreenCaptureKit 捕获系统音频；麦克风需要 macOS 15+。

桌面端只接受 `ws://` loopback 服务 URL。文件转写会从同一个 loopback 地址推导对应的 `http://` API URL。

## 验证

公共 smoke 检查不需要私有音频：

```bash
uv run python -m compileall -q qwen3_asr_runtime realtime_server.py tools tests
uv run --group test pytest
git diff --check
```

改动桌面端时运行：

```bash
make desktop-check
```

改动模型、实时服务、优化路径或质量指标时，使用 `docs/validation_and_regression.md` 中的回归和 CER gate。验证音频放在 `local_data/`，生成结果放在 `local_goldens/`；两者都被 git 忽略，公开发布不包含私有音频、转写文本或音频派生 golden。

## 文档

- `desktop/README.md`：桌面客户端。
- `docs/realtime_asr_service.md`：WebSocket 和本地文件转写 API。
- `docs/streaming_runtime.md`：流式 ASR 语义和 live profile。
- `docs/macos_mlx.md`：Apple Silicon MLX 后端。
- `docs/cpu_backend.md`：CPU fallback。
- `docs/performance_optimization.md`：优化栈和性能边界。
- `docs/validation_and_regression.md`：本地质量 gate。

## License

项目代码使用 MIT。部分 vendored 文件保留 Apache-2.0 notice。见 `LICENSE`、`LICENSES/Apache-2.0.txt` 和 `THIRD_PARTY_NOTICES.md`。
