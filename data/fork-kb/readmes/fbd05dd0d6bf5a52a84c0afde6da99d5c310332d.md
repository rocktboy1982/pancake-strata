# Strata Docker（CUDA / NVIDIA）

本项目基于 [Strata 上游仓库](https://github.com/Niko1221/Strata)
指定的 **[上游 tag，例如 vX.Y.Z]** 拉取源码，并在 Docker 镜像构建过程中编译 Strata CUDA 引擎。

这是一个 **NVIDIA CUDA 镜像**，不是适用于所有 Strata 支持硬件的通用镜像。

## 镜像包含什么

- 基于上游固定 tag 编译的 Strata CUDA 引擎
- Docker 启动入口及运行所需依赖
- 可选的视觉组件（如果该镜像构建时启用了 vision）

## 镜像不包含什么

- Qwen 模型权重：首次启动时需要下载，或按文档挂载已有模型文件
- NVIDIA 主机驱动和 NVIDIA Container Toolkit：需要在宿主机安装
- 对所有显卡、驱动版本和运行环境的兼容保证

## 兼容性范围

每个镜像 tag 的 CUDA 编译架构可能不同。请根据所下载镜像对应的
`CUDA_ARCHITECTURES` / 构建信息判断显卡是否在编译目标内。

| 项目 | 当前信息 |
|---|---|
| 上游源码版本 | `[填写上游 tag / commit]` |
| 镜像 tag | `[填写镜像地址及 tag]` |
| CUDA 编译架构 | `[填写实际构建值，例如 75;80;86;89;120]` |
| 已实际验证的 GPU | `[只填写你亲自测试过的显卡型号]` |
| 主机驱动要求 | `[填写实测/确认的最低驱动版本]` |
| 模型文件 | 不包含在镜像中；首次运行时下载或挂载已有文件 |

> 注意：CUDA 架构出现在编译列表中，只表示镜像包含对应架构的 CUDA 代码，
> 不代表该 GPU 已由本项目作者实测。不同 GPU、驱动、Docker 环境和模型配置
> 下的表现可能不同。

## 运行前提

- Linux 主机，或已正确配置 NVIDIA GPU 透传的 Docker Desktop / WSL2 环境
- NVIDIA GPU 和可用的 NVIDIA 驱动
- 已安装并配置 NVIDIA Container Toolkit
- 足够的系统内存、磁盘空间和模型文件
- Docker 可通过 `--gpus all` 将 GPU 提供给容器

## 重要限制

- 本镜像面向 **NVIDIA CUDA**；不要据此推断 AMD GPU 可用。
- 上游 Strata 支持某项硬件，不代表本 Docker 镜像也支持该硬件。
- 如果显卡架构不在该镜像的编译列表中，可能需要自行重新构建镜像并加入对应架构。
- 模型选择、内存需求、下载大小及参数配置请参考上游 Strata 文档。
- 本项目是上游 Strata 的 Docker 构建封装，不是 Strata 上游官方发布的镜像。
