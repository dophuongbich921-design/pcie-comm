# pcie-comm

独立的 PCIe 通信库原型：抽取 FlashInfer 的 PCIe IPC 通信、候选策略、
AutoTuner 和配置持久化，接入 DeepEP 使用的 DeepJIT 磁盘缓存。
运行时不依赖安装 FlashInfer 或 DeepEP。

提供 AllReduce、AllGather、ReduceScatter，以及上游包含的 Copy Engine 路径。
通信协议、候选枚举、正确性筛选和计时决策沿用固定版本的 FlashInfer。
没有增加新的 rank 协调协议、编译锁或 leader 编译机制。
上游已经存在的协作检查原样保留，来源可以通过 UPSTREAM.json 核对。

## 三层缓存

| 层 | 存的内容 | 复用条件 |
| --- | --- | --- |
| 最优配置 | 通信场景 → variant / blocks / threads | 场景和上游缓存元数据匹配 |
| 编译产物 | 源码、头文件、编译环境和参数 → module.so | 构建签名匹配，DeepJIT 条目已提交 |
| 进程内 | 已解析的配置、已加载的模块和 FFI handle | 同一个进程和 workspace |

这批 PCIe 实现的候选大多是同一个模块内已有 kernel 的不同启动配置。
调优每一个 blocks/threads 组合通常不需要再编译一个二进制。
当前编译缓存的单位是含多个 kernel 和主机辅助逻辑的共享库。

~~~text
创建 workspace
  ├─ 进程内有模块 → 复用 handle
  └─ DeepJIT 有已提交 module.so → 加载
       └─ 未命中 → 在私有临时目录编译 → 原子发布 → 加载

执行通信
  ├─ workspace 已解析该形状 → 直接执行
  ├─ 持久化配置命中 → 解析方案、缓存到内存 → 执行
  └─ 没有配置 → 使用 FlashInfer 原有默认策略

显式调用 workspace.tune(...)
  → 枚举候选 → 检查正确性 → 实测计时 → 选择赢家 → 保存配置
~~~

普通调用不会因为配置缺失而自动启动全量调优。生产运行加载配置即可；
显式 tune 用于调优阶段。尤其 AG/RS 的 tune 沿用上游行为，会重新测量，
不要在每次进程启动时无条件调用它。

### 依据什么选择配置

- AllReduce：沿用候选 variant、blocks、threads 网格和有效性过滤。
  先验证正确性并进行粗筛，再由 AutoTuner 做 CUDA Graph 计时。
  保留上游粗筛的跨 rank 最大值与精调的跨 rank 平均值逻辑。
  取当前候选、输入和计时策略下的最佳结果，不声称数学上的全局最优。
- AllGather / ReduceScatter：保留独立调优器；验证候选正确性，
  用 eager 计时和跨 rank 最大延迟选择配置。
- AR 的 key 包含形状 bucket、world size、profile、workspace 容量、
  max_blocks、dtype、相关协议版本和 memop 能力。GPU、CUDA 等由
  AutoTuner 元数据校验；batch 按上游规则映射，hidden 保持精确。
- AG/RS 的 key 包含算子/版本、精确 shape、world size、rank placement
  指纹、ordered_4plus4、容量、max_blocks 和 dtype。

AR 在 warmup < 10 或 repeat < 50 时不会持久化结果，这是上游原有约束。
这里保留默认计时次数，没有为快速演示降低它们。

### DeepJIT 接入边界

复用 third_party/DeepJIT 的原版 C++ DiskCache、哈希、文件同步、
临时目录清理、提交标记和目录原子发布代码。Pybind 适配器只连接
Python builder 和原生 DiskCache。

FlashInfer 的通信模块还包含 workspace 管理和 Copy Engine 主机调度，
因此当前 backend 生成/加载 TVM-FFI .so，**没有改写成 DeepJIT 的 cubin backend**。
调优的 delay / globaltimer 辅助 kernel 也通过同一缓存编译。

独立的 exception.hpp 适配层将 Linux elfutils 回溯替换为 C++ 异常，
避免为此安装系统库。DeepJIT 子模块本身没有修改。

编译签名覆盖模块源码和依赖头文件内容、目标架构、NVCC/C++ 编译器版本、
编译链接参数、相关环境变量、Python ABI、Torch/CUDA/TVM-FFI 版本及 FFI 头文件。
命中时不会调用 build；首次加载仍会查询环境和编译器版本。
同一进程已经加载的模块继续复用，修改源码或环境后应重启进程。

多个进程同时冷启动时，DeepJIT 允许重复编译，然后竞争原子发布；
后续重启复用成功发布的产物。没有增加跨进程编译锁。
这里的“避免重复编译”不保证“并发首次启动全局只编译一次”。

## 安装

通信需要 Linux、CUDA Toolkit/NVCC、支持 CUDA IPC/P2P 的 NVIDIA GPU，
以及 CUDA 版 PyTorch。支持的 rank 数和拓扑由原上游代码检查：
2 / 4 / 8 ranks，单机，每 rank 一个 GPU。
AR 支持 BF16 / FP16；AG/RS 另外支持 FP32。
Copy Engine 和 SM120 特定路径仍受上游能力检查限制。

原生缓存绑定需要支持 C++20 std::format 的工具链，例如 GCC 13+。
CUDA 编译器选择的 host compiler 还须与所用 CUDA Toolkit 兼容。

coe-11 上实际的环境名是 pcie（此前写作 pice）。在非 root 用户下：

~~~bash
conda activate pcie
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd ~/pcie-comm
# 当前远端副本已包含 DeepJIT 头文件和依赖。
python -m pip install --no-deps --no-build-isolation -e .
~~~

通过 git 克隆新副本时先执行 git submodule update --init --recursive，
再运行 python -m pip install -e '.[test]' 安装项目及测试依赖。

如果现有环境缺少 C++20 编译器，可以在 conda 环境内安装兼容工具链；
不需要 sudo。coe-11 已补齐 GCC 13.4、pytest 和 pybind11。
上面的 LD_LIBRARY_PATH 让进程在导入 torch 前选用 conda 内的 C++ 运行库，
避免系统 GCC 11 的 libstdc++ 缺少 GLIBCXX_3.4.31。安装时保留环境原有 CUDA 版 torch。
本地 macOS 的 .venv 用于 CPU 策略/缓存测试，不支持执行 CUDA 通信。

## 使用

下面的代码应由 torchrun 启动，每个 rank 都执行同一段程序。

~~~python
import os
import torch
import torch.distributed as dist
from pcie_comm import PcieIpcAllReduceWorkspace

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")

with PcieIpcAllReduceWorkspace(
    group=dist.group.WORLD,
    max_numel=16 * 4096,
    dtype=torch.bfloat16,
    tune_batches=(1, 2, 4, 8, 16),
    tune_cache="./ar-configs.json",
) as ws:
    # 调优阶段执行一次；后续服务进程直接省略这一行。
    ws.tune([4096], dtype=torch.bfloat16)
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    y = ws.all_reduce(x)
    # 如果要捕获 CUDA Graph，先在捕获外准备所有实际捕获形状。
    ws.prepare([(8, 4096)], dtype=torch.bfloat16)

dist.destroy_process_group()
~~~

其他两个 workspace 提供 ws.all_gather(x)、ws.reduce_scatter(x) 和 ws.tune(hiddens)。
AG 输入形状为 [local_rows, hidden]，输出为 [world_size * local_rows, hidden]；
RS 输入形状相反。max_numel 对 AG 是本 rank 输入容量，对 RS 是输出 shard 容量。
传入 out= 可复用输出张量。
workspace 不支持跨 stream 的并发使用；沿用上游生命周期和 stream 绑定约束。
同组所有 rank 必须遵守相同的调用顺序。

## 缓存位置

| 环境变量 | 含义 |
| --- | --- |
| PCIE_COMM_CACHE_DIR | 总根目录，默认 ~/.cache/pcie_comm |
| PCIE_COMM_AUTOTUNE_DIR | 配置目录，默认总根目录下 autotune/ |
| PCIE_COMM_JIT_CACHE_DIR | 编译缓存目录，默认总根目录下 jit/ |
| DJ_JIT_CACHE_DIR | 未设 PCIE_COMM_JIT_CACHE_DIR 时接受的兼容选项 |
| TVM_FFI_CUDA_ARCH_LIST | 显式编译目标，例如 "8.0"、"12.0" |
| PCIE_COMM_EXTRA_CFLAGS | 额外 C++ 编译参数 |
| PCIE_COMM_EXTRA_CUDAFLAGS | 额外 NVCC 参数 |
| PCIE_COMM_EXTRA_LDFLAGS | 额外链接参数 |
| MAX_JOBS | 每次构建的 Ninja 并行度 |

编译缓存路径支持 path1:path2；第一项可写，其余用于查询已有产物。
配置可单独通过 workspace 的 tune_cache 指定。
原 FlashInfer Python 环境变量前缀统一迁移为 PCIE_COMM_。

~~~text
~/.cache/pcie_comm/
  autotune/...
  jit/
    cache/pcie_ipc_all_reduce.<digest>/
      module.so
      signature.json
      .committed
      ...构建文件
    tmp/...                      # 成功或正常异常退出后自动清理
~~~

from pcie_comm.jit.core import cache_stats 可读取当前进程的 builds、
disk_hits、loads 计数。配置与编译缓存可以分别删除：
只删配置会失去调优结果；只删编译产物会重新编译，但匹配的配置仍可用。
使用缓存时保留完整条目目录，包括 .committed。

## 验证

CPU 测试：

~~~bash
python -m pytest -q
~~~

覆盖上游候选/合法性/缓存测试、真实 C++ 二进制的跨进程复用、
TVM-FFI 模块编译/加载、失败清理、并发发布、二级查询缓存、
源码/参数失效、配置选择与持久化。CPU 调优选择测试使用受控计时，
不代表 GPU 性能测量。

在 coe-11 的 pcie 环境中运行 GPU 验证：

~~~bash
conda activate pcie
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_VISIBLE_DEVICES=0,1 bash scripts/validate_pice.sh
# 使用 4/8 卡时同时调整 CUDA_VISIBLE_DEVICES 和 NPROC_PER_NODE。
~~~

脚本拒绝 root 用户，创建独立临时缓存，不清理已有缓存。
先运行 cold：真实调优、配置落盘、NCCL 参考正确性、Graph 重放。
再启动全新 warm 进程：禁止编译入口、验证 builds == 0、
配置来自文件且与首次结果一致，再检查通信正确性和 Graph 重放。
每个 rank 的 JSON 报告保存在脚本输出的目录中。

coe-11 已通过 2/4/8 卡的上述验证，全部 warm rank 的 builds 为 0。
另通过 examples/validate_copy_engine.py 显式验证 4/8 卡 AR CE ring 和
8 卡 AG CE 的正确性及 Graph 重放。本地、远端均为 122 passed、1 xfailed。
测试范围、8 卡首次 NCCL 连接中断记录及报告位置见 [VALIDATION.md](VALIDATION.md)。

## 来源和目录

- FlashInfer 固定提交：595bc5746cd57efc0006ff888b4bf8144f8235b9。
- DeepJIT 固定提交：cba802df081985141eeee36803685775c183a047；
  这是所检查 DeepEP 93eb6eb238127e96c6d7a4a625a6dad158348509 引用的版本。
- UPSTREAM.json 记录抽取文件、原始/迁移后 SHA256 和改动类型。
- tools/vendor_flashinfer.py 可从该固定提交重新生成抽取代码。
- pcie_comm/comm/ 和 pcie_comm/autotuner/：上游逻辑。
- pcie_comm/csrc/ 和 pcie_comm/include/：通信 kernel 和 FFI。
- pcie_comm/jit/ 与 csrc/deepjit_cache.cpp：新增缓存适配。
- third_party/DeepJIT/：未修改的上游子模块。

FlashInfer 文件保留 Apache-2.0 声明及 LICENSE / NOTICE。
DeepJIT 按原仓库固定提交保留；该提交未包含独立 LICENSE 文件，
本项目的 LICENSE 不用于重新许可该子模块。
