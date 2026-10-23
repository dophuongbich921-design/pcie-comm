# 验证记录

日期：2026-10-21。

## 本地已完成

环境：macOS arm64、Python 3.12.13、Apple Clang 16、
PyTorch 2.14.1、apache-tvm-ffi 0.1.15.post0、pybind11 3.1.0。
使用项目内 .venv，没有使用 root 或 sudo。

运行：

~~~bash
source .venv/bin/activate
python -m pytest -q --junitxml=validation-results/cpu.xml
python tools/check_vendor.py
python -m compileall -q pcie_comm tests examples tools
bash -n scripts/validate_pice.sh
python setup.py sdist bdist_wheel
~~~

- 测试结果：122 passed，1 xfailed，耗时 27.97 秒。
- 唯一 xfail 是保留的上游已知限制：候选 blocks 网格不包含 3，
  因而不能显式表达部分 seed 配置；没有修改候选网格。
- 一条 warning 来自上游“缺少匹配调优 process group”测试的预期路径。
- DeepJIT 原生绑定编译成功；真实 C++ .so 在新进程中读取并执行成功，
  新进程禁止 builder 时仍成功，builds 为 0。
- TVM-FFI 的真实 C++ 模块编译、加载、内存复用和磁盘复用通过。
- 编译失败、缺失/空产物不会发布条目；临时目录正常清理。
- 两个并发冷启动进程可得到同一个完整产物。
- 配置选择测试使用受控延迟，让 AutoTuner 选择最小值，保存 JSON，
  再在独立进程加载；第二个进程没有重新计时。
- AG/RS 配置文件恢复、内存解析缓存、placement 变化后的失效通过。
- 42 个抽取文件与 UPSTREAM.json 记录的哈希一致；DeepJIT 子模块无修改。
- sdist 和本机 wheel 已生成，已检查 CUDA 源码、头文件、原生扩展及
  构建所需 DeepJIT/异常适配头文件被包含。

CPU 测试中的受控计时和 workspace stand-in 只验证选择及缓存行为，
不证明 CUDA kernel 正确性或 GPU 性能。

## coe-11 已完成

通过 10.30.1.11 登录，用户 wangyxbh，UID 1002。
项目位于 /home/wangyxbh/pcie-comm；实际 conda 环境名为 pcie。
全程未使用 root 或 sudo。

环境：Python 3.11.16、原有 PyTorch 2.13.0+cu126、CUDA Toolkit 12.8、
apache-tvm-ffi 0.1.15.post0、8 张 NVIDIA L40S。
GPU 0–3 和 4–7 分别为 PIX 岛，跨岛为 SYS，P2P 读写探测均通过。
为编译原生缓存适配器，在该 conda 环境内补齐 GCC/G++ 13.4、pytest、
pybind11 及工具链依赖；保留原有 torch。
运行前将 conda 的 lib 放入 LD_LIBRARY_PATH，解决系统 libstdc++ 11
缺少 GLIBCXX_3.4.31 的问题。没有更改系统工具链或驱动。

Linux CPU/原生缓存测试：122 passed，1 xfailed，1 warning，21.26 秒。
xfail 和 warning 与本地相同。

### 调优、持久化与进程重启

examples/validate_gpu.py 在每种规模下启动两个独立 torchrun 生命周期。
每个 rank 验证 8 个算子/dtype 组合：AR 的 BF16、FP16；
AG/RS 的 BF16、FP16、FP32。
调优场景为本 rank/shard 的 [1, 1024]、max_blocks=4；
RS 的完整输入为 [world_size, 1024]。
使用上游默认调优次数，未降低 AR 的持久化门槛。

| ranks | cold 调优与保存 | warm 配置复用 | 每 rank 的 warm builds / disk_hits / loads |
| --- | --- | --- | --- |
| 2 | 通过；空编译缓存，编译 3 个模块 | 通过 | 0 / 2 / 2 |
| 4 | 通过；复用已有 3 个编译模块 | 通过 | 0 / 2 / 2 |
| 8 | 通过；复用已有 3 个编译模块 | 通过 | 0 / 2 / 2 |

每个用例检查 NCCL 参考结果、3 次 eager 执行、10 次 CUDA Graph 重放，
输出使用 rtol=0、atol=0 严格比较。warm 进程禁止 tvm_ffi.cpp.build，
省略 tune，断言持久化配置已命中、选出的配置与 cold 相同。
14 个 warm rank 的编译次数全部为 0。
同一进程反复执行及创建不同 dtype 的 workspace 时只加载两个通信模块；
cold 多一个调优计时辅助模块。

8 卡首次尝试在初始化 NCCL 的 all_gather_object 处发生 socket 连接中断，
尚未执行通信 kernel。保留失败日志；开启 NCCL 详细日志后重新运行，
cold 和 warm 均完整通过。没有修改通信逻辑，也没有屏蔽 NCCL 的传输方式；
该次连接中断的根因尚未确定。

### 显式 Copy Engine 路径

调优赢家不保证选中 CE，因此 examples/validate_copy_engine.py 另外使用
公开 config 参数显式执行 CE。输入元素数分别为 1024 和 262144。

| ranks | 路径 | dtype / 参数 | 每 rank 用例数 |
| --- | --- | --- | --- |
| 4 | AR COPY_ENGINE_RING | BF16、FP16；pieces=1/2/4 | 12 |
| 8 | AR COPY_ENGINE_RING + AG COPY_ENGINE | AR 同上；AG 为 BF16、FP16、FP32 | 18 |

全部与 NCCL 严格一致，并通过 3 次 eager 和 10 次 Graph 重放。
这些验证也复用了已有编译产物，builds=0。

未验证 SM120 专用 memop 路径（L40S 是 SM89）；未强行开启在当前
pcieswitch-pairs profile 下被上游调优器排除的 AR COPY_ENGINE_ISLAND。
这是功能和缓存复用验证，不是大规模性能基准；不代表所有形状、
拓扑及全部候选均已验证，也不声称已选参数是全局最优。

### 复现与报告

~~~bash
source /home/wangyxbh/miniconda3/etc/profile.d/conda.sh
conda activate pcie
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/wangyxbh/pcie-comm
CUDA_VISIBLE_DEVICES=0,1 bash scripts/validate_pice.sh
# 4/8 卡：同时设置 CUDA_VISIBLE_DEVICES 和 NPROC_PER_NODE。
~~~

报告保留在本地 validation-results/coe-11/ 和远端 validation-results/。
summary.json 汇总核对了所有 rank 的结果；gpu-ws2、gpu-ws4、gpu-ws8
保存逐 rank JSON 和持久化调优配置；ce-ws4、ce-ws8 保存 CE 报告。
cpu-linux.xml 是 Linux 测试报告，environment.txt 记录环境与拓扑。
报告目录不纳入 git。

本次编译产物位于远端 /tmp/pcie-comm-coe11.yMsO3a/jit，
测试配置原件位于同级 ws2、ws4、ws8。该路径是隔离验证目录，
实际使用时默认缓存位置是 ~/.cache/pcie_comm。
