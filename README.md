# Llama-70B：八卡 A100 稳态投机解码测试

目的只有两个：在**实际运行请求数为 128 / 256 / 512、连续没有 Prefill**的区间，比较 Chain7、原规则优化版 Adaptive2、EAGLE728 的吞吐和阶段耗时；计算保留 Adaptive2 当前输出效果、把 Draft 侧成本降至 Chain7 水平时，是否有可能超过 EAGLE728。

这是 **1 个 TP8、DP1、PP1 的模型实例**。一个 Python 主进程启动八个张量并行 rank。**不要使用 `torchrun --nproc_per_node=8` 再启动本脚本，也不要在八张卡上各启动一次脚本。**

> 本仓库已做离线代码、数据、插桩和统计口径检查；尚未在目标机器上完成 Llama-3.3-70B / TP8 / A100 的实测。CPU 检查通过不能证明模型加载、Graph 捕获、显存容量和满 BS 稳态一定通过。目标机器必须先完成环境检查和 smoke；不满足条件时脚本停止，不能把缺失结果当成功。

## 1. 固定条件

| 项目 | 本次设置 |
|---|---|
| Target | `meta-llama/Llama-3.3-70B-Instruct`，本机完整 BF16 权重 |
| Draft | `lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge`，本机配套权重 |
| 机器 | 单节点 8×A100 80GB，MIG 关闭，八卡均无其他计算任务 |
| 并行 | TP8 / DP1 / PP1 |
| 数据 | 八个 RedHat 子集混合；Target AR 按输出长度筛选后，每子集重复到 500 条，共 4000 条 |
| 解码 | greedy、seed42、自然 EOS、每条最多 1024 新 token、context4096 |
| 运行上限 | 128、256、512；验收用真实运行数，不用 Graph padding 后的形状 |
| Graph | Decode 开；Prefill breakable Graph 开；记录实际 Decode / Draft / Extend Graph 是否执行 |
| Radix Cache | 开；每次独立进程，正式负载前清空 warmup cache；负载内部重复请求允许正常命中 |
| 调度 | FCFS；chunk4096；maxprefill16384；mixed chunk 关；Prefill 优先；不人为让短请求忽略 EOS |
| Adaptive2 | `p1 < 0.5` 允许分叉；每请求七个 draft 节点预算；最多两个前沿槽位 |
| 比较次数 | 3 方法 × 3 BS × 3 遍 = 27 次正式计时；另有 9 次 smoke、9 次 kernel 诊断 |

阈值 0.5 是本次固定比较条件，**不是声称它已经在 70B 上最优**。三个方法用同一份冻结源码、输入和核心配置。方法顺序按三遍轮换，避免每次都让同一个方法最先运行。

## 2. 先准备运行环境

代码由独立源码副本运行，不替换机器的公用 SGLang，不修改模型权重。先与机器管理员确认空闲八卡及 CUDA 环境。

```bash
git clone https://github.com/12liurunze/llama-profile.git
cd llama-profile

# 激活与本仓库记录版本匹配的 Linux GPU Python 环境。
# 若使用已有匹配环境，可再建一个隔离入口；不会自动升级已有包。
python -m venv --system-site-packages .venv
source .venv/bin/activate

python -m unittest discover -s tests -v
python scripts/build_dataset.py --check
python tools/install_runtime.py --verify-only
python tools/check_release.py
python tools/install_runtime.py
```

`requirements.runtime.txt` 是本源码的核心版本约束，`vendor/requirements.observed.txt` 是开发机器的完整包版本记录。原环境为 **PyTorch 2.13.0 / CUDA13.0、Triton3.7.1、Transformers5.12.1、sglang-kernel0.4.7**。当前预检要求能支持 CUDA13 的驱动（至少 580 主版本）以及 sm80/NCCL。

**已有模型不代表软件环境也兼容。** 预检发现版本不符时会停止。不要在别人的公用环境里直接批量升级，也不要删掉检查“强行跑”。如果机器只有 CUDA12 环境，需要在新的虚拟环境/容器中安装匹配依赖或另行验证移植；本仓库没有把尚未验证的 CUDA12 兼容性写成已支持。完整 observed 清单含与本实验无关的包，它不是保证所有平台都能一键安装的锁文件。

源码安装结果：

```text
.runtime/base/sglang/       冻结原始源码
.runtime/profile/sglang/    独立计时副本
.runtime/install_receipt.json  全文件哈希与插桩 AST 审计
```

没有覆盖式安装；已存在的目录不会被替换。正式运行会重新核对源码与模型元数据。模型校验包括配置、索引、safetensors 头、尺寸及文件时间，不读取并散列 140GB 的全部 tensor 内容。

## 3. 设置模型目录并预检

把下面两行改成**该机器的本地完整目录**。它们是目录，不是待在线下载的模型 ID。

```bash
export TARGET_DIR=/actual/models/Llama-3.3-70B-Instruct
export DRAFT_DIR=/actual/models/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge
export GPU_IDS=0,1,2,3,4,5,6,7

python tools/doctor.py \
  --model "$TARGET_DIR" --draft "$DRAFT_DIR" --gpus "$GPU_IDS" \
  --output results/preflight.json
```

必须看到 `audit_pass: true`。预检检查卡型号/容量/MIG/驱动/占用、运行库、源码身份、模型结构及权重头。它不会加载模型或启动正式测试。模型实际能否加载、TP 通信和 Graph 是否正常，由后面的 smoke 检查。请预留结果磁盘空间，并观察主机 RAM；权重已在机器上，不会上传或重新下载到本仓库。

运行器还会使用本机文件锁避免本仓库重复占卡，并每两秒左右查询 GPU 进程。发现其他任务进入时，只停止自己启动的进程组，保留日志和失败目录；不会杀其他人的进程。`--wait-free` 表示等待八卡同时空闲，不表示抢占。

## 4. 准备并固定 4000 条负载

仓库已经带 `data/redhat8_4000.jsonl`：八子集各500条，由724条源记录、644个不同 prompt 重复得到。`question` 和 `writing` 的上游源文件相同，已在清单披露。**这不是4000道独立题。** 完整来源、许可证、重复编号和 SHA256 见 [data/README.md](data/README.md)。

先用目标 tokenizer 编码，不截断输入、不添加参考答案、不改写问题：

```bash
python main.py prepare --model "$TARGET_DIR" \
  --output results/llama70b_tp8/prepared4000.json
```

再运行 Target-only AR，测644个不同 prompt 的自然输出长度。只按长度挑选至少256 token 的请求，每子集重复到500条，固定成稳态负载：

```bash
python main.py calibrate \
  --model "$TARGET_DIR" --draft "$DRAFT_DIR" --gpus "$GPU_IDS" \
  --data results/llama70b_tp8/prepared4000.json \
  --output results/llama70b_tp8/calibration \
  --steady-output results/llama70b_tp8/steady4000.json \
  --minimum-output 256 --wait-free
```

这是用户批准的**长响应条件负载**：不按预测正确率、接受长度或方法快慢筛选。所有方法复用同一选定文件。达到1024上限的响应是被上限截住的长度下界，不能称其完整自然答案恰好1024。报告必须说明长响应筛选与重复会改变原始数据分布，结果不能冒充原始八个 benchmark 的总体表现。

如果某子集没有合格的长响应，程序会停止并列出该子集。可以查看已有 `calibration.json` 后，明确选择新的长度阈值，使用 `select` 生成**另一个文件**；不需要重复跑 AR。不要静默丢弃子集或强制不结束。

```bash
# 仅示例：明确改为至少128 token，生成不同文件；后续所有方法统一换用它。
python main.py select \
  --prepared results/llama70b_tp8/prepared4000.json \
  --calibration results/llama70b_tp8/calibration/runs/calibration/calibration.json \
  --minimum-output 128 --output results/llama70b_tp8/steady4000_min128.json
```

## 5. 按顺序跑三步

以下使用同一组参数，输出目录固定。每一步成功后再执行下一步。

```bash
COMMON=(--model "$TARGET_DIR" --draft "$DRAFT_DIR" --gpus "$GPU_IDS"
        --data results/llama70b_tp8/steady4000.json
        --output results/llama70b_tp8/formal)

# ① 每方法、每BS各32题预检，包含最长输入；检查加载、TP、Graph和输出计数
python main.py run "${COMMON[@]}" --stage smoke --wait-free

# ② 27次正式计时；每次4000条，三个方法×三个BS×三遍
python main.py run "${COMMON[@]}" --stage timing --wait-free --resume

# ③ 9次独立kernel诊断；不拿这部分的时间当正式吞吐
python main.py run "${COMMON[@]}" --stage kernel --wait-free --resume

python main.py report --output results/llama70b_tp8/formal
```

等价的完整流水线：设置好 `TARGET_DIR`、`DRAFT_DIR` 后执行 `bash scripts/run_all.sh`。它不会安装或升级 GPU 软件包。每个正式运行启动独立 Engine，不共享上一方法的 KV/cache。

只想检查计划、不接触 GPU：

```bash
python main.py run "${COMMON[@]}" --dry-run
```

中断后用**完全相同参数**加 `--resume`。已通过的结果保留；失败运行移到独立目录后只补缺项。修改输入、阈值、显存比例或其他条件应使用新的输出目录，禁止与旧数据混算。默认单次超时7200秒，超时会终止自己的进程并保存失败原因。

## 6. 什么才算一次有效稳态结果

不是“512张 Graph”、不是峰值碰到512，也不是运行器允许512就算达标。默认要求同时满足：

1. 八个 rank 的 step 编号、请求顺序指纹、实际 BS 一致。
2. 连续至少10个 worker step 都是纯 Decode，没有 Prefill、mixed 或 idle 插入。
3. 同一组请求连续运行，每个请求每步都有真实输出增长，没有已结束请求占槽。
4. Target、Draft 和 Draft Extend 实际执行 Graph；关键阶段和输出计数完整。
5. 全过程没有外部 GPU 计算进程干扰，运行前后源码/输入身份一致。

主结果使用满足条件的**最长连续窗口**，并同时保留所有窗口供复核。长度相同时取最早窗口，不按速度挑选。4000条的尾部不满512不能混进512统计。若没有至少10步严格窗口，给出“证据不足”并停止，不自动改成511、不降低门槛、不伪造吞吐。

`queued` 默认将全部请求一次性提交，运行器最大并发由服务器控制。`cohort` 是显式可选的分组协议，会改变 cache/入队过程，必须独立输出目录并单独标注。两种协议都保留自然 EOS；配置本身不能保证满 BS，最终只认 trace。

## 7. 最终结果与计时口径

`REPORT.md` 和 CSV 会给出三张主表：

| 表 | 内容 |
|---|---|
| 稳态性能 | BS、方法、三遍平均±样本标准差、真实输出吞吐、接受 draft 长度、七节点预算接受率、窗口步数、上下文长度 |
| 阶段耗时 | 每step的 Draft、Verify、Draft Extend、其他、step间隙；Verify 再列 Target 前向、接受处理、KV 子项 |
| 反事实估算 | Adaptive 保持当前输出效果，Draft侧降至Chain成本后的预测吞吐、相对EAGLE变化、是否可能反超 |

同时保留每遍的原始值，不能只交平均数。

**吞吐** = rank0记录的实际保留输出token数 ÷ rank0连续CUDA时间窗口（包括worker之间的间隙）。八个 rank 执行同一批请求，不能把输出量乘8。这个指标用于本次纯Decode稳态诊断，**不是客户端请求端到端时延/吞吐**。包含 Prefill 和尾部的整批 wall time 另存 `whole_workload.json`，不混入稳态表。

**接受长度**同时区分：接受的 draft 个数、不含draft的额外token加上后的接受长度、最终真正保留的输出增量。请求尾部可能因 EOS/上限丢掉部分接受结果，计算吞吐用真实保留增量。接受率按“接受draft数 /（7 × request-step数）”定义，明确是七节点预算利用率，不把加一后的接受长度除7。

**阶段定义**：

```text
worker total
├─ Draft：准备草稿、Graph重放、候选选择/树结构构建
├─ Verify（包含下列三部分及少量未归属开销）
│  ├─ target_prepare
│  ├─ target_forward
│  └─ accept_and_cache
│     └─ kv_copy（子项，不能再加一次）
├─ Draft Extend：验证后更新draft状态、产生下一轮seed
└─ 其他worker开销
连续窗口还包含worker之间的间隙
```

每张卡分别保存 CUDA Event 区间。主阶段表使用 rank0 可相加的区间；另表给每阶段最慢 rank 的耗时。不同阶段的最慢卡可能不同，**各阶段 rank 最大值不能直接相加，也不能把八张卡耗时相加当单步时延**。TP通信可能被包含在前向区间中，kernel诊断用于区分GEMM/Attention/NCCL/选择/KV等成本。

这些是主计算流上的经过时间，不是所有CUDA流的kernel忙碌时间之和；异步准备工作的等待可能归入其后依赖阶段。轻量CUDA Event与CPU记账仍有非零开销。独立kernel诊断另有更明显的profiler开销，不能与正式表混算。`kernel_totals.json`按kernel名称汇总整个捕获区间；区分Draft和Target要结合Chrome trace内的`profile.stage.*`范围与关联ID，不能仅凭总表作归因。不同A100机器的PCIe/NVLink拓扑也可能不同，预检会保存拓扑与运行日志。

### “Draft降到Chain水平”怎样计算

同BS、同repeat配对。令 `T_A` 为Adaptive连续窗口平均每step时间，`D_A/E_A` 为其Draft/Extend；`D_C/E_C`为Chain对应成本；`Y_A`为Adaptive真实保留输出token/step：

```text
T_hyp = T_A - D_A - E_A + D_C + E_C
Throughput_hyp = 1000 * Y_A / T_hyp
gain_vs_EAGLE = Throughput_hyp / Throughput_EAGLE - 1
```

这是**成本替换假设下的估算**，不是已经实现的加速。它假设验证、间隙、接受效果和通信依赖不变。不同方法会推进到不同上下文，报告提供窗口上下文范围，不能把这个估算解释为严格同状态的因果实验。不会直接套用旧8B/A6000的57ms、80ms或591ms结论。

## 8. 算法和实现在哪看

- `llama_profile/common.py`：三个方法的参数、Graph和调度配置、三遍轮换顺序。
- `llama_profile/worker.py`：warmup → 正式自然生成 → drain → trace输出 → 验收。
- `profiler/instrument.py`：只对独立副本插入计时；剥离插桩后AST必须与冻结源相同。
- `profiler/stage_hooks.py`：每rank CUDA Event、rank0真实保留输出计数；无新增每step同步、collective或额外D2H。
- `profiler/analyze.py`：连续真实满BS窗口筛选、计时嵌套审计。
- `profiler/kernel_probe.py`：独立Kineto运行，保留GPU Graph内部kernel证据。
- `.runtime/base/sglang/srt/speculative/eagle_worker_v2.py`：原始Draft/Verify/Extend流程。
- `.runtime/base/sglang/srt/speculative/spec_utils.py`：候选和自适应分叉选择。
- `.runtime/base/sglang/kernels/ops/kvcache/cache_move.py`：KV搬移实现。

**Chain7**保持单条Top1链，七个draft节点。**Adaptive2**保留原来的逐层预算扣减规则：低Top1概率允许替代分支，前沿最多两个槽位，总计七个有效draft节点预算；预算耗尽的槽位可以失活，但Graph物理形状仍是双槽。**EAGLE728**是S7/K2/N8，先产生较多候选，再选最终七个draft节点交Target验证。N8含一个非draft位置，不是八个draft节点。

本仓库沿用已冻结的融合候选处理、Graph与KV优化路径。没有加入8B专用的 `active_linear` 调参：那部分针对A6000和特定小批形状，不能当作已验证的70B/A100优化。128/256/512这几个档位原优化也会走通用稠密路径。**最终候选少，并不等于物理计算行数少**：双槽和相同前向次数是需要用本次kernel证据检验的重点，不预先写成测量结论。

该快照没有可追溯的上游Git提交号，不能冒称官方某个release。实际身份由 `vendor/provenance.json`、归档SHA256和完整文件manifest固定。安装时保留两份源码，公共源码不修改。

## 9. 应该返回什么

请返回整个 `results/llama70b_tp8` 目录，包含长度校准、固定数据、正式/诊断原始记录，而不只截图：

```bash
tar -czf llama70b_tp8_results.tar.gz results/llama70b_tp8
```

也可只打包正式运行（校准目录请另发）：

```bash
python main.py pack --output results/llama70b_tp8/formal \
  --archive results/formal_results.tar.gz
```

重要文件：`plan.json`、`input_snapshot.json`、`preflights/`、`runs/*/config.json`、`server_info.json`、`monitor.jsonl`、`analysis.json`、`trace/rank_*/rounds.json`、kernel trace和 `REPORT.md`/全部CSV。以实际生成的文件为准。失败时返回对应日志与审计JSON；不要删失败目录，不要靠改统计口径让表格“通过”。

可以用 `python main.py report --output ... --allow-partial` 查看部分结果，报告会明确标记不完整。所有27次正式计时及计划中的诊断都完成后，才可交付最终结论。

## 来源

- [配套EAGLE3模型卡](https://huggingface.co/lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge)
- [Target模型卡](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [RedHatAI数据集](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks)
- [SGLang上游](https://github.com/sgl-project/sglang)

模型权重不在仓库内；使用模型应遵守各自许可。生成成功或相同文本不等于正确率，也不证明任意输入无损。
