# KV cache 均值与 hidden states 的流式安全检测：最小实验执行计划

## 0. 给执行本计划的 code agent

本文件是实现与实验任务说明。请先阅读整个文件，再依次完成环境检查、资源下载、数据审计、代码实现、单元测试和真实模型 smoke test。用户的实验资源为**单张 24GB 或更大显存 GPU**。当前项目目录可能为空。

用户的研究目标是：在 LLM 生成过程中，直接使用已有 KV cache 的聚合特征训练轻量 classifier，比较其与 hidden-state 检测器的检测能力和**额外**显存开销。本阶段只研究均值，不实现复杂时间序列模型。

### 执行范围

- [ ] 检查实际执行主机、GPU、磁盘、Python/CUDA；建立隔离环境。
- [ ] 下载指定 backbone、tokenizer 和实验数据，记录不可变版本及文件校验信息。
- [ ] 按本计划实现完整实验管线和必要测试。
- [ ] 运行无需大模型的单元测试、真实模型对齐测试和小规模端到端 smoke test。
- [ ] 资源可用时运行 pilot；耗时较长的 extended 基准保留独立命令，不能用 smoke 结果替代。
- [ ] 交付 README、配置、测试结果、执行日志、结果表及可复现命令。

本计划本身**不表示任何模型已经下载、代码已经实现或实验已经完成**。不要预填结果，不要根据理论维度声称实际显存已经降低。

如果当前主机没有合适的 CUDA GPU，仍完成能进行的环境探测、代码和 CPU 测试，明确报告哪些 GPU 验证未执行；不要把 CPU 或其他加速设备的数据混入 CUDA 基准。不要自行启动或购买云资源。

## 1. 研究问题与固定范围

需要回答：

1. KV 前缀均值能否预测生成回答的累计有害性？
2. 与当前 token hidden state、hidden-state 前缀均值相比，准确性和检出时机如何？
3. 在双方都采用合理增量实现时，KV 方案是否降低实际额外显存或延迟？

首轮不扩展到：backbone 微调、LoRA、多模态、thinking traces、跨模型集成、多层拼接、GRU/Transformer 时序分类器、KV 压缩或量化。

### 固定 backbone

| 项目 | 默认值 |
|---|---|
| Hugging Face model ID | `Qwen/Qwen3-4B`，原始该名称的 checkpoint |
| 模型模式 | `eval()`，全部参数冻结，提取与部署使用 `torch.inference_mode()` |
| 精度 | backbone 与 KV：BF16；统计累计量与 classifier：FP32 |
| 思考模式 | `enable_thinking=False` |
| 主实验层 | 第 24 个 block，零基索引 `23` |
| hidden state 定义 | 此 block 的输出 residual stream，位于最终模型 norm 之前 |
| 提取框架 | Hugging Face Transformers，固定并记录实际测试过的版本 |
| Attention backend | 优先 SDPA；所有方法相同，禁止某方法单独切换 |
| Cache | 原生、未量化、未裁剪 KV cache；初版使用 DynamicCache |
| 提取 batch size | 1 |
| 完整样本长度限制 | chat 模板、prompt、完整 response 合计不超过 2048 tokens |

模型配置预期为 `num_hidden_layers=36`、`hidden_size=2560`、`num_key_value_heads=8`、`head_dim=128`。加载后断言实际配置与预期一致；若不一致，排查是否下载了其他变体，不要静默替换模型。

所有报告只对实际运行的模型、层位和数据范围作结论。单层效果差不能推导 KV 无安全信号。

## 2. 环境与下载

### 2.1 环境建立

1. 记录 OS、Python、GPU 型号/显存、驱动、可用 CUDA 运行时、CPU 内存、空闲磁盘。
2. 优先在已有 Linux CUDA 计算主机上执行；Windows 主机也可使用兼容环境，不能假设当前 Windows 工作目录就是 GPU 服务器。
3. 使用 Python 3.11 或 3.12 的独立虚拟环境。按目标驱动安装官方兼容的 PyTorch CUDA wheel；不覆盖系统 CUDA/Python。
4. 安装项目依赖：`torch`、`transformers`、`accelerate`、`huggingface_hub`、`safetensors`、`numpy`、`scikit-learn`、`pandas`、`pyarrow`、`PyYAML`、`pytest`、`tqdm`、`matplotlib`。按实际适配器需要添加 `datasets`，不必复制 SCM 的整套训练环境。
5. 选择支持 Qwen3、Fast tokenizer、SDPA 和 cache API 的稳定版本组合；完成 smoke 后锁定**准确版本**。不要在 README 中仅写“最新版”。
6. 交付 `pyproject.toml`、锁定文件或 `requirements.lock.txt`，以及环境重建命令。无需训练 backbone，不安装 DeepSpeed、FlashAttention 等非必要组件。

### 2.2 下载清单

| 资源 | 地址/ID | 用途 |
|---|---|---|
| Backbone、tokenizer、chat template、config | `Qwen/Qwen3-4B` | 特征提取、在线回放与真实生成 |
| FineHarm | `liyang-ict/FineHarm` | 主训练/验证/测试 |
| Qwen3GuardTest | `Qwen/Qwen3GuardTest` | `response_loc` 外部检出时机测试，后续启用 |

FineHarm 仓库已核实存在以下文件：

```text
FineHarm-train.json
FineHarm-val.json
FineHarm-test.json
README.md
```

参考 SCM 预处理代码可见字段包括 `prompt`、`sentences`、`sentence_labels`、`words`、`word_labels`。实现时仍须检查实际 JSON schema，不要假设文件名与 SCM 脚本的 `train_new.json` 等名称相同。

下载要求：

- 下载前检查磁盘空间；模型使用 HF 缓存或可配置的资源目录，避免仓库中存放权重。
- 解析并记录 model/dataset revision 的 commit SHA；后续运行使用固定 revision，不持续追踪 `main`。
- 记录 tokenizer、chat template、数据文件 SHA256 和许可信息。后续运行支持离线加载。
- 不下载 Qwen3Guard 的大模型权重、8B backbone 或 SIREN 的全部训练数据；本阶段不需要。
- 权重/数据可断点恢复；失败后报告原因，不静默换 checkpoint 或数据。
- 数据包含有害文本，日志默认只输出 ID、长度、标签和统计，不打印原始回答。

## 3. 数据处理与标签

### 3.1 划分与最小数据规模

沿用官方 train/val/test 来源；归一化 prompt 仅用于重复识别，不改变送入模型的原始文本。

跨集合同 prompt 的组不得泄漏。默认优先保留 test，其次 val，最后 train；将冲突组从较早阶段集合删除，并记录删除数量。对完全重复的 prompt-response 做去重。不要先生成前缀样本再随机划分。

| Split | Pilot 目标条数 | 用途 |
|---|---:|---|
| Train | 最多 5000 | 拟合 scaler 与 classifier |
| Val | 最多 1000 | checkpoint 选择、阈值校准 |
| Test | 最多 2000 | 最终一次性效果评估 |

从符合长度要求的完整样本中按回答安全标签分层抽样，尽量平衡安全/有害回答；有害 prompt 对应的安全拒绝必须保留。记录实际抽样比例和排除数量，不能把人工平衡后的 precision/F1 解释为自然流量指标。

不截断有害回答再沿用完整回答标签。首轮直接排除总长度超过 2048 的样本。固定采样 seed，并导出 `split_manifest.jsonl`。

### 3.2 文本、模板与 token 对齐

1. 检查是否存在可信的完整 response 字段；否则按明确规则从 `sentences` 重建，默认使用换行连接，并保留每句的字符区间。
2. 使用 Qwen3 自带 chat template，明确关闭 thinking。
3. 以最终输入字符串和 Fast tokenizer 的 offset mapping 建立句子到 token 的映射；模板特殊 token、空 offset 和句界跨 token 情况必须显式处理。
4. 保留 prompt 上下文，但均值只覆盖真正的回答文本 token。模板、padding、回答后的 EOS 不参与平均和监督。
5. 对无法无歧义对齐的记录输出错误清单，不猜边界、不静默删除单词。随机检查至少 20 个样本的对齐，涉及敏感原文的检查产物保留在本地，不放入通用日志。

统一以回答 token 的 **1-based 位置**报告检测时间，内部实现可用零基索引但必须转换清楚。

### 3.3 监督目标：累计前缀有害性

FineHarm 的词级标签使用启发式方法，因此本阶段使用其**句子标签**构造保守前缀监督。此标签是代理监督，不声称是人工逐 token 风险真值。

第一句有害句子覆盖回答 token `[a,b]`，两端包含：

```text
t < a       -> label = 0
a <= t < b  -> label = -100，忽略损失
t >= b      -> label = 1
```

完全安全回答：所有有效回答位置为 0。已经包含有害内容后，后续前缀持续为 1，即便后续句子安全。第一句有害句子内仍逐 token 输出检测分数，只是不把不确定位置用于训练 loss。

训练每条回答最多选 32 个有效位置：均匀覆盖前缀，并优先保留可用的 `a-1`、`b`、`b+32`；去重、检查越界，总数仍不超过 32。这只用于训练降采样；验证/测试逐 token 评分。

各方法共用位置清单。每条回答总 loss 权重相同，所选位置各分配 `1 / n_selected` 权重；batch 内采用能保持此权重的采样/归一化，不使长回答占优势。

### 3.4 外部数据

Qwen3GuardTest 的 `response_loc` 是全有害样本的位置测试集，不能单独计算误报率、AUROC 或宣称综合分类性能。第一阶段只下载和审计，主实验完成后再运行，阈值仍使用 FineHarm val 确定。

其边界基于原始 `input_ids` 和模板。核实 `unsafe_start_index` / `unsafe_end_index` 的端点约定，并转换到统一坐标；不能把原 token 索引直接应用于重新模板化的文本。若重建模板，需要通过原 token 到字符再到目标 token 的可验证映射。与训练数据的 prompt/回答重叠需检查并报告。首轮不启用 `thinking_loc`。

## 4. 特征与在线实现

### 4.1 五个对照组

均使用 block 23。`h_t` 指该 block 输出；`k_t,v_t` 指该 block 对 token t 写入实际 cache 的张量，head 维度仅展平，**不在 head 或 feature 维上求均值**。

| ID | 输入 | 维度 |
|---|---|---:|
| `h_last` | 当前回答 token 的 `h_t` | 2560 |
| `h_mean` | `mean(h_1,...,h_t)` | 2560 |
| `k_mean` | `mean(k_1,...,k_t)` | 1024 |
| `v_mean` | `mean(v_1,...,v_t)` | 1024 |
| `kv_mean` | 拼接 `k_mean`、`v_mean` | 2048 |

block 输出与该 block 的 K/V 投影输入不是同一个计算节点，应在报告中准确描述；本实验比较实际可用的表示，不将其误称为严格相同位置的线性投影实验。

Qwen3 实际缓存中的 K 是经过 k-norm 和 RoPE 的 K。主实验使用这些值；不能用 projection hook 取 pre-RoPE K 后仍标注为原生 cache 特征。K 可能对位置敏感，初版报告按长度分组的结果，不加入去 RoPE 变体。

### 4.2 两方都采用增量均值

```text
sum_t = sum_(t-1) + x_t
mean_t = sum_t / number_of_response_tokens_seen
```

- 累计和为 FP32；输入 cache/activation 保持 BF16，只转换当前新增片段。
- KV 方法读取 cache 当前新增位置，不复制完整 cache、不每次对整段历史求均值。
- hidden 方法只 hook 指定 block，在 hook 内完成累计更新，随后释放引用；不把历史 hidden states 存进列表。
- 性能测试 `output_hidden_states=False`、`output_attentions=False`，所有方法相同。
- 不把 cache 的 GQA heads 扩展为 query heads；使用原生 8 个 KV heads。
- 每个请求独立维护统计，明确 reset 生命周期；padding/EOS 不计入长度。
- 初版不支持 cache eviction、量化、滑动窗口或 beam reordering；遇到不兼容配置应报错。

### 4.3 离线提取与因果性

训练用冻结模型对原始回答 teacher forcing，先处理 prompt，再以默认 128 tokens 的块处理 response。按 causal mask 计算前缀累计量，只保存采样位置的**聚合特征**到 CPU/磁盘，不保存全部 raw K/V 或 hidden states。

可在同一离线 pass 中提取五组特征以节省算力；性能测量时必须每个方法独立进程运行，不能沿用同时提取五组的路径。

避免对整段序列保留 vocabulary logits；离线提取可使用基础模型模块。实时延迟基准则所有方法都使用相同的正常 CausalLM 解码路径，不能只有某方法省略 LM head。

保存格式可用分片 `.npz` 或其他明确支持类型的格式，建议存 FP32 聚合特征。附带 sample ID、prefix position、label、weight、layer、feature type。不要对保存的特征再做全序列双向处理。

### 4.4 生成时的一个 token 对齐问题

prefill 得到预测第一个回答 token 的 logits，但 cache 中尚无该回答 token。把回答 token t 输入模型后，才能得到其 K/V 和 hidden state；此次 forward 同时得到预测 token t+1 的 logits。

检测分数必须与**实际进入 cache 的前缀**对应。在线可保留一个 token 的输出缓冲；不要把 prefix t 的分数错记为 prefix t+1。终止前如要检查最后一个回答 token，需要 flush；baseline 采用相同 forward/flush 日程，相关计算计入开销。

首轮检测的是“已进入模型的回答前缀”，不是预测尚未产生 token 的有害性。

## 5. Classifier 训练

```text
StandardScaler（仅训练集拟合）
-> Linear(D, 128)
-> GELU
-> Linear(128, 1)
```

训练使用 logits，推理按需 sigmoid。

| 参数 | 值 |
|---|---|
| Loss | 加权 BCEWithLogitsLoss；忽略 `-100` |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Prefix batch size | 512 |
| Epoch 上限 | 20 |
| Early stopping | 按每条回答等权的验证 BCE，patience=3 |
| Seeds | 17、42、2026 |
| Dropout | 无 |

拟合 scaler 时也按每条回答等权，保存 mean/scale；零方差维设置安全分母。模型、scaler 都为 FP32。特征提取只需一次，三次重复只重训 classifier。训练 classifier 时可卸载 backbone，训练峰值不作为在线峰值。

基础五组均用 hidden width=128，报告实际参数量。另外训练 `kv_mean_matched`，width=160，使其第一层参数量与 `h_mean` 的 2560×128 接近。该控制用于区分参数量差异，不替代主比较。

不进行大量超参数搜索。test 不参与 checkpoint、层位或阈值选择。若更改训练策略，所有特征组必须一起重跑并记录版本。

## 6. 流式效果评估

### 6.1 决策规则与阈值

首轮采用一次超过阈值即报警，不加入每个方法不同的平滑或连续命中次数。

```text
t_alarm = 首个 score_t > threshold 的有效回答位置；不存在则为 None
```

对每个 checkpoint，使用验证集中**完全安全的回答**的 `max_t score_t` 校准阈值，选择使验证回答级误报率不超过 5% 的最低可用阈值。处理分数 ties 和 `>` / `>=` 的一致性。样本不足时报告不稳定性；不能在 test 上重新校准到 5%。

测试报告实测 FPR，即使它偏离验证目标。1% FPR 只作为后续有更多校准数据时的扩展。

### 6.2 主指标

对于有害回答，第一句有害句子为 `[a,b]`：

- `safe_response_fpr`：安全回答中任意时刻报警的比例。
- `premature_alarm_rate`：有害回答中 `t_alarm < a` 的比例；这是标注边界前报警，不能自动算安全检测成功，也不是人工确认的所有情况都无害。
- `timely_recall`：全部有害回答中 `a <= t_alarm <= b` 的比例。
- `recall_by_end_plus_32/128`：全部有害回答中 `a <= t_alarm <= min(T,b+32/128)` 的比例。
- `no_alarm_rate`：全部有害回答中整个回答无报警的比例。
- `nonpremature_detection_rate`：`a <= t_alarm <= T` 的比例；同时报告其补集，不与纯 no-alarm 混淆。
- `detected_delay_tokens`：有效检出样本的 `max(0,t_alarm-b)`，报告中位数、P90；必须与上述漏检/提前报警比例并列。

提前报警后真实系统已停止，不能再用后来某次报警给该样本计一次成功。回放仍保存全轨迹供诊断，但主指标使用第一次报警。

补充：最终前缀 AUROC、回答级 precision/recall/F1、按回答长度分组的 FPR/召回。它们不能替代流式主指标。bootstrap 以**回答**为重采样单位，不能把同一回答的 token 当独立样本；报告三训练 seeds 的结果，不只选最好 seed。

## 7. 显存与延迟实验

### 7.1 公平性

比较 `baseline_no_detector` 与五种检测器；每种方法/测试单元使用独立进程。classifier/scaler 已载入后进入测量；测量总峰值时须包含其驻留内存，不能通过减去加载后值把模型参数排除。

所有方法使用相同 prompt、token IDs、长度、cache、attention backend、dtype、batch、LM head 和最终 flush 日程。使用固定回答逐 token 回放，关闭真正的提前终止，以隔离检测开销。计时区域不包含磁盘写入、绘图、tokenization 或 per-token CPU `.item()` 同步。

显存基准中的长文本可由固定安全文本构造；这只是资源负载，不能用来报告安全检测效果。

### 7.2 两级资源矩阵

| Suite | Batch | Prompt tokens | Response tokens | 重复 |
|---|---|---:|---|---:|
| `smoke` | 1 | 128 | 64 | 1，仅验证可运行 |
| `pilot` | 1 | 128 | 512、2048 | 3 |
| `extended` | 1、4 | 128 | 512、2048、8192 | 5 |

每次先做不计入结果的预热，释放预热请求的 cache，再运行正式新请求。固定 cache 实现；若 OOM，记录该 cell 为 OOM，不静默缩短长度或改变精度。

### 7.3 测量项

- `peak_allocated_mib`、`peak_reserved_mib`：完整请求峰值。
- `decode_peak_allocated_mib`：单独 reset 后的 decode 阶段峰值；仍包括驻留模型/cache。
- `extra_peak_mib`：同负载下该方法峰值与 baseline 峰值的配对差值。
- `detector_parameter_bytes`、`scaler_bytes`、`persistent_state_bytes`：单独的张量存储审计，注意共享 storage 不重复计数。
- `decode_ms_per_token`、`tokens_per_second`、相对 baseline 延迟增量；吞吐明确是否乘 batch。
- 环境、各次原始测量值、汇总中位数与变动范围。

使用 `torch.cuda.synchronize()`、`reset_peak_memory_stats()`、`max_memory_allocated()`、`max_memory_reserved()`；同步放在整个计时区间边界，不在正常吞吐路径中每 token 强制同步。需要逐步延迟分位数时另开诊断模式，不与主吞吐混报。

allocator reserved、CUDA context 和张量 allocated 含义不同，`nvidia-smi` 仅作补充。峰值差接近分配器粒度或运行波动时，报告“未测得可靠差异”，不得写成显著收益。

### 7.4 理论大小用于 sanity check，不作为实验结论

单样本、单层、仅 FP32 累计向量：

| 方法 | 字节数 | KiB |
|---|---:|---:|
| H-mean | 2560×4 | 10 |
| K-mean | 1024×4 | 4 |
| V-mean | 1024×4 | 4 |
| KV-mean | 2048×4 | 8 |

batch 下线性增长。还需计数器、classifier、scaler 和临时内存；模型原有 KV cache 是所有方案共同的开销。H-mean 与 KV-mean 累计量只差约 2 KiB，不能预设在线峰值会大幅下降。

## 8. 代码结构与命令契约

建议目录，可在保持职责清楚的前提下调整：

```text
PLAN.md
README.md
pyproject.toml
requirements.lock.txt
.gitignore
configs/
  smoke.yaml
  pilot.yaml
  extended.yaml
src/kv_safety/
  environment.py
  download.py
  data/                 # FineHarm/QwenGuard adapters、split、offset labels
  features/             # hidden hooks、cache reader、running mean
  classifier.py
  extract.py
  train.py
  evaluate.py
  benchmark.py
  report.py
tests/
  test_labels.py
  test_running_mean.py
  test_splits.py
  test_metrics.py
  test_causality.py
  test_model_alignment.py
artifacts/              # gitignore，资源 manifest、特征、checkpoints、结果
```

提供等价于以下调用的 CLI，并在 README 写明实际参数：

```bash
python -m kv_safety.environment
python -m kv_safety.download --config configs/pilot.yaml
python -m kv_safety.data.prepare --config configs/pilot.yaml
pytest -q -m "not gpu"
pytest -q -m gpu
python -m kv_safety.extract --config configs/smoke.yaml
python -m kv_safety.train --config configs/smoke.yaml --all-features
python -m kv_safety.evaluate --config configs/smoke.yaml --all-features
python -m kv_safety.benchmark --config configs/smoke.yaml
python -m kv_safety.extract --config configs/pilot.yaml
python -m kv_safety.train --config configs/pilot.yaml --all-features
python -m kv_safety.evaluate --config configs/pilot.yaml --all-features
python -m kv_safety.benchmark --config configs/pilot.yaml
python -m kv_safety.report --config configs/pilot.yaml
```

配置至少包含：资源路径、revision、layer index、dtype、backend、max length、split budgets、seed、chunk size、feature types、classifier、训练参数、阈值规则、性能测试矩阵。禁止在代码里写死本地 Windows 绝对路径。

## 9. 必须通过的测试

### CPU/小型合成测试

- [ ] Running mean 等于显式前缀均值；逐 token 与分块更新一致；不随 T 增加永久统计量大小。
- [ ] head flatten 顺序一致；K/V 拼接顺序固定；padding/prompt/EOS 掩码正确。
- [ ] 标签在 a、b 边界及单 token 句子、第一句有害、全安全情况下正确。
- [ ] 相同 prompt 的不同回答不会跨 split；预处理前后 ID 可追踪。
- [ ] scaler 不读 val/test；每回答权重与零方差处理正确。
- [ ] 模拟安全分数验证回答级 FPR 校准、ties、无报警和提前报警指标。
- [ ] 长短回答权重一致；被忽略区间不贡献 loss。
- [ ] 多请求 reset 不残留前一请求的累计和；classifier checkpoint/scaler 往返一致。

### 真实 Qwen3 GPU 测试

- [ ] 选定层和特征维度正确，backbone 不产生梯度。
- [ ] 直接 cache 读取确实对应存储的 post-RoPE K/V，未读取其他投影或错误 layer。
- [ ] 至少 20 条短样本，逐 token 与分块提取的聚合特征/分类分数在合理容差内一致，记录最大误差；BF16 不要求 bitwise 一致。
- [ ] 同一前缀后追加两个不同后缀，前缀特征不变至数值容差内，排除未来信息泄漏。
- [ ] 明确验证 prefill、第一个回答 token、普通 decode、最后一个 token 的位置对齐。
- [ ] 监控器启用但不干预时，固定输入下的生成 logits/结果不被修改。
- [ ] H-mean 不保留历史 hidden tensors；KV 方法不 clone 完整 cache；统计量持久大小与 T 无关。
- [ ] BF16 特征无 NaN/Inf；FP32 累计量 dtype 没有被模型 dtype 转换覆盖。
- [ ] benchmark 的所有方法使用相同 token 数和 forward 次数。

不要为通过测试而不断放宽数值容差；先排查 attention mask、position IDs、模板、cache 长度和浮点累加顺序。测试用短合成文本足以验证机制，避免在测试仓库写入不必要的有害原文。

## 10. 阶段、产物与验收

### Phase A：环境和资源

产物：锁定环境、资源 manifest、数据 schema/统计报告。不得只声称“下载成功”，应验证可离线加载 tokenizer/config/权重和读取所有 split。

### Phase B：Smoke

使用约 32 train / 16 val / 16 test 的小子集，尽量包含两个类别；1 个 seed，训练 2 epochs；benchmark 采用 smoke 矩阵。必须打通下载到结果表的完整链路。**这组结果只说明代码可运行，不用于有效性或低误报结论。**

### Phase C：Pilot

5000/1000/2000 上限、五组特征、三个 classifier seeds，加 `kv_mean_matched` 参数量控制。验证测试通过后运行。训练种子不重复下载模型或重新提取相同特征。

输出至少：

```text
environment.json
resource_manifest.json
data_audit.json
split_manifest.jsonl
feature_manifest.json
checkpoints/<feature>/<seed>/
metrics/per_response.parquet
metrics/quality_summary.csv
metrics/memory_latency_raw.csv
metrics/memory_latency_summary.csv
figures/quality_vs_overhead.png
figures/memory_vs_length.png
REPORT.md
```

效果表包含：方法、seed、层位、维度、参数量、阈值、test FPR、及时检出率、+32/+128 检出率、提前报警率、无报警率、延迟统计。

资源表包含：方法、batch、P/T、总峰值、相对 baseline 峰值增量、持久统计量、每 token 延迟、吞吐、重复次数。报告所有单元的失败/OOM，不删除不利结果。

### Phase D：结果驱动的扩展，不在初版同时铺开

1. 若主层 KV 明显差，先只复核第 12、36 层的 H-mean/KV-mean；明确这是第二轮探索，不把 test 最优层当无偏结果。
2. 若效果接近，运行 extended 显存矩阵和 `response_loc` 外部测试。
3. 再用 backbone 原生生成约 200 条新回答，依据新回答重新标注，检查回放到真实生成的分布变化。原 prompt 的标签或旧回答标签不能沿用。新数据可先生成候选并输出待标注文件，缺少可靠标注时不要编造结果。
4. 完成以上后再讨论时间序列方案，且必须同时加入相同结构的 hidden-state 时间序列对照。

### 完成判据

- 代码与配置能够重建结果，CPU 测试和真实 GPU 对齐测试有日志。
- baseline 与 KV 的因果性、token 对齐、标签、数据划分、存储实现公平。
- 报告明确区分已执行、未执行、理论估计及实际测量。
- 若未测得空间优势，照实报告；不得以离线全量 hidden-state 提取作为主要在线基线。
- 不把本最小实验称为 SIREN/ShieldHead/Kelp 的完整复现；它是受控的特征比较实验。

## 11. 参考资料

- Qwen3-4B：[模型卡](https://huggingface.co/Qwen/Qwen3-4B)、[config](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)。
- Qwen3 实现：[Transformers modeling_qwen3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)。开发时以锁定版本对应源码为准。
- FineHarm：[数据仓库](https://huggingface.co/datasets/liyang-ict/FineHarm)、[SCM 论文](https://arxiv.org/abs/2506.09996)、[官方代码](https://github.com/ICTMCG/SCM)、[数据预处理参考](https://github.com/ICTMCG/SCM/blob/main/utils/data_process.py)。
- Qwen3GuardTest：[数据说明](https://huggingface.co/datasets/Qwen/Qwen3GuardTest)、[官方评估](https://github.com/QwenLM/Qwen3Guard/tree/main/eval)。
- SIREN：[论文](https://arxiv.org/abs/2604.18519)。
- ShieldHead：[论文](https://aclanthology.org/2025.findings-acl.932/)。
- Kelp：[论文](https://arxiv.org/abs/2510.09694)。
- PyTorch 显存统计：[CUDA 文档](https://docs.pytorch.org/docs/stable/cuda)。

文献和源码用于核对数据、模型与 API；本文件中的样本预算、层位、classifier、代理标签和评估规则是本实验的设计选择，不代表原论文协议。
