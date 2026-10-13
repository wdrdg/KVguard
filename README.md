# KVguard：KV cache 均值 vs hidden states 的流式安全检测（最小受控实验）

实验设计见 [PLAN.md](PLAN.md)。本仓库实现完整管线：环境探测 → 固定 revision 下载 → 数据审计/无泄漏划分/token 对齐 →
离线 teacher-forcing 特征提取 → classifier 训练 → 流式首次报警评估 → 独立进程 CUDA 显存/延迟基准 → 报告。

## 当前执行状态（如实记录）

| 项目 | 状态 |
|---|---|
| 代码与配置 | 已实现 |
| CPU 单元测试 + 微型随机 Qwen3 机制测试 + 合成数据端到端测试 | **已执行，50 passed**（`logs/pytest_dev_container.log`） |
| 真实 Qwen3-4B GPU 对齐测试（`-m gpu`，10 项） | **未执行**：开发容器无 GPU（`logs/environment_dev_container.json`） |
| 模型/数据下载 | **未完成**：开发容器的网络策略拒绝 `huggingface.co`（HTTP 403，`logs/download_attempt_dev_container.log`） |
| 数据 schema 实测审计 | **未执行**（依赖下载）。FineHarm 适配器按字段探测实现，未在真实文件上验证 |
| Smoke / Pilot / Extended 实验、显存基准 | **未执行**，没有任何结果数字 |

`requirements.lock.txt` 是在开发容器（Linux x86_64, Python 3.12）解析出的精确版本，**尚未经过真实 GPU smoke 验证**；
完成 GPU smoke 后若有调整，应重新 `uv pip freeze` 覆盖。PyPI 上的 `torch==2.14.1` 为 CUDA 13.0 构建，需要相应较新的 NVIDIA 驱动；
驱动较旧时请改用 PyTorch 官方对应 CUDA 版本的 wheel（同一 torch 版本），并在 `environment.json` 中记录。

## 环境重建

```bash
uv venv -p 3.12 .venv            # 或 python3.12 -m venv .venv
uv pip install -p .venv/bin/python -r requirements.lock.txt
uv pip install -p .venv/bin/python -e . --no-deps
source .venv/bin/activate
python -m kv_safety.environment --config configs/pilot.yaml   # -> artifacts/pilot/environment.json
```

权重和数据放在 HF 缓存（`HF_HOME` 或配置中的 `hf_cache_dir`），不进仓库。`artifacts/` 已 gitignore。

## 命令（与 PLAN §8 一致）

```bash
python -m kv_safety.environment --config configs/pilot.yaml
python -m kv_safety.download   --config configs/pilot.yaml [--verify-weights]
python -m kv_safety.data.prepare --config configs/smoke.yaml
pytest -q -m "not gpu"
pytest -q -m gpu                      # 需 CUDA + 已下载的固定 revision；写 artifacts/gpu_alignment_log.json
python -m kv_safety.extract   --config configs/smoke.yaml
python -m kv_safety.train     --config configs/smoke.yaml --all-features
python -m kv_safety.evaluate  --config configs/smoke.yaml --all-features
python -m kv_safety.benchmark --config configs/smoke.yaml
python -m kv_safety.report    --config configs/smoke.yaml
# pilot：同样顺序，--config configs/pilot.yaml（先 data.prepare）
# extended 显存矩阵（复用 pilot 的 artifacts，输出 *_extended.csv）：
python -m kv_safety.benchmark --config configs/extended.yaml
python -m kv_safety.report    --config configs/extended.yaml
```

下载完成后 `artifacts/resource_manifest.json` 记录每个资源的 commit SHA、文件 SHA256、许可证、chat template SHA256；
配置里 `revision: null` 表示使用 manifest 中的 SHA。设置 `offline: true` 后只从本地固定 revision 加载。

## 关键实现约定

- **模型输入**：`apply_chat_template([user], add_generation_prompt=True, enable_thinking=False)` 的 token ids，接上**单独 tokenize 的 response** ids
  （即模型自己生成该回答时看到的序列，边界不会落在 token 内部）。末尾 `<|im_end|>` 不输入、不计数、不监督。
  长度限制 = prompt 模板 tokens + response tokens ≤ 2048，超长直接排除。
- **句子 → token**：Fast tokenizer offset mapping；与第一句有害句子（去掉首尾空白后的字符区间）有重叠的 token 构成 `[a,b]`（1-based，含端点）。
  跨句界 token 计入并统计；response 内出现特殊 token、空 offset、非连续映射等一律进入 `alignment_errors.jsonl`，不猜测。
  完整 response 字段只有在所有句子都能按顺序找到时才使用，否则按 `\n` 连接句子重建。
- **标签**：`t<a → 0`，`a≤t<b → -100`，`t≥b → 1`；训练每条回答 ≤32 个有效位置（优先 `a-1`、`b`、`b+32`，其余均匀），每位置权重 `1/n`；
  loss 为 `Σw·BCE/Σw`，scaler 只用 train 且按回答等权；early stopping 用按回答等权的 val BCE。
- **特征**（block 23）：`h` = 该 block 输出（final norm 之前）；`k,v` = 该 block 写入 DynamicCache 的张量（K 为 k_norm + RoPE 之后），
  8 个 KV head 按 head-major 展平，不在 head/维度上求均值。均值只覆盖 response token，FP32 累加。
- **在线检测**：喂入 response token t 的那次 forward 之后得到的分数属于前缀 t；最后一次 forward 即 flush。
  KV 方法只读 cache 末尾新增位置（视图，不复制整段 cache）；H 方法在 hook 内更新累计和、不保留历史张量。持久状态与 T 无关。
- **阈值**：在 val 完全安全回答的 `max_t logit` 上取使 FPR≤5% 的最低阈值（严格 `>` 报警，ties 不计入），test 只报告实测 FPR。
- **基准**：每个 (方法, batch, 长度, 重复) 单独子进程；相同 token ids/长度/cache/SDPA/BF16/LM head/flush 日程；
  预热后释放再测；无逐 token `.item()`；OOM 记为 OOM。非 CUDA 环境直接拒绝运行，不产生 CPU 数字。
  若对应 checkpoint 尚未训练，classifier 用同结构随机初始化并在结果中标记 `classifier_source`。

## 测试

- `tests/test_running_mean.py`、`test_labels.py`、`test_splits.py`、`test_alignment.py`、`test_metrics.py`、`test_classifier.py`：纯 CPU 单元测试。
- `tests/test_causality.py`：在**随机初始化的微型 Qwen3**（与 Qwen3-4B 同一份 transformers 代码路径）上验证 cache 中是 post-RoPE K、
  分块与逐 token 提取一致、前缀不受后缀影响、在线分数与前缀位置对齐、监控不改变 logits、状态大小恒定、请求间 reset、基准 forward 次数一致。
  这些只验证机制，不代表真实模型数值。
- `tests/test_pipeline_synthetic.py`：合成数据 + 微型模型跑通 prepare→extract→train→evaluate→report。
- `tests/test_model_alignment.py`（`-m gpu`）：真实 Qwen3-4B 上的上述检查，含 20 条短文本分块/逐 token 误差记录。

## 目录

```
configs/{smoke,pilot,extended}.yaml
src/kv_safety/
  environment.py download.py config.py modeling.py
  data/{fineharm,splits,alignment,labels,prepare,qwen3guard}.py
  features/{running_mean,cache_reader,hidden_hook,monitor}.py
  extract.py classifier.py train.py metrics.py evaluate.py benchmark.py report.py
tests/
logs/          # 开发容器的环境探测、下载尝试和测试日志
artifacts/     # gitignored：manifest、特征、checkpoints、metrics、figures、REPORT.md
```

数据含有害文本：日志只输出 ID、长度、标签和统计；`artifacts/<run>/local_checks/alignment_check.md`（≥20 条对齐人工抽查，含原文）仅留本地。
