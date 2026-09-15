现在开始一个新的独立实验阶段：

# Phase 3B — PatchCore Top-1 Evidence → Qwen Anomaly Detection

本阶段只验证一个问题：

> 在 MVTec-AD 上，给 Qwen3-VL-4B 提供由 PatchCore 生成的 Top-1 suspicious patch，是否能够提升异常检测判断率？

不要实现完整 Agent。

不要实现 Planner、Reflector、Critic、Normal Reference 或多 patch reasoning。

==================================================
一、实验目标
======

比较以下两种方法：

A. Direct Qwen baseline

输入：

* query image
* anomaly detection question

输出：

* normal / anomalous

B. Qwen + PatchCore Top-1 Evidence

输入：

* query image
* PatchCore Top-1 candidate patch

输出：

* normal / anomalous

核心比较：

Direct Qwen
vs
Qwen + PatchCore Top-1 Patch

必须使用完全相同的 query set、Qwen 模型、generation configuration 和 evaluation protocol。

==================================================
二、数据范围
======

本阶段只运行两个 MVTec-AD category：

1. bottle
2. cable

暂时不要运行其他 category。

必须使用当前项目已有的 MVTec-AD 数据加载方式。

不得重新设计 dataset loader。

==================================================
三、PatchCore 配置
==============

直接复用 Phase 1 / Phase 2 已完成的 PatchCore memory。

每个 category：

* 24 张 normal training images
* 使用 Phase 1 已保存的 memory/cache
* 不重新 fit PatchCore
* 不重新抽样 normal memory

必须确认 query test image 不在 PatchCore memory 中。

==================================================
四、PatchCore candidate 配置
========================

只使用：

max_regions = 1

即：

Top-1 candidate only。

不要使用 Top-3。

必须复用 Phase 2 已经生成的 candidate region / crop。

不要重新设计：

* anomaly map threshold
* connected components
* bbox extraction
* crop generation

当前固定：

threshold = 0.5
min_area_ratio = 0.001
max_regions = 1

如果 Phase 2 文件中已经存在 Top-1 crop，优先直接读取。

不要重新运行 expensive PatchCore inference。

==================================================
五、实验输入模式
========

本阶段只测试两个模式。

---

## Mode A: Direct

Qwen 输入：

1. query image
2. question

不要提供任何 PatchCore 信息。

Prompt 中禁止出现：

* anomaly score
* anomaly map
* bbox
* candidate
* PatchCore
* region score

---

## Mode B: Global + Top-1 Patch

Qwen 输入：

1. original query image
2. Top-1 patch image
3. anomaly detection question

重要：

* 原始 query image 必须始终保留。
* patch 是辅助 visual evidence。
* patch 不是最终答案。
* PatchCore 不提供最终 anomaly label。
* 不向 Qwen 提供 PatchCore numerical score。
* 不向 Qwen 提供 bbox。
* 不向 Qwen 提供 rank。
* 不向 Qwen 提供 "this is suspicious" 之类的结论性文字。

只提供：

原图 + patch。

这样本实验只测量：

> additional local visual evidence 的价值。

而不是测量 numerical anomaly score 的价值。

==================================================
六、Qwen Prompt
=============

Mode A 使用现有 Direct baseline prompt。

不要修改 Direct baseline prompt，除非为了统一输出格式。

Mode B 使用下面的 prompt：

你是一个工业视觉异常检测模型。

你的任务是判断给定的查询图像中是否存在异常。

你将看到：

1. 原始查询图像；
2. 一个从查询图像中裁剪得到的局部区域。

局部区域只是额外的视觉观察证据，不代表该区域一定存在缺陷。

重要规则：

* 原始图像是主要视觉证据。
* 局部区域只是辅助证据。
* 不要假设局部区域一定异常。
* 不要因为看到局部纹理差异就直接判断为缺陷。
* 需要结合原始图像中的完整物体、结构和上下文理解局部区域。
* 如果局部区域与原图上下文产生冲突，以真实视觉内容为准。
* 不要编造图像中不存在的缺陷。
* 小划痕、纹理变化、光照变化、视角变化等不能自动视为异常。
* 对明显的裂纹、破损、划痕、缺失、污染、变形等真实缺陷给予更高权重。

输出只能是合法 JSON：

{
"result": "normal | anomalous",
"reason": "简洁说明最终判断依据"
}

不要输出其他字段。
不要输出 markdown。
不要输出 bbox。
不要输出坐标。

==================================================
七、重要：保持与 Direct Baseline 的公平性
=============================

Direct 与 Global+Patch 必须：

* 使用相同 Qwen3-VL-4B 模型；
* 相同 temperature；
* 相同 do_sample；
* 相同 max_new_tokens；
* 相同 system prompt（除输入证据差异外）；
* 相同 query image；
* 相同 evaluation set。

如果 Direct baseline 已经有固定配置，不要擅自修改。

如果当前项目已有 JSON recovery：
继续使用同一套 recovery。

不要为了 Patch 模式单独增加特殊恢复逻辑。

==================================================
八、实验输出
======

建立：

results/phase3b/

目录：

results/phase3b/
├── direct/
├── global_patch/
├── summary.json
├── summary.csv
└── per_image_comparison.json

每个 query 必须保存：

{
"dataset": "MVTec-AD",
"category": "bottle",
"query_image": "...",
"mode": "direct | global_patch",
"ground_truth": "normal | anomalous",
"prediction": "normal | anomalous",
"correct": true,
"raw_output": "...",
"parsed_output": {...},
"parse_ok": true,
"vlm_calls": 1,
"elapsed_s": ...
}

Global+Patch 额外记录：

{
"patch_path": "...",
"patch_bbox": [...],
"patch_rank": 1
}

注意：

patch_bbox 可以写入实验 trace，
但是不能发送给 Qwen。

==================================================
九、必须做的 paired comparison
========================

必须对完全相同的 query image：

Direct
vs
Global+Patch

逐样本进行配对比较。

统计：

1. Direct accuracy
2. Global+Patch accuracy
3. accuracy difference
4. Direct correct → Patch wrong
5. Direct wrong → Patch correct
6. both correct
7. both wrong

另外分别报告：

* bottle
* cable
* macro average

==================================================
十、异常检测指标
========

由于这是二分类 anomaly detection，至少报告：

Accuracy
Precision
Recall
Specificity
F1

同时保存：

TP
TN
FP
FN

不要只报告 accuracy。

如果当前项目已有统一 evaluator，优先复用。

==================================================
十一、正常图与缺陷图分开分析
==============

必须分别报告：

Normal test images：

* accuracy
* FP rate

Defective test images：

* recall
* FN rate

这样可以回答：

PatchCore local evidence 是：

1. 帮助发现缺陷？
   还是
2. 增加正常图误报？

不能只看 overall accuracy。

==================================================
十二、PatchCore candidate quality 关联分析
===================================

Phase 2 已经有：

* anomaly coverage
* IoU
* candidate area ratio

Phase 3B 只做分析，不修改 candidate。

对于 Global+Patch：

统计：

1. Top-1 patch area ratio
2. patch 是否来自 defective image
3. 如果对应 GT 可用于离线分析，记录：

   * patch coverage
   * patch IoU

然后分析：

Patch correct
vs
Patch wrong

是否与 candidate quality 有关系。

注意：

GT 只能用于分析，不能进入 Qwen prompt，也不能影响 prediction。

==================================================
十三、特别重要：处理“正常图也有 patch”
=======================

Phase 2.5 已经证明：

正常图也会产生 candidate。

因此：

不要过滤掉正常图的 patch。

不要根据 ground truth 删除 patch。

不要给 Qwen 提示：

“这是可疑区域”。

正常图产生的 Top-1 patch 也必须原样送给 Qwen。

这样才能真实测试：

> Qwen 能否识别一个 externally selected patch 是正常还是异常。

==================================================
十四、不要使用 PatchCore score
=======================

本阶段明确禁止将：

image_score
region_score
anomaly score

放进 Qwen prompt。

原因：

本实验只想验证：

> PatchCore localization → visual evidence

而不是：

> numerical anomaly score → classification prior

后续可以单独做 score ablation。

==================================================
十五、不要做 Normal Reference
=======================

本阶段明确禁止：

* normal reference image
* CLIP retrieval
* reference comparison
* normal memory retrieval for Qwen

虽然 PatchCore memory 使用 normal training images，
但 Qwen 只能看到：

query image + Top-1 crop

不能看到 PatchCore memory images。

==================================================
十六、不要实现 Planner
===============

本阶段：

PatchCore candidate 直接提供给 Qwen。

不要：

* tool selection
* planner prompt
* adaptive routing
* evidence demand estimation

因为当前阶段只需要验证：

> 固定加入 Top-1 suspicious patch 是否带来增益。

==================================================
十七、实验规模
=======

首先只跑：

bottle
cable

完整 test images。

不要只跑 defective images。

必须同时包含：

* good
* all defect types

保持完整测试分布。

==================================================
十八、先运行 smoke test
=================

在 full run 前：

每个 category 至少选择：

* 2 normal images
* 2 defective images

同时运行：

Direct
Global+Patch

检查：

1. 两种模式使用同一 query image。
2. Global+Patch 中 patch 文件存在。
3. patch 来自原 query image。
4. patch 是 Top-1。
5. PatchCore score 不出现在 prompt。
6. bbox 不出现在 prompt。
7. Qwen 原始输出可追踪。
8. JSON schema 合法。
9. prediction 可解析。

smoke test 通过后才能 full run。

==================================================
十九、停止条件
=======

Phase 3B 只有满足以下条件才算完成：

1. bottle full test 完成；
2. cable full test 完成；
3. Direct 与 Global+Patch 使用完全相同的 query set；
4. 所有 query 都有对应结果；
5. 两种模式 parse success 可统计；
6. 没有 Qwen retry 或 retry 数量明确记录；
7. PatchCore score 没有进入 prompt；
8. bbox 没有进入 prompt；
9. 没有 GT leakage；
10. normal images 没有被过滤；
11. paired comparison 完成；
12. Accuracy / Precision / Recall / Specificity / F1 完成；
13. TP/TN/FP/FN 完成；
14. results/phase3b 独立保存；
15. 不覆盖 Phase 2 / Phase 3 原始结果。

==================================================
二十、完成后立即停止
==========

完成 Phase 3B 后不要：

* 做 Normal Reference；
* 做 Planner；
* 做 Reflector；
* 做 Critic；
* 修改 PatchCore；
* 扩展到 LOCO；
* 扩展到其他 MVTec categories。

只报告：

1. 修改/新增文件；
2. smoke test；
3. bottle Direct vs Global+Patch；
4. cable Direct vs Global+Patch；
5. macro result；
6. paired comparison；
7. normal vs defective breakdown；
8. patch quality 与 prediction 的初步关系；
9. 是否发现任何 pipeline 问题；
10. 下一阶段建议。

停止。
