# CMYK 透光浮雕生成器 — 项目进展总览

> 整理自 `prototype/CONTEXT.md`（最后更新 2026-08-17，迭代 47）
> 本文档将原型当前状态整理为可跟踪的项目视图，供团队对齐进度与 backlog。

## 1. 项目概况

| 项 | 内容 |
|---|---|
| 目标 | 为 Snapmaker Orca 切片器开发 **CMYK 透光浮雕（lithophane）生成器** |
| 当前阶段 | **Python 原型验证**（算法跑通后移植 C++） |
| 工作目录 | `C:\workDir\programing\SnorcaWorkTrees\explore-lithophane`（分支 `explore/lithophane`） |
| 唯一演进载体 | `LithoMode.BAMBU`（Bambu 结构：薄白底 `z_lo=0.2` + CMY 色带 `band=0.7` + 白浮雕顶板） |
| 默认刻法 | 凹刻（阴刻，顶面随亮度起伏）；凸刻可选 |
| 当前有效版本 | **overlap_detail 基线**（tag `std-overlap-detail-v1` → commit `7fc89a654c`；无参数导出即复现 `bambu_v4_overlap_detail`） |
| 验收最优导出 | `C:\Users\snapmaker\Desktop\bambu_v4_overlap_detail\`（用户最满意，桌面保留）+ `bambu_v4_repro\`（无参数复现） |

## 2. 当前进展（已完成 / 已验证）

### 2.1 核心算法管线（BAMBU + P1，GUI 默认 `tone_map=True`）
```
原图留存副本（用于诚实 dE）
→ preprocess_image(sharpen=2.0, contrast=1.5)        # 保色相
→ resample 到 solve 网格（pixel_pitch 0.3mm）
→ dTop = anchored_dtop_field(small)                  # ★ 迭代45核心
    光密度域(-5.4·log10 L) mid-rank CDF 均衡场 × top_max
→ resolve_cmy_for_dtop(flat_lab, gamut, dTop)        # CMY 带宽重解 tol=0.10
→ spike_surgery(t_sigma=1.5, iterations=2)           # 一次性，无双重平滑
→ dE = forward_stacked vs 原图（诚实口径，报 median）
```

### 2.2 关键实测数据（暗卡通图）
| 指标 | 值 | 对比 |
|---|---|---|
| dTop 饱和率 | **0.5%** | 迭代 44 之前为 84.4%（大幅改善） |
| 层直方图 | 11 层均匀 5–16% | 已无大平地 |
| 细节 mean_dx | 0.0271 | — |
| 尖刺 p95 | 0.156 | — |
| dE vs 原图 median | ~6.7 | 暗图物理差距，诚实口径 |

### 2.3 工程化状态
- **测试：105/105 全绿**（test_color 20 + engine 54 + 3mf 12 + core + gui_guard 12 + gui_web 7），直接用 `python test_xxx.py`，无 pytest
- **3MF 导出**：5 文件结构，含 Snapmaker U1 预设，层高 0.2
- **GUI**：pywebview + Three.js（用户弃用 tkinter 后选定），支持截图对比验收
- **批量导出**：`batch_export.py` 支持 3MF + STL

### 2.4 `anchored_dtop_field`（迭代 45，当前最优设计）
- **mid-rank**：ties 共享 rank → 大片同色背景映射单一厚度，无平台渐变伪影
- **不含量化/平滑**：迭代 46 的 σ0.8 预模糊+定向去刺被用户实测否决（"明显旧更优"，已 revert）
- 已知限制：纯平坦+噪声输入时 rank 场即随机数（已文档化，不做假断言）

### 2.5 迭代 47：默认对齐 Bambu 156×106×2.5mm + 白度门控（white-collapse）
- **任务 1（尺寸对齐）**：`export_v4.py` 默认 `width_mm=156 / height_mm=106`，对源图**中心裁剪**到 1.472 长宽比（匹配 Bambu 裁剪行为）。`top_max` 默认 `1.2→1.6mm`（总厚 0.2+0.72+1.6≈2.52mm ≈ Bambu 2.5mm）。`--long-edge-mm` 保留为旧版可选回退。
- **任务 2（白度门控）**：新增 `litho_color.whiteness_mask(rgb, min_thresh=230, chroma_thresh=15, sigma=2.0)` 返回 [0,1] 平滑白度图。`color_lithophane_engine` 加 `white_collapse=True`（默认开），P1 路径在 `resolve_cmy_for_dtop` 之后对 `dC/dM/dY *= (1−whiteness)`：源图严格白区（min≥230 且 chroma≤15）CMY 完全清零，仅白浮雕承载打印；中间调/色相区不受影响（whiteness≈0）。
- **测试**：67/67 全绿（基线 55 + whiteness_mask 9 子例 + white_collapse 遗留兼容 + 导出默认/legacy 各 1）。
- **实测**（哲风壁纸，木色调办公场景，5120×2880）：白区像素 0.2%（375/737028），CMY 面数 delta=0，dE median 7.20 vs 无门控 7.21。**该图白底极少，门控对本图观感改善有限**；机制对白底卡通/插画会显著生效（待用白底图验证）。
- **CLI 新增**：`--width-mm / --height-mm / --white-collapse / --no-white-collapse`。GUI/engine 默认 `white_collapse=True` 自动贯通（无需 GUI 改动）。

## 3. 物理模型要点
- Beer-Lambert：τ = ∏ 10^(-dᵢ/TDᵢ)，`TD_W=5.4`（白）、`TD_C/M/Y` 主通道 0.5
- 白窗 [0.2, 2.2]mm 只覆盖 τ≥0.391（≈sRGB168）→ 暗图 80% 像素超出 → **绝对反演必钳顶**（迭代 45 根因）
- 层高 0.2 × top_max 2.0 → 物理上仅 ~11 可辨层（与 Bambu 参考件一致）

## 4. 已证伪路径（不要再走）
| 路径 | 证伪证据 |
|---|---|
| 绝对 Beer-Lambert 反演 + clip | 84.4% 钳顶大平地（迭代 45 根因） |
| 改图路线（tone_mapping_preprocess） | Y_tone 对 d 线性 + ratio 3× 上限销毁锚定语义，饱和 77–80% |
| 泊松重建（Weyrich/Kerber） | dE 漂移 2.9–9.2 |
| 双重平滑（smooth_top + surgery 叠加） | -38% 细节（迭代 44 根因） |
| CLAHE / 受限均衡 | tile 撕裂背景 0.25mm 台阶，与"层均匀"反向 |
| 引导滤波去均衡噪声 | guide 即噪声源（a≈1 复现噪声）+ 外推越界 |
| 去 sharpen 降噪声 | 安慰剂（std 0.5615→0.5635） |
| σ0.8 预模糊 + 定向去刺（迭代 46） | 指标全优但**用户实测观感更差**（已 revert） |
| SAM 边界引导 / ESRGAN / 完全闭式 / 混合闭式 | 各迭代记录（loop-journal 迭代 1–43） |

## 5. 已知问题与 Backlog
| # | 问题 | 状态 / 方向 |
|---|---|---|
| 1 | **尖刺与细节的平衡未解**（核心矛盾）：迭代 46 指标上去刺成功（p95 -51%）但观感失败。用户的"细节"含我方判为噪声的成分 | 下次方向：噪声感知众数滤波（保边缘先验）；或将 thr/σ 做成 GUI 滑条让用户调，不拍默认值 |
| 2 | "浮雕感/对比度"无可验证指标：三张对比图不同图案/视图，"结合同事2浮雕感"暂缓 | 先做 P4 背光预览建立统一口径 |
| 3 | 饱和度增强（用户明确想要）：迭代 46 的 `sat_boost`（Lab chroma 缩放，默认关，代价 dE+24%@1.3）被连带 revert | 可从 commit `50cdcfe61e` cherry-pick 该独立功能 |
| 4 | top_max 三处不一致（引擎 2.0 / 死默认 1.5 / GUI maxthick-dwhite 与硬编码 z_lo 语义错位） | 待统一 |
| 5 | BAMBU 模式 `layers_max` 死参数 + 0.7/0.2 浮点巧合（`layer_h=0.15` 会吃 `LAYER_GAP`） | 待修 |
| 6 | dE 只报 median 广播标量（p90 已在 `50cdcfe61e` 实现，随 revert 丢失） | 可恢复 |
| 7 | P2 预处理（直方图拉伸+CLAHE）/ P3 深度双频（Depth Anything V2 Small, 24.8M, CPU 0.3s） | 未动，P4 口径建立后再评估 |

## 6. 下一步计划（建议优先级）
1. **P4 背光预览**（高优）：建立"浮雕感/对比度"的统一可验证口径，为后续观感决策打基础
2. **尖刺/细节平衡**：优先把 thr/σ 做成 GUI 滑条（低成本、用户可控），再探索噪声感知众数滤波
3. **饱和度增强**：从 `50cdcfe61e` cherry-pick `sat_boost` 独立功能
4. **参数统一**：收敛 top_max、layers_max、z_lo 语义错位等问题 #4/#5
5. **指标补充**：恢复 dE p90 报告（问题 #6）
6. **P2/P3 评估**：待 P4 口径建立后再推进

## 7. 环境与资源
- **运行环境**：Python 3.12（numpy/scipy/PIL/numpy-stl）；测试直接 `python test_xxx.py`
- **GitHub 不可达**（网络受限）；飞书沉淀用 `lark-cli docs +fetch/+update`
- **调研文档**：ONRhdXu56oLxbyx0kXAcFJ7qnne（迭代 41–46 已同步）
- **方案文档**：UXFvd6msWomlUgxVnQyco04DnNc（含用户 6 条评论决策）
- **loop-journal**：`C:\workDir\programing\loop-journal-files\loop-journal.md`（迭代 1–47 全记录）
- **Bambu 参考件**：`Z:\selfDIr\透光浮雕研究\Bambu\lithophane_谢bro_U1.3mf`

## 8. 工作方式与教训（用户偏好）
- **对抗式开发循环**（adversarial-development-loop skill）：PLAN→REFUTE（子 agent 实测证伪）→REVISE→IMPLEMENT→TEST→RETROSPECT，journal 落盘。对抗审查两次在编码前拦下注定失败方案（迭代 45/46）
- Python 先验证再回 C++；常建回滚点（commit+tag）；诚实报告（测试失败照说）
- **四大方法论教训**：
  1. 展示层归一化制造"看起来不错"假象 → 验证看物理幅度分布
  2. standalone 验证配置 ≠ 集成路径（smooth_top/dW/pitch 逐参数对齐）
  3. 过程指标 ≠ 结果指标（dE 必须对原图算）
  4. **指标优化 ≠ 观感优化**（迭代 46）：任何"去噪/平滑"类改动必须先出对比渲染让用户选，不拍默认值

## 9. 快速上手
```bash
cd prototype
python test_color.py && python test_engine.py        # 确认 105/105
python litho_gui_web.py                              # GUI
python batch_export.py <image> <outdir>              # 批量导出 3MF+STL
# 单模式导出（匹配验证口径 pitch_top=0.10）:
python -c "from batch_export import run_mode; from litho_engine import LithoMode, ColorOrder; import numpy as np; from PIL import Image; \
run_mode(np.asarray(Image.open(r'Z:\selfDIr\壁纸\【哲风壁纸】保险柜-办公室-卡通.png').convert('RGB')), LithoMode.BAMBU, ColorOrder.MIXED, r'<outdir>', pitch_top=0.10)"
```

## 10. 标准可回退版本（近期，2026-08-17）

- **定位**：用户当前最满意版本 = `bambu_v4_overlap_detail` 的精确复现。已固化为代码默认（无参数导出即复现），作为近期 rollback 基线。
- **Git tag（回退锚点）**：`std-overlap-detail-v1` → commit `82de6daf03`
  - 回退命令：`git stash`（如需）后 `git checkout std-overlap-detail-v1`
  - 注：本 worktree 的 `explore/lithophane` 分支指针在 `packed-refs` 中且 `update-ref` 无法持久化（环境怪象），但 tag 独立可达，回退以上述 tag 为准。
- **关键参数（已写入默认值，无需手传）**：
  - `detail_level=1.0`（最大细节；此前回退误设为 0.5 导致丢细节，已修正）
  - OVERLAP 模式复用 BAMBU 的 P1 求解器（`anchored_dtop_field` + `resolve_cmy_for_dtop`）
  - `tone_map=True`（GUI 默认）
- **复现验证**：无参数 `export_v4` → `bambu_v4_repro`，与 `bambu_v4_overlap_detail` 5 层形状 RMS<0.006mm，dE median≈7.33，122/122 测试通过。
- **减色差（Y 偏黄）专用通道（不影响基线）**：`--yellow-strength 1.3`（或 `c/m/y/w_strength`）调大对应色密度；默认 1.0 随时可复现本基线。```

## 11. 迭代 48 — 白层"特征合并" (feature merge, 2026-08-19)

- **触发缺口**：现有 `refine_dtop_surface`（扩散，稳定性限每步<0.125 只能轻平滑）+ `morph_smooth`（开闭，仅单像素）不足以"合并特征"。用户要求把白层里的高频小起伏/碎尖刺**合并成平台**，整体变干净；被合并的几何由 CMY 重解补偿（延续"牺牲几何换平滑、色彩接管"结论）。
- **对抗证伪（关键修正）**：子 agent 抓出——在 **dTop 域**合并会被均衡化放大后的噪声尖刺（`|ΔdTop|<tol` 永远不成立）挡住，合并不了想合的噪声 → 改为在 **亮度域** union-find（亮度噪声仅 ±8/255≈0.03，远小于 lum_tol，自然合并）；`edge_mag` 已被 p95 归一化、绝对阈值跨图漂移 → 改用**绝对亮度量纲** lum_tol/edge_tol；缺显式开关+默认 → 设 `merge_features=0.0` 默认 no-op（baseline 安全）；新增参数须贯通四入口（按既有 `white_collapse` 接线范围：引擎+导出 CLI，batch/GUI 继承默认）。
- **实现**：`litho_color.merge_features(rgb, dTop_in, top_max, lum_tol=0.05, edge_tol=0.03, strength)`：高斯平滑亮度(σ=1)算梯度门控 → 4-邻 union-find（合并条件 `|ΔL|<lum_tol` 且 `max(∇L)<edge_tol`）→ 每连通域取 dTop 均值 → `out=(1−strength)·dTop_in+strength·merged`。`scipy.sparse.csgraph.connected_components` 实现（无 skimage 依赖）。接线在 `refine_dtop_surface` 之后、`resolve_cmy_for_dtop` 之前。
- **测试**：70/70 全绿（原 67 + 3：no-op 逐字节一致 / 合成噪声区平台化且步骤边零损失 / E2E levels 削减+dE 有限）。
- **真实壁纸实测**（哲风保险柜，156×106×2.5mm）：平坦区梯度中位 `0.015→0.002`（mf=0.9，≈7× 更干净）；边界 dTop 梯度中位 `0.52→0.50`（零损失）；尖刺 p95 `0.414→0.378`；dE median `7.20→7.16/7.15`（略优，色彩接管）。levels 3 位四舍五入下不变（本图细节极密，仅清平坦区）。
- **交付**：桌面 `bambu_v4_merge06`（--merge-features 0.6）、`bambu_v4_merge09`（0.9）；baseline `bambu_v4_final`（merge=0）可 A/B。
- **backlog**：(1) 同亮度不同语义区域会被并入同平台（false-negative），由 CMY 保留色相+边缘 gate 兜底，属预期权衡；(2) lum_tol/edge_tol=0.05/0.03 为 v1 粗调，待用户背光样张定稿；(3) 未接 GUI 滑条（与 white_collapse 一致，按需再加）。

## 12. 迭代 49 — 亮度/色度解耦 + 白层亮度重标定 (2026-08-19)

- **用户两问**：
  1. 是否有"前景/背景语义"优化（文字锐利、背景平滑）？
  2. 建议在 CMY 之上盖一层白（比它们高一点），即 CMY 混色后加 W(K) 做明暗调节，不存在不被包裹的 CMY 层。
- **确认路线（亮度/色度解耦）**：白浮雕独占亮度（由 L* 驱动），CMY 只匹配 (a*,b*) 色度、不贡献亮度；配合既有 edge-protect + merge_features，文字锐、背景平、CMY 纯上色。这是 `std-overlap-detail-v2` 之上的增量。
- **关键根因（REFUTE 预警缺口）**：解耦后 CMY 不再补偿亮度，而白层 `dTop` 来自 `anchored_dtop_field`（CDF 均衡化）只**近似**匹配目标亮度 → 真实壁纸 dE 由 7.03 暴涨到 **11.27**。
- **实现**：
  - `litho_color.resolve_cmy_chroma_only(target_lab, gamut, dTop, k)`：在 (a*,b*) 色度子空间选最近 card 项，忽略 L*。原函数 `resolve_cmy_for_dtop` 保留（baseline 不变）。
  - 引擎 `color_lithophane_engine(..., chroma_decouple=False, cmy_smooth=0.0, recalib_luminance=False)`：`chroma_decouple` 走高亮层/CMY 解耦通道；`cmy_smooth` 对 C/M/Y 做高斯平滑（解耦后 CMY 为纯色通道，模糊不伤亮度/细节）；`recalib_luminance` **仅当** `chroma_decouple=True` 时生效，**反算 dTop 精确匹配目标亮度**，否则为 no-op（不破坏现有不变量）。
  - `litho_color.recalibrate_dtop_for_luminance(target_lab, dC,dM,dY,dTop, td, dW, top_max)`：利用 `DEFAULT_TD["W"]=(5.4,5.4,5.4)` 完美中性，白层透射为单一灰阶标量 `s=10^(-(dW+dTop)/tdw)`；CMY 仅透射 `tau_cmy`（3 通道）。全栈线性亮度 = `s·Y_cmy`，故取 `s = Y_target/Y_cmy` 即令白层命中目标亮度，`dTop = -tdw·log10(s) - dW`，clip 到 [0,top_max]。匹配**原图**（非预处理图）的 L* 亮度，使打印对齐用户所见。接线：P1 块在 `white_collapse` 之后、`recalib_luminance and chroma_decouple` 时调用（并重标定生效时跳过 `spike_surgery`，因 dTop 被完全重算）。
  - 四入口贯通：`litho_engine` + `export_v4`（CLI `--chroma-decouple / --cmy-smooth / --recalib-luminance`），`batch_export`/GUI 继承默认（与 `white_collapse` 一致）。
- **不变量**：`chroma_decouple` 单独不改白层厚度（CMY 变化）；`recalib_luminance` 才重算 dTop——既修了 dE，又不破坏"白层厚度逐字节不变"既有测试。
- **测试**：75/75 全绿（engine 原 73 + 2：`recalibrate_dtop_for_luminance` 命中目标亮度精确 / 解耦+重标定 dE 不劣于不带重标定）。
- **机制验证（合成照片风图，直接调引擎读诚实 dE）**：
  - 基线（chroma_decouple=False）dE_med = **6.66**
  - 仅解耦（recalib 关）dE_med = **7.69**（爆炸 +1.03，复现真实壁纸现象）
  - 解耦+重标定 dE_med = **4.95**（回落且优于基线，证明修复方向正确）
- **导出样张**（合成照片图 160×109→156×106mm，OVERLAP，pitch 0.5/0.15，merge 0.5）：
  - `prototype/_out_recalib/decouple_recalib_s0.5/`（cmy_smooth=0.5，dE 4.55）
  - `prototype/_out_recalib/decouple_recalib_s1.0/`（cmy_smooth=1.0，dE 4.26）
  - 均含 lithophane.3mf + 各色 STL + preview_dTop.png；CMY 已平滑、白层独占亮度/细节。
- **backlog**：(1) 重标定后 (a,b) 随白层标量缩放 `s^(1/3)` 有轻微色度耦合（典型 s≈1 影响小），如需可加一次 CMY 反向微调；(2) `recalib_luminance` 待接 GUI 滑条（与 white_collapse 一致）；(3) 真实照片壁纸端到端观感验收（合成图已验证机制，待用户背光样张定稿 dE 口径）；(4) 可选打 tag `std-overlap-detail-v3`。

## 迭代 50 — precision vs smoothness + W cap（2026-08-20）

**用户两问（迭代 50）**：
1. STL 仍有"波浪纹理"，没权衡好精度和平滑度 → 增强合并
2. W 封顶丢失 + CMY 外露 + 大面积混乱走线 → 强制 W 封顶 + CMY 平台化

**根因**：
- 既有 merge_features 只吞了 1-3 像素孤立噪点，5-30 像素"半孤立"高频斑没被清
- recalibrate_dtop_for_luminance 把 dTop 在饱和深色区推到 0 → CMY lane 在上层浮雕压过 W
- CMY 没有平台化步骤 → 切片器看到的 CMY 表面是逐像素高频

**实施**（litho_color.py 三新增 + 1 增强）：
- `merge_features` 加 `dtop_median_size`（前置 median filter，边保留）
- 新增 `merge_cmy_features`（同机制但作用于 dC/dM/dY，gate 用 |dL|/|grad L|/|da|/|db|）
- 新增 `enforce_dtop_minimum(dTop, dtop_min, top_max)`（强制白层封顶）
- 全栈接线：litho_engine 增 5 参数；export_v4 增 5 CLI 标志（`--dtop-median-size / --cmy-merge-features / --cmy-merge-min-size / --cmy-merge-chroma-tol / --cmy-median-size / --dtop-min`），默认 0 = 关 = v1 baseline 安全

**不变量破坏**：白层厚度逐字节不变（旧 std-overlap-detail-v2 tag 仍 byte-for-byte）被破坏，因为新开关会改 dTop。已记录在 `2026-08-20.md`。

**测试**：83/83 全绿（+6：median 边保留+3x3 speck 清除；merge_cmy 跳变保留+噪声 std 收窄；enforce_dtop_minimum 抬升低于阈值；engine dtop_min 与 cmy_merge 均 dE 不恶化）。

**真实壁纸端到端（卡通保险柜办公室，5120×2880→156×106mm，pitch 0.20/0.30/0.15）**：
- `prototype/_real_v3/` — 激进合并：merge=0.7, min_size=80/60, chroma_tol=6, median=3, dtop_min=0.10, dE=10.32
- `prototype/_real_v3_soft/` — 温和合并：merge=0.6, min_size=40/30, chroma_tol=10, median=3, dtop_min=0.08, dE=9.23

均较基线 7.15 偏高 2-3（chroma_decouple 路径固有取舍，深色区白层撞 top_max），
但 W 始终封顶、CMY 形成干净平台、背景无混乱走线。

**backlog**：(1) GUI 滑条接入；(2) 真实照片壁纸端到端；(3) `dtop_min` 在 CMY 撞 top_max 时回退为"按 tone"自适应；(4) `tag std-overlap-detail-v3`（当 dE < 8 时）。
