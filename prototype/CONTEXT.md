# CMYK 透光浮雕生成器 — 交接上下文（CONTEXT）

> 最后更新：2026-08-20（迭代 50e：亮度/色度解耦 + 白层亮度重标定 + 平台合并 + 高光保护 + 高度量化 + CMY 覆盖 margin + P1 morph 平滑）
> 本文档为跨会话交接的自包含上下文。新会话请先读本文，再按需读 PROJECT_STATUS.md 与 loop-journal。

> **⚠ 2026-08-20 回退**：工作树已回退至 `std-overlap-detail-v1` 基线（tag → 82de6daf03，commit 21b8cf57e8；`litho_color.py`/`litho_engine.py`/`export_v4.py`/`test_engine.py` 与之逐字节一致）。**迭代 48–50 机制在 commit `19fe42d0a1`（HEAD）；50c/50d/50e 增量（highlight_protect / quantize_dtop / dtop_cmy_cover_margin / P1 morph）未提交**，仅存于 `git stash`（stash@{0}，"iteration 50e ... before rollback to v1"）与完整文件备份 `prototype/_rollback_backup_50e/`。恢复 50e：用备份目录或 `git stash apply`；**勿用 `git checkout 19fe42d0a1 -- prototype/`（会丢 50c/d/e）**。50e 状态全套 143 项测试通过（test_engine 86/86）。是否让 50e 开关回归默认，待背光样张 A/B 验收（Backlog A0）。

## 1. 项目定位

为 Snapmaker Orca 切片器开发 **CMYK 透光浮雕（lithophane）生成器**。当前为 **Python 原型阶段**（`prototype/`），验证算法后移植 C++。工作目录：`C:\workDir\programing\SnorcaWorkTrees\explore-lithophane`（分支 `explore/lithophane`）。

**唯一演进载体**：`LithoMode.BAMBU`（Bambu-like 结构：薄白底 `z_lo=0.2` + CMY 色带 `band=0.7` + 白浮雕顶板）。凹刻（阴刻，顶面随亮度起伏）为默认，凸刻可选。

## 2. 最终有效状态（迭代 50e）

当前 HEAD 已集成亮度/色度解耦与后续平滑加固。**推荐导出**：`prototype/_real_v6_balanced/`（细节与平滑均衡）或 `prototype/_real_v6_solid/`（最实心、尖刺最少）。

### BAMBU+P1 管线（tone_map=True，GUI 默认开）

```
原图留存副本（用于诚实 dE）
→ preprocess_image(sharpen=2.0, contrast=1.5)               # 保色相
→ resample 到 solve 网格（pixel_pitch 0.3mm）
→ dTop = anchored_dtop_field(small)                         # 迭代 45 核心
    光密度域(-5.4·log10 L) mid-rank CDF 均衡场 × top_max
→ [可选] merge_features(dTop, rgb, strength, min_size,      # 迭代 48/50
                        dtop_median_size, highlight_protect)
→ [可选] morph_smooth(dTop, detail_level)                   # 迭代 50e
→ [可选] quantize_dtop(dTop, step)                          # 迭代 50d
→ [可选] enforce_dtop_minimum(dTop, dtop_min)               # 迭代 50
→ [可选] dtop_cmy_cover_margin 强制 W 高于 CMY              # 迭代 50e
→ [chroma_decouple=True] recalibrate_dtop_for_luminance     # 迭代 49
→ [chroma_decouple=True] resolve_cmy_chroma_only            # 迭代 49
   否则 resolve_cmy_for_dtop
→ [可选] merge_cmy_features + cmy_smooth                    # 迭代 50
→ dE = forward_stacked vs 原图（诚实口径，报 median）
```

以上带 `[可选]` 的步骤默认关闭（`strength=0`、`size=0`、`step=0`、`min=0`、`margin=0`、`chroma_decouple=False`），关闭时逐字节等价于迭代 47/48 基线，可作为安全回退。

### 真实壁纸关键实测数据（哲风保险柜-办公室-卡通，5120×2880→156×106mm）

| 版本 | 关键参数 | dE median |
|---|---|---|
| `_real_baseline/` | 无解耦 | 7.15 |
| `_real_decouple_v2/` | 解耦+重标定 | 7.43 |
| `_real_v4_highlight/` | + highlight_protect 0.08, merge 0.6 | 9.12 |
| `_real_v5_quantize/` | + quantize_step 0.12 | 9.23 |
| `_real_v6_balanced/` | detail-level=0.5, cmy-cover=0.03 | 10.55 |
| `_real_v6_solid/` | detail-level=0.3, quantize=0.20, cmy-cover=0.08 | 11.18 |

> 注：dE 在 v6 上升是“平滑/实心”与“色差”之间的显式权衡；视觉上尖刺与碎平面显著减少，W 封顶完整覆盖 CMY。

### 合成照片验证（chroma_decouple+recalib 机制）

| 配置 | dE median |
|---|---|
| 基线（无解耦） | 6.66 |
| 仅解耦（无重标定） | 7.69（爆炸，复现问题） |
| 解耦+重标定 | 4.95（优于基线） |

## 3. 核心文件与算法

| 文件 | 职责 |
|---|---|
| `litho_color.py` | 色卡 `build_gamut_stacked`、求解 `solve_stacked`/`resolve_cmy_for_dtop`/`resolve_cmy_chroma_only`、**`anchored_dtop_field`**、**`recalibrate_dtop_for_luminance`**、**`merge_features`**（含 `highlight_protect`）、**`merge_cmy_features`**、**`quantize_dtop`**、**`enforce_dtop_minimum`**、Beer-Lambert 工具链 |
| `litho_engine.py` | `color_lithophane_engine` 入口；BAMBU 分支约 247–380 行为 P1 管线；新增解耦/重标定/合并/量化/覆盖 margin 开关 |
| `litho_core.py` | `LithophaneParams`、网格、网格转 mesh |
| `litho_3mf.py` / `batch_export.py` | 3MF（5 文件结构，含 Snapmaker U1 预设，层高 0.2）/ 批量导出 |
| `litho_gui_web.py` + `litho_web/` | pywebview+Three.js GUI（用户选定）|
| `export_v4.py` | CLI 主入口，已接入所有新参数 |
| `test_*.py` | 50e 状态**全套 143 项通过**（color 20 / engine 86 / 3mf 12 / gui_guard 12 / gui_web 7 / reverse 6 + core），直接 `python test_xxx.py`（无 pytest） |

### 物理模型

- Beer-Lambert：`τ = ∏ 10^(-dᵢ/TDᵢ)`，`TD_W=5.4`（白）、`TD_C/M/Y` 主通道 0.5
- 白层中性 `DEFAULT_TD["W"]=(5.4,5.4,5.4)` 使白层透射为单一标量，支持 `recalibrate_dtop_for_luminance` 反算：`dTop = -tdw·log10(Y_target/Y_cmy) - dW`
- 白窗 `[0.2, 2.2]mm` 只覆盖 `τ≥0.391`（≈sRGB168）→ 暗图 80% 像素超出 → **绝对反演必钳顶**（迭代 45 根因）

### 关键新机制

1. **`recalibrate_dtop_for_luminance`**（迭代 49）：白层独占亮度，反算 `dTop` 精确匹配原图 `L*`，把“仅解耦”爆炸的 dE 拉回并优于基线。
2. **`merge_features`**（迭代 48/50）：亮度域 + 梯度门限的连通域合并，吞孤立小碎块；`min_size` 吸收半孤立斑；`dtop_median_size` 前置中值滤波保边；`highlight_protect` 豁免局部高亮小斑（硬币/反光）。
3. **`merge_cmy_features`**（迭代 50）：对 `dC/dM/dY` 做亮度+色度联合门限合并，把 CMY 切片高频平台化。
4. **`quantize_dtop`**（迭代 50d）：按 `step` 取整高度，形成梯田，扩大 W 层单层连续面积，减少切片等高线碎路径。
5. **`dtop_cmy_cover_margin`**（迭代 50e）：`dTop = max(dTop, max(dC,dM,dY) + margin)`，强制白层封顶高于 CMY z 高度。
6. **P1 `morph_smooth`**（迭代 50e）：按 `detail_level` 计算半径，去除合并后边缘锯齿尖刺。

## 4. 证伪清单（不要再走这些路）

| 路径 | 证伪证据 |
|---|---|
| 绝对 Beer-Lambert 反演 + clip | 84.4% 钳顶大平地（迭代 45 根因） |
| 改图路线（tone_mapping_preprocess） | Y_tone 对 d 线性 + ratio 3× 上限销毁锚定语义，饱和 77–80% |
| 泊松重建（Weyrich/Kerber） | dE 漂移 2.9–9.2 |
| 双重平滑（solve smooth_top=True + surgery 叠加） | -38% 细节（迭代 44 根因） |
| CLAHE/受限均衡 | tile 撕裂背景 0.25mm 台阶；与“层均匀”同轴反向 |
| 引导滤波去均衡噪声 | guide 即噪声源（a≈1 复现噪声）+ 外推越界 |
| 去 sharpen 降噪声 | 安慰剂（std 0.5615→0.5635） |
| σ0.8 预模糊 + 定向去刺（迭代 46） | 指标全优但**用户实测观感更差**（v0.47 已 revert） |
| 色度解耦不配亮度重标定 | 真实壁纸 dE 7.15→11.27，合成图 6.66→7.69（迭代 49 根因） |
| SAM 边界引导 / ESRGAN / 完全闭式 / 混合闭式 | 各迭代记录（见 loop-journal 迭代 1–43） |

## 5. 已知问题与 backlog

1. **dE 与平滑度的权衡**：v6 为获取实心/少尖刺付出 dE 上升代价（10.55–11.18）。下一步需真实背光样张定夺最佳参数组合，或把 `merge_strength`/`min_size`/`quantize_step`/`detail_level` 做成 GUI 滑条让用户实时 A/B。
2. **GUI 滑条未接**：`chroma_decouple`、`recalib_luminance`、`merge_features`、`merge_min_size`、`dtop_median_size`、`cmy_merge_features`、`cmy_merge_min_size`、`cmy_merge_chroma_tol`、`cmy_median_size`、`dtop_min`、`dtop_quantize_step`、`dtop_cmy_cover_margin`、`highlight_protect`、`detail_level` 仅在 CLI/引擎接口可用。
3. **真实照片壁纸验证**：当前验证以卡通暗图为主，照片风图（合成图已验证机制）待端到端导出+背光验收。
4. **`dtop_min` 撞 `top_max` 回退**：当 CMY 深色区要求白层封顶 > top_max 时，当前直接 clip，可改进为“按 tone 自适应回退”。
5. **tag `std-overlap-detail-v3`**：当某组参数真实壁纸 dE < 8 且观感验收通过时打 tag。
6. `top_max` 三处不一致（引擎 2.0 / 死默认 1.5 / GUI maxthick-dwhite 与硬编码 z_lo 语义错位）待统一。
7. BAMBU 模式 `layers_max` 死参数 + 0.7/0.2 浮点巧合（`layer_h=0.15` 会吃 `LAYER_GAP`）。
8. dE 只报 median 广播标量（p90 曾在 `50cdcfe61e` 实现，随 revert 丢失）。
9. P2 预处理（直方图拉伸+CLAHE）/ P3 深度双频（Depth Anything V2 Small, 24.8M, CPU 0.3s）：未动，P4 口径建立后再评估。

## 6. 环境与工具

- Python 3.12（numpy/scipy/PIL/numpy-stl）；测试直接 `python test_xxx.py`（无 pytest）
- **GitHub 不可达**（网络受限）；飞书沉淀用 `lark-cli docs +fetch/+update`（先读 `lark-cli skills read lark-doc ...`）
- 调研文档：ONRhdXu56oLxbyx0kXAcFJ7qnne（实施进展同步至迭代 45；46+ 演进见技术方案总册）
- 方案文档：UXFvd6msWomlUgxVnQyco04DnNc（含用户评论决策）
- **技术方案总册（2026-08-20 新建）**：https://snapmaker.feishu.cn/wiki/AqyDwwoTbiN9kokBFWcc8UNznhh （全部在用算法 + 演进/证伪史 + 参数速查 + §9 效果粘贴区）
- **Backlog 跟踪表（2026-08-20 新建）**：https://snapmaker.feishu.cn/wiki/SZKnwp5kki4MVQkjHoNcuzw0nob
- 计划文档（已更新 5.6/排期/关联文档）：https://snapmaker.feishu.cn/wiki/Gkp7wvXpxisSlJkpB8oczxaHn1e
- loop-journal：`C:\workDir\programing\loop-journal-files\loop-journal.md`（迭代 1–50e 全记录）
- 本日记忆：`C:\workDir\programing\SnorcaWorkTrees\explore-lithophane\.workbuddy\memory\2026-08-20.md`
- Bambu 参考件：`Z:\selfDIr\透光浮雕研究\Bambu\lithophane_谢bro_U1.3mf`
- 图像 MCP（analyze_image）对本地图片不稳定；WinRT OCR（PowerShell）可识别文字但渲染截图无文字；Python 定量最可靠

## 7. 工作方式与教训（用户偏好）

- **对抗式开发循环**（adversarial-development-loop skill）：PLAN→REFUTE（子 agent 实测证伪）→REVISE→IMPLEMENT→TEST→RETROSPECT，journal 落盘。
- Python 先验证再回 C++；时常建回滚点（commit+tag）；诚实报告（测试失败照说）
- **核心方法论教训**：
  1. 展示层归一化制造“看起来不错”假象 → 验证看物理幅度分布
  2. standalone 验证配置 ≠ 集成路径（smooth_top/dW/pitch 逐参数对齐）
  3. 过程指标 ≠ 结果指标（dE 必须对原图算）
  4. 指标优化 ≠ 观感优化——任何“去噪/平滑”类改动必须先出对比渲染让用户选，不拍默认值
  5. 解耦必须配重标定——“让 CMY 只上色”不解决亮度守恒，会爆炸 dE

## 8. 快速上手

```bash
cd prototype
python test_color.py && python test_engine.py        # 确认 86/86
python litho_gui_web.py                              # GUI（新参数默认未接滑条）
python batch_export.py <image> <outdir>              # 批量导出 3MF+STL

# 推荐真实壁纸参数（_real_v6_balanced 复现）：
python export_v4.py \
  "Z:\selfDIr\壁纸\【哲风壁纸】保险柜-办公室-卡通.png" \
  prototype/_real_v6_balanced \
  --chroma-decouple --recalib-luminance \
  --merge-features 0.6 --merge-min-size 50 \
  --dtop-median-size 3 --highlight-protect 0.08 \
  --cmy-merge-features 0.5 --cmy-merge-min-size 30 \
  --cmy-merge-chroma-tol 8.0 --cmy-smooth 0.8 \
  --dtop-min 0.10 --dtop-quantize-step 0.12 \
  --dtop-cmy-cover-margin 0.03 --detail-level 0.5 \
  --pixel-pitch-mm 0.20 --pitch-cmy-mm 0.30

# 最实心版本（_real_v6_solid 复现）：
python export_v4.py \
  "Z:\selfDIr\壁纸\【哲风壁纸】保险柜-办公室-卡通.png" \
  prototype/_real_v6_solid \
  --chroma-decouple --recalib-luminance \
  --merge-features 0.7 --merge-min-size 80 \
  --dtop-median-size 3 --highlight-protect 0.08 \
  --cmy-merge-features 0.6 --cmy-merge-min-size 40 \
  --cmy-merge-chroma-tol 6.0 --cmy-smooth 1.0 \
  --dtop-min 0.12 --dtop-quantize-step 0.20 \
  --dtop-cmy-cover-margin 0.08 --detail-level 0.3 \
  --pixel-pitch-mm 0.20 --pitch-cmy-mm 0.30
```

## 9. 标准可回退版本

- **近期基线**：`std-overlap-detail-v1` → commit `82de6daf03`（用户最满意的 overlap_detail，无参数复现）
- **当前迭代基线**：`std-overlap-detail-v2` 对应迭代 49 解耦+重标定（未最终打 tag，待 dE < 8 观感验收后升级为 `std-overlap-detail-v3`）
- 所有新参数默认 `0`/`False`，关闭时与基线逐字节等价，可作为临时回退。
