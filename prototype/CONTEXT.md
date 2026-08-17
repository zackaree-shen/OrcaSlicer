# CMYK 透光浮雕生成器 — 交接上下文（CONTEXT）

> 最后更新：2026-08-17（迭代 47：回退迭代 46，最终有效版本 = v0.45-dtop-field）
> 本文档为跨会话交接的自包含上下文。新会话请先读本文，再按需读 loop-journal。

## 1. 项目定位

为 Snapmaker Orca 切片器开发 **CMYK 透光浮雕（lithophane）生成器**。当前为 **Python 原型阶段**（`prototype/`），验证算法后移植 C++。工作目录：`C:\workDir\programing\SnorcaWorkTrees\explore-lithophane`（分支 `explore/lithophane`）。

**唯一演进载体**：`LithoMode.BAMBU`（Bambu-like 结构：薄白底 z_lo=0.2 + CMY 色带 band=0.7 + 白浮雕顶板）。凹刻（阴刻，顶面随亮度起伏）为默认，凸刻可选。

## 2. 最终有效状态（v0.45-dtop-field，commit 1b9baadf36）

当前 HEAD = revert 后等价于该版本。**验收最优导出**：`C:\Users\snapmaker\Desktop\bambu_v2_field\`。

### BAMBU+P1（tone_map=True，GUI 默认开）管线
```
原图留存副本(用于诚实 dE)
→ preprocess_image(sharpen=2.0, contrast=1.5)   # hue-preserving
→ resample 到 solve 网格 (pixel_pitch 0.3mm)
→ dTop = anchored_dtop_field(small)             # ★ 迭代45核心
    光密度域(-5.4·log10 L) mid-rank CDF 均衡场 × top_max
→ resolve_cmy_for_dtop(flat_lab, gamut, dTop)   # CMY 带宽重解 tol=0.10
→ spike_surgery(t_sigma=1.5, iterations=2)      # 一次性,无双重平滑
→ dE = forward_stacked vs 原图(诚实口径, 报 median)
```

### 关键实测数据（暗卡通图 Z:\selfDIr\壁纸\【哲风壁纸】保险柜-办公室-卡通.png）
| 指标 | 值 |
|---|---|
| dTop 饱和率 | 0.5%（迭代44 之前 84.4%）|
| 层直方图 | 11 层均匀 5-16% |
| 细节 mean_dx | 0.0271 |
| 尖刺 p95 | 0.156 |
| dE vs 原图 median | ~6.7（暗图物理差距，诚实口径）|

## 3. 核心文件与算法

| 文件 | 职责 |
|---|---|
| `litho_color.py` | 色卡 `build_gamut_stacked`（top_step=0.08）、求解 `solve_stacked`、**`anchored_dtop_field`**（迭代45）、`resolve_cmy_for_dtop`、`spike_surgery`、`preprocess_image`、Beer-Lambert 工具链 |
| `litho_engine.py` | `color_lithophane_engine` 入口；BAMBU 分支约 247-345 行为 P1 管线；`BAMBU_WHITE_BASE=0.2` 常量 |
| `litho_core.py` | `LithophaneParams`、网格、网格转 mesh |
| `litho_3mf.py` / `batch_export.py` | 3MF（5文件结构,含 Snapmaker U1 预设,层高 0.2）/ 批量导出 |
| `litho_gui_web.py` + `litho_web/` | pywebview+Three.js GUI（用户嫌弃 tkinter 选定）|
| `test_*.py` | **105/105**（test_color 20 + engine 54 + 3mf 12 + core + gui_guard 12 + gui_web 7）|

### 物理模型
- Beer-Lambert：τ = ∏ 10^(-dᵢ/TDᵢ)，TD_W=5.4（白）、TD_C/M/Y 主通道 0.5
- 白窗 [0.2, 2.2]mm 只覆盖 τ≥0.391（≈sRGB168）→ 暗图 80% 像素超出 → **绝对反演必钳顶**（迭代45 根因）
- 层高 0.2 × top_max 2.0 → 物理上仅 ~11 可辨层（Bambu 参考件相同）

### anchored_dtop_field 设计要点（迭代45，当前最优）
- **mid-rank**：ties 共享 rank → 大片同色背景映射单一厚度（无平台渐变伪影）
- **不含量化/平滑**：迭代46 的 σ0.8 预模糊+定向去刺被用户实测否决（"明显旧更优"，已 revert）——**教训：噪声指标改善 ≠ 观感改善，mean_dx 里有用户认的"细节"**
- 已知限制：纯平坦+噪声输入 rank 场即随机数（无结构可保，文档化不做假断言）

## 4. 证伪清单（不要再走这些路）

| 路径 | 证伪证据 |
|---|---|
| 绝对 Beer-Lambert 反演 + clip | 84.4% 钳顶大平地（迭代45根因）|
| 改图路线（tone_mapping_preprocess）| Y_tone 对 d 线性 + ratio 3× 上限销毁锚定语义，端到端饱和 77-80% |
| 泊松重建（Weyrich/Kerber）| dE 漂移 2.9-9.2 |
| 双重平滑（solve smooth_top=True + surgery 叠加）| -38% 细节（迭代44根因）|
| CLAHE/受限均衡 | tile 撕裂背景 0.25mm 台阶；与"层均匀"同轴反向 |
| 引导滤波去均衡噪声 | guide 即噪声源（a≈1 复现噪声）+ 外推越界 |
| 去 sharpen 降噪声 | 安慰剂（std 0.5615→0.5635）|
| σ0.8 预模糊 + 定向去刺（迭代46）| 指标全优但**用户实测观感更差**（v0.47 已 revert）|
| SAM 边界引导 / ESRGAN / 完全闭式 / 混合闭式 | 各迭代记录（见 loop-journal 迭代 1-43）|

## 5. 已知问题与 backlog

1. **尖刺与细节的平衡未解决**（核心矛盾）：迭代46 指标上去刺成功（p95 -51%）但观感失败。用户的"细节"包含被我方判为噪声的成分。下次尝试方向：噪声感知众数滤波（保边缘先验）、或把 thr/σ 做成 GUI 滑条让用户调（而非拍默认值）。
2. **"浮雕感/对比度"无可验证指标**：三张对比图不同图案不同视图（切片灰白视图 vs 彩色渲染），"结合同事2浮雕感"暂缓——先做 P4 背光预览建立统一口径。
3. 饱和度增强（用户明确想要）：迭代46 的 sat_boost（Lab chroma 缩放,默认关,代价 dE+24%@1.3）被连带 revert；可从 commit 50cdcfe61e cherry-pick 该独立功能。
4. top_max 三处不一致（引擎 2.0 / 死默认 1.5 / GUI maxthick-dwhite 与硬编码 z_lo 语义错位）。
5. BAMBU 模式 layers_max 死参数 + 0.7/0.2 浮点巧合（layer_h=0.15 会吃 LAYER_GAP）。
6. dE 只报 median 广播标量（p90 已在 50cdcfe61e 实现过，随 revert 丢失）。
7. P2 预处理（直方图拉伸+CLAHE）/ P3 深度双频（Depth Anything V2 Small, 24.8M, CPU 0.3s）：未动，P4 口径建立后再评估。

## 6. 环境与工具

- Python 3.12（numpy/scipy/PIL/numpy-stl）；测试直接 `python test_xxx.py`（无 pytest）
- **GitHub 不可达**（网络受限）；飞书沉淀用 `lark-cli docs +fetch/+update`（先读 `lark-cli skills read lark-doc ...`）
- 调研文档：ONRhdXU56oLxbyx0kXAcFJ7qnne（7节+SVG+§8 实施进展，迭代 41-46 已同步）
- 方案文档：UXFvd6msWomlUgxVnQyco04DnNc（含用户 6 条评论决策）
- loop-journal：`C:\workDir\programing\loop-journal-files\loop-journal.md`（迭代 1-47 全记录）
- Bambu 参考件：Z:\selfDIr\透光浮雕研究\Bambu\lithophane_谢bro_U1.3mf
- 图像 MCP（analyze_image）对本地图片不稳定（CDN URL 反斜杠路径间歇 400）；WinRT OCR（PowerShell）可识别文字但渲染截图无文字；Python 定量（边缘能量/功率谱/分块统计）最可靠

## 7. 工作方式与教训（用户偏好）

- **对抗式开发循环**（adversarial-development-loop skill）：PLAN→REFUTE（子agent实测证伪）→REVISE→IMPLEMENT→TEST→RETROSPECT，journal 落盘。对抗审查两次在编码前拦下注定失败的方案（迭代45/46）
- Python 先验证再回 C++；时常建回滚点（commit+tag）；诚实报告（测试失败照说）
- **三大方法论教训**：① 展示层归一化制造"看起来不错"假象（高度图渲染教训）→ 验证看物理幅度分布；② standalone 验证配置 ≠ 集成路径（smooth_top/dW/pitch 逐参数对齐）；③ 过程指标 ≠ 结果指标（dE 必须对原图算）
- **迭代46 新增**：④ 指标优化 ≠ 观感优化——p95/孤立刺全改善但用户判"明显更差"；任何"去噪/平滑"类改动必须先出对比渲染让用户选，不拍默认值
- GUI 要现代化（pywebview+Three.js）；截图对比验收；方向转变经用户实测驱动

## 8. 快速上手

```bash
cd prototype
python test_color.py && python test_engine.py        # 确认 105/105
python litho_gui_web.py                              # GUI
python batch_export.py <image> <outdir>              # 批量导出 3MF+STL
# 单模式导出（匹配验证口径 pitch_top=0.10）:
python -c "from batch_export import run_mode; from litho_engine import LithoMode, ColorOrder; import numpy as np; from PIL import Image; \
run_mode(np.asarray(Image.open(r'Z:\selfDIr\壁纸\【哲风壁纸】保险柜-办公室-卡通.png').convert('RGB')), LithoMode.BAMBU, ColorOrder.MIXED, r'<outdir>', pitch_top=0.10)"
```
