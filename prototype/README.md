# Lithophane Prototype (M2 color - CMYW stacking)

彩色透光浮雕算法原型：CMYW 半透明耗材按层堆叠 + 背光透射（Beer-Lambert 减色模型）。
每色生成一个厚度图 STL，叠放打印后从一面打光、另一面看照片效果。

## 依赖安装

```bash
pip install numpy pillow numpy-stl scipy
```

## 启动 GUI

```bash
python prototype/litho_gui.py
```

界面三步：
1. **Choose image** 选择彩色图片
2. **Build + preview** 生成（右侧实时显示"打印后实际效果"WYSIWYG + 色准统计）
   - 默认快速模式 ~5s；勾选 Exact mode 更准但 ~35s
   - TD 参数可调：C 吸红 / M 吸绿 / Y 吸蓝 的主通道透光距离（0.3=强吸收）
3. **Export 4 STLs** 导出 `litho_W/C/M/Y.stl` 到选定文件夹

## 命令行生成

```python
import numpy as np
from PIL import Image
from litho_core import LithophaneParams
from litho_color import color_lithophane_meshes, export_stl  # 实际 export 在 litho_core

img = np.asarray(Image.open("photo.jpg").convert("RGB"))
params = LithophaneParams(width_mm=144, height_mm=108, pixel_pitch_mm=0.3)
meshes, dE, gamut, reached = color_lithophane_meshes(img, params=params, exact=True)
for color, (verts, faces) in meshes.items():
    export_stl(f"litho_{color}.stl", verts, faces, name=f"lithophane_{color}")
```

## 切片器叠放方法

把 5 个 STL 导入切片器，按同一原点对齐堆叠：
- 底面 **litho_W.stl**（白色散射底座，固定厚度 0.80mm）
- 依次叠 **litho_C / litho_M / litho_Y**（各色厚度图 0~0.64mm）
- 顶部 **litho_top_white.stl**（白色亮度浮雕 0~2.0mm）
- 四色对应 AMS/料盘 4 个槽位（W/C/M/Y），切片器按 Z 层切片自动换色
- 推荐 0.2mm 喷嘴、0.08mm 层高、100% 填充、首层 0.15mm（Bambu wiki）

### 重要：导入方式（避免切片卡死）

**层间有 0.05mm 间隙，相邻层不共面**。请用"一个对象多部件"方式导入，而不是
5 个独立对象：
1. 导入第一个 STL（如 litho_white.stl）
2. 在对象列表选中它 → 右键 → **Add Part / 添加部件** → 依次加入其余 4 个
   （或在 3D 视图选中全部 5 个 → 右键 → 合并为多部件对象）
3. 确认对象列表中只有 1 个对象、含 5 个部件

用独立对象方式导入时，切片器会对 5 个重叠对象重复做填充区域布尔运算，
进度会卡在 25%（"Generating infill regions"）——这是几何病态不是死循环。

### 如果仍卡在 25%

查看切片日志（OrcaSlicer 日志文件，最后几行 `BOOST_LOG_TRIVIAL`）确认卡在
哪个子步骤：
- `"Preparing fill surfaces..."` / `"Processing external surfaces..."` =
  prepare_infill 的布尔运算，降低分辨率（pixel pitch）或顶部浮雕层数
- 用更低的网格分辨率重新生成（pixel_pitch 0.4+）

## 颜色说明（重要，物理上限）

CMYW 减色在 0.64mm/色 预算下的饱和色与 sRGB 原色有 ~10-18 ΔE 差距——这是
染色剂选择性与厚度预算的结构上限，非调参可解（与 Bambu 成品"柔和彩色"一致）。
照片/肤色/灰调表现好（ΔE 中位 ~3-4）。想更鲜艳需实测 TD 或增加每色厚度预算。

## 文件

- `litho_core.py`  — 单色核心：sRGB/网格/STL/验证（heightfield_to_mesh 供复用）
- `litho_color.py` — 彩色核心：sRGB<->Lab、CIEDE2000、Beer-Lambert 正向、色卡、色域映射逆问题、4 色 mesh
- `litho_gui.py`   — tkinter GUI
- `test_core.py`   — M1 回归（10 项）
- `test_color.py`  — M2 验证（10 项）
