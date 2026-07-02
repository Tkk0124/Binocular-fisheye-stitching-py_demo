# Dual-Fisheye Sphere Projection + Seam Pipeline

这是一套用于双鱼眼全景算法推演的 Python 工程包。它的核心原则是：

> 先建立 `fisheye pixel → ray → sphere/equirectangular` 的几何投影，再只评估 seam overlap 区域，最后做 seam search 与融合；特征点/APAP/Mesh Warp 只作为 fallback 诊断，不作为主线。

## 目录结构

```text
common/                                  公共工具：IO、投影、指标、xlsx 解析
module_01_input_preprocess/              输入预处理：读图、白平衡、绿偏修正
module_02_circle_detect/                 鱼眼圆心/半径/有效圆 mask 检测
module_03_radial_lut/                    光学资料/默认模型生成 r(θ) LUT
module_04_sphere_projection/             左右鱼眼投到等矩形球面
module_05_pose_overlap/                  姿态初始化与 overlap component 裁剪
module_06_overlap_evaluation/            overlap energy / edge / diff 评估
module_07_pattern_evaluation/            棋盘格检测、直线弯曲、尺度一致性评估
module_08_seam_graphcut_dp/              DP seam search，输出 seam mask 与 hard composite
module_09_blending/                      Feather / Multi-band 融合
module_10_local_compensation_fallback/   局部相位相关小位移诊断，默认不改最终结果
```

每个模块都会在输出目录下生成独立文件夹，例如：

```text
output/
  01_input_preprocess/
  02_circle_detect/
  ...
  10_local_compensation_fallback/
  pipeline_summary.json
  effective_config.json
```

## 快速运行

```bash
pip install -r requirements.txt
python run_pipeline.py \
  --left /path/to/left.bmp \
  --right /path/to/right.bmp \
  --lens-xlsx /path/to/C48501B模组镜头-光学资料-20260417.xlsx \
  --output ./output \
  --pano-width 2048 \
  --pano-height 1024
```

Windows 示例：

```bat
python run_pipeline.py ^
  --left D:\your_data\left.bmp ^
  --right D:\your_data\right.bmp ^
  --lens-xlsx D:\your_data\C48501B模组镜头-光学资料-20260417.xlsx ^
  --output D:\your_data\sphere_pipeline_output ^
  --pano-width 2048 ^
  --pano-height 1024
```

最终全景默认输出在：

```text
output/09_blending/final_multiband.png
```

## 当前 MVP 的定位

这套代码是算法推演版，不是最终 C++ SDK。它严控几个核心问题：

1. 不用 Homography/APAP 作为主对齐方式；
2. 用光学资料或默认等距模型构建 `θ → r`；
3. 从左右鱼眼分别生成 `left_sphere/right_sphere/valid_mask/overlap_mask`；
4. 自动找 overlap components，而不是手写固定 seam；
5. 在 overlap 内计算 seam energy、棋盘格可检测性、直线弯曲度；
6. 用 DP seam + Feather/Multi-band 做融合；
7. 局部补偿只做诊断，避免错误 warp 造成几何爆炸。

## 关键调参项

编辑 `config/default_config.json`：

```json
{
  "lens_model": {
    "diagonal_fov_deg": 210.0,
    "effective_pixel_pitch_um": 1.6,
    "max_image_height_mm": 2.25
  },
  "pose": {
    "left_yaw_deg": 0.0,
    "right_yaw_deg": 180.0
  },
  "projection": {
    "pano_width": 2048,
    "pano_height": 1024
  }
}
```

若左右内容方向不对，优先调：

- `right_yaw_deg`
- `left_yaw_deg`
- `image_y_sign`
- 左右图是否需要交换

## 输出检查顺序

建议按这个顺序看图：

1. `02_circle_detect/*circle_overlay.png`：圆心半径是否正确；
2. `04_sphere_projection/*sphere.png`：球面展开方向是否合理；
3. `04_sphere_projection/overlap_mask.png`：overlap 是否存在且合理；
4. `05_pose_overlap/overlap_components_overlay.png`：seam 区域是否被正确识别；
5. `06_overlap_evaluation/seam_energy_heatmap.png`：高能量区域是否集中在结构错位处；
6. `08_seam_graphcut_dp/global_seam_overlay.png`：seam 是否绕开强边缘；
7. `09_blending/final_multiband.png`：最终融合结果。

## 后续 C++ 化建议

Python MVP 验证后，C++ 侧应优先固化：

- circle detector
- radial LUT
- equirectangular remap map 生成
- overlap component extraction
- DP/GraphCut seam
- MultiBand blending

不要先移植 APAP/Mesh Warp。局部补偿建议只在 seam strip 内做强约束小位移。

## Dependency choice

Default install:

```bash
pip install -r requirements.txt
```

Kornia is intentionally not part of the default dependency set. The current pipeline uses deterministic NumPy/OpenCV projection and remap. For optional differentiable geometry experiments, use:

```bash
pip install -r requirements_optional_kornia.txt
```
