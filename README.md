# driving_scene_desc
# 基于视觉语言大模型的驾驶场景语义描述系统
基于 mini-nuScenes 数据集完成驾驶场景多模态感知、LiDAR 与六相机投影对齐、多模态特征融合、CLIP 视觉语言语义对齐、FusionMLP 方向级目标识别、自然语言描述生成、实验评价和 Streamlit 可视化展示。

系统主要识别车辆、行人和障碍物三类目标，并结合六相机方向生成结构化语义结果和自然语言描述。项目同时保留道路、建筑、植被、交通灯、交通标志等环境语义作为描述补充信息，用于增强场景表达，但不参与严格 F1 评价。

---

## 1. 项目主要功能

1. **mini-nuScenes 数据读取**  
   读取关键帧中的 LiDAR 点云、六相机图像、标定参数、自车位姿和目标标注信息。

2. **LiDAR 与相机投影对齐**  
   根据 `calibrated_sensor` 和 `ego_pose` 完成坐标转换，将 LiDAR 点云投影到六个相机图像平面，并筛选有效投影点。

3. **多模态特征构建**  
   按六相机方向提取 LiDAR 几何统计特征、图像局部颜色/亮度/边缘特征、CLIP 语义分数、几何证据特征和场景上下文特征。

4. **FusionMLP 融合识别**  
   构建轻量化 FusionMLP 网络，对每个相机方向分别判断车辆、行人、障碍物是否存在，并输出方向级预测结果。

5. **视觉语言语义对齐**  
   使用 CLIP ViT-B/32 将相机图像与文本提示词库进行匹配，得到目标类别和环境类别的语义分数。

6. **语义描述生成**  
   根据预测类别、相机方向、场景条件和相邻帧变化生成驾驶场景自然语言描述。

7. **实验评价与图表生成**  
   支持 Precision、Recall、F1-score、描述连贯性、单帧核心流程耗时、晴天白天/雨天夜晚场景对比、传统规则方法对比和消融实验。

8. **GUI 可视化展示**  
   使用 Streamlit 展示六相机原图、LiDAR 投影叠加图、融合 BEV 图、语义描述、方向预测概率和单帧评价指标。

---

## 2. 项目目录结构

```text
├── app.py                              # Streamlit 可视化界面入口
├── pipeline.py                         # 系统主流程：读取、投影、融合、推理、描述、保存结果
├── requirements.txt                    # Python 依赖库
├── configs/
│   ├── paths.yaml                      # 数据集、CLIP 权重、模型权重、输出目录配置
│   ├── classes.yaml                    # 目标类别、描述类别和 CLIP 文本提示词
│   ├── scene_split.yaml                # 场景条件和训练/验证/测试划分配置
│   └── experiments.yaml                # 对比实验和消融实验方法配置
├── scripts/
│   ├── 00_check_env.py                 # 检查环境、数据集路径、CLIP 和 CUDA 状态
│   ├── 01_prepare_data_index.py        # 生成 mini-nuScenes 帧索引表
│   ├── 02_build_train_data.py          # 构建 FusionMLP 训练特征 CSV
│   ├── 02_analyze_scene_split.py       # 分析训练/验证/测试划分情况
│   ├── 03_train_fusion_mlp.py          # 训练 FusionMLP 融合模型
│   ├── 04_run_inference.py             # 运行本文方法推理并保存结果
│   ├── 05_make_charts.py               # 生成本文方法实验图表
│   ├── 06_run_traditional.py           # 运行传统规则方法对比实验
│   ├── 07_make_traditional_comparison_charts.py  # 生成传统方法对比图
│   ├── 08_run_ablation.py              # 运行特征融合消融实验
│   ├── 09_make_fusion_ablation_charts.py          # 生成消融实验图表
│   └── 10_run_gui.py                   # 启动 GUI 展示界面
└── src/
    ├── datasets/                       # mini-nuScenes 读取、标签映射和数据划分
    ├── geometry/                       # LiDAR 到相机的投影计算
    ├── features/                       # LiDAR、图像、CLIP、上下文特征提取
    ├── semantics/                      # CLIP 视觉语言对齐适配器
    ├── models/                         # FusionMLP 模型与训练工具
    ├── nlg/                            # 结构化语义和自然语言描述生成
    ├── eval/                           # 指标计算、CSV 汇总和图表绘制
    ├── baselines/                      # 传统规则方法
    ├── visualization/                  # 六相机图、投影图和 BEV 图保存
    └── common/                         # 配置读取、计时器和通用工具
```

---

## 3. 环境要求

建议使用 Python 3.10 或 3.11。主要依赖如下：

```bash
pip install -r requirements.txt
```

如果需要使用真实 CLIP 语义分数，还需要安装 OpenAI CLIP：

```bash
pip install git+https://github.com/openai/CLIP.git
```

如果网络环境无法直接安装 CLIP，可以提前将 CLIP 包和 ViT-B/32 权重放到本地环境中。正式实验建议使用真实 CLIP 权重；如果只做流程调试，可以在部分脚本中关闭 CLIP 或不启用 `--use-clip`。

---

## 4. 数据集和模型路径配置

运行前需要修改 `configs/paths.yaml`，使其与本机实际路径一致。例如：

```yaml
project_root: "D:/python/driving_scene_desc"
nuscenes_root: "D:/python/driving_scene_desc/data/raw/nuScenes"
nuscenes_version: "v1.0-mini"
clip_ckpt: "D:/python/driving_scene_desc/models/clip/ViT-B-32.pt"
fusion_ckpt: "D:/python/driving_scene_desc/models/fusion/fusion_mlp.pt"
output_root: "D:/python/driving_scene_desc/outputs"
cache_root: "D:/python/driving_scene_desc/outputs/cache"
```

其中：

- `nuscenes_root` 指向 mini-nuScenes 数据集根目录；
- `nuscenes_version` 默认为 `v1.0-mini`；
- `clip_ckpt` 指向 CLIP ViT-B/32 本地权重；
- `fusion_ckpt` 是训练后的 FusionMLP 模型保存路径；
- `output_root` 用于保存 CSV、JSON、可视化图片和实验图表。

本项目默认将 mini-nuScenes 的 10 个 scene 划分为两类场景条件：

```text
scene 1-7  ：晴天白天场景 day_sunny
scene 8-10 ：雨天夜晚场景 night_rainy
```

训练、验证和测试划分由 `configs/scene_split.yaml` 控制。当前主流程采用 `frame_stratified`，即在每个 scene 内按照帧序号划分训练集、验证集和测试集，并保证同一帧的六个方向不会被拆开。

---

## 5. 推荐运行流程

### 第一步：检查环境和路径

```bash
python scripts/00_check_env.py
```

该脚本会检查：

- 项目路径是否正确；
- mini-nuScenes 数据集是否存在；
- CLIP 权重路径是否正确；
- PyTorch 和 CUDA 是否可用；
- `nuscenes-devkit` 是否能正常读取数据集。

---

### 第二步：生成数据索引

```bash
python scripts/01_prepare_data_index.py
```

输出文件：

```text
outputs/csv/frame_index.csv
```

该文件记录每个关键帧所属 scene、场景条件和样本索引，方便后续实验统计。

---

### 第三步：构建训练特征

正式实验建议启用 CLIP：

```bash
python scripts/02_build_train_data.py --use-clip --device auto
```

如果只是快速调试流程，可以限制帧数：

```bash
python scripts/02_build_train_data.py --use-clip --device auto --limit 20
```

输出文件：

```text
outputs/csv/train_features.csv
```

该文件以“帧 × 六相机方向”为单位保存训练样本。每一行对应一个相机方向，包含 LiDAR 特征、图像特征、CLIP 语义分数、几何证据特征、上下文特征和三类标签。

---

### 第四步：训练 FusionMLP 模型

```bash
python scripts/03_train_fusion_mlp.py --epochs 60 --batch-size 64 --lr 0.001 --device auto
```

训练完成后会保存模型：

```text
models/fusion/fusion_mlp.pt
```

模型文件中包含：

- FusionMLP 网络权重；
- 三类目标的最佳判定阈值；
- 训练损失记录；
- 训练/验证/测试划分信息；
- 最佳验证集 F1-score。

FusionMLP 输入包括五个分支特征：

```text
LiDAR 几何特征 + 图像局部特征 + CLIP 语义分数 + 几何证据特征 + 场景上下文特征
```

输出为三类目标的方向级概率：

```text
vehicle / pedestrian / obstacle
```

---

### 第五步：运行本文方法推理

```bash
python scripts/04_run_inference.py --method ours --device auto
```

默认只评估测试集。如果需要评估全部 404 帧，可以使用：

```bash
python scripts/04_run_inference.py --method ours --device auto --eval-all
```

如果只想测试速度或调试，可以使用：

```bash
python scripts/04_run_inference.py --method ours --device auto --limit 10
```

输出文件主要包括：

```text
outputs/csv/predictions_ours.csv
outputs/json/ours_frame_XXXX.json
outputs/vis/ours_frame_XXXX_sixcam.jpg
outputs/vis/ours_frame_XXXX_sixcam_projected.jpg
outputs/vis/ours_frame_XXXX_bev_raw.jpg
outputs/vis/ours_frame_XXXX_bev_fused.jpg
```

其中：

- `predictions_ours.csv` 保存方向级预测结果、真实标签、Precision、Recall、F1-score 和耗时；
- `json` 文件保存单帧结构化语义结果；
- `vis` 文件夹保存六相机原图、LiDAR 投影叠加图、原始 BEV 和融合 BEV 可视化结果。

---

### 第六步：生成本文方法实验图表

```bash
python scripts/05_make_charts.py
```

输出目录：

```text
outputs/charts/
```

主要图表包括：

```text
05_ours_main_metrics_camera_mean.png
05_ours_condition_f1_camera_mean.png
05_ours_per_class_f1_camera_mean.png
05_ours_camera_f1.png
05_ours_camera_class_heatmap.png
05_fusion_mlp_train_loss.png
05_runtime_core_10frames.png
```

---

### 第七步：运行传统规则方法对比实验

```bash
python scripts/06_run_traditional.py --device auto
```

该方法只使用 LiDAR 投影点统计和几何规则，不使用 CLIP 和 FusionMLP。脚本会先在训练集或验证集上搜索规则阈值，然后在测试集上进行评价。

输出文件包括：

```text
outputs/csv/predictions_traditional_rule.csv
outputs/csv/traditional_rule_thresholds.csv
outputs/csv/traditional_rule_paper_summary.csv
```

---

### 第八步：生成传统方法对比图

```bash
python scripts/07_make_traditional_comparison_charts.py
```

输出图表包括：

```text
outputs/charts/07_traditional_vs_ours_metrics.png
outputs/charts/07_traditional_vs_ours_f1.png
```

该部分只比较 Precision、Recall 和 F1-score，不比较描述连贯性和流程耗时。

---

### 第九步：运行消融实验

```bash
python scripts/08_run_ablation.py --device auto
```

消融实验包含以下方法：

```text
image_only      仅使用图像和 CLIP 相关信息
lidar_only      仅使用 LiDAR 几何证据
normal_fusion   普通规则融合，不使用训练后的 FusionMLP
ours            本文融合方法，复用 04_run_inference.py 的结果
```

输出文件包括：

```text
outputs/csv/predictions_ablation_all.csv
outputs/csv/ablation_paper_summary.csv
outputs/csv/ablation_paper_per_class_f1.csv
outputs/csv/ablation_fixed_thresholds.csv
```

---

### 第十步：生成消融实验图表

```bash
python scripts/09_make_fusion_ablation_charts.py
```

输出图表包括：

```text
outputs/charts/09_fusion_ablation_f1.png
outputs/charts/09_fusion_ablation_per_class_f1.png
```

消融实验只比较 F1-score，不比较描述连贯性和流程耗时。

---

### 第十一步：启动 GUI 展示界面

```bash
python scripts/10_run_gui.py
```

或者直接运行：

```bash
streamlit run app.py
```

启动后访问：

```text
http://localhost:8501
```

GUI 中可以选择数据集路径、FusionMLP 模型和帧序号，并展示当前帧的：

- 六相机原图；
- LiDAR 投影叠加图；
- 原始 BEV 图；
- 融合 BEV 图；
- 语义描述文本；
- 单帧识别准确率；
- 描述连贯性；
- 核心流程耗时；
- 六方向语义预测结果。

---

## 6. 一键式推荐运行顺序

如果数据集、CLIP 权重和 Python 环境都已经配置完成，可以按下面顺序运行：

```bash
python scripts/00_check_env.py
python scripts/01_prepare_data_index.py
python scripts/02_build_train_data.py --use-clip --device auto
python scripts/03_train_fusion_mlp.py --epochs 60 --batch-size 64 --lr 0.001 --device auto
python scripts/04_run_inference.py --method ours --device auto
python scripts/05_make_charts.py
python scripts/06_run_traditional.py --device auto
python scripts/07_make_traditional_comparison_charts.py
python scripts/08_run_ablation.py --device auto
python scripts/09_make_fusion_ablation_charts.py
python scripts/10_run_gui.py
```

---

## 7. 主要输出结果说明

```text
outputs/
├── csv/
│   ├── frame_index.csv                         # 数据帧索引
│   ├── train_features.csv                      # FusionMLP 训练特征
│   ├── frame_split.csv                         # frame_stratified 数据划分结果
│   ├── predictions_ours.csv                    # 本文方法方向级预测结果
│   ├── summary_by_method_camera_mean.csv       # 本文方法总体指标
│   ├── summary_by_condition_camera_mean.csv    # 不同场景条件指标
│   ├── traditional_rule_paper_summary.csv      # 传统规则方法论文汇总
│   ├── traditional_vs_ours_paper_summary.csv   # 传统方法与本文方法对比汇总
│   ├── ablation_paper_summary.csv              # 消融实验总体 F1 汇总
│   └── ablation_paper_per_class_f1.csv         # 消融实验类别级 F1 汇总
├── json/
│   └── ours_frame_XXXX.json                    # 单帧结构化语义结果
├── vis/
│   ├── ours_frame_XXXX_sixcam.jpg              # 六相机原图拼接
│   ├── ours_frame_XXXX_sixcam_projected.jpg    # LiDAR 投影叠加图
│   ├── ours_frame_XXXX_bev_raw.jpg             # 原始 BEV 点云图
│   └── ours_frame_XXXX_bev_fused.jpg           # 融合特征强度 BEV 图
└── charts/
    ├── 05_*.png                                # 本文方法实验图
    ├── 07_*.png                                # 传统方法对比图
    └── 09_*.png                                # 消融实验图
```

---

## 8. 实验评价说明

本项目主要采用方向级评价方式。每一帧包含六个相机方向，每个方向分别判断三类目标是否存在，再计算 Precision、Recall 和 F1-score。最终结果使用六相机方向平均指标，更符合系统“按方位生成语义描述”的设计目标。

三类评价目标为：

```text
vehicle     车辆
pedestrian  行人
obstacle    障碍物
```

其中 obstacle 由 nuScenes 中的 barrier、traffic cone、debris 等可移动障碍物类别映射而来。

描述连贯性通过结构化语义完整性、方向一致性、场景条件表达、动态信息表达等规则进行评分，用于衡量生成描述是否完整、清晰和符合场景逻辑。

---

## 9. 常见问题

### 1）提示找不到 mini-nuScenes 数据集

检查 `configs/paths.yaml` 中的 `nuscenes_root` 是否指向数据集根目录。该目录下应包含 `v1.0-mini`、`samples`、`sweeps`、`maps` 等文件夹。

### 2）提示找不到 CLIP 权重

检查 `clip_ckpt` 是否指向本地 `ViT-B-32.pt` 文件。正式实验建议使用真实 CLIP 权重。如果只是调试训练特征生成，可以暂时不添加 `--use-clip`。

### 3）消融实验报错 predictions_ours.csv 不一致

消融实验会复用 `04_run_inference.py` 生成的 `predictions_ours.csv`，因此 `04` 和 `08` 必须使用相同的测试帧范围。如果 `04` 使用了 `--limit 10`，则 `08` 也需要使用 `--limit 10`。

### 4）传统方法对比图报错测试帧不一致

`07_make_traditional_comparison_charts.py` 会检查 `predictions_ours.csv` 和 `predictions_traditional_rule.csv` 是否使用同一批测试帧。如果不一致，需要使用相同的 `--eval-all` 或 `--limit` 参数重新运行 `04` 和 `06`。

### 5）GUI 页面加载失败

先确认以下文件是否存在：

```text
models/fusion/fusion_mlp.pt
configs/paths.yaml
mini-nuScenes 数据集目录
CLIP ViT-B/32 权重文件
```

如果模型文件不存在，需要先运行 `03_train_fusion_mlp.py`。

---

## 10. 说明

本项目的 SemanticKITTI 模块目前仅作为数据接口预留，主实验和 GUI 展示均基于 mini-nuScenes 完成。
