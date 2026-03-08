# 360-SL-Metasurface

360° 结构光 + 超表面联合优化项目。  
本项目包含以下核心能力：

1. 超表面相位建模与远场传播模拟
2. 可微鱼眼相机成像渲染
3. 双目结构光深度重建网络训练与评估
4. 相位图、损失曲线与指标可视化

## 目录说明

- `model/`: 超表面、重建网络与端到端模型
- `Image_formation/`: 可微渲染器
- `dataset/`: 数据读取与生成脚本
- `utils/`: 参数解析、相机模型、基础网络层与工具函数
- `train.py`, `train_v2.py`: 训练入口
- `test.py`, `test_with_metrics_*.py`: 推理与指标评估
- `vis_*.py`, `compare_*.py`: 可视化和结果对比工具

## 环境安装

```bash
conda env create -f environment.yaml
conda activate metaPolka
```

## 数据准备

默认数据根目录为 `./11518075/`，训练/验证路径由 `utils/ArgParser.py` 配置。  
你可以下载官方数据并按原目录结构放置，或替换为自己的数据。

## 常用可执行脚本

### 1) 训练

```bash
python train.py
```

```bash
python train_v2.py
```

说明：
- `train.py`: 基础训练流程
- `train_v2.py`: 增强版训练，包含更多中间结果保存逻辑

### 2) 测试与评估

```bash
python test.py
```

```bash
python test_with_metrics_cam1_only.py
python test_with_metrics_cam1_only_my.py
python test_with_metrics_cam1_only_my_v2.py
python test_with_metrics_cam_both.py
```

### 3) 可视化与对比

```bash
python vis_phasemap.py
python vis_phasemap_with_circle.py
python vis_phasemap_with_circle_v2.py
python compare_loss.py
python compare_phasemap.py
python compare_models_find_best.py
python draw_radar_metrics.py
```

### 4) 数据生成

```bash
python dataset/dataset_generator.py
```

## 上传到 GitHub（你的仓库）

目标仓库：`https://github.com/ep1phany05/360-SL-Metasurface`

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/ep1phany05/360-SL-Metasurface.git
git push -u origin main
```

如果远程已存在历史提交，先拉取再推送：

```bash
git pull --rebase origin main
git push -u origin main
```

## 参考

- Project Page: https://eschoi.com/360-SL-Metasurface/
- Paper: https://www.nature.com/articles/s41566-024-01450-x
- ArXiv: https://arxiv.org/abs/2306.13361
