# METLIN SMRT：结构指纹到保留时间预测

本仓库提供可复现的 Python 实验流程：从 METLIN SMRT 原始数据生成结构分组及两种数据划分，训练随机森林（RF）和多层感知机（MLP），再输出测试集指标、误差分析表和图。主要结果与中间文件均由代码生成，不需要预先下载模型或结果文件。

## 目录

```text
raw/       两份原始输入数据
src/       数据准备、训练、结果分析及预测脚本
outputs/   运行后生成的模型、预测、表格、图片和日志
processed/ 运行后生成的清洗数据及指纹矩阵
splits/    运行后生成的 random / scaffold 划分
```

仓库包含 Figshare 数据条目中的 `SMRT_dataset.csv` 和 `SMRT_ECFP_1024_Fingerprints.txt`。未包含作者随附的 R 代码/结果压缩包；本项目的实验流程由 `src/` 中的 Python 脚本执行。数据来源与原论文信息见 [METLIN SMRT Figshare 条目](https://doi.org/10.6084/m9.figshare.8038913)。

## 环境要求

建议使用 Python 3.12。Windows PowerShell 中，在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 从原始数据重新生成结果

```powershell
.\.venv\Scripts\python.exe src\prepare_data.py --project-root .
.\.venv\Scripts\python.exe src\train.py --project-root . --stage all --workers 1 --threads 4
.\.venv\Scripts\python.exe src\analyze_results.py --project-root .
.\.venv\Scripts\python.exe src\chemical_analysis.py --project-root .
.\.venv\Scripts\python.exe src\plot_reference_style.py --project-root .
```

`prepare_data.py` 读取 `raw/`，生成 `processed/` 和 `splits/`；`train.py` 完成两种划分协议下的调参与模型训练、测试集预测；后三个分析脚本生成指标汇总、误差/结构与化学描述符分析，以及补充的模型比较图。完整运行会占用一定时间和内存。可按设备性能调整 `--threads`；`--workers 1` 顺序运行两种协议以控制 CPU 和内存占用。

训练完成后，表格和图片位于 `outputs/tables/`、`outputs/figures/`，模型与逐分子预测位于 `outputs/models/`、`outputs/predictions/`。这些目录已加入 `.gitignore`，不会成为代码仓库内容；每次运行都会在本地产生。

## 对新分子进行预测（可选）

`src/predict.py` 使用训练好的模型及与原始数据配置一致的 1024 位指纹文件进行推断。先完成上面的数据准备和训练，再查看参数：

```powershell
.\.venv\Scripts\python.exe src\predict.py --help
```

该脚本要求输入符合原始 Dragon ECFP 配置的指纹；任意 RDKit Morgan 指纹不等价，不能直接替代。

## 实验说明

模型输入是数据集提供的 1024 位 ECFP，目标值是实验保留时间（秒）。PubChem CID 仅用于记录匹配，不作为模型特征。代码将同一结构分配到同一子集，并比较结构分组随机划分与 Murcko 骨架不相交划分；所有划分、参数选择和评价细节以 `src/prepare_data.py` 与 `src/train.py` 中的实现为准。
