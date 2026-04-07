# <D<sup>2</sup>MoRA: Diversity-Regulated Asymmetric MoE-LoRA Decomposition for Efficient Multi-Task Adaptation>

> <D<sup>2</sup>MoRA is a diversity-regulated asymmetric MoE-LoRA framework for multi-task adaptation that improves parameter efficiency and expert specialization by decoupling low-rank decomposition and explicitly encouraging diversity across experts.>

## Authors

**Jianhui Zuo**<sup>1</sup>, **Xuemeng Song**<sup>2*</sup>, **Haokun Wen**<sup>3,4</sup>, **Meng Liu**<sup>5</sup>, **Yupeng Hu**<sup>1</sup>, **Jiuru Wang**<sup>6</sup>, **Liqiang Nie**<sup>3*</sup>

<sup>1</sup> `<School of Software, Shandong University>` \
<sup>2</sup> `<Department of Computer Science and Engineering, Southern University of Science and Technology>` \
<sup>3</sup> `<School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen)>` \
<sup>4</sup> `<School of Data Science, City University of Hong Kong>` \
<sup>5</sup> `<School of Computer and Artificial Intelligence, Shandong Jianzhu University>` \
<sup>6</sup> `<School of Computer Science and Engineering, Linyi University>` \
\* Corresponding author

## Links

- **Paper**: [`Paper Link`](<https://ojs.aaai.org/index.php/AAAI/article/view/40168>)
- **Hugging Face Model**: [`Model`](<huggingface-model-link>)
- **Hugging Face Dataset**: [`Dataset`](<huggingface-dataset-link>)
- **Code Repository**: [`GitHub`](https://github.com/iLearn-Lab/AAAI26-D2MoRA)

> 如果某些链接暂时没有，可以先删掉对应条目，后续再补充。

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [Demo / Visualization](#demo--visualization)
- [TODO](#todo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [MM/YYYY] Initial release
- [MM/YYYY] Release paper / arXiv version
- [MM/YYYY] Release code
- [MM/YYYY] Release checkpoints on Hugging Face
- [MM/YYYY] Release dataset / benchmark / demo


---

## Introduction

本项目是论文 **`<Paper Title>`** 的官方实现 / 复现实现 / 项目主页。

请在这里简要说明：

- 论文要解决什么问题
- 方法的核心思想是什么
- 与现有方法相比有什么特点
- 本仓库提供了哪些内容，例如：
  - 训练代码
  - 推理代码
  - 模型权重
  - 数据处理脚本
  - 评测脚本
  - Demo

### Example Description

This project is the official implementation homepage of the paper **`D<sup>2</sup>MoRA: Diversity-Regulated Asymmetric MoE-LoRA Decomposition for Efficient Multi-Task Adaptation`**.

What problem does the paper address? \
D<sup>2</sup>MoRA aims to address the limitation of existing MoE-LoRA methods in multi-task adaptation, where experts often become insufficiently diverse and the low-rank decomposition structure is overly constrained, leading to suboptimal expert specialization and parameter utilization.

What is the core idea of the method? \
The core idea of D2MoRA is to introduce a diversity-regulated asymmetric MoE-LoRA decomposition, which decouples the low-rank adaptation structure and explicitly encourages experts to learn complementary rather than redundant knowledge through diversity regularization.

What are its key characteristics compared with existing methods? \
Compared with existing methods, D2MoRA places greater emphasis on expert diversity and asymmetric decomposition, enabling more flexible knowledge sharing and stronger expert specialization while maintaining high parameter efficiency in multi-task adaptation.

This repository provides:
  - Training Code
  - Inference Code
  - Model Weights
  - Data Processing Scripts
  - Evaluation Scripts

---

## Highlights

- 支持 `<task / domain>`
- 提供 `<training / inference / evaluation>` 脚本
- 提供 `<checkpoint / dataset / benchmark / demo>`
- 适合用于 `<论文复现 / 项目展示 / 后续研究>`

---

## Method / Framework

你可以在这里放方法框架图、模型结构图或整体 pipeline 图。

### Framework Figure

```markdown
![Framework](./assets/framework.png)
```

实际使用时，把上面这行替换成：

```markdown
![Framework](./assets/framework.png)
```

然后在下面补一句说明：

**Figure 1.** Overall framework of `<Method Name>`.

---

## Project Structure

```text
.
├── assets/                # 图片、框架图、结果图、demo 图
├── configs/               # 配置文件
├── data/                  # 数据说明（不建议直接上传大数据本体）
├── scripts/               # 训练、推理、评测脚本
├── src/                   # 核心源码
├── README.md
├── requirements.txt
└── LICENSE
```

如果你的项目结构不同，请按实际情况修改。

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iLearn-Lab/<repo-name>.git
cd <D2MoRA>
```

### 2. Create environment

```bash
conda create -n D2MoRA python=3.10
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


---

## Checkpoints / Models

如果你们发布了模型权重，可以写：

- **Main checkpoint**: [`Model Link`](<huggingface-model-link>)
- **Additional checkpoint**: [`Other Checkpoint`](<other-checkpoint-link>)

下载后请放入如下目录：

```text
checkpoints/
```

如果需要修改配置路径，也可以说明：

- 修改 `config.yaml` 中的 checkpoint 路径
- 或在运行脚本时通过参数传入

---

## Dataset / Benchmark

如果你们还提供数据集，可以写：

- **Dataset**: [`Dataset Link`](<huggingface-dataset-link>)
- **Benchmark**: [`Benchmark Link`](<benchmark-link>)

并说明数据组织方式，例如：

```text
data/
├── train/
├── val/
└── test/
```

> 如果数据集不能直接公开，请在这里说明申请方式或访问限制。

---

## Usage

### Training

```bash
sh llama2_7B_D2MoRA.sh 16 32
```

### Evaluation

```bash
sh llama2_7B_D2MoRA_eval.sh
```


---


### Example Results


| Model | PEFT Method | Param | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-c | ARC-e | OBQA | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaMA-7B | LoRA<sub>{M=1, N=1, r=64}</sub> | 50.3M | 68.47 | 80.09 | 76.56 | 78.83 | 78.69 | 60.75 | 76.56 | 74.60 | 74.32 |
| LLaMA-7B | DoRA<sub>{M=1, N=1, r=64}</sub> | 51.7M | 68.13 | 79.92 | 77.64 | 82.25 | 80.58 | 62.80 | 76.01 | 76.20 | 75.44 |
| LLaMA-7B | MoSLoRA<sub>{M=1, N=1, r=64}</sub> | 50.7M | 66.82 | 81.39 | **78.40** | 81.79 | **80.98** | 62.63 | 78.28 | 77.80 | 76.01 |
| LLaMA-7B | MOELoRA<sub>{M=8, N=8, r=8}</sub> | 50.4M | 69.39 | 79.90 | 76.21 | 81.14 | 80.76 | 62.41 | 78.53 | 78.70 | 75.88 |
| LLaMA-7B | MOELoRA<sub>{M=4, N=4, r=16}</sub> | 50.4M | 68.47 | 80.20 | 77.99 | 80.81 | 80.66 | 63.48 | 79.00 | 75.40 | 75.75 |
| LLaMA-7B | HydraLoRA<sub>{M=1, N=8, r=12}</sub> | 45.6M | 68.59 | 81.56 | 77.94 | 83.20 | 78.61 | 63.91 | 78.58 | 77.40 | 76.22 |
| LLaMA-7B | HydraLoRA<sub>{M=1, N=6, r=16}</sub> | 46.4M | 68.07 | 81.99 | 77.64 | 79.44 | 79.32 | 63.82 | 79.00 | **79.20** | 76.06 |
| LLaMA-7B | D<sup>2</sup>MoRA<sub>{M=3, N=8, r=8}</sub> | **35.8M** | 69.48 | 81.34 | 78.25 | 83.89 | 79.72 | 64.33 | **79.21** | 78.40 | 76.83 |
| LLaMA-7B | D<sup>2</sup>MoRA<sub>{M=3, N=4, r=16}</sub> | 45.6M | **69.66** | **82.86** | 77.22 | **85.95** | 80.58 | **64.68** | **79.21** | 77.20 | **77.17** |
| LLaMA2-7B | LoRA<sub>{M=1, N=1, r=64}</sub> | 50.3M | 70.91 | 81.34 | 76.20 | 81.41 | 80.19 | 63.99 | 77.31 | 76.80 | 76.02 |
| LLaMA2-7B | DoRA<sub>{M=1, N=1, r=64}</sub> | 51.7M | 68.65 | 81.12 | 78.45 | 86.64 | 81.06 | 65.02 | 78.24 | 79.20 | 77.30 |
| LLaMA2-7B | MoSLoRA<sub>{M=1, N=1, r=64}</sub> | 50.7M | 68.64 | 82.05 | 77.52 | 87.66 | 80.61 | 67.36 | 81.62 | 79.42 | 78.11 |
| LLaMA2-7B | MOELoRA<sub>{M=8, N=8, r=8}</sub> | 50.4M | 70.26 | 82.15 | **78.81** | 86.23 | 80.96 | 65.15 | 82.81 | 78.20 | 78.07 |
| LLaMA2-7B | MOELoRA<sub>{M=4, N=4, r=16}</sub> | 50.4M | 70.69 | 81.60 | 77.43 | 83.35 | **82.06** | 66.55 | 83.54 | 78.70 | 77.99 |
| LLaMA2-7B | HydraLoRA<sub>{M=1, N=8, r=12}</sub> | 45.6M | 69.52 | 82.81 | 78.56 | 87.82 | 80.58 | 67.28 | 81.29 | 79.80 | 78.46 |
| LLaMA2-7B | HydraLoRA<sub>{M=1, N=6, r=16}</sub> | 46.4M | 70.07 | 82.66 | **78.81** | 87.53 | 80.34 | 66.27 | 81.82 | 78.40 | 78.24 |
| LLaMA2-7B | D<sup>2</sup>MoRA<sub>{M=3, N=8, r=8}</sub> | **35.8M** | 70.40 | 82.26 | 78.76 | 87.72 | 81.53 | **70.65** | **84.01** | 78.80 | 79.27 |
| LLaMA2-7B | D<sup>2</sup>MoRA<sub>{M=4, N=3, r=16}</sub> | 45.6M | **71.31** | **82.86** | 78.40 | **90.11** | 81.68 | 67.06 | 83.38 | **81.00** | **79.48** |

---

## TODO

- [ ] 完善文档
- [ ] 补充训练脚本说明
- [ ] 补充推理脚本说明
- [ ] 上传模型权重
- [ ] 上传结果图
- [ ] 发布 demo / project page

---

## Citation


```bibtex
@inproceedings{zuo2026d2mora,
  title={D2MoRA: Diversity-Regulated Asymmetric MoE-LoRA Decomposition for Efficient Multi-Task Adaptation},
  author={Zuo, Jianhui and Song, Xuemeng and Wen, Haokun and Liu, Meng and Hu, Yupeng and Wang, Jiuru and Nie, Liqiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={34},
  pages={29286--29294},
  year={2026}
}
```

---

## Acknowledgement

- Thanks to our supervisor Xuemeng Song and Haokun Wen for valuable support.
- Thanks to the open-source community for providing useful baselines and tools.

---

## License

This project is released under the Apache License 2.0.
