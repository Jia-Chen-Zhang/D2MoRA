# `<D<sup>2</sup>MoRA: Diversity-Regulated Asymmetric MoE-LoRA Decomposition for Efficient Multi-Task Adaptation>`

> `<D<sup>2</sup>MoRA is a diversity-regulated asymmetric MoE-LoRA framework for multi-task adaptation that improves parameter efficiency and expert specialization by decoupling low-rank decomposition and explicitly encouraging diversity across experts.>`

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

> 如果你使用的是 conda、poetry、uv 或 docker，请改成自己的实际安装方式。

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

你可以插入结果图：

```markdown
![Result](./assets/result.png)
```

或者放一个简单结果表：

| Setting | Result |
|---|---:|
| Baseline | xx.x |
| Ours | xx.x |

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
