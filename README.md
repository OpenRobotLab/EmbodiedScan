<br>
<p align="center">
<h1 align="center"><strong>MMScan: A Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations</strong></h1>

</p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2312.16170-blue)](https://arxiv.org/abs/2312.16170)
[![](https://img.shields.io/badge/Paper-%F0%9F%93%96-blue)](./assets/2024_NeurIPS_MMScan_Camera_Ready.pdf)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-blue)](https://tai-wang.github.io/mmscan)

</div>

## 🤖 [Demo](https://tai-wang.github.io/mmscan)

[![demo](assets/demo.png "demo")](https://tai-wang.github.io/mmscan)

<!-- contents with emoji -->

## 📋 Contents

1. [News](#-news)
2. [About](#-about)
3. [Getting Started](#-getting-started)
4. [MMScan Tutorial](#-mmscan-api-tutorial)
5. [MMScan Benchmark](#-mmscan-benchmark)
6. [TODO List](#-todo-list)

## 🔥 News

- \[2025-06\] We are co-organizing the CVPR 2025 3D Scene Understanding Challenge. You're warmly invited to participate in the MMScan Hierarchical Visual Grounding track!
The challenge test server is now online [here](https://huggingface.co/spaces/rbler/3d-iou-challenge). We look forward to your strong submissions!

- \[2025-01\] We are delighted to present the official release of [MMScan-devkit](https://github.com/OpenRobotLab/EmbodiedScan/tree/mmscan), which encompasses a suite of data processing utilities, benchmark evaluation tools, and adaptations of some models for the MMScan benchmarks. We invite you to explore these resources and welcome any feedback or questions you may have!

## 🏠 About

<!-- ![Teaser](assets/teaser.jpg) -->

<div style="text-align: center;">
    <img src="assets/MMScan_teaser.png" alt="Dialogue_Teaser" width=100% >
</div>

With the emergence of LLMs and their integration with other data modalities,
multi-modal 3D perception attracts more attention due to its connectivity to the
physical world and makes rapid progress. However, limited by existing datasets,
previous works mainly focus on understanding object properties or inter-object
spatial relationships in a 3D scene. To tackle this problem, this paper builds <b>the
first largest ever multi-modal 3D scene dataset and benchmark with hierarchical
grounded language annotations, MMScan.</b> It is constructed based on a top-down
logic, from region to object level, from a single target to inter-target relation
ships, covering holistic aspects of spatial and attribute understanding. The overall
pipeline incorporates powerful VLMs via carefully designed prompts to initialize
the annotations efficiently and further involve humans’ correction in the loop to
ensure the annotations are natural, correct, and comprehensive. Built upon exist
ing 3D scanning data, the resulting multi-modal 3D dataset encompasses 1.4M
meta-annotated captions on 109k objects and 7.7k regions as well as over 3.04M
diverse samples for 3D visual grounding and question-answering benchmarks. We
evaluate representative baselines on our benchmarks, analyze their capabilities in
different aspects, and showcase the key problems to be addressed in the future.
Furthermore, we use this high-quality dataset to train state-of-the-art 3D visual
grounding and LLMs and obtain remarkable performance improvement both on
existing benchmarks and in-the-wild evaluation.

## 🚀 Getting Started

- ### Installation

1. Clone Github repo.

   ```shell
   git clone git@github.com:rbler1234/MMScan.git
   cd MMScan
   ```

2. Install requirements.

   Your environment needs to include Python version 3.8 or higher.

   ```shell
   conda activate your_env_name
   python intall.py all/VG/QA
   ```

   Use `"all"` to install all components and specify `"VG"` or `"QA"` if you only need to install the components for Visual Grounding or Question Answering, respectively.

- ### Data Preparation

1. Download the Embodiedscan and MMScan annotation. (Fill in the [form](https://docs.google.com/forms/d/e/1FAIpQLScUXEDTksGiqHZp31j7Zp7zlCNV7p_08uViwP_Nbzfn3g6hhw/viewform) to apply for downloading)

   Create a folder `mmscan_data/` and then unzip the files. For the first zip file, put `embodiedscan` under `mmscan_data/embodiedscan_split` and rename it to `embodiedscan-v1`. For the second zip file, put `MMScan-beta-release` under `mmscan_data/MMScan-beta-release` and `embodiedscan-v2` under `mmscan_data/embodiedscan_split`.

   The directory structure should be as below:

   ```
   mmscan_data
   ├── embodiedscan_split
   │   ├──embodiedscan-v1/   # EmbodiedScan v1 data in 'embodiedscan.zip'
   │   ├──embodiedscan-v2/   # EmbodiedScan v2 data in 'embodiedscan-v2-beta.zip'
   ├── MMScan-beta-release   # MMScan data in 'embodiedscan-v2-beta.zip'
   ```

2. Prepare the point clouds files.

   Please refer to the [guide](data_preparation/README.md) here.

## 👓 MMScan Tutorial

The **MMScan Toolkit** provides comprehensive tools for dataset handling and model evaluation in  tasks.

### MMScan Dataset

The dataset tool in MMScan allows seamless access to data required for various tasks within MMScan.

- #### Usage

  Initialize the dataset for a specific task with:

  ```bash
  from mmscan import MMScan

  # (1) The dataset tool
  my_dataset = MMScan(split='train'/'test'/'val', task='MMScan-VG'/'MMScan-QA')
  # Access a specific sample
  print(my_dataset[index])
  ```

  *Note:*  For the test split, we have only made the VG portion publicly available, while the QA portion has not been released.

- #### Data Access

  Each dataset item is a dictionary containing data information from three modalities: language, 2D, and 3D.（[Details](https://rbler1234.gitbook.io/mmscan-devkit-tutorial#data-access)）

### MMScan  Evaluator

Our evaluation tool is designed to streamline the assessment of model outputs for the MMScan task, providing essential metrics to gauge model performance effectively. We provide three evaluation tools: `VisualGroundingEvaluator`, `QuestionAnsweringEvaluator`, and `GPTEvaluator`. ([Details](https://rbler1234.gitbook.io/mmscan-devkit-tutorial/evaluator))

```bash
from mmscan import MMScan

# (2) The evaluator tool ('VisualGroundingEvaluator', 'QuestionAnsweringEvaluator', 'GPTEvaluator')
from mmscan import VisualGroundingEvaluator, QuestionAnsweringEvaluator, GPTEvaluator
```


### MMScan HVG Challenge Submission

To participate and submit your results in our MMScan Visual Grounding challenge, please refer to the instructions provided on our [test server](https://huggingface.co/spaces/rbler/3d-iou-challenge).
We welcome any feedback — feel free to contact us via [Google email](linjingli@166.com).

## 🏆 MMScan Benchmark

<div align=center>
<img src="assets/mix.png" width=95%>
</div>

### MMScan Visual Grounding Benchmark

| Methods | gTop-1 | gTop-3 | AP<sub>sample</sub> | AP<sub>box</sub> | AR | Release | Download |
|---------|----------------|-----------|---------------------|------------------|----|-------|----|
| ScanRefer | 4.74 | 9.19 | 9.49 | 2.28 | 47.68 | [code](./models/Scanrefer/README.md) | [model](https://drive.google.com/file/d/1C0-AJweXEc-cHTe9tLJ3Shgqyd44tXqY/view?usp=drive_link) | [log](https://drive.google.com/file/d/1ENOS2FE7fkLPWjIf9J76VgiPrn6dGKvi/view?usp=drive_link) |
| MVT | 7.94 | 13.07 | 13.67 | 2.50 | 86.86 | - | - |
| BUTD-DETR  | 15.24 | 20.68 | 18.58 | 9.27 | 66.62 |  - | - |
| ReGround3D  | 16.35 | 26.13 | 22.89 | 5.25 | 43.24 | - | - |
| EmbodiedScan  | 19.66 | 34.00 | 29.30 | **15.18** | 59.96 | [code](./models/EmbodiedScan/README.md) |  [model](https://drive.google.com/file/d/1F6cHY6-JVzAk6xg5s61aTT-vD-eu_4DD/view?usp=drive_link) | [log](https://drive.google.com/file/d/1Ua_-Z2G3g0CthbeBkrR1a7_sqg_Spd9s/view?usp=drive_link) |
| 3D-VisTA | 25.38 | 35.41 | 33.47 | 6.67 | 87.52 |  - | - |
| ViL3DRef | **26.34** | **37.58** | **35.09** | 6.65 | 86.86 | - | - |

### MMScan Question Answering Benchmark

| Methods | Overall | ST-attr | ST-space | OO-attr | OO-space | OR| Advanced | Release | Download |
|---|--------|--------|--------|--------|--------|--------|-------|----|----|
| LL3DA | 45.7 | 39.1 | 58.5 | 43.6 | 55.9 | 37.1 | 24.0| [code](./models/LL3DA/README.md) | [model](https://drive.google.com/file/d/1mcWNHdfrhdbtySBtmG-QRH1Y1y5U3PDQ/view?usp=drive_link) | [log](https://drive.google.com/file/d/1VHpcnO0QmAvMa0HuZa83TEjU6AiFrP42/view?usp=drive_link) |
| LEO |54.6 | 48.9 | 62.7 | 50.8 | 64.7 | 50.4 | 45.9 | [code](./models/LEO/README.md) | [model](https://drive.google.com/drive/folders/1HZ38LwRe-1Q_VxlWy8vqvImFjtQ_b9iA?usp=drive_link)|
| LLaVA-3D |**61.6** | 58.5 | 63.5 | 56.8 | 75.6 | 58.0 | 38.5|- | - |

*Note:* These two tables only show the results for main metrics; see the paper for complete results.

We have released the codes of some models under [./models](./models).

## 📝 TODO List

- \[ \] MMScan annotation and samples for ARKitScenes.
- \[ \] Codes of more MMScan Visual Grounding baselines and Question Answering baselines.
- \[ \] Full release and further updates.
