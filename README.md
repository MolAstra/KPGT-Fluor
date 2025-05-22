# KPGT-Fluor

Fluorescent dyes play a crucial role in various fields, including biology, chemistry, and material science, due to their unique optical properties. Accurately predicting these properties is essential for the rational design of high-performance dyes. The introduction of Deep Learning (DL) has revolutionized Artificial Intelligence-driven Drug Design (AIDD), enabling powerful tools for molecular property prediction. Building on foundational models in molecular property prediction, we introduce KPGT-Fluor, a novel adaptation of the Knowledge-guided Pre-training of Graph Transformer (KPGT) framework, specifically tailored for fluorescent dye property prediction. While the original KPGT framework was developed for general molecular property estimation, KPGT-Fluor extends its capabilities by integrating solvent representations, allowing the model to better capture environmental effects that influence dye behavior. KPGT-Fluor achieves strong predictive performance, with Root Mean Square Errors (RMSE) of 18.91 nm and 18.56 nm for absorption and emission wavelengths, respectively. For the logarithm of the extinction coefficient and quantum yield, the RMSE values are 0.159 and 0.126, demonstrating high accuracy. Additionally, the model exhibits robust generalization across multiple downstream dye datasets. These capabilities make KPGT-Fluor a powerful and versatile tool for the data-driven design of next-generation fluorescent dyes.

![KPGT-Fluor](./framework.png)

## Setup

**1. Install the KPGT Framework**

```bash
git clone https://github.com/MolAstra/KPGT-Fluor.git
cd KPGT
mamba env create  # CUDA11.3, torch1.10
mamba activate KPGT  # 

pip install transformers
```

**2. Download the dataset and pretrained model into corresponding folders**

For further details, please refer to the [KPGT](https://github.com/lihan97/KPGT) repository. And in this paper we reuse the pretrained model and dataset from the KPGT repository.

## Experiments

```bash
bash train.sh  # train KPGT-Fluor on consolidation dataset
bash predict.sh  # predict KPGT-Fluor on consolidation dataset

bash train_external.sh  # train KPGT-Fluor on external dataset with trained KPGT-Fluor on FluorDB
bash predict_external.sh  # predict KPGT-Fluor on external dataset with trained KPGT-Fluor on FluorDB
bash predict_direct.sh  # predict KPGT-Fluor on external dataset with zero-shot KPGT-Fluor

python train_ml.py  # for all datasets consolidation, cyanine, xanthene
python predict_ml.py  # for all datasets consolidation, cyanine, xanthene

python predict_ml_direct.py  # predict direct on external dataset with zero-shot ml
```

## Useful Jupyter Notebooks for Visualization

please refer `notebooks` and `plots` folder for more details. And also you can find some `case_study` in the paper.

```bash
bash case_study.sh
```
