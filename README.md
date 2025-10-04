# Margin-Adaptive Direct Preference Optimization (MADPO)

This repository contains the official code and resources for the paper: **"Margin Adaptive DPO: Leveraging Reward Model for Granular Control in Preference Optimization."**

## Abstract

Direct Preference Optimization (DPO) has emerged as a simple and effective method for aligning large language models. However, its reliance on a fixed temperature parameter leads to suboptimal training on diverse preference data, causing overfitting on easy examples and under-learning from informative ones. While IPO addresses general overfitting, its uniform regularization can be overly conservative. The more targeted approach of $\beta$-DPO suffers from its own limitations: its batch-level adaptation applies a single, compromised temperature to mixed-margin pairs, its linear update rule can produce unstable negative $\beta$ values, and its filtering mechanism discards potentially useful training signals.

In this work, we introduce Margin-Adaptive Direct Preference Optimization (MADPO), a novel method that provides a stable, data-preserving, and instance-level solution. MADPO employs a practical two-step approach: it first trains a reward model to estimate preference margins and then uses these margins to apply a continuous, adaptive weight to the DPO loss for each individual training sample. This re-weighting scheme creates an effective target margin that is amplified for hard pairs and dampened for easy pairs, allowing for granular control over the learning signal.

We provide a comprehensive theoretical analysis, proving that MADPO has a well-behaved optimization landscape and is robust to reward model estimation errors. We validate our theory with experiments on a sentiment generation task, where MADPO consistently and significantly outperforms strong baselines across datasets of varying quality. It achieves performance gains of up to +33.3% on High Quality data and +10.5% on Low Quality data over the next-best method. Our results establish MADPO as a more robust and principled approach to preference alignment.

## Main Results

Our experiments show that MADPO consistently and significantly outperforms standard baselines across all three data quality tiers. The chart below visualizes the main findings, with annotations highlighting the percentage improvement over the next-best method, `β`-DPO.

![Main Results Chart](figure/model_performance_chart_sorted.png)

The key takeaways from our results are:
- **Superior Performance**: MADPO achieves the highest average reward in all settings.
- **Robustness**: MADPO's performance is the most stable, showing minimal degradation as data quality decreases, a scenario where other methods falter.
- **Quantitative Gains**: The performance improvement is substantial, reaching up to **+33.3%** over the next-best baseline on the High Quality dataset.

The table below provides the precise mean reward scores for each method and dataset.

| Method | High Quality | Medium Quality | Low Quality |
| :--- | :--- | :--- | :--- |
| **MADPO** | **2.23** | **2.23** | **1.95** |
| `β`-DPO | 1.67 | 1.84 | 1.76 |
| DPO | 1.62 | 1.71 | 1.48 |
| IPO | 0.35 | 0.31 | 0.10 |

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[sitano1004]/Margin-Apative-Direct-Preference-Optimization.git
    cd MADPO
    ```

2.  **Create and activate a conda environment:**
    ```bash
    conda create -n madpo python=3.10
    conda activate madpo
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    The preference datasets used for training are available on Kaggle. Please download the data and place it in a `./data/` directory.
    
    **Kaggle URL**: `https://www.kaggle.com/datasets/sirano1004/madpo-data-set/`

---

## How to Reproduce the Experiments

The experiments are conducted via a series of Jupyter Notebooks, which should be run in the following order to reproduce the full pipeline. All notebooks are located in the `notebooks/` directory.

1.  **Supervised Fine-Tuning (SFT)**
    Run `SFT.ipynb` to fine-tune the base Gemma model. This creates the base SFT policy used in subsequent steps.

2.  **Data Generation**
    Run `Data Generation.ipynb` to use the SFT model to generate the preference pairs and create the three distinct datasets (High, Medium, and Low quality).

3.  **Reward Model Training**
    Run `Reward Model.ipynb` to train the reward models on the datasets created in the previous step.

4.  **Preference Alignment (DPO, IPO, `β`-DPO, MADPO)**
    Run the corresponding notebook for each alignment method to fine-tune the SFT policy:
    * `DPO.ipynb`
    * `IPO.ipynb`
    * `beta DPO.ipynb`
    * `MADPO.ipynb`

5.  **Evaluation**
    The evaluation logic to generate the final results and figures is included at the end of each of the alignment notebooks (e.g., in `MADPO.ipynb`).