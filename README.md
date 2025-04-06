# ABFR-KAN
This repository contains the implementation of ABFR-KAN. The associated publication for this project is listed below:

["Improving brain disorder diagnosis with advanced brain function representation and Kolmogorov-Arnold Networks,"](https://openreview.net/forum?id=YmUDkDQhCW) by Tyler Ward and Abdullahh-Al-Zubaer Imran. In [MIDL](https://2025.midl.io/), 2025.

ABFR-KAN implements a novel deep learning pipeline for diagnosing ASD from resting state functional magnetic resonance imaging (rs-fMRI) data. This pipeline uses randomized anchor patch selection and iterative patch sampling to create individualized, unbiased representations of brain functional connectivity. Additionally, we introduce a transformer-based classifier with Kolmogorov-Arnold Networks (KANs) replacing traditional MLP layers for better function approximation.

## Abstract
Quantifying functional connectivity (FC), a vital metric for the diagnosis of various brain disorders traditionally relies on the use of a pre-defined brain atlas. However, using such atlases can lead to issues regarding selection bias and lack of regard for specificity. Addressing this, we propose a novel transformer-based classification network (AFBR-KAN) with effective brain function representation to aid in diagnosing autism spectrum disorder (ASD). AFBR-KAN leverages Kolmogorov-Arnold Network (KAN) blocks replacing traditional multi-layer perceptron (MLP) components. Thorough experimentation reveals the effectiveness of AFBR-KAN in improving the diagnosis of ASD under various configurations of the model architecture.

## Model
![Figure](https://github.com/tbwa233/ABFR-KAN/blob/main/images/abfrkanarch6.png)

Below, we highlight key components of the ABFR-KAN pipeline and their benefits.

### Random Anchor Selection
ABFR-KAN avoids structural bias by randomly selecting anchor patches from the gray matter region rather than using fixed grids or atlas-based regions of interest (ROIs). This randomization enhances individual specificity and ensures that FC representations are not constrained by pre-defined anatomical assumptions. The result is a more flexible and generalizable model that captures meaningful subject-specific variation in brain function.

### Iterative Patch Sampling
To capture rich multi-scale functional features, ABFR-KAN employs an iterative sampling strategy. Patches are repeatedly drawn from gray matter using varying sizes, and their FC with anchor patches is computed and aggregated. This process improves anatomical coverage and reduces the impact of any single patch sampling configuration, producing more robust and informative FC representations.

### Kolmogorov-Arnold Networks
Traditional transformer networks rely on MLPs for encoding and classification. ABFR-KAN replaces these with Kolmogorov-Arnold Networks—neural layers that use learnable spline-based activation functions on edges. Inspired by the Kolmogorov-Arnold representation theorem, KANs offer enhanced expressiveness, faster convergence, and better interpretability. Their ability to model complex, high-dimensional relationships makes them especially well-suited for analyzing the subtle patterns involved in diagnosing conditions like ASD.

## Results
A brief summary of our results are shown below. Our ABFR-KAN model is evaluated under various different experimental configurations. In the table, the best scores are bolded and the second-best scores are underlined.
![Results](https://github.com/tbwa233/ABFR-KAN/blob/main/images/abfrkanresults.png)

# How to Use
There are a lot of different experiments that can be run from the code in this repository. Let's start by discussing the baseline. In order to prepare and train the baseline model, follow the following steps:

1. git clone https://github.com/tbwa233/ABFR-KAN.git
2. cd ABFR-KAN
3. cd Data_Preparation
4. python 1.Create_Coordinate_ForAnchor_181217181.py
5. python 2.Create_AnchorRegion_WithGMmask_617361.py
6. cd Data_Preparation
7. python 3.ClcFCMatrix_BasedRandomPatch_Anchor.py
8. cd Code
9. cd data
10. python generate_subjectwise_position.py
11. cd Code
12. python NYU_train.py

When doing this, if you encounter any issues, make sure the paths are what you expect them to be.

So, that was the baseline model. If you want to run the data preparation code again, this time selecting a random anchor patch instead of a fixed one, you can run the code in this order:

1. git clone https://github.com/tbwa233/ABFR-KAN.git
2. cd ABFR-KAN
3. cd Data_Preparation
4. python random_anchor1.py
5. python random_anchor2.py
6. python random_anchor3.py
7. cd Code
8. cd data
9. python generate_subjectwise_position.py
10. cd Code
11. python NYU_train.py

If you want to run the data preparation code again, this time iteratively representing brain function, you can run the code in this order:

1. git clone https://github.com/tbwa233/ABFR-KAN.git
2. cd ABFR-KAN
3. cd Data_Preparation
4. python 1.Create_Coordinate_ForAnchor_181217181.py
5. python 2.Create_AnchorRegion_WithGMmask_617361.py
6. python ibfr.py
7. cd Code
8. cd data
9. python generate_subjectwise_position_it.py
10. cd Code
11. python NYU_train.py

In order to evaluate different architectures, please look at the commented out lines near the top of the NYU_train.py file. All you have to do is just select the one you want, and run the training program.

If you'd like to use element-wise multiplication in place of element-wise addition to combine the function and position embeddings, simply refer to the commented out line near the bottom of each of the architecture files.

# Important
In order to run the Data_Preparation code, you have to have the ABIDE data available to you. The link to download the data is here: http://preprocessed-connectomes-project.org/abide/download.html, simply follow the instructions on the page. The setup you need to use for the data is as follows:

pipeline: dparsf
strategy: nofilt_noglobal
derivative: func_prepoc

Patients are selected from the NYU site with IDs from 0050954 to 0051156.

If you're unable to get the raw data downloaded, I have provided a zip file (https://www.dropbox.com/scl/fi/jkygnqcjv99tnbgtzel1g/fcmatrices_and_subjectwisepositions.zip?rlkey=beeyk6tvt6aa5umbvcko16oat&st=csbgxy1m&dl=0) that you can place in the Code/data folder that will allow you to train the model. Simply unzip the files, and make sure the paths are properly set in the NYU_dataset.py file.

## Acknowledgements
This project was built off the backbone of the [RandomFR](https://github.com/mjliu2020/RandomFR) repository provided by Mengjun Liu, Huifeng Zhang, Mianxin Liu, Dongdong Chen, Zixu Zhuang, and Xin Wang. We greatly thank these authors for open-sourcing their project. We highly recommend checking out their respository, as well as their [paper](https://ieeexplore.ieee.org/document/10440630) published in [IEEE T-MI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=42).

Additionally, we would like to thank the authors of the [Vision-KAN](https://github.com/chenziwenhaoshuai/Vision-KAN) and [FasterKAN](https://github.com/AthanasiosDelis/faster-kan) repositories, whose work was greatly beneficial for providing a guideline for proper KAN integration.
