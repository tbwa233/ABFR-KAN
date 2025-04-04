# ABFR-KAN
This repository contains the implementation of ABFR-KAN. The associated publication for this project is listed below:

["Improving brain disorder diagnosis with advanced brain function representation and Kolmogorov-Arnold Networks,"](https://openreview.net/forum?id=YmUDkDQhCW) by Tyler Ward and Abdullahh-Al-Zubaer Imran. In [MIDL](https://2025.midl.io/), 2025.

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
