This repoistory contains the code for statistical analyses and figures, along with the computational models used in "Sex-dependent effects on object interaction, territorial, and cooperative behaviors in C57BL/6J mice using machine learning." This project was executed in The SOCIAL Neurobiology Lab at Arizona State University under Dr. Jessica Verpeut (https://verpeutlab.org/).

**A Guide to the Repository:**

Each folder contains an Overview document that summarizes its contents. This information is also listed below.

**Manuscript Figures** - contains the Adobe Illustrator files for each figure in the manuscript; contains additional folder titled **Colab Files (for graphs and statistics)**

**Colab Files (for graphs and statistics)** - contains code to generate figures and run statistical analyses, along with the .csv files containing data

**Three Chamber** - contains the training config file and skeleton for model; contains code to evaluate model and to extract data from videos that were predicted on

**Open Field** - contains three folders: **SLEAP**, **SimBA**, and **Tracking Object Interaction (TOI)**

**SLEAP** - folders **centered instance** and **centroid** contain training config for the centered instance and centroid, respectively (multi-animal SLEAP models have both a centered instance and a centroid); skeleton for SLEAP model; contains code to evaluate model and to extract data from videos that were predicted on

**SimBA**" - contains .sav files for each model created in SimBA

**Tracking Object Interaction (TOI)** - contains code for TOI and lists the order to run the scripts in; contains example video generated after preprocessing
