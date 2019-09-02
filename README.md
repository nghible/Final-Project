## This repository contains the code and data for my "Machine Learning for Handling Missing Data in Wearable Electromyographic Systems" final project.

The project can be summarized as follows. First, the missing data is simulated under two different assumptions: Missing Completely at Random, and Missing Under Gait Assumptions. For each assumption, this project implements five different imputation methods: zero-imputation, mean-imputation, K-Nearest Neighbors, Gaussian Process, and Predictive Mean Matching on the raw EMG signal and study their effectiveness.

The final report is available upon request.

#### TABLE OF CONTENTS:

1. FuncAndParams.py - This file contains functions and parameters needed to run the rest of the files. Please run this file first.

2. GPboundssim.py - This file contains the simulation to find the optimal bounds for Gaussian Process length scale hyperparameters. Please run this file second.
3. MainSimulation.py - This file contain code that generate results when data under missing-completely-at-random assumptions.

4. GaitMISim.py - This file contain code that generate results when data under gait assumptions.

5. visknndistanalysis.py - This file contains the visualizations, and distributional analysis in the Results section.

6. PCAvsLDA.py - This file contains the simulation to determine between LDA and PCA.

7. CompAna.py - This file generate running time analysis.

8. The rest are data files. Please run the codes in the same fold with the data files.
