# Accelerating_Conformal_Prediction_via_Approximate_Leave-One-Out
Code for the numerical experiments section of the paper ''Accelerating Conformal Prediction via Approximate Leave-One-Out''.


The simulation folder contains MATLAB scripts corresponding to the Synthetic data section of the paper.

All scripts are fully self-contained and can be directly executed in MATLAB without additional dependencies.
Each file represents one specific simulation setting discussed in the paper.

File Descriptions:

jkp_L01.m
Simulation code for Jackknife+ and Fast Jackknife+ when λ = 0.1.

jkp_L1.m
Simulation code for Jackknife+ and Fast Jackknife+ when λ = 1.

jkmm_L01.m
Simulation code for Jackknife-minmax and Fast Jackknife-minmax when λ = 0.1.

jkmm_L1.m
Simulation code for Jackknife-minmax and Fast Jackknife-minmax when λ = 1.


The real data folder contains the data and code for conducting experiments on the Concrete Compressive Strength Dataset (Yeh, 1998) and the Energy Efficiency Dataset (Tsanas & Xifara, 2012).
The code for the experiments on these two datasets is provided in the files concrete.m and energy.m, respectively.


Usage:
Simply open any .m file in MATLAB and run it.
Each script will automatically generate the results corresponding to the numerical experiments described in the paper.
