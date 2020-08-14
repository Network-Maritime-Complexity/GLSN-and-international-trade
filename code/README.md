# Estimating international trade status of countries from global liner shipping networks

# Overview

The repository allows one to reproduce the results reported in the manuscript. The repository is organized as follows:

| Subdirectory | Description |
| --- | --- |
| **data** | This folder contains all the data adopted and generated in our study. |
| **expected output** | This folder contains the expected results of the analysis. |
| **src** | This folder contains all the scripts used for calculation, including:<br><li>the script "[CalculateExplanatoryVariables.py](./src/CalculateExplanatoryVariables.py)" for calculating the explanatory variables,</li> <li>the script "[CalculatePearsonCorrelationCoefficient.py](./src/CalculatePearsonCorrelationCoefficient.py)" for calculating the Pearson correlation coefficients,</li> <br><li>the script "[MultivariateLinearRegression.py](./src/MultivariateLinearRegression.py)" for running the multivariate linear regressions,</li> <br><li>the script "[Validation2017.py](./src/Validation2017.py)" for validation using the dataset of 2017,</li> <br><li>the script "[TradeValueChanges.py](./src/TradeValueChanges.py)" for estimating countries’ trade value changes by the GLSN betweenness,</li> <br><li>the script "[GravityModel.py](./src/GravityModel.py)" for comparison with the gravity model.</li> |
| **output** | After running a script, you will get a folder named "output". Results will be saved in this folder. Please find out below [How to Use](#How-to-Use). |

# System Requirements 

## OS Requirements

These scripts have been tested on *Windows10* operating system.

### Installing Python on Windows

Before setting up the package, users should have Python version 3.6 or higher, and several packages set up from Python 3.6. The latest version of python can be downloaded from the official website: https://www.python.org/

## Hardware Requirements 

The package requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 4 GB of RAM. For optimal performance, we recommend a computer with the following specs:

RAM: 8+ GB  
CPU: 4+ cores, 3.4+ GHz/core

The runtimes are generated in a computer with the recommended specs (8 GB RAM, 4 cores@3.4 GHz).

# Installation Guide

## Package dependencies

Users should install the following packages prior to running the code:

```
networkx==2.3
numpy==1.17.4
openpyxl==2.6.2
pandas==0.25.0
scipy==1.3.1
statsmodels==0.10.1
xlrd==1.2.0
xlwt==1.3.0
```

For a *Windows10* operating system, users can install the packages as follows:

To install all the packages, open the *cmd* window in the root folder and type:

```
pip install -r requirements.txt
```

To install only one of the packages, type:

```
pip install pandas==0.25.0
```

# How to Use

The script [`main.py`](main.py) is used for reproducing the results reported in the manuscript. Open the *cmd* window in the root folder, then run the following command:

```
python main.py
```

The results will be saved in a folder called "output".

### Code performance

It takes approximately 70 minutes to reproduce the results reported in the manuscript, using a computer with a recommendable spec (8 GB RAM, 4 cores@3.4 GHz).

# Contact

* Mengqiao Xu: <stephanie1996@sina.com>
* Qian Pan: <qianpan_93@163.com>

# Acknowledgement

We appreciate a lab member for carefully testing the code:

- Jia Song: <songjiavv@163.com>
