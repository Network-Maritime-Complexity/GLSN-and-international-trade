# Estimating international trade status of countries from global liner shipping networks

# Overview

The repository allows one to reproduce the quantitative results reported in the manuscript. The repository is organized as follows:

| Subdirectory | Description |
| --- | --- |
| **data** | This folder contains all the data adopted and generated in our study. |
| **expected output** | The expected results of the analysis are saved in the [expected output](./expected%20output) folder. |
| **src** | This folder contains all the scripts used for calculation, including the script "[CalculateExplanatoryVariables.py](./src/CalculateExplanatoryVariables.py)" for calculating the explanatory variables, the script "[CalculatePearsonCorrelationCoefficient.py](./src/CalculatePearsonCorrelationCoefficient.py)" for calculating the Pearson correlation coefficients, the script "[MultivariateLinearRegression.py](./src/MultivariateLinearRegression.py)" for running the multivariate linear regressions, the script "[Validation2017.py](./src/Validation2017.py)" for validation using the dataset of 2017, the script "[TradeValueChanges.py](./src/TradeValueChanges.py)" for estimating countries’ trade value changes by the GLSN betweenness, and the script "[GravityModel.py](./src/GravityModel.py)" for comparison with the gravity model. |
| **output** | After running a script, you will get a folder named "output". Results will be saved in this folder. Please find out below [How To Use](#How-To-Use). |
| **data/publicly available online datasets** | The publicly available online datasets include: [UNCTAD maritime transport indicators.xlsx](./data/publicly%20available%20online%20datasets/UNCTAD%20maritime%20transport%20indicators.xlsx) (available at [UNCTAD](https://unctadstat.unctad.org/wds/ReportFolders/reportFolders.aspx), accessed on 9 August 2019 and 16 July 2020), [Trade statistics.xlsx](./data/publicly%20available%20online%20datasets/Trade%20statistics.xlsx) (available at [UN Comtrade](https://comtrade.un.org/data/), accessed on 4 July 2018), [GDP at current prices.xlsx](./data/publicly%20available%20online%20datasets/GDP%20at%20current%20prices.xlsx) (available at [UNdata](https://data.un.org/), accessed on 14 May 2019), [GDP (current US$).xlsx](./data/publicly%20available%20online%20datasets/GDP%20(current%20US%24).xlsx) (available at [World Bank](https://data.worldbank.org/), accessed on 6 May 2019), and [Merchandise imports and exports (current US$).xlsx](./data/publicly%20available%20online%20datasets/Merchandise%20imports%20and%20exports%20(current%20US%24).xlsx) (available at [World Bank](https://data.worldbank.org/), accessed on 6 May 2019 and 2 January 2020). |

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

If using computers with a *Windows10* operating system, users can install the packages in the following ways:

To install all the packages, please open the *cmd* window in the root folder and type:

```
pip install -r requirements.txt
```

To install only one of the packages, type:

```
pip install pandas==0.25.0
```

# How To Use

The script [`main.py`](main.py) is used to reproduce the quantitative results reported in the manuscript. Please open the *cmd* window in the root folder, then run the following command:

```
python main.py
```

After performing the code, the results will be saved in a folder called "output". All the results will be saved in the folder "output".

### Code performance

It will take approximately 70 minutes to reproduce the quantitative results reported in the manuscript, in a computer with the recommended specs (8 GB RAM, 4 cores@3.4 GHz).

# Contact

* Mengqiao Xu: <stephanie1996@sina.com>
* Qian Pan: <qianpan_93@163.com>

# Acknowledgement

We appreciate a lab member for carefully testing the code：

- Jia Song: <songjiavv@163.com>
