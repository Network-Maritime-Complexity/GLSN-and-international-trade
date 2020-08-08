# -*- coding: utf-8 -*-
"""
Created on 2020/2/5
Python 3.6

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


if __name__ == "__main__":
    import time
    start_time = time.perf_counter()
    print()
    print('**************************************** RUN TIME WARNING ****************************************')
    print('It needs approximately 70 minutes for the whole experiment.')
    print()
    print('======================================================================================================')
    print('Output:')
    print()
    from src import CalculateExplanatoryVariables
    from src import CalculatePearsonCorrelationCoefficient
    from src import MultivariateLinearRegression
    from src import Validation2017
    from src import TradeValueChanges
    from src import GravityModel

    CalculateExplanatoryVariables.startup()
    CalculatePearsonCorrelationCoefficient.startup()
    MultivariateLinearRegression.startup()
    Validation2017.startup()
    TradeValueChanges.startup()
    GravityModel.startup()

    print('======================================================================================================')
    print()
    print('Code performance: {:.0f}s.'.format(time.perf_counter() - start_time))
