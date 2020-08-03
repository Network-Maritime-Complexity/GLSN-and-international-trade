#! python3
# -*- coding: utf-8 -*-
"""

Created on 2019/7/31

Code Performance:
 

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def cal_pearson_r(x, y):
    pr = stats.pearsonr(x, y)
    corr = round(pr[0], 3)
    pval = pr[1]
    pmarker = pval_marker(pval)

    return corr, pval, pmarker


def pval_marker(pval):
    if pval < 0.001:
        pmarker = '**'
    elif 0.001 <= pval < 0.01:
        pmarker = '*'
    elif 0.01 <= pval < 0.05:
        pmarker = '+'
    elif pval >= 0.05:
        pmarker = np.nan
    else:
        pmarker = pval
    return pmarker


def cal_pr1(data):
    x_cols = [
        'GLSN connectivity (Edge weight=None)',
       'GLSN connectivity (Edge weight=1)',
       'GLSN connectivity (Edge weight=1/(n-1))',
       'GLSN connectivity (Edge weight=1/[n(n-1)/2])',
       'GLSN connectivity (Edge weight=C)',
       'GLSN connectivity (Edge weight=C/(n-1))',
       'GLSN connectivity (Edge weight=C/[n(n-1)/2])',
        'Normalized GLSN connectivity (Edge weight=None)',
        'Normalized GLSN connectivity (Edge weight=1)',
        'Normalized GLSN connectivity (Edge weight=1/(n-1))',
        'Normalized GLSN connectivity (Edge weight=1/[n(n-1)/2])',
        'Normalized GLSN connectivity (Edge weight=C)',
        'Normalized GLSN connectivity (Edge weight=C/(n-1))',
        'Normalized GLSN connectivity (Edge weight=C/[n(n-1)/2])',
      'GLSN betweenness (Lmax=2)', 'GLSN betweenness (Lmax=3)',
       'GLSN betweenness (Lmax=4)', 'GLSN betweenness (Lmax=5)',
      'Freeman betweenness', 'normalized Freeman betweenness', 'LSCI']

    y_col = 'Trade value'
    y = data[y_col]

    list_corr = []
    list_pval = []
    list_pmk = []
    for x_col in x_cols:
        x = data[x_col]
        corr = cal_pearson_r(x, y)[0]
        pval = cal_pearson_r(x, y)[1]
        pmk = cal_pearson_r(x, y)[2]
        list_corr.append(corr)
        list_pval.append(pval)
        list_pmk.append(pmk)
    df_res = pd.DataFrame()
    df_res['Variable'] = x_cols
    df_res['r'] = list_corr
    df_res['p-value'] = list_pval
    df_res['superscript'] = list_pmk

    sheet_name = 'Table 1'
    filename = 'Table 1. Pearson correlation coefficient between the trade value and each ' \
               'explanatory variable....xlsx'
    writer = pd.ExcelWriter(save_path + filename)
    df_res.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    writer.close()
    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    print()


def cal_pr2(data):
    data.rename(columns={'GLSN connectivity (Edge weight=None)': 'GLSN connectivity',
                         'GLSN betweenness (Lmax=2)': 'GLSN betweenness'}, inplace=True)
    x_cols = ['GLSN connectivity', 'GLSN betweenness', 'Freeman betweenness', 'LSCI']

    y_cols = ['Export value', 'Import value', 'Net export', 'GDP']
    print('******************************************************************')
    print('The in-text result:')
    print('Subsection titled "Estimating countries\' trade values by multivariate linear regression"')
    print('Section titled "Results"')
    print('******************************************************************')
    for y_col in y_cols:
        print('=========== Pearson correlation coefficient between the {} and each explanatory variable ==========='.format(y_col))
        y = data[y_col]

        list_corr = []
        list_pval = []
        list_pmk = []
        for x_col in x_cols:
            x = data[x_col]
            corr = cal_pearson_r(x, y)[0]
            pval = cal_pearson_r(x, y)[1]
            pmk = cal_pearson_r(x, y)[2]
            list_corr.append(corr)
            list_pval.append(pval)
            list_pmk.append(pmk)
        df_res = pd.DataFrame()
        df_res['Variable'] = x_cols
        df_res['r'] = list_corr
        df_res['p-value'] = list_pval
        df_res['superscript'] = list_pmk
        print(df_res)
        print()


def startup():
    datasets = ['2015']
    for dataset in datasets:
        dataname = 'Regression_Variables_' + dataset + '.xlsx'
        data = pd.read_excel(save_path + dataname)
        cal_pr1(data)
        cal_pr2(data)
