# -*- coding: utf-8 -*-
"""
Created on 2020/2/4
Python 3.6

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def z_score(x):
    x = (x - np.mean(x, axis=0)) / np.std(x, ddof=1, axis=0)
    return x


def cal_pearson_r(x, y):
    pr = stats.pearsonr(x, y)
    corr = round(pr[0], 3)
    pval = pr[1]
    pmarker = pval_marker(pval)
    return corr, pmarker


def pval_marker(pval):
    if pval < 0.001:
        pmarker = '**'
    elif 0.001 <= pval < 0.01:
        pmarker = '*'
    elif 0.01 <= pval < 0.05:
        pmarker = '+'
    elif pval >= 0.05:
        pmarker = 'NaN'
    else:
        pmarker = pval
    return pmarker


def tables4():
    dataset = '2017'
    dataname = 'Regression_Variables_' + dataset + '.xlsx'
    data = pd.read_excel(save_path + dataname)
    data.rename(columns={'GLSN betweenness (Lmax=2)': 'Gb', 'GLSN connectivity (Edge weight=None)': 'Gc',
                           'LSCI': 'L', 'Freeman betweenness': 'Fb'}, inplace=True)
    y_col = 'Trade value'
    y = data[y_col]
    y = z_score(y)

    param_cols = ['Gc', 'Gb', 'Fb', 'L']
    N = len(param_cols)

    list_nobs = []
    list_adj_r2 = []
    list_fpval = []
    list_vif = []
    list_aic = []
    list_col_idx = []
    for n in np.arange(1, N+1, 1):
        x_cols = list(itertools.combinations(param_cols, r=int(n)))
        for x in x_cols:
            sep = ', '
            xname = sep.join(x)
            list_col_idx.append(xname)
            xs = list(x)
            x = z_score(data[xs])
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            fit_res = model.fit()
            # print(fit_res.summary())

            list_adj_r2.append(fit_res.rsquared_adj)
            f_pvalue = pval_marker(fit_res.f_pvalue)
            list_fpval.append(f_pvalue)
            list_nobs.append(fit_res.nobs)

            list_vif.append([round(variance_inflation_factor(X.values, i), 2) for i in range(X.shape[1])])

            # AIC
            predict_y = fit_res.predict()
            RSS = sum((y - predict_y) ** 2)
            num_obs = fit_res.nobs
            K = fit_res.df_model + fit_res.k_constant
            aic = round(num_obs * np.log(RSS / num_obs) + 2 * K, 2)
            list_aic.append(aic)

    cols = ['variable' + str(i) for i in range(1, N+1)]
    cols.insert(0, 'const')

    df_vif = pd.DataFrame(list_vif, columns=cols)

    v_cols = [col for col in df_vif.columns if 'variable' in col]
    max_vif = df_vif[v_cols].max(axis=1)

    df_res = pd.DataFrame()
    df_res['Adjusted R2'] = list_adj_r2
    df_res['p-value'] = list_fpval
    df_res['AIC'] = list_aic
    df_res['Max VIF'] = max_vif
    df_res['# observations'] = list_nobs
    df_res.index = pd.Series(list_col_idx)

    sheetname = "Supplementary Table S4"
    filename = 'Supplementary Table S4. Results for multivariate linear regressions when the dependent variable is ' \
               'the trade value in 2017....xlsx'
    df_res.to_excel(save_path + filename, sheet_name=sheetname,
                    float_format='%.3f', index=True, index_label='Explanatory variable')
    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    print()


def startup():
    tables4()
