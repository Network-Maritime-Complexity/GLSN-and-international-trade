# -*- coding: utf-8 -*-
"""
Created on 2020/1/17
Python 3.6

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def gravity_model():
    dataset = '2015'

    def _cal_corr(x, y):
        corr = round(stats.pearsonr(x, y)[0], 3)
        pval = round(stats.pearsonr(x, y)[1], 3)

        return corr, pval

    dataname = 'Gravity_model.xlsx'
    df_btv = pd.read_excel(data_path + dataname)
    y_col = 'BTVij'
    y = np.log(df_btv[y_col])

    x_cols = ['GDPi x GDPj', 'dij']
    x = np.log(df_btv[x_cols])
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    fit_res = model.fit()
    print('******************************************************************')
    print('The in-text result:')
    print("Location in the manuscript text: ")
    print('Subsection titled "Comparison with the gravity model"')
    print('Section titled "Results"')
    print('******************************************************************')
    print()
    print('"The model yielded an adjusted R^2 value of {:.3f}, where the qualified (i, j) pairs were regarded as samples."'.format(fit_res.rsquared_adj))
    print()
    predict_btv = np.exp(fit_res.fittedvalues)
    df_btv['predicted BTVij'] = predict_btv

    df_btv_copy = df_btv.copy()
    tmp = df_btv['Partner ISO']
    df_btv_copy['Partner ISO'] = df_btv['Reporter ISO']
    df_btv_copy['Reporter ISO'] = tmp
    df_btv = pd.concat([df_btv, df_btv_copy])
    df_predicted = df_btv.groupby('Reporter ISO', as_index=False)['predicted BTVij'].sum()
    df_predicted.columns = ['Country Code', 'Estimated TV']

    dataname = 'Regression_Variables_' + dataset + '.xlsx'
    df_tv = pd.read_excel(save_path + dataname)
    df_tv = df_tv[df_tv['GravityModel'] == 'T']
    df_res = pd.merge(df_tv, df_predicted, on='Country Code')
    pr, pval = _cal_corr(df_res['Trade value'], df_res['Estimated TV'])
    r2 = pr ** 2
    n = 144
    k = 1
    adj_r2 = 1 - ((1-r2)*(n-1) / (n-k-1))
    print(
        '"The Pearson correlation coefficient between the empirical and estimated trade value of countries was equal to {:.3f}, resulting in an adjusted R^2 value of {:.3f}."'.format(pr, adj_r2))
    print()


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


def tables3():
    dataset = '2015'
    dataname = 'Regression_Variables_' + dataset + '.xlsx'
    data = pd.read_excel(save_path + dataname)
    data = data[data['GravityModel'] == 'T']
    data.rename(columns={'GLSN betweenness (Lmax=2)': 'Gb', 'GLSN connectivity (Edge weight=None)': 'Gc',
                         'LSCI': 'L', 'Freeman betweenness': 'Fb'}, inplace=True)

    y_col = 'Trade value'
    y = data[y_col]
    y = z_score(y)

    param_cols = ['Gc', 'Gb', 'Fb', 'L']
    N = len(param_cols)

    list_adj_r2 = []
    list_r2 = []
    list_fpval = []
    list_vif = []
    list_aic = []
    list_col_idx = []
    list_nobs = []
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

            list_adj_r2.append(fit_res.rsquared_adj)
            list_r2.append(fit_res.rsquared)
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

    cols = ['variable' + str(i) for i in range(1, N + 1)]
    cols.insert(0, 'const')
    df_vif = pd.DataFrame(list_vif, columns=cols)

    v_cols = [col for col in df_vif.columns if 'variable' in col]
    max_vif = df_vif[v_cols].max(axis=1)
    df_res = pd.DataFrame()
    df_res['Adjusted R2'] = list_adj_r2
    # df_res['R2'] = list_r2
    df_res['p-value'] = list_fpval
    df_res['AIC'] = list_aic
    df_res['Max VIF'] = max_vif
    df_res['# observations'] = list_nobs
    df_res.index = pd.Series(list_col_idx)

    sheetname = 'Supplementary Table S3'
    filename = 'Supplementary Table S3. Results for multivariate linear regressions....xlsx'
    df_res.to_excel(save_path + filename, sheet_name=sheetname,
                    float_format='%.3f', index=True,
                    index_label='Explanatory variable')
    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    print()


def table3():
    data = pd.read_excel('data/Gravity_model_lsbci.xlsx')
    data = data.dropna()

    y_col = 'BTVij'
    y = data[y_col]
    y = np.log(y)

    models = [['ln(GDPi x GDPj)', 'ln(dij)'], ['ln(GDPi x GDPj)', 'ln(dij)', 'ln(LSBCIij)'],
              ['ln(GDPi x GDPj)', 'ln(dij)', 'ln(Gbi x Gbj)'],
              ['ln(GDPi x GDPj)', 'ln(dij)', 'ln(LSBCIij)', 'ln(Gbi x Gbj)']]
    N = 4

    list_nobs = []
    list_adj_r2 = []
    list_fpval = []
    list_vif = []
    list_aic = []
    list_col_idx = []
    sep = ', '
    for model in models:
        xname = sep.join(model)
        list_col_idx.append(xname)

        x = z_score(data[model])
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

    sheetname = "Table 3"
    filename = 'Table 3. Multivariate regression results for gravity models....xlsx'
    df_res.to_excel(save_path + filename, sheet_name=sheetname,
                    float_format='%.3f', index=True, index_label='Explanatory variable')
    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    print()


def tables6():
    data = pd.read_excel('data/Gravity_model_lsbci.xlsx')
    y_col = 'BTVij'
    y = data[y_col]
    y = np.log(y)

    models = [['ln(GDPi x GDPj)', 'ln(dij)'], ['ln(GDPi x GDPj)', 'ln(dij)', 'ln(LSBCIij)'],
              ['ln(GDPi x GDPj)', 'ln(dij)', 'ln(Gci x Gcj)'],
              ['ln(GDPi x GDPj)', 'ln(dij)', 'ln(LSBCIij)', 'ln(Gci x Gcj)']]
    N = 4

    list_nobs = []
    list_adj_r2 = []
    list_fpval = []
    list_vif = []
    list_aic = []
    list_col_idx = []
    sep = ', '
    for model in models:
        xname = sep.join(model)
        list_col_idx.append(xname)

        x = z_score(data[model])
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

    sheetname = "Supplementary Table S6"
    filename = 'Supplementary Table S6. Multivariate regression results for gravity models....xlsx'
    df_res.to_excel(save_path + filename, sheet_name=sheetname,
                    float_format='%.3f', index=True, index_label='Explanatory variable')
    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    print()


def startup():
    gravity_model()
    tables3()
    table3()
    tables6()
