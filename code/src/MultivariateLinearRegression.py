#! python3
# -*- coding: utf-8 -*-
"""
Created on 2019/5/9

Note:

Code Performance:


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


def my_zip(*args, fillvalue=None):
    from itertools import repeat
    # zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-
    iterators = [iter(it) for it in args]
    num_active = len(iterators)
    if not num_active:
        return
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                value = next(it)
            except StopIteration:
                num_active -= 1
                if not num_active:
                    return
                iterators[i] = repeat(fillvalue)
                value = fillvalue
            values.append(value)
        yield list(values)


def table2(data):

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

    sheetname = "Table 2"
    filename = 'Table 2. Results for multivariate linear regressions....xlsx'
    df_res.to_excel(save_path + filename, sheet_name=sheetname,
                    float_format='%.3f', index=True, index_label='Explanatory variable')
    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    print()


def tables1(data):
    y_cols = ['Export value', 'Import value', 'Net export', 'GDP']

    list_res = []
    for y_col in y_cols:
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
        list_res.append(df_res)

    df_res_all = pd.concat([list_res[0], list_res[1], list_res[2], list_res[3]], axis=1, keys=y_cols)
    df_res_all['Max VIF'] = max_vif
    df_res_all['# observations'] = list_nobs
    df_res_all.index = pd.Series(list_col_idx)
    filename = 'Supplementary Table S1. Results for multivariate linear regressions when the export value....xlsx'
    sheetname = 'Supplementary Table S1'
    df_res_all.to_excel(save_path + '/' + filename, sheet_name=sheetname, float_format='%.3f',
                        freeze_panes=(3, 1), index=True,
                        index_label='Explanatory variable')
    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    print()


def tables2():
    models = {'2015': [['Trade value', 'Gc', 'Fb'], ['Trade value', 'Gc', 'Gb'],
                       ['Export value', 'Gc', 'Fb'], ['Export value', 'Gc', 'Gb', 'L'],
                       ['Import value', 'Gc', 'Fb'], ['Import value', 'Gc', 'Gb'],
                       ['Net export', 'Gc', 'Gb', 'L'],
                       ['GDP', 'Gc', 'L'], ['GDP', 'Gc', 'Fb', 'L']],
              '2017': [['Trade value', 'Gc', 'Fb'], ['Trade value', 'Gc', 'Gb']]}

    list_nobs = []
    list_col_idx = []

    dict_conf_ints = {}
    dict_param_coef = {}
    dict_tpval = {}
    ix = 0

    list_dependent_variable = []
    for year, model in models.items():
        dataname = 'Regression_Variables_' + year + '.xlsx'
        data = pd.read_excel(save_path + dataname)
        data.rename(columns={'GLSN betweenness (Lmax=2)': 'Gb', 'GLSN connectivity (Edge weight=None)': 'Gc',
                               'LSCI': 'L', 'Freeman betweenness': 'Fb'}, inplace=True)

        for mm in model:
            y_col = mm[0]

            dependent_variable = y_col + ' (' + year + ')'
            list_dependent_variable.append(dependent_variable)

            y = data[y_col]
            y = z_score(y)
            x_cols = mm[1:]
            sep = ', '
            xname = sep.join(x_cols)
            list_col_idx.append(xname)
            x = data[x_cols]
            x = z_score(x)
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            fit_res = model.fit()
            # print(fit_res.summary())

            list_nobs.append(fit_res.nobs)
            coefs = fit_res.params.values.tolist()
            param = fit_res.params.index.tolist()
            tpvals = fit_res.pvalues.values.tolist()

            dict_param_coef[ix] = dict(zip(param, coefs))
            dict_tpval[ix] = dict(zip(param, tpvals))

            lower_conf_int = round(fit_res.conf_int(alpha=0.05)[0], 3).values.tolist()
            upper_conf_int = round(fit_res.conf_int(alpha=0.05)[1], 3).values.tolist()
            conf_int = list(my_zip(lower_conf_int, upper_conf_int))
            dict_conf_ints[ix] = dict(zip(param, conf_int))

            ix += 1

    df_coef = pd.DataFrame(dict_param_coef).T
    df_conf_ints = pd.DataFrame(dict_conf_ints).T
    df_tpval = pd.DataFrame(dict_tpval).T

    cols = ['Gc', 'Gb', 'Fb', 'L']
    df_coef = df_coef[cols]
    df_tpval = df_tpval[cols]
    df_conf_ints = df_conf_ints[cols]

    for col in df_tpval.columns:
        for ix in df_tpval.index:
            df_tpval.loc[ix, col] = pval_marker(df_tpval.loc[ix, col])

    df_res = pd.concat([df_coef, df_tpval, df_conf_ints], axis=1,
                       keys=['Coefficient', 'p-value', 'confidence interval'])

    df_res['# observations'] = list_nobs

    df_res['Dependent variable'] = list_dependent_variable
    df_res['Explanatory variable'] = pd.Series(list_col_idx)
    df_res = df_res.set_index(['Dependent variable', 'Explanatory variable'])
    df_res.fillna('—', inplace=True)

    filename = 'Supplementary Table S2. Coefficients for representative multivariate linear regression models....xlsx'
    sheetname = 'Supplementary Table S2'
    df_res.to_excel(save_path + '/' + filename, sheet_name=sheetname, freeze_panes=(3, 2), float_format='%.3f',
                    index=True)
    print('The result file "{}" saved at: "{}"'.format(filename, save_path))
    print()


def startup():
    dataset = '2015'
    dataname = 'Regression_Variables_' + dataset + '.xlsx'
    data = pd.read_excel(save_path + dataname)
    data.rename(columns={'GLSN betweenness (Lmax=2)': 'Gb', 'GLSN connectivity (Edge weight=None)': 'Gc',
                         'LSCI': 'L', 'Freeman betweenness': 'Fb'}, inplace=True)
    table2(data)
    tables1(data)
    tables2()
