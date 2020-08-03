#! python3
# -*- coding: utf-8 -*-
"""

Created on: 2020/1/16

Code Performance:
 

@author: Qian Pan
@e-mail: qianpan_93@163.com
"""


from configure import *


def calculate_glsn_connectivity(df_edges, df_nodes):
    num_ports = df_nodes.groupby('Country Code', as_index=False)['id'].count()
    dict_num_ports = dict(zip(num_ports['Country Code'], num_ports['id']))

    df_edges = df_edges[df_edges['CountryCode_port1'] != df_edges['CountryCode_port2']]

    df_edges_copy = df_edges.copy()
    s_cols = [col for col in df_edges.columns if 'port1_id' in col or 'CountryCode_port1' in col]
    t_cols = [col for col in df_edges.columns if 'port2_id' in col or 'CountryCode_port2' in col]
    tmp = df_edges[s_cols]
    df_edges_copy[s_cols] = df_edges[t_cols]
    df_edges_copy[t_cols] = tmp
    df_edges = pd.concat([df_edges, df_edges_copy], axis=0)

    ew_cols = [col for col in df_edges.columns if 'Edge weight=' in col]
    df_gc = df_edges.groupby('CountryCode_port1', as_index=False)[ew_cols].sum()

    df_ew_none = df_edges.groupby('CountryCode_port1', as_index=False)['CountryCode_port2'].count()
    df_gc = pd.merge(df_gc, df_ew_none, on=['CountryCode_port1'])
    df_gc.rename(columns={'CountryCode_port2': 'GLSN connectivity (Edge weight=None)',
                          'CountryCode_port1': 'Country Code'}, inplace=True)

    norm_cols = []
    for col in ew_cols:
        new_col = 'GLSN connectivity (' + col + ')'
        df_gc.rename(columns={col: new_col}, inplace=True)
        norm_cols.append(new_col)

    df_gc['# ports'] = df_gc['Country Code'].apply(dict_num_ports.get)
    norm_cols.insert(0, 'GLSN connectivity (Edge weight=None)')
    for col in norm_cols:
        df_gc['Normalized ' + col] = round(df_gc[col] / df_gc['# ports'], 4)

    df_gc.drop(columns='# ports', inplace=True)

    return df_gc


def calculate_glsn_betweennes(df_edges, df_nodes):

    def _dump_sp(df_edges, df_nodes):
        dict_port_country = dict(zip(df_nodes['id'], df_nodes['Country Code']))
        graph = nx.from_pandas_edgelist(df_edges, 'port1_id', 'port2_id', edge_attr=None, create_using=nx.Graph())
        nodelist = list(graph.nodes)
        list_spl = []
        list_valid_sp = []
        list_country = []
        for i, port_s in enumerate(nodelist[:-1]):
            for port_t in nodelist[i + 1:]:
                all_sp = nx.all_shortest_paths(graph, source=port_s, target=port_t, weight=None)
                for path in all_sp:
                    spl = len(path) - 1
                    if spl == 1:
                        continue
                    else:
                        country_list = list(map(dict_port_country.get, path))
                        inner_countries = country_list[1:-1]
                        head_country = country_list[0]
                        tail_country = country_list[-1]
                        if (head_country in inner_countries) or (tail_country in inner_countries) \
                                or (head_country == tail_country):
                            continue
                        else:
                            list_spl.append(spl)
                            list_valid_sp.append(path)
                            list_country.append(country_list)
        df_sp = pd.DataFrame(list_valid_sp)
        df_sp.columns = ['id' + str(col + 1) for col in df_sp.columns]
        df_sp['SPL'] = list_spl
        df_country = pd.DataFrame(list_country)
        df_country.columns = ['Country' + str(col + 1) for col in df_country.columns]
        df_sp = pd.concat([df_sp, df_country], axis=1)
        return df_sp

    def _cal_glsn_bc_spl2(df_sp):
        df_sp = df_sp[df_sp['SPL'] == 2]
        df_sp['Edge'] = df_sp['id1'].astype(str) + '--' + df_sp['id3'].astype(str)

        num_total_sp = df_sp.groupby(['Edge'], as_index=False)['id1'].count()
        dict_num_sp = dict(zip(num_total_sp['Edge'], num_total_sp['id1']))

        df_gb = df_sp.groupby(['Edge', 'Country2'], as_index=False)['id1'].count()
        df_gb.rename(columns={'id1': '# country_SP', 'Country2': 'Country Code'}, inplace=True)
        df_gb['# SP'] = df_gb['Edge'].apply(dict_num_sp.get)
        df_gb['GLSN betweenness (SPL=2)'] = df_gb['# country_SP'] / df_gb['# SP']
        df_gb = df_gb.groupby('Country Code', as_index=False)['GLSN betweenness (SPL=2)'].sum()
        dict_gb = dict(zip(df_gb['Country Code'], df_gb['GLSN betweenness (SPL=2)']))
        return dict_gb

    def _cal_glsn_bc_spl(df_sp, spl):
        data = df_sp[df_sp['SPL'] == spl]
        data['Edge'] = data['id1'].astype(str) + '--' + data['id' + str(spl + 1)].astype(str)
        if spl == 3:
            data['countries'] = data['Country2'].astype(str) + ',' + data['Country3'].astype(str)
        elif spl == 4:
            data['countries'] = data['Country2'].astype(str) + ',' + data['Country3'].astype(str) + \
                                ',' + data['Country4'].astype(str)
        else:
            data['countries'] = data['Country2'].astype(str) + ',' + data['Country3'].astype(str) + \
                                ',' + data['Country4'].astype(str) + ',' + data['Country5'].astype(str)

        data['countries'] = data['countries'].str.split(',')
        data['unique_countries'] = data['countries'].apply(pd.unique)
        df_country = pd.DataFrame(data['unique_countries'].values.tolist())
        df_country.columns = ['Country' + str(col+2) for col in df_country.columns]

        inner_country_cols = df_country.columns
        data.drop(columns=inner_country_cols, inplace=True)
        data.index = range(0, len(data))
        data = pd.concat([data, df_country], axis=1)

        num_total_sp = data.groupby(['Edge'], as_index=False)['id1'].count()
        dict_num_sp = dict(zip(num_total_sp['Edge'], num_total_sp['id1']))

        df_gb_all = pd.DataFrame()
        for col in inner_country_cols:
            df_gb = data.groupby(['Edge', col], as_index=False)['id1'].count()
            df_gb.rename(columns={'id1': '# country_SP', col: 'Country Code'}, inplace=True)
            df_gb['# SP'] = df_gb['Edge'].apply(dict_num_sp.get)
            df_gb['GLSN betweenness (SPL=' + str(spl) + ')'] = df_gb['# country_SP'] / df_gb['# SP']
            df_gb_all = pd.concat([df_gb_all, df_gb], axis=0)

        df_gb = df_gb_all.groupby('Country Code', as_index=False)['GLSN betweenness (SPL=' + str(spl) + ')'].sum()
        dict_gb = dict(zip(df_gb['Country Code'], df_gb['GLSN betweenness (SPL=' + str(spl) + ')']))

        return dict_gb

    def _merge_all(data):

        cols1 = ['GLSN betweenness (SPL=2)', 'GLSN betweenness (SPL=3)', 'GLSN betweenness (SPL=4)',
                 'GLSN betweenness (SPL=5)']
        for i in range(2, 6):
            data['GLSN betweenness (Lmax=' + str(i) + ')'] = data[cols1[:i - 1]].sum(axis=1)

        data.drop(columns=['GLSN betweenness (SPL=2)', 'GLSN betweenness (SPL=3)',
                           'GLSN betweenness (SPL=4)', 'GLSN betweenness (SPL=5)'], inplace=True)
        return data

    df_country = df_nodes[['Country Code']]
    df_country.drop_duplicates(inplace=True)
    df_sp = _dump_sp(df_edges, df_nodes)
    dict_gb2 = _cal_glsn_bc_spl2(df_sp)
    df_country['GLSN betweenness (SPL=2)'] = df_country['Country Code'].apply(dict_gb2.get)

    max_spl = df_sp['SPL'].max()
    list_spl = range(3, max_spl + 1)
    for spl in list_spl:
        dict_gb = _cal_glsn_bc_spl(df_sp, spl)
        df_country['GLSN betweenness (SPL=' + str(spl) + ')'] = df_country['Country Code'].apply(dict_gb.get)

    df_country.fillna(0, inplace=True)
    df_gb_res = _merge_all(df_country)

    return df_gb_res


def calculate_freeman_bc(df_edges, df_nodes):
    g = nx.from_pandas_edgelist(df_edges, 'port1_id', 'port2_id', create_using=nx.Graph())
    dict_bc = nx.betweenness_centrality(g, normalized=False)
    df_nodes['Freeman betweenness'] = df_nodes['id'].apply(dict_bc.get)
    num_ports = df_nodes.groupby('Country Code', as_index=False)['id'].count()
    dict_num_ports = dict(zip(num_ports['Country Code'], num_ports['id']))

    df_bc = df_nodes.groupby('Country Code', as_index=False)['Freeman betweenness'].sum()
    df_bc['# ports'] = df_bc['Country Code'].apply(dict_num_ports.get)
    df_bc['normalized Freeman betweenness'] = df_bc['Freeman betweenness'] / df_bc['# ports']
    df_bc.drop(columns='# ports', inplace=True)
    return df_bc


def startup():
    datasets = ['2015', '2017']

    for dataset in datasets:
        edgedata = pd.read_excel(data_path + 'Edges_GLSN_' + dataset + '.xlsx')
        edgedata_copy = edgedata.copy()
        tmp = edgedata[['port1_id', 'CountryCode_port1']]
        edgedata_copy[['port1_id', 'CountryCode_port1']] = edgedata[['port2_id', 'CountryCode_port2']]
        edgedata_copy[['port2_id', 'CountryCode_port2']] = tmp

        edgedata_copy = pd.concat([edgedata, edgedata_copy], axis=0)
        nodedata = edgedata_copy[['port1_id', 'CountryCode_port1']]
        nodedata.drop_duplicates(inplace=True)
        nodedata.columns = ['id', 'Country Code']

        gc = calculate_glsn_connectivity(edgedata, nodedata)
        bc = calculate_freeman_bc(edgedata, nodedata)
        gb = calculate_glsn_betweennes(edgedata, nodedata)

        df_res = pd.merge(gc, bc, on='Country Code')
        df_res = pd.merge(df_res, gb, on='Country Code')

        df_ec = pd.read_excel(data_path + 'TV_GDP_LSCI_' + dataset + '.xlsx')

        df_all = pd.merge(df_ec, df_res, on='Country Code')
        filename2 = 'Regression_Variables_' + dataset + '.xlsx'
        df_all.to_excel(save_path + filename2, index=False)
        print('The result file "{}" saved at: "{}"'.format(filename2, save_path))
        print()
