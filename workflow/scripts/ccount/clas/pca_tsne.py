def pca_tsne(df_gene_col, cluster_info=None, title='data', 
             #dir='plots',
             num_pc=50, num_tsne=2, ncores=8):
    '''
    PCA and tSNE plots for DF_cell_row, save projections.csv
    :param df_cell_row: data matrix, features as columns, e.g. [cell, gene]
    :param cluster_info: cluster_id for each cell_id
    :param title: figure title, e.g. Late
    :param num_pc: 50
    :param num_tsne: 2
    :return: tsne_df, plots saved, pc_projection.csv, tsne_projection.csv saved
    '''

#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     title = './' + dir + '/' + title
    import time
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE  # single core
    #from MulticoreTSNE import MulticoreTSNE as TSNE  # MCORE
    df = df_gene_col
    if cluster_info is None:
        cluster_info = pd.DataFrame(0, index=df.index, columns=['cluster_id'])

    tic = time.time()
    # PCA
    pca = PCA(n_components=num_pc)
    pc_x = pca.fit_transform(df)
    df_pc_df = pd.DataFrame(data=pc_x, index=df.index, columns=range(num_pc))
    df_pc_df.index.name = 'cell_id'
    df_pc_df.columns.name = 'PC'
    #df_pc_df.to_csv(title + '.pca.csv')
    print('dim before PCA', df.shape)
    print('dim after PCA', df_pc_df.shape)
    print('explained variance ratio: {}'.format(
        sum(pca.explained_variance_ratio_)))

    colors = cluster_info.reindex(df_pc_df.index)
    colors = colors.dropna().iloc[:, 0]
    print('matched cluster_info:', colors.shape)
    print('unmatched data will be excluded from the plot')  # todo: include unmatched

    df_pc_ = df_pc_df.reindex(colors.index)  # only plot labeled data?
    cluster_scatterplot(df_pc_, colors.values.astype(str), title=title + ' (PCA)')

#     # tSNE
#     print('MCORE-TSNE, with ', ncores, ' cores')
#     df_tsne = TSNE(n_components=num_tsne, n_jobs=ncores).fit_transform(df_pc_)
#     print('tsne done')
#     df_tsne_df = pd.DataFrame(data=df_tsne, index=df_pc_.index)
#     print('wait to output tsne')
#     df_tsne_df.to_csv(title + '.tsne.csv')
#     print('wrote tsne to output')
#     cluster_scatterplot(df_tsne_df, colors.values.astype(str), title=title + ' ('
#                                                                              't-SNE)')
    toc = time.time()
    print('took {:.1f} seconds\n'.format(toc - tic))

    return df_pc_df




def cluster_scatterplot(df2d, labels, title):
    '''
    PCA or t-SNE 2D visualization

    `cluster_scatterplot(tsne_projection, cluster_info.Cluster.values.astype(int),
                    title='projection.csv t-SNE')`

    :param df2d: PCA or t-SNE projection df, cell as row, feature as columns
    :param labels:
    :param title:
    :return:
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    legends = np.unique(labels)
    print('all labels:', legends)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    for i in legends:
        _ = df2d.iloc[labels == i]
        num_blobs = str(len(_))
        percent_cells = str(round(int(num_blobs) / len(df2d) * 100, 1)) + '%'
        ax.scatter(_.iloc[:, 0], _.iloc[:, 1],
                   alpha=0.5, marker='.',
                   label='c' + str(i) + ':' + num_blobs + ', ' + percent_cells
                   )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel('legend format:  cluster_id:num-cells')

    #plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()
    plt.close('all')



