        #!/usr/bin/env python3
# Use this to boot up your scRNA-seq notbooks. Use it before loading adata


import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import yaml
import time
import anndata
import scrublet as scr
import pickle as pk
#import plotnine as p9
import re
import os
import sklearn
from collections import Counter
from collections import defaultdict
import scipy
import seaborn
#from plotnine.data import economics, mtcars, mpg

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
#sc.logging.print_versions()

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(name='gene_cmap', colors=['lightgrey', 'thistle', 'red', 'darkred']) 

sc.settings.set_figure_params(dpi=80, color_map='magma', vector_friendly=True,  dpi_save=700)


# Here to be more robust, define the number of columns for multi-plots
# numcol = 2
# cr_dir = '../align'
# order_of_conditions = ['WT', 'Aldh2/Fancd2 homo', 'Aldh2/Fancd2/p53 homo']

get_batch_ids = lambda adata: np.unique(adata.obs['batch'])
get_nrow_plots = lambda batchids, numcol: int(np.ceil(len(batchids)/numcol))

#### Transform functions
def rotate_map_2D(my_adata, degrees = 90, key_umap = 'X_umap', key_umap_rot_add = 'X_umap'):
    """
    Rotate a UMAP (2D) by specified degrees
    Parameters:
        my_adata: annData
        degrees: Angle to rotate by in degrees
        key_umap: key in anndata.obsm that contains umap coordinates
        key_umap_rot_add: add a key to anndata with rotated coordinates
    Returns:
        annData
    """
    import scipy
    p = my_adata.obsm['X_umap']
    origin = np.mean(p, axis = 0) # Centroid of the umap
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    rotated_mat = np.squeeze((R @ (p.T-o.T) + o.T).T)
    #k = key_umap_rot_add + str(degrees)
    my_adata.obsm[key_umap_rot_add] = rotated_mat
    return my_adata
    



#### Plotting functions
def sns_barplot(data, x = 'index', y = 'sample_name', xrot = 45, colors = ""):
    """
    Plots a simple barplot:
    Params:
        data: pandas core series or dataframe
        x: colname of the x axis
        y: colname of the y axis
        xrot: rotate x tick labels (degrees)
        colors: a pandas dataframe where index is x colnames and column is named 'hexcode'
    Returns:
        None
    """
    sns.set_style("ticks")
    temp = pd.DataFrame(data)
    if len(colors) > 0:
        sns.set_palette(sns.color_palette(colors.hexcode))
        temp = temp.reindex(index=colors.index)
    if x == 'index':
        xax = temp.index
    else:
        xax = temp[x]
    seaborn.barplot( x = xax, y = temp[y])
    sns.despine(offset=10, trim=True)
    loc, lab = plt.xticks()
    loc, lab = plt.xticks(rotation = xrot, ha = 'right')
    return None


##### Saving functions
def save_adata(annda, name, path="./"):
    annda.write(path+'/'+name+'.h5ad')
    
def load_adata(name, path="./"):
    f = path+'/'+name+'.h5ad'
    annda = sc.read(path+'/'+name+'.h5ad')
    return annda

#### Project data functions

# Load project yaml file
def load_project(pfile = 'project.yaml'):
    """
    Loads a project.yaml file.
    pfile: str, read in file name
    returns: list, project related vars
    """
    project_file = open("project.yaml")
    project_dict = yaml.load(project_file, Loader=yaml.Loader) # Loader class can be yaml.FullLoader but only if you have after v
    project_dict
    align_dir = project_dict["cellranger_aligndir"]
    numcol = project_dict['numcols_for_plotting']
    samplefile = project_dict['samplefile']
    cell_cycle_file=project_dict['cell_cycle_file']
    genemarker_file = project_dict['genemarker_file']
    return [project_dict, align_dir, numcol, samplefile, cell_cycle_file, genemarker_file]

def load_sample_data(samplefile):
    sample_data = pd.read_csv(samplefile,
                          names=['sample_id', 'run_id','sample_name','condition','expected_cells']
                         )
    return sample_data


# Get subplot indices
def get_subplot_indices(ncols, total, ind = 0, rtype = 'indices'):
    '''
    Useful for plt.subplots to get nrows and individual indices
    Params:
        ncols: number of columns
        total: total plots
        ind: an index starting from 1. Needed if rtype = 'indices'
        rtype: str
            size: returns a list [numrows, numcols] needed to plot
            indices: returns a list [rowInd, colInd] for plotting ax = ax[rowInd, colInd]
    Returns:
        list of 2 values. [row, col]
    '''
    c,r = [0,-1]
    cl = rl = []
    rlist = []
    for i in range(total):
        c = i % ncols
        if c == 0:
            r += 1
        #print(r,c)
        cl.append(c)
        rl.append(r)
    if rtype == 'size':
        rlist = [max(rl)+1,ncols]
    if rtype == 'indices':
        c,r = [0,-1]
        for i in range(ind):
            c = i % ncols
            if c == 0:
                r += 1
        rlist = [r,c]
    return rlist

# Reading the project yaml file

def readProjectFile(filename = "project.yaml"):
    project_file = open(filename)
    project_dict = yaml.load(project_file, Loader=yaml.Loader) # Loader class can be yaml.FullLoader but only if you have after v
    return project_dict

# Doublet scoring using scrublet
def runScrubletByBatch(adata, nrow = 2, ncol = 2):
    """
    Runs scrublet by batch
    :param adata: AnnData object
    :return: list
    """
    db_scores = []
    predicted_db = []
    Cells = np.array([])
    for i in np.unique(adata.obs['batch']):
        fig, ax = plt.subplots()
        idx_sub = (adata.obs['batch']==i)
        Cells = np.append(Cells, adata[idx_sub,:].obs_names)
        scrub = scr.Scrublet(adata[idx_sub,:].X)
        doublet_scores, predicted_doublets = scrub.scrub_doublets()
        db_scores.append(doublet_scores)
        predicted_db.append(predicted_doublets)
        pd.DataFrame(doublet_scores).plot.hist(bins=200,ax = ax).axvline(x=scrub.threshold_, color='b')
        plt.title("Batch id = " + i)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.tight_layout()
    return scrub,db_scores,predicted_db,Cells


def resultScrublet(db_scores, predicted_db, Cells):
    db_results = {'doublet_scores': np.concatenate(db_scores),
              'predicted_doublets': np.concatenate(predicted_db)
             }
    db_results = pd.DataFrame.from_dict(db_results)
    db_results.index = Cells
    # save the doublet detection results due to randomness
    db_results.to_csv('db_results.csv')
    return db_results


def scrublet_custom_filter(db_scores, db_cutoffs):
    pred_db = []
    if len(db_cutoffs) != len(db_scores):
        print("Cutoff list and scores list are not of same length")
        print("Please rerun after fixing the error")
        return 0
    else:
        for i in range(len(db_cutoffs)):
            pred_db.append(db_scores[i] >= db_cutoffs[i]) 
        return pred_db
        
    

def removeDoublets(adata, db_results):
    return adata[~db_results['predicted_doublets'].values,:].copy()


def checkSex(adata_hvg, numrow , numcol):
    numrow = 2
    fig, ax = plt.subplots(numrow,numcol, figsize=(7,5), sharex=True, sharey=True, squeeze=False)
    Condition_unique = adata_hvg.obs['sample_name'].cat.categories
    print("Plotting expression of Xist")
    for i in range(len(Condition_unique)):
        rowidx, colidx = plt_idx(i, numcol = numcol)
        # Plot for gene Xist
        ax1 = sc.pl.umap(adata_hvg[adata_hvg.obs['sample_name'] == Condition_unique[i]], 
                         title=Condition_unique[i], color='Xist', ncols=2, ax=ax[rowidx][colidx],
                          show=False, s=10, color_map=cmap
                        )
        #ax1.plot()
    plt.tight_layout() 


def calcMito(adata, ylim = 100):
   # global adata
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, 
                           qc_vars=['mt'], 
                           percent_top=None, 
                           log1p=False, 
                           inplace=True)
    ax = sc.pl.violin(adata, 'pct_counts_mt', jitter=0.4, show = False)
    ax.set_ylim(0,ylim)
    return adata


def filterMito(adata, mitFilterPerc = 2.5, **kwargs):
    # mit percentage shows some dying cells, remove them
    adata.obs['mit_pass'] = ~(adata.obs["pct_counts_mt"] >= mitFilterPerc)
    adata.obs['mit_pass'] = adata.obs['mit_pass'].astype('category')
    f = sc.pl.scatter(adata, x='n_genes_by_counts', y='pct_counts_mt', 
          color="mit_pass", show=False, 
          title= "Number of genes covered vs Mitochondrial gene perc",
          color_map = cmap,
          palette = ['#3346ff', '#ff4233' ], size = 1200000/adata.n_obs, 
         )
    f.set_xlabel('Genes by counts')
    f.set_ylabel('% Mit')
    #adata = adata[adata.obs['mit_pass'].astype("bool"),:]
    adata = adata[adata.obs['mit_pass']==True, :]
    return adata

def assignCellCycle(adata, cell_cycle_file):
    """
    If adata.raw is present then think about using use_raw = F because scoregenes uses scaled data
    """
    cell_cycle_genes = [ reformat_GN(x.strip().title()) for x in open(cell_cycle_file, "r")]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    sc.tl.score_genes_cell_cycle(adata, s_genes = s_genes, g2m_genes= g2m_genes, use_raw = False)
    print("Writing output to cell_cycle_results.csv")
    adata.obs[['phase', 'S_score', 'G2M_score']].to_csv('cell_cycle_results.csv')
    return adata

def plotCellCycle(adata):
    #global adata
    x = pd.crosstab(adata.obs['phase'], adata.obs['sample_name'], normalize=1)*100
    ax = x.T.plot.bar(stacked=True)
    ax.grid(False)
    ax.set_ylabel('%')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# misc
genes_to_plot=["Mcm5", "Pcna", "Tyms", "Fen1", "Mcm2", "Mcm4", "Rrm1"]

# Write to log file

def writeLog(string, logfile = "logs.txt"):
    t = time.ctime()
    with open(logfile, "a+") as f:
        print(t + " : " + string)


def plot_some_genes(adata, genes_to_plot, ylabel = "Counts", title="Genes"):
    #genes_to_plot = ["Erg","Rps9","Malat1"]
    temp_nparray = adata[:,genes_to_plot].X.toarray()
    d = plt.boxplot(temp_nparray, labels = genes_to_plot)
    d = plt.ylabel(ylabel)
    d = plt.title(title)
    return(d)

def ReadInFiles(cr_dir, folder_name, mouseID, Sample, Condition, Expected_cells):
    adata = sc.read_10x_h5(cr_dir+'/'+folder_name+'/outs/raw_feature_bc_matrix.h5')
    print(f"Reading >> {folder_name} {mouseID} {Condition}")
    #adata.var_names_make_unique()
    adata.obs['mouseID'] = mouseID
    adata.obs['Condition'] = Condition
    adata.obs['Expected_cells'] = Expected_cells
    adata.obs['Sample'] = Sample
    adata.var_names_make_unique()
    return adata


def cutoff_qc(n_counts, N=10000):
    selectedCells = np.sort(n_counts)[::-1][0:N]
    m = np.percentile(selectedCells, 99)/10
    passidx = n_counts>m
    return(m, passidx)

def dump_pickle(filename = "temp.pickle", **to_pickle):
    print("Pickling data")
    #to_pickle = {"adata": adata, "numcol": numcol, "numrow": numrow, "unique_batch_ids":list(unique_batch_ids),"project_dict": project_dict}
    # Write as a binary pickle
    with open(filename, "wb") as f:
        pk.dump(to_pickle, f)


# reformat gene list
def reformat_GN(s):
    s = s[0].upper() + s[1:].lower()
    return(s)

def gen_adata(sample_data, align_dir):
    adata_list = []
    for i in sample_data.sample_id:
        align_folder= align_dir + "/" + i + "/outs/" + readin_type+ "_feature_bc_matrix"
        temp = sc.read_10x_mtx(
        align_folder,  # the directory with the `.mtx` file
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)
        cell_ids = temp.obs.index
        obsdata = sample_data[sample_data.sample_id==i]
        obsdata = obsdata.append([obsdata]* (temp.n_obs-1))
        obsdata = obsdata.set_index(cell_ids)
        temp.obs = obsdata
        adata_list.append(temp)
        print(i)
    adata = adata_list[0].concatenate(adata_list[1:])
    return adata



def plot_qc(axid, x,y, m, batchid):
    axid.plot(x, y)
    axid.set_yscale('log',basey=10)
    axid.set_xscale('log',basex=10)
    axid.axhline(y=m, color='b')
    axid.set_title(batchid)
    axid.set_xlabel("Cells sorted by read count")
    axid.set_ylabel("Read count")

def old_10x_filter(adata, numcol, numrow, **kwargs):
    """
    Filter using 10X old filter method.
    adata: anndata
    numcol: Total cols for plotting
    numrow: Total rows for plotting
    Batch_unique: a list of unique batch ids
    """
    print("ncol = "+ str(numcol) + " nrow = "+ str(numrow))
    fig, ax = plt.subplots(numrow,numcol, figsize=(10,8))
    temp = adata.obs[["sample_name","batch"]].drop_duplicates()
    Batch_unique = list(temp.batch)
    Sample_unique = list(temp.sample_name)
    
    #Batch_unique = np.unique(adata.obs['batch'])
    Cells = np.array([])
    passIDidx = []
    for i in range(len(Batch_unique)):
        # save to np.array()
        cellidx = adata.obs['batch'] == Batch_unique[i]
        Cells = np.append(Cells, adata[cellidx,:].obs_names)
        n_counts = np.array(np.sum(adata[cellidx,:].X,axis=1)).flatten()
        exp_cells = int(adata[cellidx,:].obs["expected_cells"].unique())
        m, passidx = cutoff_qc(n_counts, N=exp_cells) 
        print('Batch '+str(i)+': QC threshold with counts > '+str(m))
        print('Number of cells passed QC: '+str(np.sum(passidx)))
        writeLog('Number of cells passed QC: '+str(np.sum(passidx)))
        passIDidx.append(passidx)
        # plot scatter plot
        colidx = i%numcol
        rowidx = np.floor(i/numcol).astype(int)
        n_counts_rank = np.sort(n_counts, axis=None)[::-1]
        x = range(len(n_counts_rank))
        y = n_counts_rank
        plot_qc(ax[rowidx, colidx], x,y, m, Sample_unique[i])

    plt.tight_layout()
    return [Cells, passIDidx]



def filter_cellcycle(adata, cell_cycle_genes):
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]

def plt_idx(i, numcol):
    colidx = i%numcol
    rowidx = np.floor(i/numcol).astype(int)
    return([rowidx, colidx])

def plot_cellcycle_bar(adata, x, y):
    # plot cell cycle per sample
    tab = pd.crosstab(adata.obs[x], adata.obs[y], normalize='index') * 100
    #tab = tab.T
    ax = tab.plot.bar(stacked=True)
    ax.legend(loc='center right', bbox_to_anchor=(1.3,0.5))
    ax.grid(False)
    return(ax)

def minmax(adata):
    """
    A convenient way to print
    """
    print("min = " + str(np.min(adata.X)))
    print("max = " + str(np.max(adata.X)))


def Obs_sublabels(ref_data, proj_data, proj_data_obs, cl_assigned, prefix):
    CT = np.unique(proj_data.obs[proj_data_obs])
    print(CT)
    for ct in CT:
        cl_assigned_sub = [cl_assigned[i] for i in np.where(proj_data.obs[proj_data_obs] == ct)[0]]
        cl_flat_sub = [item for sublist in cl_assigned_sub for item in sublist]
        freq1 = Counter(cl_flat_sub)
        freq2 = np.array([0] * ref_data.X.shape[0])
        for k, i in freq1.items():
            if k in ref_data.obs_names:
                idx = np.where(ref_data.obs_names==k)
                freq2[idx] = i
        ref_data.obs[prefix+'_'+ct] = np.log2(freq2+1)


def plotMA(adata, unsName, cidx=0, Cells = None, save=False, padj_cutoff=0.05, logFC_cutoff=1, exp_cutoff=-6, s=1, fig_dir = "figures"):
    if Cells is not None:
        adata_sub = adata[Cells,:]
    else:
        adata_sub = adata
    print(adata_sub.shape)
    gnames = pd.DataFrame(adata.uns[unsName]['names']).iloc[:,cidx]
    logFC = pd.DataFrame(adata.uns[unsName]['logfoldchanges']).iloc[:,cidx]
    pvals = pd.DataFrame(adata.uns[unsName]['pvals']).iloc[:,cidx]
    padj = pd.DataFrame(adata.uns[unsName]['pvals_adj']).iloc[:,cidx]
    adata_sub = adata_sub.raw[:, gnames].X
    print(adata_sub.shape)
    normExp = np.mean(np.exp(adata_sub.toarray())-1, axis=0)
    del adata_sub
    #print(len(normExp))
    #print(len(logFC))
    abs_logFC = logFC.copy()
    abs_logFC[abs_logFC > 4] = 4
    abs_logFC[abs_logFC < -4] = -4
    #import seaborn as sns
    #sns.kdeplot(np.log(normExp))
    logExp = np.log2(normExp)
    idx = (padj < padj_cutoff) & (abs(abs_logFC) > logFC_cutoff)
    upidx = (padj < padj_cutoff) & (abs_logFC > logFC_cutoff) & (logExp > exp_cutoff)
    downidx = (padj < padj_cutoff) & (abs_logFC < -logFC_cutoff) & (logExp > exp_cutoff)
    print('upRegulated gene: '+str(sum(upidx)))
    print('downRegulated gene: '+str(sum(downidx)))
    
    fig = plt.figure()
    plt.scatter(x=logExp, y=abs_logFC, s=s)
    plt.scatter(x=logExp[idx & (logExp > exp_cutoff)], y=abs_logFC[idx & (logExp > exp_cutoff)], c='red',s=s)
    plt.axhline(y=0, color='black')
    plt.axhline(y=logFC_cutoff, color='grey', linestyle = '--')
    plt.axhline(y=-logFC_cutoff, color='grey', linestyle = '--')
    plt.axvline(x=exp_cutoff, color='grey', linestyle = '--')
    plt.xlabel('log2 Mean Exp')
    plt.ylabel('log2 Fold Change')
    plt.grid(b=None)
    plt.show()
    if save:
        fig.savefig(fig_dir+'/DEres_'+unsName+'_idx_'+str(cidx)+'.pdf')
    
    Ftable = pd.DataFrame(np.column_stack([gnames, logExp, logFC, pvals, padj]), columns=['GN','logMeanExp', 'logFC', 'pvals', 'padj'])
    return gnames[upidx], gnames[downidx], Ftable

def get_avgExp_cluster(adata, obs_key, cluster_id):
    temp = adata[adata.obs[obs_key]==cluster_id,:]
    temp1 = scipy.sparse.csr_matrix.todense(temp.X)
    return np.average(temp1, axis = 0).tolist()[0]

def gen_adata(sample_data, align_dir):
    adata_list = []
    for i in sample_data.sample_id:
        align_folder= align_dir + "/" + i + "/outs/" + readin_type+ "_feature_bc_matrix"
        temp = sc.read_10x_mtx(
        align_folder,  # the directory with the `.mtx` file
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)
        cell_ids = temp.obs.index
        obsdata = sample_data[sample_data.sample_id==i]
        obsdata = obsdata.append([obsdata]* (temp.n_obs-1))
        obsdata = obsdata.set_index(cell_ids)
        temp.obs = obsdata
        adata_list.append(temp)
        print(i)
    adata = adata_list[0].concatenate(adata_list[1:])
    return adata

def writeDE(adata, key = 'louvain', key_added='DE_louvain'):
    de_genes = pd.DataFrame(adata.uns[key_added]['names'])
    de_lfc = pd.DataFrame(adata.uns[key_added]['logfoldchanges'])
    de_padj = pd.DataFrame(adata.uns[key_added]['pvals_adj'])
    cluster_id = '3'
    writer = pd.ExcelWriter(key_added+'.xlsx', engine='xlsxwriter')
    df = pd.DataFrame()
    for cluster_id in adata.obs[key].unique():
        temp = pd.DataFrame({"Cluster": cluster_id, 
                       "Gene": de_genes.loc[:, cluster_id], 
                       "lfc": de_lfc.loc[:, cluster_id], 
                       "padj": de_padj.loc[:,cluster_id]})
        temp.to_excel(writer, sheet_name="Cluster "+cluster_id)
        df = df.append(temp)
    writer.save()


def map_niki_landscape(niki_hvg, adata_hvg):
    niki_hvg.obs['condn'] = 'ref'
    #adata_hvg = adata_hvg_orig.copy()
    #overlap_genes = np.intersect1d(niki_hvg.var_names, adata_hvg.var_names)
    #overlap_genes = list(overlap_genes)
    #adata_hvg = adata_hvg[:,overlap_genes]
    #niki_hvg = niki_hvg[:,overlap_genes]
    #print("Overlapping genes:")
    #print(len(overlap_genes))
    # Concatenate common genes
    alldata = niki_hvg.concatenate(adata_hvg)
    print(alldata.shape)
    # scale them together
    sc.pp.scale(alldata)
    # split them again
    adata = alldata[alldata.obs['condn']!='ref']

    niki_data = alldata[alldata.obs['condn']=='ref']
    print(niki_data.shape)
    print('Running PCA')
    from sklearn.decomposition import PCA
    pca_ = PCA(n_components=50, svd_solver='auto', random_state=0)
    nikiX = niki_data.X
    adataX = adata.X
    pca_.fit(nikiX)
    X_pca1 = pca_.transform(nikiX)
    X_pca2 = pca_.transform(adataX)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(X_pca1[:,0], X_pca1[:,1], c='black', alpha=0.5)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    ax1.scatter(X_pca2[:,0], X_pca2[:,1], c='red', alpha=0.5)
    plt.show()
    print("Computing Euclidean distances")
    from sklearn.metrics.pairwise import euclidean_distances
    D_sub = euclidean_distances(X_pca2, X_pca1)
    from collections import Counter
    from collections import defaultdict
    cl_assigned = []
    print("Assigning cells")
    Rstore = defaultdict(list) # dictionary to store results
    for i in range(D_sub.shape[0]):
        CellDis = D_sub[i,:]
        CellDis_sorted = np.argsort(CellDis)[:15]
        max_samples = niki_data.obs_names[CellDis_sorted]
        cl_assigned.append(max_samples)
        Rstore['MinDist'].append(np.min(CellDis[CellDis_sorted]))
        Rstore['MedianDist'].append(np.median(CellDis[CellDis_sorted]))
        Rstore['MaxDist'].append(np.max(CellDis[CellDis_sorted]))
        Rstore['SD'].append(np.std(CellDis[CellDis_sorted]))
        Rstore['NikiAssign'].append(Counter(niki_data.obs['CellSubType'][CellDis_sorted]).most_common(1)[0][0])
    Rstore = pd.DataFrame.from_dict(Rstore)
    Rstore.index = adata_hvg.obs_names
    temp = adata_hvg.obs
    temp = pd.concat([temp, Rstore], axis =1 )
    adata_hvg.obs = temp
    return([cl_assigned, Rstore, adata_hvg])

def plot_gene_umap(adata, gene_name, nrow, ncol, groupby='sample_name', figsize=(7,5)):
    #numrow = 2
    #numcol = 2
    fig, ax = plt.subplots(numrow,numcol, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    Condition_unique = adata.obs[groupby].cat.categories
    for i in range(len(Condition_unique)):
        rowidx, colidx = plt_idx(i, numcol = numcol)
        # Plot for gene Xist
        ax1 = sc.pl.umap(adata_all[adata_all.obs[groupby] == Condition_unique[i]], 
                         title=Condition_unique[i], color=gene_name, ncols=2, ax=ax[rowidx][colidx],
                          show=False, s=10, color_map=cmap, vmin=0.1
                        )
        #ax1.plot()
    plt.tight_layout() 

    




#### Store data
import shelve


def shelveit(filename):
    #filename='/tmp/shelve.out'
    my_shelf = shelve.open(filename,'n') # 'n' for new

    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()


#### Paired voting method
import scanpy as sc
import pandas as pd
import numpy as np
import yaml
import time
import anndata
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict

class Projector:
    """
    Projector class:
    ## Paired Voting System
    * For every cell in the dataset its NN in the reference dataset gets 1 vote.
    * Smooth out the votes for each cell in the reference dataset by sharing them equally among 100 NN in it. So if its vote is 40 you share 40/100 among each of them.
    * Compute this for WT and mutant datasets. 
    * Plot their difference or logFC etc on the reference landscape (UMAP)
    """
    def __init__(self, adata, ref_adata, ref_hvg, npcs = 25):
        self.adata = adata
        self.ref_adata = ref_adata
        self.ref_hvg = ref_hvg
        self.npcs = npcs
        self.dist_mtx = dict()
    
    def project_on_ref(self, npcs = 50):
        """
        Input
        -----
        adata: data to be projected. Needs to have adata.raw instantiated. Ideally it should have all the genes not just hvg.
        ref_adata: reference dataset such as Niki or Nesterowa
        ref_hvg: HVG in reference dataset
        npcs: number of PC
        """
        adata_niki = self.ref_adata
        niki_hvg = self.ref_hvg
        adata = self.adata
        self.npcs = npcs
        adata = anndata.AnnData(X=np.exp(adata.raw.X.todense())-1,
                            obs=adata.obs, 
                            var=adata.raw.var, 
                            obsm=adata.obsm,
                            uns = adata.uns)

        print("Finding overlapping gene set")
        OLG = np.intersect1d(niki_hvg, adata.var_names)
        print("Genes common to both:")
        print(len(OLG))
        adata = adata[:,OLG].copy()
        adata_niki = adata_niki[:,OLG].copy()

        print("Normalising & log'ing data")
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000)
        sc.pp.normalize_per_cell(adata_niki, counts_per_cell_after=10000)
        sc.pp.log1p(adata)
        sc.pp.log1p(adata_niki)

        # scale them together
        print("Combining & Scaling them together")
        data_comb = adata.concatenate(adata_niki)
        sc.pp.scale(data_comb)

        adata = anndata.AnnData(X=data_comb[data_comb.obs['batch'] == '0',:].X, 
                                obs=adata.obs, 
                                var=adata.var, 
                                obsm=adata.obsm, 
                                uns=adata.uns)

        adata_niki = anndata.AnnData(X=data_comb[data_comb.obs['batch'] == '1',:].X, 
                                     obs=adata_niki.obs, 
                                     var=adata_niki.var, 
                                     obsm=adata_niki.obsm, 
                                     uns=adata_niki.uns)
        # PCs for Niki's data
        print("Computing PCA for reference dataset")
        pca_ = sklearn.decomposition.PCA(n_components=npcs, svd_solver='auto', random_state=0)
        pca_.fit(adata_niki.X)

        # Project data and niki's data onto Niki's PCs
        print("Projecting data on the reference PCA space")
        adata_proj = pca_.transform(adata.X)
        adata_niki_proj = pca_.transform(adata_niki.X)
        self.adata_proj = adata_proj
        self.ref_proj = adata_niki_proj
        self.pca = pca_
    
    def plotPCA(self):
        """
        Some PCA plots. First plot is used to alter pcs
        Second plot is used to determine if projection worked well
        """
        pca_ = self.pca
        adata_niki_proj = self.ref_proj
        adata_proj =  self.adata_proj
        print("Plotting explained variance. Useful to alter npcs parameter")
        plt.plot(pca_.explained_variance_)

        fig = plt.figure()
        print("Plotting the projected cells. They should fall within the reference phases")
        ax1 = fig.add_subplot(111)
        ax1.scatter(adata_niki_proj[:,0], adata_niki_proj[:,1], c='black', alpha=0.5)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        ax1.scatter(adata_proj[:,0], adata_proj[:,1], c='red', alpha=0.5)
        plt.show()
    
    def pwdist(self, rtype='across'):
        """
        Computes pairwise distance between adata and reference data.
        type:   across - between adata (row) and reference data (col)
                adata - between cells of adata
                ref - between cells of reference data

        returns: self.dist_mtx - a dict of {type: numpy mtx}        
        """
        adata_proj = self.adata_proj
        ref_proj = self.ref_proj
        npcs = self.npcs
        if rtype == "across":
            D_sub= euclidean_distances(adata_proj[:,0:npcs], ref_proj[:,0:npcs])
        if rtype == "adata":
            D_sub= euclidean_distances(adata_proj[:,0:npcs])
        if rtype == "ref":
            D_sub= euclidean_distances(ref_proj[:,0:npcs])
        self.dist_mtx[rtype] = D_sub
        self.type_to_work = rtype
    
    def voting(self, tot_votes = 500000):
        """
        voting:
        for each cell in adata, find NN in ref data. 
        add a vote to the each of the cell in NN.
        Needs pwdist(rtype='across') already performed.
        
        """
        adata_niki = self.ref_adata
        D_sub = self.dist_mtx['across']
        adata_proj = self.adata
        tot_cells_adata = adata_proj.shape[0]
        tot_cells_ref = adata_niki.shape[0]
        votes_per_cell = np.int(np.floor(tot_votes/tot_cells_adata))
        print(f"votes per cell = {votes_per_cell}")
        
        ref_names = adata_niki.obs_names
        votes = np.zeros(tot_cells_ref)
        for i in range(D_sub.shape[0]):
            CellDis = D_sub[i,:]
            CellDis_sorted = np.argsort(CellDis)[:votes_per_cell]
            votes[CellDis_sorted] += 1
        print(f"Run for {i} cells in the data") 

        self.tot_votes = tot_votes
        self.votes_per_cell = votes_per_cell
        self.votes_raw = votes
    
    def vote_smoothing(self, NN=100, hscNN = 20, celltype_col="CellSubType"):
        
        adata_niki = self.ref_adata
        ref_dist_matx = self.dist_mtx['ref']
        votes = self.votes_raw.copy()
        
        ref_nn = []
        nn_no = NN
        hsc_nn_no = hscNN
        cell_types = list(adata_niki.obs[celltype_col])
        for i in range(adata_niki.shape[0]):
            CellDis = ref_dist_matx[i,:]
            celltype = cell_types[i]
            if celltype == "HSCs":
                lim = hsc_nn_no
            else:
                lim = nn_no
            CellDis_sorted = list(np.argsort(CellDis)[:lim])
            ref_nn.append(CellDis_sorted)
        
        for i in range(len(ref_nn)):
            nn = ref_nn[i]
            K = len(nn)
            votes_for_i = votes[i]
            to_share = votes_for_i/K
            votes[nn] += to_share
        
        self.NN = 100
        self.hsc_nn = 20
        self.votes_smooth = votes

class Project_landscape:
    """
    Projecting onto landscapes and performing label transfer
    For Niki's dataset scale together as they are both 10x
    For Nestorowa's dataset scale separately
    
    Run methods flowchart:
    -----------------------
    normalise, log, scale_(together/separately), 
    run_pca_ref, pca_project_myadata, calc_pcadist,
    labelling, plot_on_ref, cluster_assignment
    """
    def __init__(self, my_adata, ref_adata, genes_to_consider, ref_data_obs_label = 'CellSubType', npcs = 25, knn = 15):
        OLG = np.intersect1d(genes_to_consider, my_adata.var_names)
        print("Genes common to both:")
        print(len(OLG))
        my_adata = my_adata[:,OLG].copy()
        ref_adata = ref_adata[:,OLG].copy()
        self.my_adata = my_adata
        self.ref_adata = ref_adata
        self.genes_to_consider = genes_to_consider
        self.npcs = npcs
        self.dist_mtx = dict()
        self.ref_data_obs_label = ref_data_obs_label
        self.knn = knn
        
    def normalise(self):
        """
        Normalise per cell
        """
       # sc.pp.normalize_per_cell(self.my_adata, counts_per_cell_after=10000)
       # sc.pp.normalize_per_cell(self.ref_adata, counts_per_cell_after=10000)
        sc.pp.normalize_total(self.my_adata, target_sum=1e4, exclude_highly_expressed=True)
        sc.pp.normalize_total(self.ref_adata, target_sum=1e4, exclude_highly_expressed=True)
        
    def log(self):
        """
        Logp data
        """
        sc.pp.log1p(self.my_adata)
        sc.pp.log1p(self.ref_adata)
        
    def scale_together(self):
        """
        Combine the two anndata.
        Scale them together. 
        Adviced for Niki's dataset integration
        """
        # scale them together
        data_comb = self.my_adata.concatenate(self.ref_adata)
        sc.pp.scale(data_comb)
        # Split them back again
        self.my_adata = anndata.AnnData(X=data_comb[data_comb.obs['batch'] == '0',:].X, 
                            obs=self.my_adata.obs, 
                            var=self.my_adata.var)
        self.ref_adata = anndata.AnnData(X=data_comb[data_comb.obs['batch'] == '1',:].X, 
                                 obs=self.ref_adata.obs, 
                                 var=self.ref_adata.var, 
                                 obsm=self.ref_adata.obsm, 
                                 uns=self.ref_adata.uns)
        self.data_comb = data_comb
        
    def scale_separately(self):
        """
        Scales the two data separately.
        Adviced if the datasets are from two different sequencing methods, for eg Nesterowa and 10x
        """
        # Scale them separately because they are from different technologies
        sc.pp.scale(self.my_adata)
        sc.pp.scale(self.ref_adata)

    def run_pca_ref(self):
        """
        Run PCA and optain pca object for ref data
        """
        # PCs for Niki's data
        pca_ = sklearn.decomposition.PCA(n_components=50, svd_solver='auto', random_state=0)
        pca_.fit(self.ref_adata.X)
        # Plot variance contrib
        plt.plot(pca_.explained_variance_)
        self.pca = pca_
    
    def pca_project_myadata(self):
        """
        Using PCs from ref data predict the coords for my_adata
        """
        # Project data and niki's data onto Niki's PCs
        pca_ = self.pca
        self.my_adata_proj = pca_.transform(self.my_adata.X)
        self.ref_adata_proj = pca_.transform(self.ref_adata.X)
        # Plot projection
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.ref_adata_proj[:,0], self.ref_adata_proj[:,1], c='black', alpha=0.5)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        ax1.scatter(self.my_adata_proj[:,0], self.my_adata_proj[:,1], c='red', alpha=0.5)
        plt.show()
    
    def calc_pca_dist(self):
        """
        Compute euclidean dist between each cell in my_adata to each cell in ref_adata in the PCA space
        """
        # High memory process
        npcs=self.npcs
        from sklearn.metrics.pairwise import euclidean_distances
        self.pca_dist = euclidean_distances(self.my_adata_proj[:,0:npcs], self.ref_adata_proj[:,0:npcs])
        
    def labelling(self):
        """
        Using the distances find knn nearest neighbours of ref_adata for each cell in my_adata.
        The most common NN in ref_adata is returned as the label for each cell in my_adata.
        """
        knn = self.knn
        from collections import Counter
        from collections import defaultdict
        cl_assigned = []
        D_sub = self.pca_dist
        ref_label = self.ref_data_obs_label
        Rstore = defaultdict(list) # dictionary to store results
        for i in range(D_sub.shape[0]):
            if i%5000 == 0 and i > 0:
                print(f"Label transfer complete for {i} cells")
            CellDis = D_sub[i,:]
            CellDis_sorted = np.argsort(CellDis)[:knn]
            max_samples = self.ref_adata.obs_names[CellDis_sorted]
            cl_assigned.append(max_samples)
            Rstore['MinDist'].append(np.min(CellDis[CellDis_sorted]))
            Rstore['MedianDist'].append(np.median(CellDis[CellDis_sorted]))
            Rstore['MaxDist'].append(np.max(CellDis[CellDis_sorted]))
            Rstore['SD'].append(np.std(CellDis[CellDis_sorted]))
            Rstore['label_ct'].append(Counter(self.ref_adata.obs[ref_label][CellDis_sorted]).most_common(1)[0][0])
        Rstore = pd.DataFrame.from_dict(Rstore)
        Rstore.index = self.my_adata.obs_names
        self.label_store = Rstore
        self.cl_assigned = cl_assigned
        
    def plot_on_ref(self, my_adata_cluster_obs = 'leiden', prefix = 'niki'):
        """
        Using the distances map each cluster in my_adata onto cells in niki dataset.
        """
        ref_data = self.ref_adata
        proj_data = self.my_adata
        proj_data_obs = my_adata_cluster_obs
        cl_assigned = self.cl_assigned
        self.my_adata_cluster_obs = my_adata_cluster_obs
        obs_names = []
        #CT = np.unique(proj_data.obs[proj_data_obs])
        CT = np.sort(proj_data.obs[proj_data_obs].unique().astype('int')) 
        #print(CT)
        for ct in CT:
            ct = str(ct)
            cl_assigned_sub = [cl_assigned[i] for i in np.where(proj_data.obs[proj_data_obs] == ct)[0]]
            cl_flat_sub = [item for sublist in cl_assigned_sub for item in sublist]
            freq1 = Counter(cl_flat_sub)
            freq2 = np.array([0] * ref_data.X.shape[0])
            for k, i in freq1.items():
                if k in ref_data.obs_names:
                    idx = np.where(ref_data.obs_names==k)
                    freq2[idx] = i
             
            ref_data.obs[prefix+'_'+ct] = np.log2(freq2+1)
            obs_names.append(prefix+'_'+ct)
        self.ref_obsnames = obs_names
        sc.pl.draw_graph(self.ref_adata, color=obs_names, size = 30)
    
    def cluster_assignment(self):
        """
        Based on cell labelling, use the most common label for the cells in each cluster of my_adata as the cluster name.
        """
        cluster_assignment = dict()
        cluster_obs = self.my_adata_cluster_obs
        Rstore = self.label_store
        ref_adata = self.ref_adata
        my_adata = self.my_adata 
        for i in my_adata.obs[cluster_obs].unique():
            #print(Counter(np.array(cl_assigned_ct)[adata.obs['louvain_v2'] == i]))
            ct = Counter(np.array(Rstore['label_ct'])[my_adata.obs[cluster_obs] == i]).most_common(1)[0][0]
            print(i+':'+ct)
            cluster_assignment[int(i)] = ct
        self.classign = cluster_assignment
        
