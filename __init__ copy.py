# scOmni/__init__.py
try:
    from .tool_base import SingleToolBase
except ImportError as e:
    print(f"Exception: Error importing SingleToolBase: {e}")

try:
    from .tool_base import MultiToolBase
except ImportError as e:
    print(f"Exception: Error importing MultiToolBase: {e}")

# try:
#     from .SpatialTranscriptomics import SpatialTranscriptomics
# except ImportError as e:
#     print(f"Exception: Error importing SpatialTranscriptomics: {e}")

### not for registering
# from .pythonnotebook import PythonNoteBook
# from .pythonnotebook import NBCellExcutionException


##############################################################
# SingleCell API
##############################################################

import scanpy as sc
import pandas as pd
import numpy as np
import openai
import re
# 去批次 API
def batch_integration(adata, batch_key, method=["liger"]): 
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, svd_solver='arpack')
    # ["harmony", "liger", "scvi"]
    def run_harmony(adata, batch):
        import scanpy.external as sce
        sce.pp.harmony_integrate(adata, key=batch)
        print("Harmony integration finished.")
        adata.obsm["X_harmony"]= adata.obsm["X_pca_harmony"]  # Harmony 的结果存储在 X_pca_harmony 中
        # res will be stored in adata.obsm["X_pca_harmony"]
        return adata

    def run_liger(adata, batch):
        import pyliger
        import scipy.sparse

        bdata = adata.copy()
        bdata.obs[batch] = bdata.obs[batch].astype('category')
        batch_cats = bdata.obs[batch].cat.categories

        # 确保是稠密矩阵再转稀疏
        if isinstance(bdata.X, np.ndarray):
            bdata.X = scipy.sparse.csr_matrix(bdata.X)

        adata_list = [bdata[bdata.obs[batch] == b].copy() for b in batch_cats]
        for i, ad in enumerate(adata_list):
            ad.uns["sample_name"] = batch_cats[i]
            ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)
        liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
        liger_data.var_genes = bdata.var_names
        pyliger.normalize(liger_data)
        pyliger.scale_not_center(liger_data)
        # k = min(30, min(ad.shape[0] for ad in liger_data.adata_list)-1)  # 确保k合法
        pyliger.optimize_ALS(liger_data, k=30, max_iters=30)
        pyliger.quantile_norm(liger_data)
        adata.obsm["X_pca_liger"] = np.zeros((adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
        for i, b in enumerate(batch_cats):
            adata.obsm["X_pca_liger"][adata.obs[batch] == b] = liger_data.adata_list[i].obsm["H_norm"]
        adata.obsm["X_liger"]= adata.obsm["X_pca_liger"]  # LIGER 的结果存储在 X_pca_liger 中
        print("LIGER integration finished.")
        return adata

    def run_scvi(adata, batch):
        import scvi
        if 'counts' not in adata.layers:
            # 获取当前 adata.var_names 在 raw.var_names 里的索引
            raw_index = [np.where(adata.raw.var_names == v)[0][0] for v in adata.var_names]
            adata.layers['counts'] = adata.raw.X[:, raw_index]

        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch)
        vae = scvi.model.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=30)
        vae.train()
        adata.obsm["X_pca_scVI"] = vae.get_latent_representation()
        adata.obsm["X_scVI"] = adata.obsm["X_pca_scVI"]

    # 鲁棒地处理 method 入参
    if isinstance(method, str):
        method = [method]
    elif not isinstance(method, (list, tuple)):
        raise ValueError("method must be str or list/tuple of str")

    # 依次运行所需方法
    for m in method:
        m_lower = m.lower()
        if m_lower == "harmony":
            adata = run_harmony(adata, batch_key)
        elif m_lower == "liger":
            adata = run_liger(adata, batch_key)
        elif m_lower == "scvi":
            adata = run_scvi(adata, batch_key)
        else:
            print(f"Unknown method: {m}, skipped.")

    return adata

# 细胞类型注释 API
def celltype_annotation(adata,species='',tissue_type='',cancer_type='Normal',groupby='',method=["gpt4"],openai_api_key=None ): 
    # 可为单字符串或列表，如 ["gpt4", "cellmarker", "act"]
    def gpt4_method(adata, species, tissue_type, cancer_type, groupby, openai_api_key):
        if "rank_genes" not in adata.uns.keys():
            sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added='rank_genes')
        result = adata.uns['rank_genes']
        groups = result['names'].dtype.names
        dat = pd.DataFrame({group: result['names'][group] for group in groups})
        df_first_10_rows = dat.head(10)
        rows_as_strings = df_first_10_rows.T.apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        gene_list = '\n'.join([f"{i + 1}.{row}" for i, row in enumerate(rows_as_strings)])
        parts = []
        if tissue_type:
            parts.append(f"of {tissue_type} cells")
        meta = []
        if species:
            meta.append(f"species: {species}")
        if cancer_type:
            meta.append(f"cancer type: {cancer_type}")
        meta_str = f" ({', '.join(meta)})" if meta else ""
        prompt = (
            f"Markers for each cluster ({groupby}):\n"
            f"{gene_list}\n\n"
            f"Identify each cluster cell types {' '.join(parts)}{meta_str} using these markers separately for each row. "
            "Only return the cell type name. Do not show numbers before the cell types name. Some can be a mixture of multiple cell types.\n"
        )
        if openai_api_key is not None:
            client = openai.OpenAI(api_key=openai_api_key)
        else:
            client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        lines = completion.choices[0].message.content.split('\n')
        names = [re.sub(r'^\s*\d+\s*[\.\-:]?\s*', '', line).strip() for line in lines if line.strip()]
        n_cluster = len(groups)
        cell_types = (names + [names[-1]] * n_cluster)[:n_cluster] if names else ["Unknown"] * n_cluster
        cluster2type = dict(zip(groups, cell_types))
        adata.obs['gpt4_predict_type'] = adata.obs[groupby].map(cluster2type).astype('category')
        return adata

    def cellmarker_method(adata, species, tissue_type, cancer_type, groupby):
        
        marker = pd.read_excel("/root/CellAgent/scOmni/maker_database/Cell_marker_Seq.xlsx")
        filtered_df = marker[
            (marker['species'] == species) &
            (marker['tissue_type'] == tissue_type) &
            (marker['cancer_type'] == cancer_type)
        ]
        result_dict = {}
        for cell_name in filtered_df['cell_name'].unique():
            cell_df = filtered_df[filtered_df['cell_name'] == cell_name]
            markers = cell_df['marker'].tolist()
            result_dict[cell_name] = markers

        if "rank_genes" not in adata.uns.keys():
            sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added='rank_genes')

        cell_annotation_norm = sc.tl.marker_gene_overlap(
            adata, result_dict,
            key='rank_genes', 
            normalize='reference',
            adj_pval_threshold=0.05
        )
        max_indexes = cell_annotation_norm.idxmax()
        final = max_indexes.to_dict()
        adata.obs['CellMarker_predict_type'] = adata.obs[groupby].map(final).astype('category')
        adata.obs['cellmarker_predict_type'] = adata.obs['CellMarker_predict_type']
        return adata

    def act_method(adata, species, tissue_type, groupby):
        marker = pd.read_csv("/root/CellAgent/scOmni/maker_database/ACT.csv")
        filtered_df = marker[
            (marker['Species'] == species) &
            (marker['Tissue'] == tissue_type)
        ]
        result_dict = {}
        for cell_name in filtered_df['CellType'].unique():
            cell_df = filtered_df[filtered_df['CellType'] == cell_name]
            markers = cell_df['Marker'].tolist()
            result_dict[cell_name] = markers
        if "rank_genes" not in adata.uns.keys():
            sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', key_added='rank_genes')
        cell_annotation_norm = sc.tl.marker_gene_overlap(
            adata, result_dict,
            key='rank_genes', 
            normalize='reference',
            adj_pval_threshold=0.05
        )
        max_indexes = cell_annotation_norm.idxmax()
        final = max_indexes.to_dict()
        adata.obs['ACT_predict_type'] = adata.obs[groupby].map(final).astype('category')
        adata.obs['act_predict_type'] = adata.obs['ACT_predict_type']
        return adata

    if isinstance(method, str):
        methods = [method]
    else:
        methods = method

    for m in methods:
        m_lower = m.lower()
        if m_lower == "gpt4":
            adata = gpt4_method(adata, species, tissue_type, cancer_type, groupby, openai_api_key)
        elif m_lower == "cellmarker":
            adata = cellmarker_method(adata, species, tissue_type, cancer_type, groupby)
        elif m_lower == "act":
            adata = act_method(adata, species, tissue_type, groupby)
        else:
            print(f"Unknown annotation method: {m}, skipped.")
    return adata

# 将单独获取推荐的method也设置为一个函数
def trajectory_top_k_methods(
    adata,
    time_limit='20m',
    memory_limit='25GB',
    fixed_n_methods='3',
    expected_topology=None,
    start_id=None,
    end_id=None,
    groups_id=None,
    PCA=None,
    X_umap=None
):
    # 初始化
    import scipy.sparse
    import os
    ti_output_path = "./temp/trajectory_inference"
    if not os.path.exists(ti_output_path):
        os.makedirs(ti_output_path, exist_ok=True)
        if 'counts' not in adata.layers:
            # 找到当前所有细胞在 raw.obs_names 里的索引
            cell_indices = [adata.raw.obs_names.get_loc(x) for x in adata.obs_names]
            gene_indices = [adata.raw.var_names.get_loc(v) for v in adata.var_names]
            adata.layers['counts'] = adata.raw.X[cell_indices, :][:, gene_indices]

        # 保证 adata.X、adata.layers['counts'] 都为 csc_matrix，R侧兼容
        if not isinstance(adata.X, np.ndarray):
            if scipy.sparse.isspmatrix_csr(adata.X):
                adata.X = adata.X.tocsc()
            elif scipy.sparse.isspmatrix(adata.X) and not scipy.sparse.isspmatrix_csc(adata.X):
                adata.X = adata.X.tocsc()
        if 'counts' in adata.layers:
            if scipy.sparse.isspmatrix_csr(adata.layers['counts']):
                adata.layers['counts'] = adata.layers['counts'].tocsc()
            elif scipy.sparse.isspmatrix(adata.layers['counts']) and not scipy.sparse.isspmatrix_csc(adata.layers['counts']):
                adata.layers['counts'] = adata.layers['counts'].tocsc()

        # 保存h5ad，供R调用
        sc.write(os.path.join(ti_output_path, 'adata.h5ad'), adata)       



    import subprocess
    import re
    """
    自动组装命令行并调用R脚本，返回方法列表
    """
    # Basic command list
    command = [
        "conda", "run", "-p", "/opt/conda/envs/DYNO_R",
        "Rscript", "/root/CellAgent/scOmni/TI_R/Ti_method.R"
    ]

    # Dynamically build command list based on parameter values
    params = [
        ("time_limit", time_limit),
        ("memory_limit", memory_limit),
        ("fixed_n_methods", fixed_n_methods),
        ("expected_topology", expected_topology),
        ("start_id", start_id),
        ("end_id", end_id),
        ("groups_id", groups_id),
        ("dimred", PCA),
        ("X_umap", X_umap),
        ("adata_path", ti_output_path)
    ]

    for param, value in params:
        if value is not None:
            command.append(f"{param}={value}")

    # 打印实际将要在shell中执行的命令行
    print("Executing command:", " ".join(command))

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Use regex to find all content within double quotes
        print("STDOUT:")
        print(result.stdout)
        methods = re.findall(r'"([^"]+)"', re.search(r'\[\d+\]\s+("[^"]+"(\s+)*)+', result.stdout).group()) if re.search(r'\[\d+\]\s+("[^"]+"(\s+)*)+', result.stdout) else print("No match found")
        # 你可以根据实际脚本输出，决定返回parse_last_line还是methods
        print("Methods found:", methods)
        return methods
    except subprocess.CalledProcessError as e:
        print("ERROR:")
        print(e.stderr)
        raise RuntimeError(f"Command '{command}' failed with error:\n{e.stderr}") from e            

#  轨迹推断 API  直接一步调用的方案 
def trajectory_inference(adata, groupby=None, obsm_embedding=None, start_id=None,expected_topology=None,fixed_n_methods=3):
    def top_k_methods(
        adata,
        time_limit='20m',
        memory_limit='25GB',
        fixed_n_methods='3',
        expected_topology=None,
        start_id=None,
        end_id=None,
        groups_id=None,
        PCA=None,
        X_umap=None
    ):
        
        # # 初始化
        # ti_output_path = "./temp/trajectory_inference"
        # if not os.path.exists(ti_output_path):
        #     os.makedirs(ti_output_path, exist_ok=True)
        #     if 'counts' not in adata.layers:
        #         # 找到当前所有细胞在 raw.obs_names 里的索引
        #         cell_indices = [adata.raw.obs_names.get_loc(x) for x in adata.obs_names]
        #         gene_indices = [adata.raw.var_names.get_loc(v) for v in adata.var_names]
        #         adata.layers['counts'] = adata.raw.X[cell_indices, :][:, gene_indices]
        #     if not isinstance(adata.X, np.ndarray):
        #         adata.X = adata.X.toarray()
        #     sc.write(os.path.join(ti_output_path, 'adata.h5ad'), adata)

        # 初始化
        import scipy.sparse
        import os
        ti_output_path = "./temp/trajectory_inference"
        if not os.path.exists(ti_output_path):
            os.makedirs(ti_output_path, exist_ok=True)
            if 'counts' not in adata.layers:
                # 找到当前所有细胞在 raw.obs_names 里的索引
                cell_indices = [adata.raw.obs_names.get_loc(x) for x in adata.obs_names]
                gene_indices = [adata.raw.var_names.get_loc(v) for v in adata.var_names]
                adata.layers['counts'] = adata.raw.X[cell_indices, :][:, gene_indices]

            # 保证 adata.X、adata.layers['counts'] 都为 csc_matrix，R侧兼容
            if not isinstance(adata.X, np.ndarray):
                if scipy.sparse.isspmatrix_csr(adata.X):
                    adata.X = adata.X.tocsc()
                elif scipy.sparse.isspmatrix(adata.X) and not scipy.sparse.isspmatrix_csc(adata.X):
                    adata.X = adata.X.tocsc()
            if 'counts' in adata.layers:
                if scipy.sparse.isspmatrix_csr(adata.layers['counts']):
                    adata.layers['counts'] = adata.layers['counts'].tocsc()
                elif scipy.sparse.isspmatrix(adata.layers['counts']) and not scipy.sparse.isspmatrix_csc(adata.layers['counts']):
                    adata.layers['counts'] = adata.layers['counts'].tocsc()

            # 保存h5ad，供R调用
            sc.write(os.path.join(ti_output_path, 'adata.h5ad'), adata)       



        import subprocess
        import re
        """
        自动组装命令行并调用R脚本，返回方法列表
        """
        # Basic command list
        command = [
            "conda", "run", "-p", "/opt/conda/envs/DYNO_R",
            "Rscript", "/root/CellAgent/scOmni/TI_R/Ti_method.R"
        ]

        # Dynamically build command list based on parameter values
        params = [
            ("time_limit", time_limit),
            ("memory_limit", memory_limit),
            ("fixed_n_methods", fixed_n_methods),
            ("expected_topology", expected_topology),
            ("start_id", start_id),
            ("end_id", end_id),
            ("groups_id", groups_id),
            ("dimred", PCA),
            ("X_umap", X_umap),
            ("adata_path", ti_output_path)
        ]

        for param, value in params:
            if value is not None:
                command.append(f"{param}={value}")

        # 打印实际将要在shell中执行的命令行
        print("Executing command:", " ".join(command))

        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Use regex to find all content within double quotes
            print("STDOUT:")
            print(result.stdout)
            methods = re.findall(r'"([^"]+)"', re.search(r'\[\d+\]\s+("[^"]+"(\s+)*)+', result.stdout).group()) if re.search(r'\[\d+\]\s+("[^"]+"(\s+)*)+', result.stdout) else print("No match found")
            # 你可以根据实际脚本输出，决定返回parse_last_line还是methods
            print("Methods found:", methods)
            return methods
        except subprocess.CalledProcessError as e:
            print("ERROR:")
            print(e.stderr)
            raise RuntimeError(f"Command '{command}' failed with error:\n{e.stderr}") from e            
    
    def run_method(method_name=None):
        import subprocess
        # Basic command list
        command = ["conda", "run", "-p", "/opt/conda/envs/DYNO_R", "Rscript", "/root/CellAgent/scOmni/TI_R/Ti_Run.R"]
        # Dynamically build command list
        if method_name is not None:
            command.append(f"Methods_selected={method_name}")
        ti_output_path = "./temp/trajectory_inference"
        command.append(f"adata_path={ti_output_path}")

        # Execute the R script and capture output
        print("Executing command:", " ".join(command))
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("STDOUT:")
            print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print("ERROR:")
            print(e.stderr)
            raise RuntimeError(f"Command '{command}' failed with error:\n{e.stderr}") from e
    
    method_list=top_k_methods(adata,start_id=start_id, expected_topology=expected_topology, fixed_n_methods=fixed_n_methods,X_umap=obsm_embedding,groups_id=groupby)

    for method in method_list:
        try:
            run_method(method_name=method)
        except Exception as e:
            print(f"Error running method {method}: {e}")
    

##############################################################
# Spatial API
##############################################################
import scanpy as sc
try:
    import squidpy as sq
except ImportError as e:
    print(f"Exception: Error importing squidpy: {e}")
import subprocess
import os
import numpy as np
import random
import scanpy as sc
from .tool_base import MultiToolBase
import pandas as pd

def save_filtered_raw_data(adata, train_genes, predict_genes,value_genes,filename):
    """
    Save filtered raw data to a CSV file.
    Parameters:
    adata (AnnData): AnnData object containing the raw data.
    train_genes (list): List of genes used for training.
    predict_genes (list): List of genes to be predicted.
    value_genes (list): List of genes used for validation.
    filename (str): The file path where the filtered data will be saved.

    Returns:
    None: This function saves the filtered data to a CSV file.
    """
    try:
        # 提取 raw 数据并转换为 DataFrame
        raw_data = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
        # 过滤基因
        genes = train_genes + predict_genes + value_genes
        filtered_data = raw_data[genes].copy()
        filtered_data.to_csv(filename, sep=',', index=True, header=True)
        print(f"Filtered raw data saved to {filename}")
    except Exception as e:  
        raise ValueError("No raw data available in adata.")
        
def save_filtered_raw_data_st(adata, train_genes, predict_genes,value_genes,filename):
    """
    Save filtered raw data for spatial transcriptomics to a CSV file.

    Parameters:
    adata (AnnData): AnnData object containing the raw data.
    train_genes (list): List of genes used for training.
    predict_genes (list): List of genes to be predicted.
    value_genes (list): List of genes used for validation.
    filename (str): The file path where the filtered data will be saved.

    Returns:
    None: This function saves the filtered data to a CSV file.
    """
    # 检查 adata 是否具有 raw 属性
    try:
        # 提取 raw 数据并转换为 DataFrame
        raw_data = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
        # 过滤基因
        genes = train_genes + value_genes
        filtered_data = raw_data[genes].copy()
        # 将 predict_genes 对应的列设置为 -1
        for gene in predict_genes:
            filtered_data.loc[:, gene] = -1.0
        filtered_data.to_csv(filename, sep=',', index=True, header=True)
        print(f"Filtered raw data saved to {filename}")
    except Exception as e:
        raise ValueError("No raw data available in adata.")
                
def save_spatial_coordinates_as_txt(adata, filename):
    """
    Save the spatial coordinates from adata.obsm['spatial'] into a txt file.
    The first row will contain column headers 'x' and 'y'.

    Parameters:
    - adata: AnnData object containing spatial coordinates in adata.obsm['spatial']
    - filename: The file path where the spatial coordinates will be saved

    Returns:
    None: This function saves the spatial coordinates to a text file.
    """
    # Check if spatial coordinates exist in adata.obsm
    if 'spatial' not in adata.obsm:
        print("Error: Spatial data not found in adata.obsm['spatial']")
        return

    # Extract the spatial coordinates from adata.obsm['spatial']
    spatial_coords = adata.obsm['spatial']

    # Convert to a DataFrame with 'x' and 'y' as column names
    spatial_df = pd.DataFrame(spatial_coords, columns=['x', 'y'])

    # Save the DataFrame to a tab-separated text file (.txt)
    spatial_df.to_csv(filename, sep='\t', index=False, header=True)
    print(f"Spatial coordinates saved to {filename}")

def to_adata_raw(adata):
    import pandas as pd
    import scanpy as sc

    if adata.raw is not None:
        raw_data = pd.DataFrame(adata.raw.X.toarray(), columns=adata.raw.var_names, index=adata.raw.obs_names)
        adata_raw = sc.AnnData(raw_data)

    else:
        raw_data = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
        adata_raw = sc.AnnData(raw_data)

    if 'spatial' in adata.obsm:
        adata_raw.obsm['spatial'] = adata.obsm['spatial']

    return adata_raw 

def to_adata_raw(adata):
    """
    Creates a new AnnData object from adata.raw if available; otherwise uses adata.X.
    Keeps the 'spatial' data in adata.obsm if present.
    """
    import pandas as pd
    import scanpy as sc

    # Decide which data (raw or X) to use
    if adata.raw is not None:
        data_source = adata.raw.X
        var_names = adata.raw.var_names
        obs_names = adata.raw.obs_names
    else:
        data_source = adata.X
        var_names = adata.var_names
        obs_names = adata.obs_names

    # Convert sparse matrix (if present) to a dense array
    if hasattr(data_source, "toarray"):
        data_source = data_source.toarray()

    # Build pandas DataFrame and then create the AnnData object
    raw_data = pd.DataFrame(data_source, columns=var_names, index=obs_names)
    adata_raw = sc.AnnData(raw_data)

    # Retain 'spatial' data from the original AnnData's .obsm
    if "spatial" in adata.obsm:
        adata_raw.obsm["spatial"] = adata.obsm["spatial"]

    return adata_raw

def ensure_highly_variable_genes(adata):
    # Copy the original data matrix
    original_X = adata.X.copy()
    # Check if 'highly_variable' column exists in adata.var
    if 'highly_variable' not in adata.var.columns:
        # Normalize the total counts per cell
        sc.pp.normalize_total(adata, target_sum=1e4)
        # Logarithmize the data
        sc.pp.log1p(adata)
        # Identify highly variable genes
        sc.pp.highly_variable_genes(adata)
    # Restore the original data matrix
    adata.X = original_X

# 空间域（空间聚类） API
def spatial_domain_identify(adata_st, n_domains=None, methods=None):
    """
    Identify spatial domains in spatial transcriptomics data using specified methods.

    This function takes an AnnData object containing spatial transcriptomics data
    and applies one or more domain identification methods to it. It also saves
    relevant visualizations and intermediate results.

    Parameters:
    adata_st (anndata.AnnData): The AnnData object containing spatial transcriptomics data.
                            It should have the necessary information such as the spatial coordinates
                            and gene expression values.
    n_domains (int, optional): The number of domains to identify. If not provided, it will be
                            determined by the individual methods. Defaults to None.
    methods (list of str, optional): A list of methods to use for spatial domain identification.
                                    Supported methods include 'DeepST', 'stLearn', 'SEDR'.
                                    If not provided, the default list ['DeepST', 'stLearn', 'SEDR'] will be used.
                                    Defaults to None.

    Returns:
    None: This function does not return a value directly. However, it saves the following:
        - A histopathological plot showing the spatial data to './tmp/spatial_domains/histopathological_plot'.
    Example:
    >>> # specify the number of domains and methods to use
    >>> # here assume the user specifies n_domains as 7, but if usr do not specify the number of domains, n_domains should be None !
    >>> spatial_domain_identify(adata, n_domains=7, methods=['DeepST', 'stLearn', 'SEDR']) # specify the number of domains and methods to use 
    >>> # Note:  
    >>> try:
            adata_DeepST = sc.read('./tmp/spatial_domains/DeepST_results/adata.h5ad')
            sc.pl.spatial(adata_DeepST, color='DeepST_clusters', title='Spatial Domains by DeepST')
        except Exception as e:
            print(f"An error occurred: {e}")
        try:
            adata_stLearn = sc.read('./tmp/spatial_domains/stLearn_results/adata.h5ad')
            sc.pl.spatial(adata_stLearn, color='stLearn_clusters', title='Spatial Domains by stLearn')
        except Exception as e:
            print(f"An error occurred: {e}")
        try:
            adata_SEDR = sc.read('./tmp/spatial_domains/SEDR_results/adata.h5ad')
            sc.pl.spatial(adata_SEDR, color='SEDR_clusters', title='Spatial Domains by SEDR')
        except Exception as e:
            print(f"An error occurred: {e}")
    """
    adata_st.write('./adata_spatial_domain.h5ad')
    import matplotlib.pyplot as plt
    import os 
    # 检测文件夹是否存在，如果不存在则创建
    os.makedirs('./tmp/spatial_domains', exist_ok=True)
    sc.pl.spatial(adata_st, img_key="hires",show=False)
    plt.savefig('./tmp/spatial_domains/histopathological_plot', bbox_inches='tight')
    plt.close() 
    if methods is None:
        methods = ['DeepST', 'stLearn', 'SEDR']
        
    import subprocess
    python_interpreters = {
        'DeepST': '/data/cellana/anaconda3/envs/deepst_env/bin/python',
        'stLearn': '/home/cellana/anaconda3/envs/cellana/envs/stlearn/bin/python',
        'SEDR': '/home/cellana/anaconda3/envs/SEDR/bin/python',
    }
    script_paths = {
        'DeepST': '/data/cellana/空转单细胞分析/空转分析脚本/DeepST_cluster.py',
        'stLearn': '/data/cellana/空转单细胞分析/空转分析脚本/stLearn_cluster.py',
        'SEDR': '/data/cellana/空转单细胞分析/空转分析脚本/SEDR_cluster.py',
    }

    for method in methods:
        if method not in python_interpreters or method not in script_paths:
            print(f"Method {method} is not recognized.")
            continue
        try:
            # Construct the command
            command = [python_interpreters[method], script_paths[method]]
            command.append(str(n_domains))
            # Execute the script using subprocess
            result = subprocess.run(
                command,
                text=True,              # Capture output as a string
                stdout=subprocess.PIPE, # Capture standard output
                stderr=subprocess.PIPE, # Capture standard error
                check=True,             # Raise an exception if the command fails
            )

            # Print the script's output
            #print(result.stdout.strip())

        except subprocess.CalledProcessError as e:
            print(f"Error running the script for method {method}:\n{e.stderr}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# 空间聚类 绘图 API
def spatial_domain_plot(method=None):
    """
    Function to display the image visualization results corresponding to specific spatial domain methods.

    Parameters:
    method (str or None): Specifies the name of the method for which the image is to be displayed. Valid options are 'DeepST', 'stLearn', 'SEDR'. If it is None, the function will iterate through and display the images for all available methods.
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    plots_paths = {
        'DeepST': './tmp/spatial_domains/DeepST_results/DeepST_refine_plot.png',
        'stLearn': './tmp/spatial_domains/stLearn_results/stLearn_plot.png',
        'SEDR': './tmp/spatial_domains/SEDR_results/SEDR_leiden_plot.png',
    }

    # 如果method为None，则遍历所有方法
    if method is None:
        methods = plots_paths.keys()
    else:
        methods = [method] if method in plots_paths else []  # 确保method在字典中

    # 遍历并显示图片，若文件不存在则跳过
    for method in methods:
        img_path = plots_paths[method]
        if os.path.exists(img_path):  # 判断文件是否存在
            img = mpimg.imread(img_path)
            plt.imshow(img)
            # plt.title(f'{method} Visualization')
            plt.axis('off')
            plt.show()
        else:
            print(f"Image for {method} does not exist, skipping.")         

# 空间轨迹 API
def stLearn_spatial_trajectory(adata, use_label="louvain_morphology", cluster=0):
    """
    Perform spatial trajectory analysis using stLearn.
    
    Parameters:
    adata (AnnData): The input data.
    use_label (str): The label to use for clustering. Defaults to "louvain_morphology".
    cluster (int): The cluster to set as root for trajectory analysis. Defaults to 0.
    
    Returns:
    dict: A dictionary of available paths for trajectory analysis.

    Example:
    >>> stLearn_spatial_trajectory(adata, use_label="louvain", cluster=0) 
    >>> # The function will automatically calculate the trajectory and plot the trajectory, so there is no need to generate code to plot further.
    """
    
    # Define paths for output
    # temp_path = "/data/cellana/空转单细胞分析/stlearn_tmp"
    # Write the AnnData object to a temporary file
    import subprocess
    import os 
    # 检测文件夹是否存在，如果不存在则创建
    temp_path = "./tmp/spatial_trajectory" 
    os.makedirs(temp_path, exist_ok=True)
    # adata_ = adata.copy()
    adata.write(f'{temp_path}/adata.h5ad')

    # 创建 TI_plot 文件夹，如果不存在的话
    os.makedirs("./tmp/spatial_trajectory/TI_plot" , exist_ok=True)
    ti_plot_path = "./tmp/spatial_trajectory/TI_plot"

    # Specify the virtual environment's Python interpreter
    python_interpreter = "/home/cellana/anaconda3/envs/cellana/envs/stlearn/bin/python"
    script_path = '/data/cellana/空转单细胞分析/空转分析脚本/stLearn_spatial_trajectory.py'
    
    #Call the script using subprocess
    subprocess.run([
        python_interpreter, 
        script_path, 
        temp_path, 
        use_label, 
        str(cluster),
        ti_plot_path,
    ])
    # Display images in the TI_plot folder
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    
    for img_file in os.listdir(ti_plot_path):
        if img_file.endswith(".png"):
            display(Image(filename=os.path.join(ti_plot_path, img_file)))   

# 空间邻域分析 API
def Compute_interaction_matrix(adata, cluster_key):
    """
    Compute and visualize spatial neighborhood richness analysis for clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data.
    cluster_key : str
        Key in `adata.obs` where clustering information is stored.

    details of this function:
    def Compute_interaction_matrixCompute_interaction_matrix(adata, cluster_key):
        sq.gr.spatial_neighbors(adata)
        sq.gr.interaction_matrix(adata, cluster_key=cluster_key)
        sq.pl.interaction_matrix(adata, cluster_key=cluster_key)
    -------
    None
        Modifies the `adata` object to include the interaction matrix,
        stored in `adata.uns['{cluster_key}_interactions']`,
        and generates a plot of the interaction matrix.

    Example:
    >>> # The function will automatically calculate interaction_matrix and plot the image, so there is no need to generate code further.
    >>> Compute_interaction_matrix(adata, cluster_key="louvain") 
    """
    # Compute spatial neighbors
    sq.gr.spatial_neighbors(adata)
    # Compute interaction matrix for clusters
    sq.gr.interaction_matrix(adata, cluster_key=cluster_key)
    # Plot the interaction matrix
    sq.pl.interaction_matrix(adata, cluster_key=cluster_key)

def Compute_co_occurrence_probability(adata, cluster_key, chosen_clusters):
    """
    Compute the co-occurrence probability for specified clusters and visualize the results.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data and spatial information.
    cluster_key : str
        Key in `adata.obs` where clustering information is stored.
    chosen_clusters : str | Sequence[str]
        Specific cluster instances to plot, can be a single cluster or a list of multiple clusters.

    details of this function:
    def Compute_co_occurrence_probability(adata, cluster_key, chosen_clusters):
        sq.gr.co_occurrence(adata, cluster_key=cluster_key)
        sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters=chosen_clusters)
        sq.pl.spatial_scatter(adata, color=cluster_key, shape=None)

    Returns
    -------
    None
        This function directly displays the results in the plotting interface and does not return any value.
    """
    sq.gr.co_occurrence(adata, cluster_key=cluster_key)
    sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters=chosen_clusters)
    sq.pl.spatial_scatter(adata, color=cluster_key, shape=None)

def receptor_ligand_analysis(adata, cluster_key, source_groups, n_perms=1000, threshold=0, alpha=0.005):
    """
    Perform receptor-ligand analysis on the given AnnData object.
    
    Parameters:
    - adata: AnnData object containing the spatial transcriptomics data.
    - cluster_key: Key to identify the clusters in the data.
    - source_groups: Groups to visualize in the ligand-receptor analysis.
    - n_perms: Number of permutations for significance testing (default is 1000).
    - threshold: Minimum significance threshold for receptor-ligand interactions (default is 0).
    - alpha: Significance level for the analysis (default is 0.005).
    
    Returns:
    - res: Results of the receptor-ligand analysis as a new AnnData object.

    details of this function:
    def receptor_ligand_analysis( adata, cluster_key, source_groups, n_perms=1000, threshold=0, alpha=0.005):
        res = sq.gr.ligrec(
        adata,
        n_perms=n_perms,  # Number of permutations for significance testing
        cluster_key=cluster_key,  # Key to identify clusters
        copy=True,  # Copy results to a new AnnData object
        use_raw=True,  # Use raw data for the analysis
        transmitter_params={"categories": "ligand"},  # Parameters for ligand (transmitter)
        receiver_params={"categories": "receptor"},  # Parameters for receptor (receiver)
        threshold=threshold,  # Minimum threshold for significant interactions
        )
        sq.pl.ligrec(res, source_groups=source_groups, alpha=alpha)
        return res
    
    """
    # Perform ligand-receptor analysis on the data
    res = sq.gr.ligrec(
        adata,
        n_perms=n_perms,  # Number of permutations for significance testing
        cluster_key=cluster_key,  # Key to identify clusters
        copy=True,  # Copy results to a new AnnData object
        use_raw=True,  # Use raw data for the analysis
        transmitter_params={"categories": "ligand"},  # Parameters for ligand (transmitter)
        receiver_params={"categories": "receptor"},  # Parameters for receptor (receiver)
        threshold=threshold,  # Minimum threshold for significant interactions
    )
    # Visualize the results of the receptor-ligand analysis
    sq.pl.ligrec(res, source_groups=source_groups, alpha=alpha)
    
    return res  # Return the results of the analysis

def analyze_spatial_autocorr(adata):
    """
    Analyze spatial autocorrelation using Moran's I score and identify the most spatially correlated genes.
    
    Parameters:
    - adata: AnnData object containing the spatial transcriptomics data.
            
    Returns:
    -  Moran's I score will save in adata.uns["moranI"].

    This function will compute the Moran's I score to identify genes with strong spatial autocorrelation,
    similar to how spatially variable genes are identified.

    details of this function:
    def analyze_spatial_autocorr( adata):
        ...
        # Omit the intermediate code. 
        sq.gr.spatial_autocorr(
            adata,
            mode="moran",  # Use Moran's I method for spatial autocorrelation
            genes=genes,  # Genes to analyze
            n_perms=100,  # Number of permutations for significance testing
            n_jobs=4,  # Number of parallel jobs to run 4 for sequential)
        )
        adata.uns["moranI"].head(10)
        print('top 10 genes with the highest Moran\'s I score\n',adata.uns["moranI"].head(10))

    Example:
    >>> ...
    >>> SpatialTranscriptomics.analyze_spatial_autocorr(adata_st)
    >>> top_genes = adata_st.uns["moranI"].head(10).index
    >>> sq.pl.spatial_scatter(adata_st, color=top_genes[:3])

            
    """
    if 'highly_variable' not in adata.var.columns:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    # Select the top 100 highly variable genes
    genes = adata[:, adata.var.highly_variable].var_names.values[:100]
    # Compute spatial neighbors for the AnnData object
    sq.gr.spatial_neighbors(adata)
    # Calculate Moran's I score for the selected genes
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",  # Use Moran's I method for spatial autocorrelation
        genes=genes,  # Genes to analyze
        n_perms=100,  # Number of permutations for significance testing
        n_jobs=4,  # Number of parallel jobs to run (1 for sequential)
    )
    # Display the top 10 genes with the highest Moran's I scores
    adata.uns["moranI"].head(10)
    print('top 10 genes with the highest Moran\'s I score\n',adata.uns["moranI"].head(10))
    # Visualize the spatial distribution of the top genes based on Moran's I score
    #top_genes = adata.uns["moranI"].head(10).index
    #print('adata.uns["moranI"].head(10).index\n',top_genes)
    #sq.pl.spatial_scatter(adata, color=top_genes[:3])  # Scatter plot of spatial distribution

def compute_ripley( adata, cluster_key, mode="L"):
    """
    Compute Ripley's statistics to analyze the spatial distribution patterns of cells.
    
    Parameters:
    - adata: AnnData object containing the spatial transcriptomics data.
    - cluster_key: Key in `adata.obs` that indicates the clustering of cells.
    - mode: The mode of Ripley's statistic to compute. Default is "L", which is commonly used.
    
    This function calculates Ripley's K-function or L-function to assess the spatial distribution
    of cells and determine whether they exhibit randomness, clustering, or regularity.

    details of this function:
    def compute_ripley( adata, cluster_key, mode="L"):
        sq.gr.ripley(adata, cluster_key=cluster_key, mode=mode)
        sq.pl.ripley(adata, cluster_key=cluster_key, mode=mode)
    """
    # Calculate Ripley's statistics for the specified clustering
    sq.gr.ripley(adata, cluster_key=cluster_key, mode=mode)
    
    # Visualize the results of Ripley's statistics
    sq.pl.ripley(adata, cluster_key=cluster_key, mode=mode)


# 空间插补
first_impute_method_run = True
def impute_method_run(adata_sc,adata_st,train_genes=None,predict_genes=None,value_genes=None,methods=['Tangram', 'gimVI', 'SpaGE']):
    """
    Run imputation methods on the given single-cell and spatial transcriptomics data.

    Parameters:
    adata_sc (AnnData): AnnData object containing single-cell data.
    adata_st (AnnData): AnnData object containing spatial transcriptomics data.
    train_genes (list, optional): List of genes used for training. Defaults to None.
    predict_genes (list, optional): List of genes to be predicted. Defaults to None.
    value_genes (list, optional): List of genes used for validation. Defaults to None.
    methods (list of str, optional): List of imputation methods to use. Defaults to ['Tangram', 'gimVI', 'SpaGE']. 

    Returns:
    None: This function runs the imputation methods and saves the results.
    Example:
    >>> methods=['Tangram','gimVI']
    >>> # specify the methods to run , if ues require these methods, otherwise the default methods should be run.
    >>> # if train_genes and predict_genes are provided, use them, otherwise, the function will automatically select genes.
    >>> SpatialTranscriptomics.impute_method_run(adata_sc,adata_st,methods=methods) 
    >>> # 尝试可视化预测结果
    >>> try: 
            # if Tangram imputation is available
            adata_Tangram = SpatialTranscriptomics.get_imputed_anndata(adata_st=adata_st,method='Tangram')
            genes = list(adata_Tangram.var_names)[:3]
            # 可视化这三个基因
            for gene in genes:
                sc.pl.spatial(adata_Tangram, color=gene, title=f'Expression of {gene}', show=True)
                # if 用户指定  spot_size = 50 ，请使用 sc.pl.spatial(adata_Tangram, color=gene, title=f'Expression of {gene}', spot_size = 50,show=True)
        except Exception as e:
            print(f"An error occurred: {e}") 
    
        try:
            adata_gimVI = SpatialTranscriptomics.get_imputed_anndata(adata_st=adata_st,method='gimVI')
            genes = list(adata_gimVI.var_names)[:3]
            for gene in genes:
                sc.pl.spatial(adata_gimVI, color=gene, title=f'Expression of {gene}', show=True)
        except Exception as e:
            print(f"An error occurred: {e}")
        ...
    """
    adata_sc = to_adata_raw(adata_sc)
    adata_st = to_adata_raw(adata_st) 

    # global first_impute_method_run
    if first_impute_method_run:
        # 标记为 False，确保后续调用不再执行
        first_impute_method_run = False
        if train_genes is None:
            ensure_highly_variable_genes(adata_st)
            ensure_highly_variable_genes(adata_sc)

            # 获取adata_st_和adata_sc_的高变基因
            highly_variable_genes_st = set(adata_st.var[adata_st.var['highly_variable']].index)
            highly_variable_genes_sc = set(adata_sc.var[adata_sc.var['highly_variable']].index)

            # 取交集
            highly_variable_genes_intersection = highly_variable_genes_st & highly_variable_genes_sc

            # 获取原始数据的基因交集
            raw_genes_intersection = set(adata_sc.var_names) & set(adata_st.var_names)

            # 合并高变基因交集和原始数据基因交集
            combined_genes = highly_variable_genes_intersection | raw_genes_intersection

            # 如果合并后的基因数超过1000，优先选择高变基因
            if len(combined_genes) > 1000:
                combined_genes = list(highly_variable_genes_intersection)[:1000] + list(raw_genes_intersection - highly_variable_genes_intersection)[:1000-len(highly_variable_genes_intersection)]

            # 最终选取的1000个基因
            train_genes = list(combined_genes)[:1000]

        if predict_genes is None:
            predict_genes = list(set(adata_sc.var_names)-set(adata_st.var_names))[:10]

        if value_genes is None:
            value_genes_ = list(set(adata_sc.var_names) & set(adata_st.var_names) - set(train_genes))
            value_genes_len = int(len(predict_genes)*0.2)+1  #保证至少选中一个
            value_genes = value_genes_[:value_genes_len]

        input_dir='./tmp/imputation'
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        save_filtered_raw_data(adata_sc,train_genes, predict_genes,value_genes, input_dir + '/Rna_data.csv')
        save_filtered_raw_data_st(adata_st,train_genes, predict_genes,value_genes,input_dir + '/Spatial_data.csv')
        save_spatial_coordinates_as_txt(adata_st, input_dir+'/Locations.txt')
        np.save(input_dir + '/train_list.npy', np.array(train_genes))
        np.save(input_dir + '/test_list.npy', np.array(predict_genes+value_genes))
        np.save(input_dir + '/value_list.npy', np.array(value_genes))
        
    # if methods is None:
    #     # methods = ['SpaGE','novoSpaRc','stPlus','Tangram','gimVI']
    #     methods = ['Tangram','gimVI','SpaGE']
    methods = str(methods)


    import subprocess
    python_interpreter = "/home/cellana/anaconda3/envs/Benchmarking/bin/python"
    script_path = "/data/cellana/空转单细胞分析/空转分析脚本/基因插补运行脚本-Copy1.py"
    try:
        # Construct the command
        command = [python_interpreter, script_path,methods]
        # /home/cellana/anaconda3/envs/Benchmarking/bin/python /data/cellana/空转单细胞分析/空转分析脚本/基因插补运行脚本-Copy1.py "['Tangram', 'gimVI', 'SpaGE']"
        # Execute the script using subprocess
        result = subprocess.run(
            command,
            text=True,              # Capture output as a string
            stdout=subprocess.PIPE, # Capture standard output
            stderr=subprocess.PIPE, # Capture standard error
            check=True,             # Raise an exception if the command fails
            # env=os.environ  # Ensure the environment is correctly passed
        )

        # Return the script's output, stripped of extra whitespace
        # return  ast.literal_eval(result.stdout.strip())

    except subprocess.CalledProcessError as e:
        print(f"Error running the script:\n{e.stderr}")
        raise
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def get_imputed_anndata(adata_st,method):
    """
    Get the imputed AnnData object for the specified method.

    Parameters:
    adata_st (AnnData): AnnData object containing spatial transcriptomics data.
    method (str): The imputation method to use.

    Returns:AnnData
        The imputed AnnData object. In this object:
        - `adata.X` contains the imputed expression matrix. where all genes are imputed (predicted) unknown genes.
    
    Example:
    >>> try: 
            # if Tangram imputation is available
            adata_Tangram = SpatialTranscriptomics.get_imputed_anndata(adata_st=adata_st,method='Tangram')
            genes = adata_Tangram.var_names
            random_genes = random.sample(list(genes), 3)
            # 可视化这三个基因
            for gene in random_genes:
                sc.pl.spatial(adata_Tangram, color=gene, title=f'Expression of {gene}', show=True)
        except Exception as e:
            print(f"An error occurred: {e}")    
    """
    impute_genes_paths = {
        'Tangram': './tmp/imputation/Tangram_impute.csv',
        'gimVI': './tmp/imputation/gimVI_impute.csv',
        'SpaGE': './tmp/imputation/SpaGE_impute.csv',
        'stPlus': './tmp/imputation/stPlus_impute.csv',
        'novoSpaRc': './tmp/imputation/novoSpaRc_impute.csv',
    }
    
    if method in impute_genes_paths:
        file_path = impute_genes_paths[method]
        if os.path.exists(file_path):
            imputed_data = pd.read_csv(file_path,index_col=0)
            adata_imputation = sc.AnnData(imputed_data) 
            try:
                # 尝试复制 spatial 数据
                adata_imputation.uns['spatial'] = adata_st.uns['spatial'].copy()
            except Exception as e:
                print(f"KeyError: 'spatial' not found in adata_st.uns: {e}")
            try:
                adata_imputation.obsm['spatial'] = adata_st.obsm['spatial'].copy()
            except Exception as e:
                print(f"KeyError: 'spatial' not found in adata_st.obsm: {e}")
    else:
        print(f"Unknown method: {method}")
    print('imputed adata info:')
    # 打印 var 的前 5 行
    print(adata_imputation)
    return adata_imputation


















