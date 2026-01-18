import itertools
import os
import hashlib

def generate_configs(fixed_params, variable_params, param_filters=None, constraints=None, output_dir="configs", id = 0):
    """
    生成配置文件，支持针对 path 的 param_filter、跨参数约束，并自动去重
    """
    os.makedirs(output_dir, exist_ok=True)

    # 提取 path 值
    path_values = []
    if "path" in fixed_params:
        path_values = [fixed_params["path"]]
    elif "path" in variable_params:
        path_values = variable_params["path"]
    else:
        raise ValueError("必须在 fixed_params 或 variable_params 里指定 path 参数")

    seen_configs = set()  # 去重用

    for path in path_values:
        # 根据 path 过滤参数
        active_variable_params = {k: v for k, v in variable_params.items() if k != "path"}
        if param_filters and path in param_filters:
            for k, v in param_filters[path].items():
                if k in active_variable_params:
                    active_variable_params[k] = [val for val in active_variable_params[k] if val in v]

        # 笛卡尔积
        keys = list(active_variable_params.keys())
        values = list(active_variable_params.values())
        all_combinations = itertools.product(*values)

        generated = 0
        for combo in all_combinations:
            config = {}
            config.update(fixed_params)
            config["path"] = path
            for k, v in zip(keys, combo):
                config[k] = v

            # 应用跨参数约束
            if constraints and not all(rule(config) for rule in constraints):
                continue

            # 去重
            config_key = tuple(sorted(config.items()))
            print(config_key)
            if config_key in seen_configs:
                continue
            seen_configs.add(config_key)

            # 文件名
            config_str = str(sorted(config.items()))
            short_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

            filename = f"cfg_{id}_{os.path.basename(path.strip('/'))}_{short_hash}.conf"

            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                for k, v in config.items():
                    f.write(f"{k} = {v}\n")

            generated += 1

        print(f"[{path}] 生成 {generated} 个唯一配置文件")


if __name__ == "__main__":
    # 示例：固定参数
    fixed_super_grid = {
        "enable_multi_gpu": 0,
        "break_down": 1,
        "is_calc_mem_predictor_coeff": 0,
        "pq_dim" : 8, 
        "n_clusters" : 256,
        "lowest_query_batch" : 125,
        "label_dim" : 1,
        "max_grids" : 30000000, 
        "mem_bound" : 5000000000, 
    }

    fixed_ivfpq = {
        "enable_multi_gpu": 0,
        "break_down": 1,
        "is_calc_mem_predictor_coeff": 0,
        "pq_dim" : 4, 
        "pq_bits" : 8,
        "kmeans_n_iters" : 8,
        "lowest_query_batch" : 125, 
        "label_dim" : 1,
        "mem_bound" : 5000000000
    }

    variables_ivfpq = [
        {
            "data_width" : [1], 
            "path" : ["\"dataset/ag_news/\""],
            "n_list" : [128],
            "n_probes" : [8, 16],
            "beta" : [1, 5, 10, 20, 30],
            "refine" : [4, 8], 
            "filter_dim" : [1],
            "topk" : [100]
        }, 
        {
            "data_width" : [1], 
            "path" : ["\"dataset/cc_news/\""],
            "n_list" : [768],
            "n_probes" : [16, 32, 64],
            "beta" : [1, 5, 10, 20, 35, 50],
            "refine" : [8, 15, 30], 
            "filter_dim" : [1],
            "topk" : [100]
        }, 
        {
            "data_width" : [1], 
            "path" : ["\"dataset/app_reviews/\""],
            "n_list" : [512],
            "n_probes" : [32, 64],
            "beta" : [5, 10, 15, 20, 35, 50],
            "refine" : [5, 10, 15, 20], 
            "filter_dim" : [1],
            "topk" : [100]
        }, 
        {
            "data_width" : [4], 
            "path" : ["\"dataset/amazon/\""],
            "n_list" : [1024],
            "n_probes" : [64, 128],
            "beta" : [50, 60, 90],
            "refine" : [5, 10, 15, 30], 
            "filter_dim" : [1],
            "topk" : [100]
        }, 
        {
            "data_width" : [1], 
            "path" : ["\"dataset/msong/\""],
            "n_list" : [1024],
            "n_probes" : [128],
            "beta" : [2],
            "refine" : [1, 5, 10, 15, 30, 50], 
            "filter_dim" : [2],
            "topk" : [200]
        }, 
        {
            "data_width" : [1], 
            "path" : ["\"dataset/deep1m/\""],
            "n_list" : [1024],
            "n_probes" : [128],
            "beta" : [2],
            "refine" : [1, 5, 10, 15, 30, 50], 
            "filter_dim" : [2],
            "topk" : [200]
        }, 
        {
            "data_width" : [1], 
            "path" : ["\"dataset/gist/\""],
            "n_list" : [1024],
            "n_probes" : [128],
            "beta" : [100, 200, 400],
            "refine" : [1, 2, 3, 5, 10], 
            "filter_dim" : [2],
            "topk" : [200]
        }, 
        {
            "data_width" : [5], 
            "path" : ["\"dataset/sift5m/\""],
            "n_list" : [1024],
            "n_probes" : [128],
            "beta" : [35, 50, 100],
            "refine" : [1, 3, 5, 10, 15, 30], 
            "filter_dim" : [2],
            "topk" : [200]
        }, 
    ]

    variables_super_grid_scalabity = [
        {
            "path": ["\"dataset/sift10m_1label/1/\""],
            "clusters": [999],
            "sub_clusters": [10], 
            "n_list": [249], 
            "sub_lists" : [249, 311, 373, 435, 498], 
            "exps0" : [29], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/sift10m_1label/2/\""],
            "clusters": [1413],
            "sub_clusters": [10], 
            "n_list": [353], 
            "sub_lists" : [353, 441, 529, 617, 706], 
            "exps0" : [41], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/sift10m_1label/3/\""],
            "clusters": [1731],
            "sub_clusters": [10],
            "n_list": [432], 
            "sub_lists" : [432, 540, 648, 756, 864], 
            "exps0" : [50], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
         {
            "path": ["\"dataset/sift10m_1label/4/\""],
            "clusters": [1999],
            "sub_clusters": [10], 
            "n_list": [499], 
            "sub_lists" : [499, 623, 748, 873, 998], 
            "exps0" : [58], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/sift10m_1label/5/\""],
            "clusters": [2235],
            "sub_clusters": [10], 
            "n_list": [558], 
            "sub_lists" : [558, 697, 837, 976, 1116], 
            "exps0" : [65], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/sift10m_1label/6/\""],
            "clusters": [2448],
            "sub_clusters": [10], 
            "n_list": [612], 
            "sub_lists" : [612, 765, 918, 1071, 1224], 
            "exps0" : [71], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/sift10m_1label/7/\""],
            "clusters": [2644],
            "sub_clusters": [10], 
            "n_list": [661], 
            "sub_lists" : [661, 826, 991, 1156, 1322], 
            "exps0" : [77], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/sift10m_1label/8/\""],
            "clusters": [2827],
            "sub_clusters": [10], 
            "n_list": [706], 
            "sub_lists" : [706, 882, 1059, 1235, 1412], 
            "exps0" : [82], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/sift10m_1label/9/\""],
            "clusters": [2998],
            "sub_clusters": [10], 
            "n_list": [749], 
            "sub_lists" : [749, 936, 1123, 1310, 1498], 
            "exps0" : [87], 
            "label_mask" : [0], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
    ]

    # 示例：可变参数
    variables_super_grid = [
        # {
        #     "path": ["\"dataset/sift5m_new/\""],
        #     "clusters": [1024],
        #     "sub_clusters": [10], 
        #     "n_list": [32, 64, 128, 192, 256], 
        #     "sub_lists" : [64, 128, 256, 384, 512], 
        #     "exps0" : [10, 15, 30], 
        #     "label_mask" : [0], 
        #     "data_width" : [5], 
        #     "is_split" : [1], 
        #     "n_split" : [32], 
        #     "filter_dim" : [2], 
        #     "topk": [200],
        # }, 
        # {
        #     "path": ["\"dataset/cc_news/\""],
        #     "clusters": [768],
        #     "sub_clusters": [10], 
        #     "n_list": [128], 
        #     "sub_lists" : [256], 
        #     "exps0" : [10, 20, 30, 40, 50], 
        #     "label_mask" : [0], 
        #     "data_width" : [1], 
        #     "is_split" : [0], 
        #     "n_split" : [1], 
        #     "filter_dim" : [1], 
        #     "topk": [100],
        # }, 
        # {
        #     "path": ["\"dataset/msong/\""],
        #     "clusters": [999],
        #     "sub_clusters": [10], 
        #     "n_list": [128], 
        #     "sub_lists" : [128, 256], 
        #     "exps0" : [30, 40, 50, 60, 70, 80, 90], 
        #     "label_mask" : [0], 
        #     "data_width" : [1], 
        #     "is_split" : [0], 
        #     "n_split" : [1], 
        #     "filter_dim" : [2], 
        #     "topk": [200],
        # }, 
        # {
        #     "path": ["\"dataset/sift10m/slice/5/\""],
        #     "clusters": [1413],
        #     "sub_clusters": [10], 
        #     "n_list": [353], 
        #     "sub_lists" : [353, 441, 529, 617, 706], 
        #     "exps0" : [41], 
        #     "label_mask" : [0], 
        #     "data_width" : [2], 
        #     "is_split" : [1], 
        #     "n_split" : [32], 
        #     "filter_dim" : [2], 
        #     "topk": [200],
        # }, 
        # """"
        # {
        #     "path": ["\"dataset/cc_news/\""],
        #     "clusters": [768],
        #     "sub_clusters": [5], 
        #     "n_list": [128], 
        #     "sub_lists" : [8, 16, 32, 64, 128, 256], 
        #     "exps0" : [15, 20], 
        #     "label_mask" : [0], 
        #     "data_width" : [1], 
        #     "is_split" : [0], 
        #     "n_split" : [1], 
        #     "filter_dim" : [1], 
        #     "topk": [100],
        # }, 
        # {
        #     "path": ["\"dataset/sift10m/\""],
        #     "clusters": [2048],
        #     "sub_clusters": [10], 
        #     "n_list": [64, 128, 256], 
        #     "sub_lists" : [64, 128, 256, 512], 
        #     "exps0" : [10, 15], 
        #     "label_mask" : [62], 
        #     "data_width" : [8], 
        #     "is_split" : [0], 
        #     "n_split" : [1], 
        #     "filter_dim" : [2], 
        #     "topk": [200],
        # }, 
        # """
        # '''
        {
            "path": ["\"dataset/ag_news/\""],
            "clusters": [128],
            "sub_clusters": [1, 10], 
            "n_list": [4, 8, 16, 32, 64, 128], 
            "sub_lists" : [8, 16, 32, 64, 128, 256, 384], 
            "exps0" : [10, 15, 30, 50, 60, 80], 
            "label_mask" : [0], 
            "data_width" : [1], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [1], 
            "topk": [100],
        }, 
        # {
        #     "path": ["\"dataset/app_reviews/\""],
        #     "clusters": [512],
        #     "sub_clusters": [10], 
        #     "n_list": [4, 8, 16, 32, 64, 128], 
        #     "sub_lists" : [8, 16, 32, 64, 128, 256, 384], 
        #     "exps0" : [10, 15, 30, 50], 
        #     "label_mask" : [0], 
        #     "data_width" : [1], 
        #     "is_split" : [1], 
        #     "n_split" : [32], 
        #     "filter_dim" : [1], 
        #     "topk": [100],
        # },
        # {
        #     "path": ["\"dataset/app_reviews/\""],
        #     "clusters": [512],
        #     "sub_clusters": [1], 
        #     "n_list": [4, 8, 16, 32, 64, 128], 
        #     "sub_lists" : [4, 8, 16, 32, 64, 128], 
        #     "exps0" : [10, 15, 30, 50], 
        #     "label_mask" : [0], 
        #     "data_width" : [1], 
        #     "is_split" : [1], 
        #     "n_split" : [32], 
        #     "filter_dim" : [1], 
        #     "topk": [100],
        # },
        # {
        #     "path": ["\"dataset/gist/\""],
        #     "clusters": [1024],
        #     "sub_clusters": [10], 
        #     "n_list": [32, 64, 128, 192, 256], 
        #     "sub_lists" : [32, 64, 128, 256, 384, 512], 
        #     "exps0" : [5],
        #     "label_mask" : [0], 
        #     "data_width" : [2], 
        #     "is_split" : [1], 
        #     "n_split" : [32], 
        #     "filter_dim" : [2], 
        #     "topk": [200],
        # }, 
        # {
        #     "path": ["\"dataset/msong/\""],
        #     "clusters": [256, 512],
        #     "sub_clusters": [1], 
        #     "n_list": [16, 32, 64, 128,256], 
        #     "sub_lists" : [16, 32, 64, 128, 256], 
        #     "exps0" : [30,40,60,80], 
        #     "label_mask" : [0], 
        #     "data_width" : [1], 
        #     "is_split" : [0], 
        #     "n_split" : [1], 
        #     "filter_dim" : [2], 
        #     "topk": [200],
        # }, 
        # {
        #     "path": ["\"dataset/deep1m/\""],
        #     "clusters": [1000],
        #     "sub_clusters": [10],
        #     "n_list": [16,32,64,128],
        #     "sub_lists" : [32,64,128,256],
        #     "exps0" : [15, 30,50,70],
        #     "label_mask" : [0],
        #     "data_width" : [1],
        #     "is_split" : [1],
        #     "n_split" : [32,64],
        #     "filter_dim" : [2],
        #     "topk": [200],
        # },
        #  {
        #     "path": ["\"dataset/amazon/\""],
        #     "clusters": [1024],
        #     "sub_clusters": [1], 
        #     "n_list": [64, 128, 256,512], 
        #     "sub_lists" : [64, 128, 256, 512], 
        #     "exps0" : [10, 20, 30], 
        #     "label_mask" : [20], 
        #     "data_width" : [4], 
        #     "is_split" : [1], 
        #     "n_split" : [32],
        #     "filter_dim" : [1], 
        #     "topk": [100],
        #  }, 
        # '''
    ]

    fixed_cagra = {
        "enable_multi_gpu": 0,
        "break_down": 1,
        "is_calc_mem_predictor_coeff": 0,
        "degree" : 64, 
        "i_degree" : 128,
        "search_width" : 8, 
        "lowest_query_batch" : 125,
        "label_dim" : 1,
        "mem_bound" : 5000000000, 
    }

    variables_cagra = [
        # {
        #     "path": ["\"dataset/amazon/\""],
        #     "filter_dim" : [1], 
        #     "topk": [100],
        #     "beta": [20, 30, 50, 70, 100, 110, 150, 200, 300, 400],
        #     "itopk_size": [2000, 3000, 3500, 4000, 50000, 10000, 15000, 25000, 30000, 40000, 45000],
        #     "data_width" : [4],
        # },
        # {
        #     "path": ["\"dataset/sift5m_new/\""],
        #     "filter_dim" : [2], 
        #     "topk": [200],
        #     "beta": [20, 30, 50, 70, 100, 110, 150, 200, 300, 400],
        #     "data_width" : [5], 
        #     "itopk_size": [5000, 6000, 7000, 8000, 10000, 20000,  30000, 50000, 60000, 80000, 90000],
        # },
        # """
        # {
        #     "path": ["\"dataset/msong/\""],
        #     "filter_dim" : [2], 
        #     "topk": [200],
        #     "beta": [1,5,10,20, 30, 50],
        #     "itopk_size": [200,210,1100,1300,2200,4300,4500,6500,11000,15000,22000,24000,25000,30000,40000,60000,62000,70000,85000],
        # },
        # {
        #     "path": ["\"dataset/deep1m/\""],
        #     "filter_dim" : [2], 
        #     "topk": [200],
        #     "beta": [1,5,10,20, 30, 50],
        #     "itopk_size": [200,210,1100,1300,2200,4300,4500,6500,11000,15000,22000,24000,25000,30000,40000,60000,62000,70000,85000],
        # },
        # {
        #     "path": ["\"dataset/ag_news/\""],
        #     "filter_dim" : [1], 
        #     "topk": [100],
        #     "beta": [1,5,10,20, 30, 50],
        #     "itopk_size": [110,510,1200,2300,3300,5500,6000,7500,11000,11500,16000,23000,34000,45000],
        # },
        # {
        #     "path": ["\"dataset/cc_news/\""],
        #     "filter_dim" : [1], 
        #     "topk": [100],
        #     "beta": [1,5,10,20, 30, 50, 70, 100, 110, 150, 200, 300, 400],
        #     "itopk_size": [110,510,1200,2300,3300,5500,7500,11000,11500,16000,23000,34000,45000],
        # },
        {
            "path": ["\"dataset/gist/\""],
            "filter_dim" : [2], 
            "topk": [200],
            "beta": [1,5,10,20, 30, 50, 70, 100, 110, 150, 200, 300, 400],
             "itopk_size": [210,1010,2200,4300,6300,10500,14500,21000,21500,26000,43000,64000,85000],
        },
        # {
        #     "path": ["\"dataset/app_reviews/\""],
        #     "filter_dim" : [1], 
        #     "topk": [100],
        #     "beta": [1,5,10,20, 30, 50, 70, 100, 110, 150, 200, 300, 400],
        #     "itopk_size": [110,510,1200,2300,3300,5500,7500,11000,11500,16000,23000,34000,45000],
        # },
        # """
    ]

    constraints_cagra = [
        lambda cfg: cfg["beta"] * cfg["topk"] <= cfg["itopk_size"],              # n_list <= clusters
        lambda cfg: cfg["beta"] * cfg["topk"] * 1.2 >= cfg["itopk_size"],              # n_list <= clusters
    ]

    constraints_ivfpq = [
        lambda cfg: cfg["beta"] * cfg["refine"] * cfg["topk"] / 100 <= 600,              # n_list <= clusters
    ]


    # 跨参数约束
    constraints_super_grid = [
        lambda cfg: cfg["n_list"] <= cfg["clusters"],              # n_list <= clusters
        lambda cfg: cfg["sub_lists"] >= cfg["n_list"],             # sub_lists >= n_list
        lambda cfg: cfg["sub_lists"] <= cfg["n_list"] * cfg["sub_clusters"],  # sub_lists <= n_list * sub_cluster
        lambda cfg: cfg["sub_lists"] <= cfg["n_list"] * 2,  # sub_lists <= n_list * sub_cluster
        lambda cfg: (cfg["is_split"] == 0 and cfg["n_split"] == 1) or (cfg["is_split"] == 1 and cfg["n_split"] > 1), 
        # lambda cfg: not (cfg["n_list"] >= 64) or (cfg["exps0"] >= 60),
        # lambda cfg: cfg["sub_lists"] == cfg["n_list"],
    ]

    id = 0
    for config in variables_super_grid: 
        generate_configs(fixed_super_grid, config, constraints=constraints_super_grid, output_dir="/home/zwang/parafilter-cuda/build/all_config/super-grid/ag_news/",id=id)
        id += 1
