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
        "enable_multi_gpu": 1,
        "break_down": 1,
        "is_calc_mem_predictor_coeff": 0,
        "pq_dim" : 8, 
        "n_clusters" : 256,
        "lowest_query_batch" : 125,
        "label_dim" : 1,
        "max_grids" : 30000000, 
        "mem_bound" : 5000000000, 
    }

    # 示例：可变参数
    variables_super_grid = [
        {
            "path": ["\"dataset/ag_news/\""],
            "clusters": [128],
            "sub_clusters": [10], 
            "n_list": [32], 
            "sub_lists" : [64], 
            "exps0" : [10, 20, 30, 40], 
            "label_mask" : [0], 
            "data_width" : [1], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [1], 
            "topk": [100],
        }, 
        """"
        {
            "path": ["\"dataset/cc_news/\""],
            "clusters": [768],
            "sub_clusters": [5], 
            "n_list": [128], 
            "sub_lists" : [8, 16, 32, 64, 128, 256], 
            "exps0" : [15, 20], 
            "label_mask" : [0], 
            "data_width" : [1], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [1], 
            "topk": [100],
        }, 
        {
            "path": ["\"dataset/sift10m/\""],
            "clusters": [2048],
            "sub_clusters": [10], 
            "n_list": [64, 128, 256], 
            "sub_lists" : [64, 128, 256, 512], 
            "exps0" : [10, 15], 
            "label_mask" : [62], 
            "data_width" : [8], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        """
        '''
        {
            "path": ["\"dataset/gist/\""],
            "clusters": [1024],
            "sub_clusters": [10], 
            "n_list": [16, 32, 64, 128, 256], 
            "sub_lists" : [16, 32, 64, 128, 256, 512], 
            "exps0" : [30, 50, 70, 100], 
            "label_mask" : [0], 
            "data_width" : [2], 
            "is_split" : [1], 
            "n_split" : [32], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/ag_news/\""],
            "clusters": [128, 256],
            "sub_clusters": [5, 10], 
            "n_list": [16, 32, 64, 128], 
            "sub_lists" : [16, 32, 64, 128, 256], 
            "exps0" : [10, 15, 30], 
            "label_mask" : [0], 
            "data_width" : [1], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [1], 
            "topk": [100],
        }, 
        {
            "path": ["\"dataset/msong/\""],
            "clusters": [128, 256, 512],
            "sub_clusters": [5, 10], 
            "n_list": [16, 32, 64, 128], 
            "sub_lists" : [16, 32, 64, 128, 256], 
            "exps0" : [10, 15, 30], 
            "label_mask" : [0], 
            "data_width" : [1], 
            "is_split" : [0], 
            "n_split" : [1], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/deep1m/\""],
            "clusters": [128, 256, 512],
            "sub_clusters": [5, 10], 
            "n_list": [16, 32, 64, 128], 
            "sub_lists" : [16, 32, 64, 128, 256], 
            "exps0" : [10, 15, 30], 
            "label_mask" : [0], 
            "data_width" : [1], 
            "is_split" : [0, 1], 
            "n_split" : [1, 32, 64], 
            "filter_dim" : [2], 
            "topk": [200],
        }, 
        {
            "path": ["\"dataset/amazon/\""],
            "clusters": [512, 1024],
            "sub_clusters": [5, 10], 
            "n_list": [64, 128, 256], 
            "sub_lists" : [64, 128, 256, 512], 
            "exps0" : [10], 
            "label_mask" : [20], 
            "data_width" : [1, 2, 4], 
            "is_split" : [0, 1], 
            "n_split" : [1, 32, 64],
            "filter_dim" : [1], 
            "topk": [100],
        }, 
        '''
    ]

    fixed_cagra = {
        "enable_multi_gpu": 1,
        "break_down": 1,
        "is_calc_mem_predictor_coeff": 0,
        "degree" : 64, 
        "i_degree" : 128,
        "search_width" : 8, 
        "lowest_query_batch" : 125,
        "label_dim" : 1,
        "mem_bound" : 5000000000, 
        "data_width" : 1, 
    }

    variables_cagra = [
        {
            "path": ["\"dataset/msong/\""],
            "filter_dim" : [2], 
            "topk": [200],
            "beta": [2],
            "itopk_size": [400, 600, 900, 1000, 1100, 1200, 1300],
        },
        {
            "path": ["\"dataset/deep1m/\""],
            "filter_dim" : [2], 
            "topk": [200],
            "beta": [4],
            "itopk_size": [900, 1000, 1100, 1200, 1300],
        },
        {
            "path": ["\"dataset/tiny5m/\""],
            "filter_dim" : [2], 
            "topk": [200],
            "beta": [8, 10],
            "itopk_size": [1600, 1700, 1800, 1900, 2000, 2100],
        },
        {
            "path": ["\"dataset/sift10m/\""],
            "filter_dim" : [2], 
            "topk": [200],
            "beta": [15, 20],
            "itopk_size": [3000, 3200, 3500, 3800, 4000, 4200],
        },
        {
            "path": ["\"dataset/ag_news/\""],
            "filter_dim" : [1], 
            "topk": [100],
            "beta": [5],
            "itopk_size": [500, 600, 700, 800, 900],
        },
        {
            "path": ["\"dataset/cc_news/\""],
            "filter_dim" : [1], 
            "topk": [100],
            "beta": [30, 50],
            "itopk_size": [3000, 3500, 4000, 4500, 5000, 5200],
        },
        {
            "path": ["\"dataset/app_reviews/\""],
            "filter_dim" : [1], 
            "topk": [100],
            "beta": [50, 70, 90],
            "itopk_size": [5000, 5500, 6000, 6500, 7000, 7500, 9000, 9500],
        },
        {
            "path": ["\"dataset/amazon/\""],
            "filter_dim" : [1], 
            "topk": [100],
            "beta": [1000, 5000],
            "itopk_size": [100000, 200000, 500000],
        },
    ]

    constraints_cagra = [
        lambda cfg: cfg["beta"] * cfg["topk"] <= cfg["itopk_size"],              # n_list <= clusters
    ]


    # 跨参数约束
    constraints_super_grid = [
        lambda cfg: cfg["n_list"] <= cfg["clusters"],              # n_list <= clusters
        # lambda cfg: cfg["sub_lists"] >= cfg["n_list"],             # sub_lists >= n_list
        lambda cfg: cfg["sub_lists"] <= cfg["n_list"] * cfg["sub_clusters"],  # sub_lists <= n_list * sub_cluster
        # lambda cfg: cfg["sub_lists"] <= cfg["n_list"] * 2,  # sub_lists <= n_list * sub_cluster
        lambda cfg: (cfg["is_split"] == 0 and cfg["n_split"] == 1) or (cfg["is_split"] == 1 and cfg["n_split"] > 1), 
    ]

    id = 0
    for config in variables_super_grid : 
        generate_configs(fixed_super_grid, config, constraints=constraints_super_grid, output_dir="/home/zwang/parafilter-cuda/build/all_config/super_grid", id = id)
        id += 1
