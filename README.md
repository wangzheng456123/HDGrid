### HDGrid:

- This repository provides a reference implementation of HDGrid for evaluation purposes.

#### BUILD:

- Simply run build.sh.


##### Requirements:

- C++ 17 or higher

- CUDATookit 12.0+

- CMake >= 3.26.4

#### Run HDGrid:

1. ##### parse dataset:

   HDGrid uses an HDF5 dataset as input. Before running HDGrid, you need to parse the HDF5 dataset into raw data. This can be done by simply running the read_hdf5.py script.

2. ##### config HDGrid: 
   Before running HDGrid, you need to create a HDGrid.conf file. A sample configuration file is shown below:

```c
enable_multi_gpu = 0
break_down = 1
is_calc_mem_predictor_coeff = 0
pq_dim = 4
n_clusters = 256
lowest_query_batch = 125
label_dim = 1
max_grids = 30000000
mem_bound = 5000000000
path = "dataset/example_dataset/"
clusters = 512
sub_clusters = 10
n_list = 64
sub_lists = 128
exps0 = 30
label_mask = 0
data_width = 1
is_split = 0
n_split = 1
filter_dim = 1
topk = 100
```

Here are some notes for the configuration:

- **break_down**: Specifies whether to log performance data.

- **mem_bound**: Determines the appropriate query and data batch sizes. It indicates the upper bound of the GPU memory used by HDGrid.

- **is_calc_mem_predictor_coeff**: HDGrid requires certain configuration parameters to determine the optimal query and data batch sizes. If set to `1`, HDGrid will estimate these parameters without performing the actual ANN search. You should set this to `1` and run SearchEngine before starting the first time actual ANN search.

- **path**: Specifies the relative path to the dataset. The path is relative to the build folder where the HDGrid binary is located.

The meanings of other configuration parameters correspond directly to the HDGrid algorithm.

3. ##### config filter:
   The filter.conf file is used to configure filters for the ANN search. A sample file is shown below:
```c
l = 5
filter0:
	type = 0
	shift_val = [604800, 0]
filter1:
	type = 0
	shift_val = [30, 30]
filter2:
	type = 0
	shift_val = [107321744, 107321744]
filter3:
	type = 1
	interval_map = [1, 2, 1, 2, 3, 4, 3, 5, 4, 5]
filter4:
	type = 2
	div_value = [10]
```
HDGrid supports three types of filters:

- **Shift type**: `type = 0` for this filter, `shift_val = [left_shift, right_shift]`. The data is filtered by `label - left_shift <= data <= label + right_shift`.

- **Map type**: `type = 1`, each data is filtered by `interval_map[2 * label] <= data <= interval_map[2 * label + 1]`.

- **Div type**: `type = 2`, data is filtered by `data >= label / div_value`.

4. ##### calculate coeficients :
Set is_calc_mem_predictor_coeff to 1 and run HDGrid. After that, set it back to 0 to begin the actual ANN search.

# Run

## 1. After building

Change directory to the build folder:

```bash
cd build
```

## 2. Run the binary

Run the `search-engine` executable with the following command line arguments:

```bash
./search-engine [Algorithm] [path-to-APP-config-file] [path-to-filter-config-file]
```

- **[Algorithm]**: Name of the algorithm, e.g., `super-grid`(Note: Our retrieval engine includes other high-performance filtering and retrieval algorithms in addition to HDGrid, such as cagra-post and ivfpq-pos)
- **[path-to-APP-config-file]**: Path to the application config file, e.g., `HDGrid.conf`
- **[path-to-filter-config-file]**: Path to the filter config file, e.g., `example_dataset.conf`

## 3. Example

```bash
./search-engine Super-Grid HDGrid.conf cc_news.conf
```


