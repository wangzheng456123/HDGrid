#include "index/2-level_index/Super-Grid-IVF_PQ.cuh"
namespace super_grid {
    __device__ __forceinline__ int flatten_index(const int* coords, const int* dims, int ndims) 
    {
        int idx = 0;
        int stride = 1;
    #pragma unroll
        for (int i = ndims - 1; i >= 0; --i) {
            idx += coords[i] * stride;
            stride *= dims[i];
        }
        return idx;
    }

    __device__ __forceinline__ void unflatten_index(int idx, const int* dims, int ndims, int* coords) 
    {
    #pragma unroll
        for (int i = ndims - 1; i >= 0; --i) {
            coords[i] = idx % dims[i];
            idx /= dims[i];
        }
    }

    __device__ __forceinline__ int get_grid_id_device(float x, uint64_t l, uint64_t len, int M) {
        return ((uint64_t)((x - l) * M)) / len;
    }

    __global__ void get_grid_ids_kernel(
        const int* grid_dims, const float* data_labels, const uint64_t* inv_id_list, const uint64_t* label_spans,  
        const uint64_t* label_left, uint64_t* o_ids, int l, uint64_t n_data, uint64_t cluster_len, int* grid_coords, 
        const int* shuffle
    )
    {
        uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x >= cluster_len) return;
        uint64_t id = inv_id_list[x];
        //todo: how to process id large than n_data
        if (id >= n_data) return;
        int* coords = &grid_coords[l * x];
        const float* label = &data_labels[l * id];

    #pragma unroll
        for (int i = 0; i < l; i++) {
            int dim = shuffle == nullptr ? i : shuffle[i];
            coords[i] = get_grid_id_device(label[dim], label_left[i], label_spans[i], grid_dims[i]);
            assert(coords[i] <= grid_dims[i]);
            coords[i] = std::min(coords[i], grid_dims[i] - 1);
        }

        o_ids[x] = flatten_index(coords, grid_dims, l);
    }

    void get_grid_id(
        raft::device_vector_view<int, uint64_t> const& grid_dims_device,
        raft::device_vector_view<uint64_t, uint64_t> const& label_span_device,  
        raft::device_vector_view<uint64_t, uint64_t> const& label_left_device,  
        raft::device_vector_view<uint64_t, uint64_t> const& sub_cluster, 
        raft::device_matrix_view<float, uint64_t> const& data_labels, 
        raft::device_vector_view<uint64_t, uint64_t> &grid_ids, 
        raft::device_matrix_view<int, uint64_t> &grid_coords, 
        const int* shuffle = nullptr
    )
    {
        int l = data_labels.extent(1);
        uint64_t n_data = data_labels.extent(0);
        uint64_t sub_cluster_len = sub_cluster.extent(0);

        dim3 blocks(128);
        dim3 grids((sub_cluster_len + blocks.x - 1) / blocks.x);

        get_grid_ids_kernel<<<grids, blocks>>>(grid_dims_device.data_handle(), data_labels.data_handle(), 
            sub_cluster.data_handle(), label_span_device.data_handle(), label_left_device.data_handle(), 
            grid_ids.data_handle(), l, n_data, sub_cluster_len, grid_coords.data_handle(), shuffle);
        
        cudaDeviceSynchronize();
    } 

    __global__ void write_grid_kernel(
        uint64_t* cnt_grids, const uint64_t* unique_values, const int* counts, uint64_t n_elements, 
        uint64_t tot_grids
    )
    {
        uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= n_elements) return;
        assert(unique_values[x] < tot_grids);
        cnt_grids[unique_values[x]] = counts[x];
    }

    static uint64_t write_grid(
        raft::device_vector_view<uint64_t, uint64_t> &flatten_cnt_grid, 
        raft::device_vector_view<uint64_t, uint64_t> const& sorted_grid_ids
    )
    {
        uint64_t cluster_len = sorted_grid_ids.extent(0);
        uint64_t tot_grids = flatten_cnt_grid.extent(0);

        uint64_t* d_unique_values = (uint64_t*)parafilter_mmr::mem_allocator(cluster_len * sizeof(uint64_t));
        int* d_counts = (int*)parafilter_mmr::mem_allocator(cluster_len * sizeof(int));
        
        uint64_t n_elements = deduplicate_sorted_keys(sorted_grid_ids.data_handle(), (size_t)cluster_len, d_unique_values, d_counts);
        
        dim3 blocks(128);
        dim3 grids((n_elements + blocks.x - 1) / blocks.x);

        write_grid_kernel<<<grids, blocks>>>(flatten_cnt_grid.data_handle(), d_unique_values, d_counts, n_elements, tot_grids);

        return n_elements;
    }

    __device__ void unpack_coords_except_dim(
        int flat_idx, const int* shape, int dims, int dim,
        int* coords_out)
    {
        // 从 0~(prod of shape except dim) compute
    #pragma unroll 
        for (int i = dims - 1; i >= 0; --i) {
            int is_dim = (i == dim);
            coords_out[i] = (1 - is_dim) * flat_idx % shape[i];
            flat_idx /= (1 - is_dim) * shape[i] + is_dim;
        }
    }

    __global__ void compute_diff_nd(
        const uint64_t* __restrict__ prefixSum,  
        int* __restrict__ diff,                   
        const int* __restrict__ shape,       
        const int flatten_id_except_dim,  
        int2* dim_sums, // array saving summation along axis dim    
        int dims,
        int axis,                                
        uint64_t total_size                            
    ) 
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total_size) return;
        int coords[MAX_DIMS];
        unpack_coords_except_dim(flatten_id_except_dim, shape, dims, axis, coords);

        coords[axis] = tid;

        int cur_id = flatten_index(coords, shape, dims);
        if (coords[axis] == 0) {
            diff[tid] = 1;  // border
            dim_sums[tid] = make_int2(tid, prefixSum[cur_id]);
            return;
        }

        //index for previouse
        coords[axis] -= 1;
        int prev_idx = flatten_index(coords, shape, dims);
        coords[axis] += 1;

        long long curr = prefixSum[cur_id];
        long long prev = prefixSum[prev_idx];

        diff[tid] = (curr - prev != 0) ? 1 : 0;
        dim_sums[tid] = (curr - prev != 0) ? make_int2(tid, curr) : make_int2(0, 0);
    }

    __global__ void filter_zero_tail_kernel(
        const uint64_t* __restrict__ prefix,
        int* __restrict__ mask,
        const int* __restrict__ shape,
        int dims,
        int diff_dim,  
        uint64_t total_tail_size,  
        int diff_dim_size
    ) 
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total_tail_size) return;

        int coords[MAX_DIMS];
        unpack_coords_except_dim(tid, shape, dims, diff_dim, coords);

        // early exit: tail slice at boundary always preserved
        int last_dim = diff_dim - 1;
        while (last_dim >= 0 && shape[last_dim] <= 1) last_dim--;

        if (coords[last_dim] == 0) {
            mask[tid] = 1;
            return;
        }

        int diff_val = 0;

        for (int i = 0; i < diff_dim_size; ++i) {
            coords[diff_dim] = i;

            int idx1 = flatten_index(coords, shape, dims);
            coords[last_dim]--;
            int idx2 = flatten_index(coords, shape, dims);
            coords[last_dim]++;  // restore

            if (prefix[idx1] != prefix[idx2]) {
                diff_val = 1;
                break;
            }
        }

        mask[tid] = diff_val;
    }

    struct IsOne {
        __device__ bool operator()(int x) const { return x == 1; }
    };
    
    void index::compress_HOPS_data(
        HOPS const& sums, 
        raft::device_vector_view<prefix_sum_node, uint64_t>& compressed_sums,
        int diff_dim
    )
    {
        const uint64_t* sums_data = sums.data_handle();
        uint64_t tot_grids = sums.size();
        uint64_t total_tail_size = compressed_sums.extent(0);
        int dims = sums.rank(); 
        int* mask_host = new int[total_tail_size];
        dim3 blocks(128);
        if (dims > 1) {
            int* mask = (int*)parafilter_mmr::mem_allocator(sizeof(int) * total_tail_size);
            dim3 grids((total_tail_size + blocks.x - 1) / blocks.x);

            filter_zero_tail_kernel<<<grids, blocks>>>(sums_data, mask, grid_dims_device.data_handle(), dims, diff_dim, total_tail_size, sums.dim(diff_dim));
            cudaDeviceSynchronize();
            checkCUDAError("compress prefix sum cols failed");
            cudaMemcpy(mask_host, mask, total_tail_size * sizeof(int), cudaMemcpyDeviceToHost);
        }
        else mask_host[0] = 1;
        
        prefix_sum_node *nodes_host = new prefix_sum_node[total_tail_size];

        thread_local int2* compressed_sums_pool = nullptr;

        uint64_t init_offset = offset;
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_OUTPUT);
#endif
        thread_local int2* tmp_sums = (int2*)parafilter_mmr::mem_allocator(sums.dim(diff_dim) * sizeof(int2));
        thread_local int* mask_diff_dim = (int*)parafilter_mmr::mem_allocator(sizeof(int) * sums.dim(diff_dim));
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_DEFAULT);
#endif

        uint64_t last = 0;
        
        for (uint64_t i = 0; i < total_tail_size; i++) {
            nodes_host[i].pre = last;
            nodes_host[i].sums = nullptr;
            if(mask_host[i]) {
                last = i;

                cudaMemset(tmp_sums, 0, sizeof(int2) * sums.dim(diff_dim));
                cudaMemset(mask_diff_dim, 0, sizeof(int) * sums.dim(diff_dim));
                dim3 grids = dim3((sums.dim(diff_dim) + blocks.x - 1) / blocks.x);
                compute_diff_nd<<<grids, blocks>>>(sums_data, mask_diff_dim, 
                    grid_dims_device.data_handle(), i, tmp_sums, dims, diff_dim, sums.dim(diff_dim));
                
                checkCUDAError("compress prefix sum row failed");
                
                thrust::device_ptr<const int> d_mask(mask_diff_dim);
                thrust::device_ptr<const int2> d_data(tmp_sums);

                int num_selected = thrust::count(d_mask, d_mask + sums.dim(diff_dim), 1);
                assert(num_selected > 0);
                nodes_host[i].len = num_selected;

                if (remain == 0 || num_selected > remain) {
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_INDEX);
#endif
                    compressed_sums_pool = (int2*)parafilter_mmr::mem_allocator(2 * tot_grids * sizeof(int2));
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_DEFAULT);
#endif
                    offset = 0, remain = 2 * tot_grids, init_offset = offset;
                    LOG(TRACE) << "prefix sum poll re-allocated";
                }

                int2* cur_node_ptr = compressed_sums_pool + offset;
                assert(((uintptr_t)cur_node_ptr % alignof(int2)) == 0);
                nodes_host[i].sums = cur_node_ptr;
                thrust::device_ptr<int2> d_out(nodes_host[i].sums);

                thrust::copy_if(
                    d_data,                         // begin of int2 array
                    d_data + sums.dim(diff_dim),          // end
                    d_mask,                         // stencil (bool mask)
                    d_out,                          // output: int2*
                    IsOne()
                );

                offset += num_selected;
                remain -= num_selected;
            }
            assert(nodes_host[last].sums != nullptr);
        }
        cudaMemcpy(compressed_sums.data_handle(), nodes_host, sizeof(prefix_sum_node) * total_tail_size, cudaMemcpyHostToDevice);

        LOG(INFO) << "ratio for compress is: " << (float)(offset - init_offset) / total_tail_size; 
        checkCUDAError("copy nodes to device failed");
    }

    void index::calc_grid_shape(raft::device_matrix_view<float, uint64_t> const& data_labels)
    {
        int l = f_config.l;
        std::vector<float> global_min(l);
        std::vector<float> global_max(l);
        calculate_batch_min_max(data_labels, global_min, global_max, l);

        uint8_t label_mask = ~0;
        uint8_t mask =  super_grid_config.label_mask;  
        label_mask &= ~mask;
        int cnt = 0; 

        for (int i = 0; i < l; i++) {
            if ((label_mask >> i) & 1)
                cnt++;
        }

        std::vector<uint64_t> label_span_host(l);
        std::vector<uint64_t> label_left_host(l);
        std::vector<std::pair<uint64_t, int>> span_pairs(l);
        for (int i = 0; i < l; i++) {
            uint64_t span = ceil(global_max[i]) - floor(global_min[i]);
            span_pairs[i] = std::make_pair(span, i);
        }

        int k = 1;
        uint64_t max_grids = super_grid_config.max_grids;
        int main_grids = k * floor(sqrt(max_grids));

        std::sort(span_pairs.begin(), span_pairs.end());
        grid_dims_host.resize(l);
        if (l == 1)
            main_grids = std::min(max_grids, span_pairs[0].first);

        int remain = 1, remain_cnt = cnt - 1, last_valid = l - 1;
        if (l > 1)
            remain = max_grids / main_grids;
        uint8_t morton_mask = ~0;
        for (int i = 0; i < l; i++) {
            int id = span_pairs[i].second;
            uint64_t len = span_pairs[i].first;

            if ((label_mask >> id) & 1) {
                if (remain_cnt == 0) {
                    last_valid = i;
                    continue;
                }
                int dims_len = std::min((uint64_t)remain, std::min(len, (uint64_t)std::floor(std::pow(main_grids, 1.0 / float(cnt - 1)))));
                //uisng all remains if it is the last dim
                if (remain_cnt == 1)
                    dims_len = std::min((uint64_t)remain, len);
                remain_cnt--;
                grid_dims_host[i] = dims_len;
                remain /= dims_len;
            }
            else {
                morton_mask &= ~(1 << i);
                grid_dims_host[i] = 1;
            }
        }

#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_INDEX);
#endif
        grid_dims_device = parafilter_mmr::make_device_vector_view<int, uint64_t>(l);
        label_span_device = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(l);
        label_left_device = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(l);
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_DEFAULT);
#endif
        grid_dims_host[last_valid] = main_grids;
        cudaMemcpy(grid_dims_device.data_handle(), grid_dims_host.data(), sizeof(int) * l, cudaMemcpyHostToDevice);
        tot_grids = 1;
        for (auto dim : grid_dims_host)
            tot_grids *= dim;

        for (int i = 0; i < l; i++) {
            int id = span_pairs[i].second;
            label_span_host[i] = span_pairs[i].first;
            label_left_host[i] = floor(global_min[id]);
        }

        cudaMemcpy(label_span_device.data_handle(), label_span_host.data(), sizeof(uint64_t) * l, cudaMemcpyHostToDevice);
        cudaMemcpy(label_left_device.data_handle(), label_left_host.data(), sizeof(uint64_t) * l, cudaMemcpyHostToDevice);

        int* shuffle_host = new int[l];
        for (int i = 0; i < l; i++) {
            int id = span_pairs[i].second;
            shuffle_host[i] = id;
        }
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_INDEX);
#endif
        shuffle = (int*)parafilter_mmr::mem_allocator(sizeof(int) * l);
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_DEFAULT);
#endif
        cudaMemcpy(shuffle, shuffle_host, sizeof(int) * l, cudaMemcpyHostToDevice);

        morton_config.dims = static_cast<int>(grid_dims_host.size());
        morton_config.max_bits = static_cast<int>(
            std::ceil(std::log2(*std::max_element(grid_dims_host.begin(), grid_dims_host.end()) + 1))
        );

        uint8_t final_mask = (1ULL << l) - 1;
        morton_config.valid_map = morton_mask & final_mask;

    }

    void index::init_state()
    {
        offset = 0;
        remain = 0;
    }

    void index::build(
        build_input_t const& in
    ) 
    {   
        // todo: build pq uppon residual
        init_state();
        calc_grid_shape(in.data_labels);
        prefiltering::index::build(in);
        uint64_t n_dim = in.dataset.extent(1);
        uint64_t n_data = in.dataset.extent(0);

        raft::cluster::kmeans::KMeansParams params;
        params.metric = raft::distance::DistanceType::L2Expanded;
        int clusters =  super_grid_config.clusters;
        int sub_clusters = super_grid_config.sub_clusters;
        params.n_clusters = clusters * sub_clusters;

        float interia;
        uint64_t niters;
        uint64_t inv_offset = 0; 

        auto& sub_centers_view = super_grid_index_data.sub_clusters_data.sub_centers;
        auto& centers_view = super_grid_index_data.centers;
        auto sub_centers_tmp_view = parafilter_mmr::make_device_matrix_view<float, uint64_t>(params.n_clusters, n_dim);
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_INDEX);
#endif
        centers_view = parafilter_mmr::make_device_matrix_view<float, uint64_t>(clusters, n_dim);
        sub_centers_view = parafilter_mmr::make_device_matrix_view<float, uint64_t>(params.n_clusters, n_dim);
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_DEFAULT);
#endif
        auto inv_list_view = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_data);

        raft::cluster::kmeans::fit_predict<float, uint64_t>(
            in.dev_resources, 
            params, 
            in.dataset, 
            std::nullopt, 
            sub_centers_tmp_view, 
            inv_list_view,
            raft::make_host_scalar_view(&interia), 
            raft::make_host_scalar_view(&niters)
        );

        cudaDeviceSynchronize();
        raft::cluster::kmeans_balanced_params balanced_params;
        balanced_params.n_iters = 200;
        auto sub_cluster_ids = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(params.n_clusters);

        raft::cluster::kmeans_balanced::fit_predict<float, float, uint64_t, uint64_t>(
            in.dev_resources, 
            balanced_params, 
            sub_centers_tmp_view, 
            centers_view, 
            sub_cluster_ids
        );
        cudaDeviceSynchronize();

        auto sub_cluster_ids_matrix_view = raft::make_device_matrix_view<uint64_t, uint64_t>(sub_cluster_ids.data_handle(), params.n_clusters, 1);

        std::vector<raft::device_vector_view<uint64_t, uint64_t>> grouped_clusters(params.n_clusters);
        group_by_cluster_id(inv_list_view, grouped_clusters);

        std::vector<raft::device_vector_view<uint64_t, uint64_t>> sub_cluster_ids_grouped(clusters);
        group_by_cluster_id(sub_cluster_ids, sub_cluster_ids_grouped);
        auto ids_host = new uint64_t[params.n_clusters];

        uint64_t* sub_cluster_offsets_host = new uint64_t[clusters];
        uint64_t* sub_cluster_counts_host = new uint64_t[clusters];

        int ids_host_offset = 0;

        for (int i = 0; i < clusters; i++) {
            int len = sub_cluster_ids_grouped[i].size();
            sub_cluster_offsets_host[i] = ids_host_offset;
            sub_cluster_counts_host[i] = len;
            if (len == 0) continue;
            cudaMemcpy(&ids_host[ids_host_offset], sub_cluster_ids_grouped[i].data_handle(), len * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            ids_host_offset += len;
        }
        assert(ids_host_offset == params.n_clusters);
        auto ids_device = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(params.n_clusters, 1);
        cudaMemcpy(ids_device.data_handle(), ids_host, sizeof(uint64_t) * params.n_clusters, cudaMemcpyHostToDevice);
        select_elements<float, uint64_t>(in.dev_resources, sub_centers_tmp_view, ids_device, sub_centers_view);
        // the index data
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_INDEX);
#endif
        auto zorder_offsets = (uint64_t*)parafilter_mmr::mem_allocator(sizeof(uint64_t) * clusters * sub_clusters);
        auto zorder_lengthes = (uint64_t*)parafilter_mmr::mem_allocator(sizeof(uint64_t) * clusters * sub_clusters);
        uint64_t* sub_cluster_offsets = (uint64_t*)parafilter_mmr::mem_allocator(sizeof(uint64_t) * clusters);
        uint64_t* sub_cluster_counts = (uint64_t*)parafilter_mmr::mem_allocator(sizeof(uint64_t) * clusters);
        uint64_t* zorder_code_pool = (uint64_t*)parafilter_mmr::mem_allocator((sizeof(uint64_t) * n_data));
        uint64_t* zcode_ids_pool = (uint64_t*)parafilter_mmr::mem_allocator(sizeof(uint64_t) * n_data);
        prefix_sum_node** sums_device = (prefix_sum_node**)parafilter_mmr::mem_allocator(sizeof(prefix_sum_node*) * clusters * sub_clusters);
#ifdef TAGGED_MMR
        parafilter_mmr::set_tag(MEM_DEFAULT);
#endif

        uint64_t zorder_pool_offset = 0;
        uint64_t n_znode = 0;
        auto zorder_offsets_host = new uint64_t[clusters * sub_clusters];
        auto zorder_lengthes_host = new uint64_t[clusters * sub_clusters];

        prefix_sum_node** sums_host = new prefix_sum_node*[clusters * sub_clusters];

        auto cnt_grid = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(tot_grids);
        int sub_center_id = 0;

        for (int i = 0; i < params.n_clusters; i++) {
            auto &cluster = grouped_clusters[ids_host[i]];
            uint64_t cluster_len = cluster.extent(0);
            /*processing null sub-clusters*/
            if (cluster_len == 0) {
                LOG(TRACE) << "sub-center id:" << sub_center_id << "is null!";
                zorder_offsets_host[sub_center_id] = zorder_pool_offset;
                zorder_lengthes_host[sub_center_id] = 0;
                sums_host[sub_center_id] = nullptr;
                auto cur_centers_view = raft::make_device_matrix_view<float, uint64_t>(sub_centers_view.data_handle() + ids_host[i] * n_dim, 
                    params.n_clusters, n_dim);
                thrust::fill_n(thrust::device_pointer_cast(cur_centers_view.data_handle()) + n_dim * sub_center_id, n_dim, 1e6 / n_dim);
                sub_center_id++;
                continue;
            }
            cudaMemset(cnt_grid.data_handle(), 0, tot_grids * sizeof(uint64_t));
            
            assert(zorder_pool_offset + cluster_len <= n_data);
            
            auto& data_ids = cluster;
            // int my_array[] = {
            //     460, 324, 261, 17, 216, 270, 675, 183, 144, 437,
            //     589, 187, 209, 42, 293, 661, 690, 727, 312, 300,
            //     532, 135, 710, 503, 455, 356, 499, 397, 468, 51,
            //     285, 731, 738, 535, 392, 544, 87, 150, 489, 240,
            //     473, 358, 654, 171, 196, 605, 562, 290, 553, 453,
            //     655, 673, 480, 311, 696, 697, 205, 272, 742, 246,
            //     529, 586, 572, 101, 204, 13, 349, 158, 611, 430,
            //     96, 699, 734, 186, 388, 298, 425, 383, 334, 635,
            //     420, 528, 239, 348, 403, 549, 514, 752, 613, 543,
            //     175, 705, 178, 550, 234, 408, 378, 283, 577, 755
            // }; 
            // for (int j = 0; j < 100; j++) {
            //     auto p = thrust::find(thrust::device, data_ids.data_handle(), data_ids.data_handle() + cluster_len, my_array[j]);
            //     if (p != data_ids.data_handle() + cluster_len) {
            //         LOG(TRACE) << "find " << my_array[j] << " in cluster" << i << "offset: " << p - data_ids.data_handle() << " pool offset: " << zorder_pool_offset;
            //     }
            // }
            
            auto grid_ids = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(cluster_len);
            auto grid_coords = parafilter_mmr::make_device_matrix_view<int, uint64_t>(cluster_len, grid_dims_host.size());

            get_grid_id(grid_dims_device, label_span_device, label_left_device, data_ids, in.data_labels, grid_ids, grid_coords, shuffle);

            thrust::sort(
                thrust::device_pointer_cast(grid_ids.data_handle()), 
                thrust::device_pointer_cast(grid_ids.data_handle()) + cluster_len
            ); 
            uint64_t valid_grids = write_grid(cnt_grid, grid_ids);
            cudaDeviceSynchronize();

            HOPS grid_sums(grid_dims_host);
            grid_sums.load(cnt_grid.data_handle());
            grid_sums.multi_dim_scan();

            uint64_t diff_dim = grid_dims_host.size() - 1;
            for (int i = grid_dims_host.size() - 1; i >= 0; i--) {
                if (grid_dims_host[i] > 1) {
                    diff_dim = i;
                    break;
                }
            }
            int max_len = grid_dims_host[diff_dim];

#ifdef TAGGED_MMR
            parafilter_mmr::set_tag(MEM_INDEX);
#endif
            auto grid_sums_compress = parafilter_mmr::make_device_vector_view<prefix_sum_node, uint64_t>(tot_grids / max_len);
#ifdef TAGGED_MMR
            parafilter_mmr::set_tag(MEM_DEFAULT);
#endif
            sums_host[sub_center_id] = grid_sums_compress.data_handle();
            compress_HOPS_data(grid_sums, grid_sums_compress, diff_dim);

            raft::device_vector_view<uint64_t, uint64_t> zcodes = raft::make_device_vector_view<uint64_t, uint64_t>(zorder_code_pool + zorder_pool_offset, cluster_len);
            encode_points(grid_coords, zcodes, morton_config);

            thrust::sort_by_key(
                thrust::device_pointer_cast(zcodes.data_handle()),      
                thrust::device_pointer_cast(zcodes.data_handle())  + cluster_len, 
                thrust::device_pointer_cast(data_ids.data_handle())   
            );   
            cudaMemcpy(zcode_ids_pool + zorder_pool_offset, data_ids.data_handle(), cluster_len * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            
            zorder_offsets_host[sub_center_id] = zorder_pool_offset;
            zorder_lengthes_host[sub_center_id] = cluster_len;
            
            zorder_pool_offset += cluster_len;
            sub_center_id++;
        }

        assert(zorder_pool_offset == n_data);

        cudaMemcpy(zorder_offsets, zorder_offsets_host, sizeof(uint64_t) * params.n_clusters, cudaMemcpyHostToDevice);
        super_grid_index_data.sub_clusters_data.zorder_array_offsets = raft::make_device_vector_view<uint64_t, uint64_t>(zorder_offsets, params.n_clusters);

        cudaMemcpy(zorder_lengthes, zorder_lengthes_host, sizeof(uint64_t) * params.n_clusters, cudaMemcpyHostToDevice);
        super_grid_index_data.sub_clusters_data.zorder_array_lengthes = raft::make_device_vector_view<uint64_t, uint64_t>(zorder_lengthes, params.n_clusters);

        cudaMemcpy(sums_device, sums_host, sizeof(prefix_sum_node*) * params.n_clusters, cudaMemcpyHostToDevice);
        super_grid_index_data.sub_clusters_data.compressed_sums = raft::make_device_vector_view<prefix_sum_node*, uint64_t>(sums_device, params.n_clusters);

        cudaMemcpy(sub_cluster_offsets, sub_cluster_offsets_host, sizeof(uint64_t) * clusters, cudaMemcpyHostToDevice);
        cudaMemcpy(sub_cluster_counts, sub_cluster_counts_host, sizeof(uint64_t) * clusters, cudaMemcpyHostToDevice);

        super_grid_index_data.sub_cluster_offsets = raft::make_device_vector_view<uint64_t, uint64_t>(sub_cluster_offsets, clusters);
        super_grid_index_data.sub_cluster_counts = raft::make_device_vector_view<uint64_t, uint64_t>(sub_cluster_counts, clusters);

        super_grid_index_data.sub_clusters_data.zorder_code_arrays = raft::make_device_vector_view<uint64_t, uint64_t>(zorder_code_pool, n_data);
        super_grid_index_data.sub_clusters_data.zorder_node_arrays = raft::make_device_vector_view<uint64_t, uint64_t>(zcode_ids_pool, n_data);

        delete zorder_offsets_host;
        delete zorder_lengthes_host; 
        delete sums_host; 
        delete sub_cluster_offsets_host;
        delete sub_cluster_counts_host;
    }

    __device__ uint64_t query_compressed_sum_device(
        const prefix_sum_node* sum_nodes,
        const int* coords, 
        const int* shape, 
        int diff_dim, int dims, 
        int thresh_hold = 0 // if longer than thresh hold, using binary search
    )
    {
        uint64_t flatten_remain = flatten_index(coords, shape, dims);
        flatten_remain /= shape[diff_dim];
        int key = coords[diff_dim];
        prefix_sum_node node = sum_nodes[flatten_remain];
        
        node = (node.sums == nullptr) ? sum_nodes[node.pre] : node;
        
        uint64_t res = 0; 
        assert(node.sums != nullptr);
        assert(((uintptr_t)node.sums % alignof(int2)) == 0);
        if (node.len <= thresh_hold) {
            int last_valid = -1;
    #pragma unroll
            for(int i = 0; i < node.len; ++i) {
                if(node.sums[i].x <= key) {
                    last_valid = i;
                } else {
                    break;
                }
            }
            res = (last_valid >= 0) ? node.sums[last_valid].y : 0;
        }
        else {
            int l = 0, r = node.len - 1;
            while(l <= r) {
                int mid = (l + r) >> 1;
                if(node.sums[mid].x <= key) {
                    res = node.sums[mid].y; 
                    l = mid + 1;            
                } else {
                    r = mid - 1;
                }
            }
        }
        return res;
    } 

    __device__ uint64_t get_prefix_sum_device(
        const prefix_sum_node* sum_nodes, const int* shape, 
        const int* query, int d, int real_dim, int mask, int diff_dim, int valid_map = 0) 
    {
        // mask: 0~2^d-1, the corner points for the query box
        int coords[MAX_DIMS];
    #pragma unroll
        for (int i = 0; i < MAX_DIMS; ++i) coords[i] = 0;
        int sign = 1;
        if (sum_nodes == nullptr)  {
            return 0;
        }
        int cur = 0;
        for (int i = 0; i < d; ++i) {
            if (valid_map) {
                while (!((valid_map >> cur) & 1)) cur++;
                valid_map &= ~(1 << cur); 
            }
            if ((mask >> i) & 1) {
                coords[cur] = query[2 * cur];  // l_i
                sign *= -1;
                if (coords[cur] > 0) coords[cur]--;
                else { 
                    return 0;
                }
            } else {
                coords[cur] = query[2 * cur + 1];   // r_i
            }
        }
        int sum = query_compressed_sum_device(sum_nodes, coords, shape, diff_dim, real_dim);
        return sign * sum;
    }

    // main kernel for processing multi-dimensional and multi query prefix sum
    __global__ void get_prefix_sum_kernel(
        const prefix_sum_node*const* prefixSum, 
        const int* shape,
        const uint64_t* cluster_offsets,  
        const uint64_t* cluster_counts, 
        const int* queries, // shape: [n_queries][2*dims]
        int dims, int real_dims, 
        uint64_t* sub_result, // shape: [n_queries][n_trees] 
        int diff_dim,
        int n_sub_trees,
        int n_tree, int n_queries, 
        int valid_map = 0
    ) 
    {
        __shared__ int partial[1024];  // max 2^d threads per block
    
        int query_blocks = blockDim.x / (1 << dims);
        int qid = blockIdx.x * query_blocks + threadIdx.x / (1 << dims);
        int tid = threadIdx.x % (1 << dims);
        int tree_id = blockIdx.y;
        int sub_tree_id = blockIdx.z;
        int tot_cluster = n_tree * n_sub_trees;

        if (tid >= (1 << dims) || qid >= n_queries) return;

        const int* query = &queries[qid * real_dims * 2];
        int prefix_tree_offset = tree_id * n_sub_trees  + sub_tree_id;
        int local_offset = (threadIdx.x / (1 << dims)) * (1 << dims);
        partial[local_offset + tid] = get_prefix_sum_device(prefixSum[prefix_tree_offset], shape, query, dims, real_dims, tid, diff_dim, valid_map);

        __syncthreads();
        if (tid == 0) {
            uint64_t sum = 0;
        #pragma unroll
            for (int i = 0; i < (1 << dims); ++i)
                sum += partial[local_offset + i];
            sub_result[qid * tot_cluster + prefix_tree_offset] = sum;
        }
    }

    __global__ void get_cluster_sum_kernel(
        const uint64_t* cluster_counts, 
        const uint64_t* cluster_offsets,  
        const uint64_t* sub_result, // shape: [n_queries][n_trees]
        uint64_t n_tree, uint64_t n_queries,  
        uint64_t tot_clusters, 
        uint64_t* result
    )
    {
        int qid = blockIdx.x *  blockDim.x + threadIdx.x;
        int tree_id = blockIdx.y * blockDim.y + threadIdx.y;

        if (qid >= n_queries || tree_id >= n_tree) return;

        int sum = 0;
        uint64_t start = cluster_offsets[tree_id];
        uint64_t counts = cluster_counts[tree_id];
#pragma unroll
        for (int i = 0; i < counts; i++) {
            const uint64_t* cur_result = &sub_result[qid * tot_clusters];
            sum += cur_result[start + i];
        }
        result[qid * n_tree + tree_id] = sum;
    }

    void get_prefix_sum(
        raft::device_vector_view<prefix_sum_node*, uint64_t> const& compressed_sums,
        raft::device_matrix_view<int, uint64_t> const& queries,  
        raft::device_vector_view<uint64_t, uint64_t> const& cluster_starts, 
        raft::device_vector_view<uint64_t, uint64_t> const& cluster_lengthes,
        raft::device_vector_view<int, uint64_t> const& grid_dims_device, 
        raft::device_matrix_view<uint64_t, uint64_t> &sub_cluster_counts,
        raft::device_matrix_view<uint64_t, uint64_t> &clutser_counts, 
        int diff_dim, int valid_map = 0 
    )
    {
        uint64_t n_queries = queries.extent(0);
        uint64_t l = queries.extent(1) / 2;
        uint64_t real_dim = l;
        if (valid_map) {
            int new_dim = 0;
            for (int i = 0; i < l; i++) {
                if ((valid_map >> i) & 1) 
                    new_dim++;
            }
            l = new_dim;
        }
        int n_clusters = clutser_counts.extent(1); 
        int n_sub_clusters = sub_cluster_counts.extent(1) / n_clusters; 
        int tot_clusters = n_clusters * n_sub_clusters;

        dim3 blocks(std::max(1 << (int)l, 256));
        int query_blocks = blocks.x / (1 << l);
        dim3 grids((n_queries + query_blocks - 1) / query_blocks, n_clusters, n_sub_clusters);

        get_prefix_sum_kernel<<<grids, blocks>>>(
            compressed_sums.data_handle(), 
            grid_dims_device.data_handle(), 
            cluster_starts.data_handle(), 
            cluster_lengthes.data_handle(), 
            queries.data_handle(), l, real_dim,
            sub_cluster_counts.data_handle(), 
            diff_dim, 
            n_sub_clusters,
            n_clusters, n_queries, valid_map
        );
        cudaDeviceSynchronize();
        checkCUDAError("prefix sum kernel failed");

        dim3 blocks2(8, 32);
        dim3 grids2((n_queries + blocks2.x - 1) / blocks2.x , (n_clusters + blocks2.y - 1) / blocks2.y);

        get_cluster_sum_kernel<<<grids2, blocks2>>>(
            cluster_lengthes.data_handle(),
            cluster_starts.data_handle(),  
            sub_cluster_counts.data_handle(),
            n_clusters, n_queries, 
            tot_clusters,
            clutser_counts.data_handle()
        );
    }

    static void select_sub_clusters(
        raft::device_resources const& dev_resources,
        raft::device_matrix_view<float, uint64_t> const& queries, 
        raft::device_matrix_view<uint64_t, uint64_t> const& cluster_candies, 
        raft::device_vector_view<uint64_t, uint64_t> const& cluster_starts, 
        raft::device_vector_view<uint64_t, uint64_t> const& cluster_lengthes, 
        raft::device_matrix_view<uint64_t, uint64_t> const& sub_cluster_counts, 
        raft::device_matrix_view<float, uint64_t> const& sub_cluster_centers, 
        raft::device_matrix_view<uint64_t, uint64_t> &sub_cluster_candies, 
        int sub_lists
    )
    {
        int n_list = cluster_candies.extent(1);
        uint64_t n_queries = cluster_candies.extent(0);
        uint64_t clusters = cluster_starts.extent(0);
        int sub_clusters = sub_cluster_counts.extent(1) / clusters;

        raft::device_matrix_view<uint64_t, uint64_t> cluster_candi_starts = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_list);
        raft::device_matrix_view<uint64_t, uint64_t> cluster_candi_lengths = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_list);

        select_elements<uint64_t, uint64_t>(dev_resources, cluster_starts, cluster_candies, cluster_candi_starts);
        select_elements<uint64_t, uint64_t>(dev_resources, cluster_lengthes, cluster_candies, cluster_candi_lengths, 0);

        auto length_sums = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, cluster_candi_lengths.extent(1));
        matrix_scan_cub(cluster_candi_lengths, length_sums, false);
        raft::device_matrix_view<uint64_t, uint64_t> offset_table{};
        build_offset_table(cluster_candi_lengths, length_sums, cluster_candi_starts, offset_table);
        auto valid_indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, offset_table.extent(1));

        raft::device_matrix_view<uint64_t, uint64_t> dst_data{};
        collect_segments_data<uint64_t, uint64_t, uint64_t>(dev_resources, cluster_candi_lengths, cluster_candi_starts, sub_cluster_counts, dst_data, (uint64_t)0, offset_table);

        int max_counts = filter_none_zero_data(dst_data, valid_indices);
        auto valid_sub_clusters = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, valid_indices.extent(1));
        select_elements<uint64_t, uint64_t>(dev_resources, offset_table, valid_indices, valid_sub_clusters, false);

        int n_candi_centers = std::min(sub_lists, max_counts);
        sub_cluster_candies = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_candi_centers);
        auto center_distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_candi_centers);

        refine(
            dev_resources, 
            sub_cluster_centers, 
            queries, 
            valid_sub_clusters, 
            sub_cluster_candies, 
            center_distances
        );

        cudaDeviceSynchronize();
        auto sub_cluster_candies_host = new uint64_t[n_queries * n_candi_centers];
        cudaMemcpy(sub_cluster_candies_host, sub_cluster_candies.data_handle(), n_queries * n_candi_centers * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    __global__ void extract_queries_border_kernel(
        const int* ranges, int* borders, int l, int n_queries
    )
    {
        int q = blockIdx.x * blockDim.x + threadIdx.x;
        int dim = blockIdx.y * blockDim.y + threadIdx.y;

        if (q >= n_queries || dim >= l) return;

        borders[q * 2 * l + dim] = ranges[q * 2 * l + 2 * dim];
        borders[q * 2 * l + dim + l] = ranges[q * 2 * l + 2 * dim + 1];
    }

    static void extract_queries_border(
        raft::device_matrix_view<int, uint64_t> const& ranges,
        raft::device_matrix_view<int, uint64_t> &borders 
    )
    {
        uint64_t n_queries = ranges.extent(0);
        uint64_t l = ranges.extent(1) / 2;

        dim3 blocks(64, 4);
        dim3 grids((n_queries + blocks.x - 1) / blocks.x, (l + blocks.y - 1) / blocks.y);

        extract_queries_border_kernel<<<grids, blocks>>>(ranges.data_handle(), borders.data_handle(), l, n_queries);
    } 

    __global__ void get_filter_grid_ranges_kernel(
        const float* __restrict__ box_matrix,  // [n_intervals][2 * dims]
        int* __restrict__ output_grid_bounds,  // [n_intervals][2 * dims]
        const int* __restrict__ dims_array,    // [dims] 
        const uint64_t* __restrict__ label_spans, 
        const uint64_t* __restrict__ label_left, 
        const int dims,
        const int n_intervals,
        const int* shuffle 
    )
    {
        int box_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (box_id >= n_intervals) return;

        const float* box = box_matrix + box_id * 2 * dims;
        int* output = output_grid_bounds + box_id * 2 * dims;

        for (int i = 0; i < dims; ++i) {
            int d = shuffle == nullptr ? i : shuffle[i];
            float l = max((uint64_t)box[2 * d], label_left[i]);
            float r = min((uint64_t)box[2 * d + 1], label_left[i] + label_spans[i]);

            // Clamp and floor/ceil to grid range
            int grid_l = get_grid_id_device(l, label_left[i], label_spans[i], dims_array[i]);
            int grid_r = get_grid_id_device(r, label_left[i], label_spans[i], dims_array[i]);

            assert(grid_l >= 0 && grid_r <= dims_array[i]);

            output[2 * i + 1] = min(grid_r, dims_array[i] - 1);
            output[2 * i] = min(output[2 * i + 1], grid_l);
        }
    }

    void get_filter_grid_ranges(
        raft::device_matrix_view<float, uint64_t> const& filters,
        raft::device_vector_view<int, uint64_t> const& grid_dims_device,  
        raft::device_vector_view<uint64_t, uint64_t> const& label_span_device, 
        raft::device_vector_view<uint64_t, uint64_t> const& label_left_device, 
        raft::device_matrix_view<int, uint64_t> &out, 
        const int* shuffle = nullptr
    )
    {
        uint64_t n_queries = filters.extent(0);
        uint64_t dims = filters.extent(1) / 2;

        dim3 blocks(128);
        dim3 grids((n_queries + blocks.x - 1) / blocks.x);

        get_filter_grid_ranges_kernel<<<grids, blocks>>>(filters.data_handle(), out.data_handle(), 
            grid_dims_device.data_handle(), label_span_device.data_handle(), label_left_device.data_handle(), 
            dims, n_queries, shuffle);
        cudaDeviceSynchronize();
        checkCUDAError("get filter grid id kernel failed");
    }

    void index::query(query_input_t const& in, query_output_t & out) const 
    {
        uint64_t n_queries = in.queries.extent(0);
        uint64_t n_dim = in.queries.extent(1);
        int l = grid_dims_host.size();
        int clusters = super_grid_config.clusters;
        int sub_clusters = super_grid_config.sub_clusters;
        int n_list = super_grid_config.n_list;
        int sub_lists = super_grid_config.sub_lists;
        int is_split = super_grid_config.is_split;

        auto& znode_arrays = super_grid_index_data.sub_clusters_data.zorder_node_arrays;
        auto& zcode_arrays = super_grid_index_data.sub_clusters_data.zorder_code_arrays;
        auto& zcodes_array_offsets = super_grid_index_data.sub_clusters_data.zorder_array_offsets;
        auto& zcodes_array_lengthes = super_grid_index_data.sub_clusters_data.zorder_array_lengthes;
        auto& compressed_sums = super_grid_index_data.sub_clusters_data.compressed_sums;

        auto& cluster_starts = super_grid_index_data.sub_cluster_offsets;
        auto& cluster_lengthes = super_grid_index_data.sub_cluster_counts;

        auto& cluster_centers = super_grid_index_data.centers;
        auto& sub_cluster_centers = super_grid_index_data.sub_clusters_data.sub_centers;

        auto cluster_counts = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, clusters);
        auto sub_cluster_counts = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, sub_clusters * clusters);

        auto filters_range = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, 2 * l);
        recovery_filters(in.query_labels, filters_range, f_config);
        auto filters = parafilter_mmr::make_device_matrix_view<int, uint64_t>(n_queries, 2 * l);
        get_filter_grid_ranges(filters_range, grid_dims_device, label_span_device, label_left_device, filters, shuffle);

        uint64_t diff_dim = grid_dims_host.size() - 1;
        for (int i = grid_dims_host.size() - 1; i >= 0; i--) {
            if (grid_dims_host[i] > 1) {
                diff_dim = i;
                break;
            }
        }
        int max_size = grid_dims_host[diff_dim];

        get_prefix_sum(compressed_sums, filters, cluster_starts, cluster_lengthes, grid_dims_device, sub_cluster_counts, cluster_counts, diff_dim, morton_config.valid_map);

        auto valid_indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, clusters);
        int max_counts = filter_none_zero_data(cluster_counts, valid_indices);

        int n_candi_centers = std::min(n_list, max_counts);
        auto candi_centers = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_candi_centers);
        auto center_distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_candi_centers);

        ::refine(
            in.dev_resources, 
            cluster_centers, 
            in.queries, 
            valid_indices, 
            candi_centers, 
            center_distances,
            true
        );

        raft::device_matrix_view<uint64_t, uint64_t> sub_cluster_candies{};

        select_sub_clusters(
            in.dev_resources, 
            in.queries, 
            candi_centers, 
            super_grid_index_data.sub_cluster_offsets,  
            super_grid_index_data.sub_cluster_counts, 
            sub_cluster_counts, 
            sub_cluster_centers, 
            sub_cluster_candies, 
            sub_lists
        );

        auto borders = parafilter_mmr::make_device_matrix_view<int, uint64_t>(2 * n_queries, l);
        extract_queries_border(filters, borders);

        auto query_zcode_ranges = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, 2);
        auto query_zcode_vector = raft::make_device_vector_view<uint64_t, uint64_t>(query_zcode_ranges.data_handle(), n_queries * 2);
        encode_points(borders, query_zcode_vector, morton_config);

        raft::device_matrix_view<uint64_t, uint64_t> out_counts{};  
        raft::device_matrix_view<uint64_t, uint64_t> out_offsets{}; 

        if (!is_split) {
            int n_splits = 1;
            sub_lists = std::min((int)sub_cluster_candies.extent(1), sub_lists);
            out_counts = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, sub_lists * n_splits);  
            out_offsets = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, sub_lists * n_splits); 

            search_zorder_ranges(
                zcode_arrays, 
                zcodes_array_offsets,
                zcodes_array_lengthes,
                query_zcode_ranges, 
                filters, 
                sub_cluster_candies, 
                out_offsets, 
                out_counts
            ); 
        }
        else {
            int n_splits = super_grid_config.n_split;
            out_counts = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, sub_lists * n_splits);  
            out_offsets = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, sub_lists * n_splits); 
            sub_lists = std::min((int)sub_cluster_candies.extent(1), sub_lists);

            search_zorder_ranges(
                zcode_arrays, 
                zcodes_array_offsets,
                zcodes_array_lengthes,
                query_zcode_ranges, 
                filters, 
                sub_cluster_candies,
                out_offsets, 
                out_counts, 
                morton_config
            );
        }


        auto& dense_out_counts = out_counts;
        auto& dense_out_offsets = out_offsets;

        /*a null place holder, current build offset table call not build the offset table*/
        int n_batch_split = 1;
        uint64_t refine_ratio = pq_config.exps0;

        if (n_batch_split > 1) {
            raft::device_matrix_view<uint64_t, uint64_t> offset_table_null{};
            auto length_sums = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, dense_out_counts.extent(1));
            matrix_scan_cub(dense_out_counts, length_sums, false);
            auto row_lengthes = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_queries);
            calc_row_lengthes<uint64_t, uint64_t>(dense_out_counts, length_sums, row_lengthes);
            auto row_indices = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_queries);

            thrust::sequence(
                thrust::device_pointer_cast(row_indices.data_handle()), 
                thrust::device_pointer_cast(row_indices.data_handle()) + n_queries
            );        
            thrust::sort_by_key(
                thrust::device_pointer_cast(row_lengthes.data_handle()), 
                thrust::device_pointer_cast(row_lengthes.data_handle()) + n_queries, 
                thrust::device_pointer_cast(row_indices.data_handle())
            );

            int* batch_size = new int[n_batch_split];
            batch_size[0] = n_queries * 1 / 3;
            batch_size[1] = n_queries * 1 / 3;
            batch_size[2] = n_queries - (batch_size[0] + batch_size[1]);
            int batch_offset = 0;

            for (int i = 0; i < n_batch_split; i++) {
                auto cur_indices_view = raft::make_device_matrix_view<uint64_t, uint64_t>(row_indices.data_handle() + batch_offset, batch_size[i], 1);
                auto cur_counts_view = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(batch_size[i], sub_lists);
                auto cur_offsets_view = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(batch_size[i], sub_lists);
                select_elements<uint64_t, uint64_t>(in.dev_resources, dense_out_counts, cur_indices_view, cur_counts_view, true, 0);
                select_elements<uint64_t, uint64_t>(in.dev_resources, dense_out_offsets, cur_indices_view, cur_offsets_view);
                
                auto length_sums = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, cur_counts_view.extent(1));
                matrix_scan_cub(cur_counts_view, length_sums, false);
                raft::device_matrix_view<uint64_t, uint64_t> offset_table{};
                build_offset_table<uint64_t, uint64_t>(cur_counts_view, length_sums, cur_offsets_view, offset_table);
                raft::device_matrix_view<uint64_t, uint64_t> out_idx = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(batch_size[i], offset_table.extent(1));
                select_elements<uint64_t, uint64_t>(in.dev_resources, znode_arrays, offset_table, out_idx);

                auto cur_queries = parafilter_mmr::make_device_matrix_view<float, uint64_t>(batch_size[i], n_dim);
                select_elements<float, uint64_t>(in.dev_resources, in.queries, cur_indices_view, cur_queries);
                auto cur_labels = parafilter_mmr::make_device_matrix_view<float, uint64_t>(batch_size[i], l);
                select_elements<float, uint64_t>(in.dev_resources, in.query_labels, cur_indices_view, cur_labels);
                
                valid_indices_t candidate_indices{}; 

                candidate_indices.valid_cnt = out_idx.extent(1);
                candidate_indices.indices = out_idx;

                query_output_t cur_out{};
                cur_out.neighbors = raft::make_device_matrix_view<uint64_t, uint64_t>(out.neighbors.data_handle() + topk * batch_offset, batch_size[i], topk);
                cur_out.distances = raft::make_device_matrix_view<float, uint64_t>(out.distances.data_handle() + topk * batch_offset, batch_size[i], topk);
                query_input_t cur_in{in.dev_resources};
                cur_in.queries = cur_queries;
                cur_in.query_labels = cur_labels;
                cur_in.dataset = in.dataset;
                if (refine_ratio > 1) {
                    query_output_t tmp_out;
                    tmp_out.neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(batch_size[i], refine_ratio * topk);
                    tmp_out.distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(batch_size[i], refine_ratio * topk);
                    
                    query_prefilter(cur_in, candidate_indices, tmp_out);
                    refine(cur_in, tmp_out, cur_out);
                }
                else query_prefilter(cur_in, candidate_indices, cur_out);
                batch_offset += batch_size[i];
            }
        }
        else {
            raft::device_matrix_view<uint64_t, uint64_t> offset_table{};
            auto row_lengthes = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_queries); 
            auto length_sums = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, dense_out_counts.extent(1));
            matrix_scan_cub(dense_out_counts, length_sums, false);
            calc_row_lengthes(dense_out_counts, length_sums, row_lengthes);
            build_offset_table<uint64_t, uint64_t>(dense_out_counts, length_sums, dense_out_offsets, offset_table, row_lengthes);

            raft::device_matrix_view<uint64_t, uint64_t> out_idx = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, offset_table.extent(1));
            LOG(TRACE) << "candidate points counts after zsearch is: " << offset_table.extent(1); 
            select_elements<uint64_t, uint64_t>(in.dev_resources, znode_arrays, offset_table, out_idx);
            valid_indices_t candidate_indices{}; 

            candidate_indices.valid_cnt = out_idx.extent(1);
            candidate_indices.indices = out_idx;

            if (refine_ratio > 1) {
                query_output_t tmp_out;
                
                if (l > 1) {   
                    auto candi_counts = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, sub_lists);
                    select_elements<uint64_t, uint64_t>(in.dev_resources, sub_cluster_counts, sub_cluster_candies, candi_counts, false, 0);
                    auto count_sums = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_queries);
                    
                    matrix_reduce_sum(candi_counts.data_handle(), count_sums.data_handle(), n_queries, (uint64_t)sub_lists);

                    uint64_t real_candies = device_reduce(
                        count_sums.data_handle(), 
                        n_queries, 
                        (uint64_t)0, 
                        thrust::plus<uint64_t>()
                    );

                    uint64_t coarse_candies = device_reduce(
                        row_lengthes.data_handle(), 
                        n_queries, 
                        (uint64_t)0, 
                        thrust::plus<uint64_t>()
                    );

                    float beta = (float)coarse_candies / (float)real_candies;
                    LOG(TRACE) << "current batch coarse candidate is : " << coarse_candies << "real candidate is : " << real_candies << "ratio is : " << 1.f / beta;

                    uint64_t n_candies = std::min(uint64_t(refine_ratio * topk * beta), offset_table.extent(1));
                    query_output_t tmp_out_pq;
                    tmp_out_pq.neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_candies);
                    tmp_out_pq.distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_candies);
                    
                    query_pq(in, candidate_indices, tmp_out_pq);
                    tmp_out.neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_candies);
                    tmp_out.distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, n_candies);
                    valid_indices_t valid_indices_in{};
                    valid_indices_in.indices = tmp_out_pq.neighbors;
                    valid_indices_in.valid_cnt = refine_ratio * topk * beta;

                    valid_indices_t valid_indices{};
                    valid_indices.indices = tmp_out.neighbors;
                    prefilter(in, valid_indices_in, valid_indices); 
                }
                else {

                    tmp_out.neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, refine_ratio * topk);
                    tmp_out.distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, refine_ratio * topk);
                    query_pq(in, candidate_indices, tmp_out);
                }
                refine(in, tmp_out, out);
            }
            else query_prefilter(in, candidate_indices, out);
        }
    }
};