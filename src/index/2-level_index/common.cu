#include <index/2-level_index/common.cuh>

static __global__ void select_rows_by_flag_kernel(const int* flags, const uint64_t *prefix_sum, 
                                               uint64_t* selected_data, uint64_t n_data) 
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_data) return;

    if (flags[x]) {
        uint64_t loc = prefix_sum[x];
        selected_data[loc - 1] = x;
    }
}

static __global__ void label_data_with_value_kernel(int* flags, int val, const uint64_t* ids, uint64_t n_data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_data) return;

    if (ids[x] == val) {
        flags[x] = 1;
    }
    else {
        flags[x] = 0;
    }
}

void group_by_cluster_id(
        raft::device_vector_view<uint64_t, uint64_t> const& cluster_ids, 
        std::vector<raft::device_vector_view<uint64_t, uint64_t>>& grouped_data
    ) 
{
    uint64_t n_data = cluster_ids.extent(0);
    uint64_t n_clusters = grouped_data.size();

    int* tmp_flags = static_cast<int*>(parafilter_mmr::mem_allocator(n_data * sizeof(int)));
    uint64_t* prefix_sum = static_cast<uint64_t*>(parafilter_mmr::mem_allocator(n_data * sizeof(uint64_t)));

    dim3 threads_block_size(256);
    dim3 blocks_per_grid(n_data + threads_block_size.x - 1 / threads_block_size.x);

    uint64_t tot_valid = 0;
    for (int i = 0; i < n_clusters; i++) {
        cudaMemset(tmp_flags, 0, n_data * sizeof(int));

        label_data_with_value_kernel<<<blocks_per_grid, threads_block_size>>>(tmp_flags, i, cluster_ids.data_handle(), n_data);

        thrust::device_ptr<int> thrust_flags(tmp_flags);
        thrust::device_ptr<uint64_t> thrust_prefix_sum(prefix_sum);
        thrust::inclusive_scan(thrust::device, thrust_flags, thrust_flags + n_data, thrust_prefix_sum, thrust::plus<uint64_t>());
        uint64_t n_valid = thrust::reduce(thrust::device, thrust_prefix_sum, thrust_prefix_sum + n_data, 0, thrust::maximum<uint64_t>());
        if (n_valid == 0) continue;
        tot_valid += n_valid;

        // todo: allocate these mem from a local pool.
        auto cur_cluster_group = parafilter_mmr::make_device_vector_view<uint64_t, uint64_t>(n_valid);
        select_rows_by_flag_kernel<<<blocks_per_grid, threads_block_size>>>(tmp_flags, prefix_sum, cur_cluster_group.data_handle(), n_data);
        grouped_data[i] = cur_cluster_group;
    }
    assert(tot_valid == n_data);
}