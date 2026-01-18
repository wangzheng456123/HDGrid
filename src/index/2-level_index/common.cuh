#include <core/mat_operators.cuh>

void group_by_cluster_id(
    raft::device_vector_view<uint64_t, uint64_t> const& cluster_ids, 
    std::vector<raft::device_vector_view<uint64_t, uint64_t>>& grouped_data
); 