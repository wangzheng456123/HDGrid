#include "index/baseline/ivfpq-post.cuh"

namespace ivf_pq {
    void index::build(const build_input_t& in) {
        index_data.index_params.n_lists = ivfpq_config.n_list;
        index_data.index_params.kmeans_n_iters = ivfpq_config.kmeans_n_iters;
        index_data.index_params.pq_bits = ivfpq_config.pq_bits;
        index_data.index_params.pq_dim = ivfpq_config.pq_dim;

        index_data.search_params.n_probes = ivfpq_config.n_probes;
        index_data.data_labels = in.data_labels;

        auto const_dataset = raft::make_device_matrix_view<const float, int64_t>(in.dataset.data_handle(), 
            in.dataset.extent(0), in.dataset.extent(1));

        auto built_index = raft::neighbors::ivf_pq::build<float>(
            in.dev_resources,
            index_data.index_params,
            const_dataset
        );

        index_data.index_ptr = std::make_unique<raft::neighbors::ivf_pq::index<int64_t>>(
           std::move(built_index)
        );
    }
    void index::query(const query_input_t& in, query_output_t& out) const {
        uint64_t n_queries = in.queries.extent(0);
        uint64_t l = f_config.l;

        uint64_t refine_topk = topk * ivfpq_config.refine * ivfpq_config.beta;
        auto i_distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, refine_topk);
        /*current version raft not support uint64_t type index when searching*/
        auto i_neighbors_raft = parafilter_mmr::make_device_matrix_view<int64_t, uint64_t>(n_queries, refine_topk);

        raft::neighbors::ivf_pq::search<float, int64_t>(in.dev_resources, index_data.search_params, 
                *index_data.index_ptr, in.queries, i_neighbors_raft, i_distances);
        auto i_neighbors = raft::make_device_matrix_view<uint64_t, uint64_t>((uint64_t*)i_neighbors_raft.data_handle(), 
                i_neighbors_raft.extent(0), i_neighbors_raft.extent(1));
        auto filter_neighbors = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * ivfpq_config.beta);
        auto filter_distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * ivfpq_config.beta);

        query_output_t tmp_out{};
        tmp_out.distances = i_distances;
        tmp_out.neighbors = i_neighbors;

        query_output_t filter_out{};
        filter_out.distances = filter_distances;
        filter_out.neighbors = filter_neighbors;
        refine(in, tmp_out, filter_out);

        auto filters = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, 2 * l);
        recovery_filters(in.query_labels, filters, f_config);

        auto filtered_indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * ivfpq_config.beta);
        auto filtered_indices_indirect = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk * ivfpq_config.beta);

        uint64_t valid_cnt = filter_valid_data(in.dev_resources, index_data.data_labels, filters, filtered_indices, filter_neighbors, true, filtered_indices_indirect.data_handle());
        auto filtered_distances = parafilter_mmr::make_device_matrix_view<float, uint64_t>(n_queries, topk * ivfpq_config.beta);
        select_elements<float, uint64_t>(in.dev_resources, filter_distances, filtered_indices_indirect, filtered_distances, false);    
        
        auto indirect_indices = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, topk);
        raft::matrix::select_k<float, uint64_t>(in.dev_resources, filtered_distances, std::nullopt, out.distances, indirect_indices, true);

        select_elements<uint64_t, uint64_t>(in.dev_resources, filtered_indices, indirect_indices, out.neighbors, false);
    }
}