#include <raft/neighbors/ivf_pq.cuh>
#include <core/interfaces.cuh>
#include <core/mat_operators.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/select_k.cuh>
#include <utils/config_utils.h>
#include <core/filter.cuh>
#include <core/preprocessing.cuh>
#include <core/mmr.cuh>

namespace ivf_pq {
    struct index_data_t {
        raft::neighbors::ivf_pq::search_params search_params;
        raft::neighbors::ivf_pq::index_params index_params;
        std::unique_ptr<raft::neighbors::ivf_pq::index<int64_t>> index_ptr;
        raft::device_matrix_view<float, uint64_t> data_labels;
    };

    struct index_params {
        uint64_t n_list;
        uint64_t kmeans_n_iters;
        uint64_t pq_bits;
        uint64_t pq_dim;
        uint64_t n_probes;
        uint64_t beta;
        uint64_t refine;
    };

    struct config_t: common_config {
        // parameters for build pq index
        index_params ivfpq_config;

        config_t(std::string const& path):
            common_config(path, [](std::map<std::string, int> &m) {
               m.insert({ 
                            {"N_LIST", OFFSETOF_NESTED(config_t, ivfpq_config, index_params, n_list)}, 
                            {"KMEANS_N_ITERS", OFFSETOF_NESTED(config_t, ivfpq_config, index_params, kmeans_n_iters)},
                            {"PQ_BITS", OFFSETOF_NESTED(config_t, ivfpq_config, index_params, pq_bits)}, 
                            {"PQ_DIM", OFFSETOF_NESTED(config_t, ivfpq_config, index_params, pq_dim)}, 
                            {"N_PROBES", OFFSETOF_NESTED(config_t, ivfpq_config, index_params, n_probes)}, 
                            {"BETA", OFFSETOF_NESTED(config_t, ivfpq_config, index_params, beta)},
                            {"REFINE", OFFSETOF_NESTED(config_t, ivfpq_config, index_params, refine)}
                        }); 
            }) {}
    };

    class index : public index_base_t {
    public:
        index(index_params const& ivfpq_config, std::string const& filter_config_path, uint64_t topk) : 
            ivfpq_config(ivfpq_config), index_base_t(topk), f_config(filter_config_path)  {}
        void build(const build_input_t& in) override;
        void query(const query_input_t& in, query_output_t& out) const override;
    private:
        index_params ivfpq_config;
        index_data_t index_data;
        filter_config f_config;
    };

};