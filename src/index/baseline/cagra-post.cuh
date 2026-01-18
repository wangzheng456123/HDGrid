#include <core/interfaces.cuh>
#include <core/mat_operators.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/select_k.cuh>
#include <utils/config_utils.h>
#include <core/filter.cuh>
#include <core/preprocessing.cuh>
#include <core/mmr.cuh>

namespace cagra {
    struct index_data_t {
        raft::neighbors::cagra::search_params search_params;
        raft::neighbors::cagra::index_params index_params;
        std::unique_ptr<raft::neighbors::cagra::index<float, int64_t>> index_ptr;
        raft::device_matrix_view<float, uint64_t> data_labels;
    };

    struct index_params {
        uint64_t degree;
        uint64_t i_degree;
        uint64_t itopk_size; // key componets for trade-off acuracy and performance
        uint64_t search_width;
        uint64_t beta;
    };

    struct config_t: common_config {
        // parameters for build pq index
        index_params cagra_config;

        config_t(std::string const& path):
            common_config(path, [](std::map<std::string, int> &m) {
               m.insert({ 
                            {"DEGREE", OFFSETOF_NESTED(config_t, cagra_config, index_params, degree)}, 
                            {"I_DEGREE", OFFSETOF_NESTED(config_t, cagra_config, index_params, i_degree)},
                            {"ITOPK_SIZE", OFFSETOF_NESTED(config_t, cagra_config, index_params, itopk_size)}, 
                            {"SEARCH_WIDTH", OFFSETOF_NESTED(config_t, cagra_config, index_params, search_width)}, 
                            {"BETA", OFFSETOF_NESTED(config_t, cagra_config, index_params, beta)}
                        }); 
            }) {}
    };

    class index : public index_base_t {
    public:
        index(index_params const& cagra_config, std::string const& filter_config_path, uint64_t topk) : 
            cagra_config(cagra_config), index_base_t(topk), f_config(filter_config_path)  {}
        void build(const build_input_t& in) override;
        void query(const query_input_t& in, query_output_t& out) const override;
    private:
        index_params cagra_config;
        index_data_t index_data;
        filter_config f_config;
    };
    

};