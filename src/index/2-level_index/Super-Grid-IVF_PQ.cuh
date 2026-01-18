#pragma conce
#include <index/prefiltering.cuh>
#include <raft/distance/distance.cuh>
#include <raft/cluster/kmeans_balanced.cuh>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <index/2-level_index/common.cuh>
#include <core/HOPS.cuh>
#include <core/zorder.cuh>
#include <cmath>
#include <algorithm>
#include <sstream>

namespace super_grid {
    struct index_params {
        uint64_t clusters;
        uint64_t sub_clusters;
        uint64_t max_grids;
        uint64_t n_list;
        uint64_t sub_lists;
        uint64_t n_split;
        uint64_t label_mask;
        uint64_t is_split;

        std::string to_string() const {
            std::ostringstream oss;
            oss << "clusters = " << clusters
                << ", sub_clusters = " << sub_clusters
                << ", n_list = " << n_list 
                << ", sub_lists = " << sub_lists
                << ", label_mask = " << label_mask
                << ", n_split = " << n_split
                << ", is_split = " << is_split;
            return oss.str();
        }
    };

    struct config_t: pq::config_t {
        index_params super_grid_config;

        config_t(std::string const& path):
            pq::config_t(path, [](std::map<std::string, int> &m) {
               m.insert({ 
                            {"CLUSTERS", OFFSETOF_NESTED(config_t, super_grid_config, index_params, clusters)}, 
                            {"SUB_CLUSTERS", OFFSETOF_NESTED(config_t, super_grid_config, index_params, sub_clusters)}, 
                            {"MAX_GRIDS", OFFSETOF_NESTED(config_t, super_grid_config, index_params, max_grids)}, 
                            {"N_LIST", OFFSETOF_NESTED(config_t, super_grid_config, index_params, n_list)}, 
                            {"SUB_LISTS", OFFSETOF_NESTED(config_t, super_grid_config, index_params, sub_lists)}, 
                            {"N_SPLIT", OFFSETOF_NESTED(config_t, super_grid_config, index_params, n_split)},
                            {"LABEL_MASK", OFFSETOF_NESTED(config_t, super_grid_config, index_params, label_mask)},
                            {"IS_SPLIT", OFFSETOF_NESTED(config_t, super_grid_config, index_params, is_split)},
                        }); 
            }) {  }
    };

    struct prefix_sum_node {
        int2* sums;
        int pre;
        int len; 
    };

    // device data per sub cluster
    struct cluster_data {
        raft::device_vector_view<prefix_sum_node*, uint64_t> compressed_sums;
        raft::device_vector_view<uint64_t, uint64_t> zorder_node_arrays; 
        raft::device_vector_view<uint64_t, uint64_t> zorder_code_arrays;
        raft::device_vector_view<uint64_t, uint64_t> zorder_array_offsets;
        raft::device_vector_view<uint64_t, uint64_t> zorder_array_lengthes;
        raft::device_matrix_view<float, uint64_t> sub_centers;
    };

    struct index_data_t {
        cluster_data sub_clusters_data;
        raft::device_matrix_view<float, uint64_t> centers;
        raft::device_vector_view<uint64_t, uint64_t> sub_cluster_offsets;
        raft::device_vector_view<uint64_t, uint64_t> sub_cluster_counts;    
    };

    class index : public prefiltering::index {
    public:
        index(pq::index_params const& pq_params, index_params const& super_grid_config, std::string const& filter_config_path, uint64_t topk) :
            prefiltering::index(pq_params, filter_config_path, topk), super_grid_config(super_grid_config) {
                LOG(TRACE) << super_grid_config.to_string();
            }
        void build(build_input_t const&) override;
        void query(query_input_t const&, query_output_t &) const override;
    protected:
        index_data_t super_grid_index_data;
        index_params super_grid_config;
    private:
        std::vector<int> grid_dims_host;
        raft::device_vector_view<int, uint64_t> grid_dims_device;
        raft::device_vector_view<uint64_t, uint64_t> label_span_device;
        raft::device_vector_view<uint64_t, uint64_t> label_left_device;
        uint64_t tot_grids;
        MortonConfig_t morton_config;
        void calc_grid_shape(raft::device_matrix_view<float, uint64_t> const& data_labels);
        void compress_HOPS_data(
            HOPS const& sums, 
            raft::device_vector_view<prefix_sum_node, uint64_t>& compressed_sums,
            int diff_dim
        );
        void init_state();
        uint64_t offset = 0;
        uint64_t remain = 0;
        int* shuffle;
    };
};