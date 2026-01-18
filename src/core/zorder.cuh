#pragma once
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <raft/core/device_mdspan.hpp>
#include <core/mmr.cuh>
#include <core/mat_operators.cuh>
#include <utils/debugging_utils.cuh>

#define MAX_DIMS 8

struct Box_GPU 
{
    int2 bounds[MAX_DIMS];

    __device__ __host__ int2& operator[](int i) { return bounds[i]; }
    __device__ __host__ const int2& operator[](int i) const { return bounds[i]; }
};

struct __host__ __device__ MortonConfig_t 
{
    int dims;
    int max_bits;
    int valid_map;
};

void encode_points(
    raft::device_matrix_view<int, uint64_t> const& points,
    raft::device_vector_view<uint64_t, uint64_t> &zcodes, 
    MortonConfig_t config
);

void search_zorder_ranges(
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_arrays, // A second level zcode array splited by offsets and lengthes array 
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array_offsets, 
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array_lengthes, 
    raft::device_matrix_view<uint64_t, uint64_t> const& query_ranges, 
    raft::device_matrix_view<int, uint64_t> const& query_boxes, 
    raft::device_matrix_view<uint64_t, uint64_t> const& candi_arrays,   
    raft::device_matrix_view<uint64_t, uint64_t> &out_starts, 
    raft::device_matrix_view<uint64_t, uint64_t> &out_sizes, 
    MortonConfig_t config
); 

void search_zorder_ranges(
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_arrays, // A second level zcode array splited by offsets and lengthes array 
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array_offsets, 
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array_lengthes, 
    raft::device_matrix_view<uint64_t, uint64_t> const& query_ranges, 
    raft::device_matrix_view<int, uint64_t> const& query_boxes, 
    raft::device_matrix_view<uint64_t, uint64_t> const& candi_arrays,  
    raft::device_matrix_view<uint64_t, uint64_t> &end_points, 
    raft::device_matrix_view<uint64_t, uint64_t> &counts
);