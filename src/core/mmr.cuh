#pragma once
#include <raft/core/device_mdarray.hpp>
#include <map>
#include <stdint.h>
#include <vector>
#include <array>
#include <utils/debugging_utils.cuh>
#include "cuda_runtime.h"
#include "easylogging++.h"

#ifdef TAGGED_MMR
#define INIT_PARAFILTER_MMR_STATIC_MEMBERS \
    std::vector<std::array<std::vector<void*>, MEM_TAG_COUNT>> parafilter_mmr::taged_mem; \
    std::vector<std::array<uint64_t, MEM_TAG_COUNT>> parafilter_mmr::allocated_size; \
    std::vector<mem_type> parafilter_mmr::tags;

enum mem_type {
  MEM_DEFAULT, // temperary memory can be freed after used 
  MEM_INDEX, // index memory allocated when build and freed after query
  MEM_INPUT, // input memory that can be freed after query finish
  MEM_OUTPUT,  // output memory that can be freed after query finish
  MEM_TAG_COUNT
};
#else
#define INIT_PARAFILTER_MMR_STATIC_MEMBERS                         \
    std::vector<uint64_t> parafilter_mmr::total;                   \
    std::vector<uint64_t> parafilter_mmr::available;               \
    std::vector<std::map<uint64_t, std::vector<void *>>> parafilter_mmr::cur_mems; \
    std::vector<std::map<uint64_t, int>> parafilter_mmr::cur_offset;
#endif

class parafilter_mmr {
public:
    static void init_mmr();
  
    template<typename ElementType, typename IndexType>
    static auto make_device_matrix_view(IndexType n_row, 
                                        IndexType n_dim) 
    {
      ElementType *device_ptr;
      uint64_t size = sizeof(ElementType) * n_row * n_dim;
      device_ptr = (ElementType*)mem_allocator(size);
      LOG(INFO) << "alloc memory with size: " << size;
      return raft::make_device_matrix_view<ElementType, IndexType>(device_ptr, n_row, n_dim);
    }
    
    template<typename ElementType, typename IndexType>
    static auto make_device_vector_view(IndexType n_elements) {
      ElementType *device_ptr;
      uint64_t size = sizeof(ElementType) * n_elements;
      device_ptr = (ElementType*)mem_allocator(size);
      LOG(INFO) << "alloc memory with size: " << size;
      return raft::make_device_vector_view<ElementType, IndexType>(device_ptr, n_elements);
    }

    static void* mem_allocator(uint64_t size);

#ifdef TAGGED_MMR
    static void add_mem_with_tag(void *mem, uint64_t size) {
      int id; 
      cudaGetDevice(&id);
      mem_type tag = tags[id];
      taged_mem[id][tag].push_back(mem);
      allocated_size[id][tag] += size;
    }

    static void set_tag(mem_type tag) {
      int id; 
      cudaGetDevice(&id);
      tags[id] = tag;
    } 

    static void free_mem_with_tag(mem_type tag) {
      int id; 
      cudaGetDevice(&id);
      for (auto block : taged_mem[id][tag])
        cudaFree(block);
      allocated_size[id][tag] = 0;
      taged_mem[id][tag].clear();
    }

    static void print_mem_statistic_with_tag() {
      int id; 
      cudaGetDevice(&id);
      for (uint64_t i = 0; i < MEM_TAG_COUNT; i++) {
        LOG(TRACE) << "tag: " << i << "alloacted: " << allocated_size[id][i] / (1024 * 1024)
          << "MB";
      }
    }
#else 
    static uint64_t get_current_workspace_free() 
    {
      int id;
      cudaGetDevice(&id);
      return available[id];
    }   
  
    static uint64_t get_current_workspace_total() 
    {
      int id;
      cudaGetDevice(&id);
      return total[id];
    }

    static uint64_t get_current_workspace_used() {
      int id;
      cudaGetDevice(&id);
      return total[id] - available[id];
    }
  
    static void reset_current_workspace(uint64_t size) {
      int id;
      cudaGetDevice(&id);
      total[id] = size; 
      available[id] = size; 
    }
    static void workspace_add_mem(void *mem, uint64_t size) {
      int id;
      cudaGetDevice(&id);
      available[id] -= size;
      LOG(INFO) << "add device mem with" << mem;
      cur_mems[id][size].push_back(mem);
    }
    static void free_cur_workspace_device_mems(bool free_mems = true); 
#endif
    
#ifdef TAGGED_MMR
    // taged memory manager
    static std::vector<std::array<std::vector<void*>, MEM_TAG_COUNT>> taged_mem;
    static std::vector<std::array<uint64_t, MEM_TAG_COUNT>> allocated_size;
    static std::vector<mem_type> tags;
#else
    // todo: make these data thread safe
    static std::vector<uint64_t> total;
    static std::vector<uint64_t> available;
    // todo: implement mmr deconstructor to avoid GPU mem leak 
    static std::vector<std::map<uint64_t, std::vector<void *>>> cur_mems;
    static std::vector<std::map<uint64_t, int>> cur_offset;
#endif
    ~parafilter_mmr();
private:
};