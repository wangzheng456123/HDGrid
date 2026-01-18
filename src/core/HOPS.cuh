#pragma once
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <vector>
#include <cassert>
#include <numeric>
#include <core/mmr.cuh>

using Index = int;
using Value = uint64_t;
using Shape = std::vector<int>;
class HOPS {
public:
    HOPS(const Shape& shape)
        :shape_(shape)
    {
        // walk-around for different data types
        total_size_ = std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<Index>());
        Value* scratch_raw_; 
        cudaMalloc(&scratch_raw_, sizeof(Value) * total_size_);
        scratch_ = thrust::device_pointer_cast(scratch_raw_);
    }

    ~HOPS() 
    {
        cudaFree(thrust::raw_pointer_cast(scratch_));
    }

    void load(Value* data_ptr) {
        data_ = thrust::device_pointer_cast(data_ptr);
    }

    Value* data_handle() const
    {
        return thrust::raw_pointer_cast(data_);
    }

    Index size() const {
        return total_size_;
    }

    Index dim(int i) const {
        return shape_[i];
    }; 

    Index rank() const {
        return shape_.size();
    }


    void scan_along_axis(Index axis);
    void multi_dim_scan();

private:
    thrust::device_ptr<Value> data_;
    Shape shape_;
    Value total_size_;
    thrust::device_ptr<Value> scratch_;
};