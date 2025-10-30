#include <core/HOPS.cuh>

void HOPS::scan_along_axis(Index axis) {
    assert(axis < shape_.size());

    Index ndim = shape_.size();
    Index axis_size = shape_[axis];

    Index axis_stride = 1;
    for (Index i = axis + 1; i < ndim; ++i)
        axis_stride *= shape_[i];

    Index outer = 1;
    for (Index i = 0; i < axis; ++i)
        outer *= shape_[i];

    Index inner = axis_stride;

    thrust::device_vector<Index> gather_idx(total_size_);
    thrust::device_vector<Index> scatter_idx(total_size_);
    thrust::device_vector<Index> segment_keys(total_size_);

    thrust::counting_iterator<Index> idx_first(0);

    thrust::transform(
        idx_first, idx_first + total_size_,
        gather_idx.begin(),
        [=] __host__ __device__(Index flat_idx) {
        Index i = flat_idx;
        Index c = i % inner;
        i /= inner;
        Index b = i % axis_size;
        Index a = i / axis_size;

        // new index: (a * inner + c) * axis_size + b
        return (a * inner + c) * axis_size + b;
    }
    );
    cudaDeviceSynchronize();

    thrust::device_vector<Index> inverse_idx(total_size_);
    thrust::sequence(inverse_idx.begin(), inverse_idx.end(), 0);
    thrust::scatter(inverse_idx.begin(), inverse_idx.end(),
        gather_idx.begin(), scatter_idx.begin());
    Index* scatter_idx_ptr = scatter_idx.data().get();
    
    thrust::gather(scatter_idx.begin(), scatter_idx.end(),
        data_, scratch_);

    thrust::transform(
        idx_first, idx_first + total_size_,
        segment_keys.begin(),
        [=] __host__ __device__(Index i) {
        return i / axis_size;  
    }
    );

    thrust::inclusive_scan_by_key(
        segment_keys.begin(), segment_keys.end(),
        scratch_,
        scratch_,
        thrust::equal_to<Index>(),
        thrust::plus<float>()
    );

    thrust::gather(gather_idx.begin(), gather_idx.end(),
        scratch_, data_);
}

void HOPS::multi_dim_scan() {
    for (Index axis = 0; axis < shape_.size(); ++axis) {
        scan_along_axis(axis);
    }
}