#pragma once
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <core/mmr.cuh>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>
#include <utils/debugging_utils.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <optional>

const int block_size = 128;
const int block_size_x = 32;
const int block_size_y = 16;

template <typename m_t, typename idx_t = int>
RAFT_KERNEL slice(const m_t* src_d, idx_t lda, m_t* dst_d, idx_t x1, idx_t y1, idx_t x2, idx_t y2)
{
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx_t dm = x2 - x1, dn = y2 - y1;
  if (idx < dm * dn) {
    idx_t i = idx % dm, j = idx / dm;
    idx_t is = i + x1, js = j + y1;
    dst_d[idx] = src_d[is + js * lda];
  }
}

template <typename idx_t>
struct slice_coordinates {
  idx_t row1;  ///< row coordinate of the top-left point of the wanted area (0-based)
  idx_t col1;  ///< column coordinate of the top-left point of the wanted area (0-based)
  idx_t row2;  ///< row coordinate of the bottom-right point of the wanted area (1-based)
  idx_t col2;  ///< column coordinate of the bottom-right point of the wanted area (1-based)

  slice_coordinates(idx_t row1_, idx_t col1_, idx_t row2_, idx_t col2_)
    : row1(row1_), col1(col1_), row2(row2_), col2(col2_)
  {
  }
};

template<typename ElmentType, typename IndexType>
__global__ void min_max_reduce_kernel(ElmentType *input, ElmentType* output, IndexType n, bool is_min = true)
{
    IndexType block_size = blockDim.x;
    IndexType thread_id = threadIdx.x;
    IndexType block_id = blockIdx.x;

    IndexType chunk_size = block_size * 2;
    IndexType block_start = block_id * chunk_size;
    IndexType left;  // holds index of left operand
    IndexType right; // holds index or right operand
    IndexType threads = block_size;
    for (IndexType stride = 1; stride < chunk_size; stride *= 2, threads /= 2)
    {
        left = block_start + thread_id * (stride * 2);
        right = left + stride;

        if (thread_id < threads 
            && right < n) 
        {
            if (is_min)
                input[left] = std::min(input[right], input[left]);
            else input[left] = std::max(input[right], input[left]);
        }
        __syncthreads();
    }

    if (!thread_id)
    {
        output[block_id] = input[block_start];
    }
}

template <typename T>
__global__ void fill_kernel(T* data, T value, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

template <typename T>
__global__ void init_rng_kernel(curandState* states, T seed) {
    T idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

template <typename T>
__global__ void fill_random_kernel(T* out, curandState* states, T range, T size) {
    T idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int r = curand(&states[idx]) % range;
        out[idx] = r;
    }
}

template<typename IndexType> 
__global__ void modify_data_patch_offset_kernel(IndexType* indices, 
                                     IndexType last_offset, 
                                     IndexType stride, 
                                     IndexType n_row, 
                                     IndexType n_dim, 
                                     IndexType topk) 
{
    IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n_row || y >= n_dim) return;

    IndexType batch_cnt = n_dim / topk;
    IndexType batch_id = y / topk;

    IndexType id = x * n_dim + y;
    
    indices[id] += last_offset - (batch_cnt - batch_id - 1) * stride;
} 

template<typename ElementType, typename IndexType>
__global__ void shuffle_data_kernel(const ElementType* in, ElementType* out,
                                    IndexType n_queries, IndexType l)
{
    IndexType row = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n_queries && col < l) {
        ElementType lo = in[row * 2 * l + 2 * col];
        ElementType hi = in[row * 2 * l + 2 * col + 1];

        out[col * n_queries + row] = lo;
        out[(col + l) * n_queries + row] = hi;
    }
}

// add C = w1 * A + w2 * B with matrix A, B, C
template<typename ElementType, typename IndexType>
__global__ void matrix_weight_add_kernel(const ElementType *A, const ElementType *B, ElementType *C,
                                       IndexType n_row, IndexType n_dim, ElementType w1, ElementType w2) 
{
    IndexType x = blockDim.x * blockIdx.x + threadIdx.x;
    IndexType y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= n_row || y >= n_dim) return ;

    IndexType idx = x * n_dim + y;
    C[idx] = w1 * A[idx] + w2 * B[idx]; 
} 

// select a batch of idices of row from the input matrix
template<typename ElementType, typename IndexType>
__global__ void select_elements_kernel(const ElementType* input, const IndexType* indices, 
                                        ElementType *output, IndexType n_row_o, IndexType n_row_i, 
                                        IndexType n_dim_o, IndexType n_dim_i, ElementType invalid_value) 
{
    IndexType x = blockIdx.y * blockDim.y + threadIdx.y;
    IndexType y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= n_row_o || y >= n_dim_o) return ;
    IndexType idx = x;
    IndexType idy = indices[x * n_dim_o + y]; 

    
    IndexType o_idx = x * n_dim_o + y;
    if (idy < n_dim_i)
        output[o_idx] = input[n_dim_i * idx + idy];
    else output[o_idx] = invalid_value;
}

template<typename ElementType, typename IndexType>
__global__ void select_elements_row_kernel(const ElementType* input, const IndexType* indices, 
                                        ElementType *output, IndexType n_row_o, IndexType n_row_i, 
                                        IndexType n_dim_o, IndexType n_dim_i, ElementType invalid_value) 
{
    IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType y = blockIdx.y * blockDim.y + threadIdx.y;
    IndexType z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= n_row_o || y >= n_dim_o || z >= n_dim_i) return;
    IndexType idx = x;
    IndexType idy = indices[x * n_dim_o + y]; 

    IndexType o_idx = x * n_dim_i * n_dim_o + y * n_dim_i + z;
    if (idy < n_row_i) 
        output[o_idx] = input[n_dim_i * idy + z];
    else output[o_idx] = std::sqrt(invalid_value / static_cast<ElementType>(n_dim_i));
}

template <typename T, typename IndexType>
__global__ void transpose_kernel(const T* __restrict__ input, T* __restrict__ output,
                                  IndexType rows, IndexType cols) {
    // Thread and block indices
    IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType y = blockIdx.y * blockDim.y + threadIdx.y;

    // Transpose the matrix directly using global memory
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

template<typename ElementType, typename IndexType>
void shuffle_data(raft::device_matrix_view<ElementType, IndexType> in_mat, 
                  raft::device_matrix_view<ElementType, IndexType> out_mat) 
{
    IndexType n_queries = in_mat.extent(0);
    IndexType l = in_mat.extent(1) / 2;

    dim3 block_dim(32, 8); 
    dim3 grid_dim((n_queries + block_dim.x - 1) / block_dim.x,
                  (l + block_dim.y - 1) / block_dim.y);

    shuffle_data_kernel<<<grid_dim, block_dim>>>(
        in_mat.data_handle(),
        out_mat.data_handle(),
        n_queries,
        l
    );
}


//todo: benchmark these scans, since scan is a very common interface in super grid index
// matrix scan thrsut low performance version
template <typename InputType, typename OutputType, typename IndexType>
void matrix_scan_naive(raft::device_matrix_view<InputType, IndexType> const &in_matrix,
                 raft::device_matrix_view<OutputType, IndexType> &out_matrix) {
    IndexType rows = in_matrix.extent(0);
    IndexType cols = in_matrix.extent(1);

    // Ensure in_matrix and out_matrix have the same shape
    assert(in_matrix.extent(0) == out_matrix.extent(0));
    assert(in_matrix.extent(1) == out_matrix.extent(1));

    // Launch parallel scan for each row
    for (IndexType row = 0; row < rows; ++row) {
        InputType *in_row_ptr = in_matrix.data_handle() + row * cols;
        OutputType *out_row_ptr = out_matrix.data_handle() + row * cols;

        thrust::inclusive_scan(thrust::device, in_row_ptr, in_row_ptr + cols, out_row_ptr);
        
    }
}

template <typename InputType, typename OutputType, typename IndexType>
void matrix_scan_thrust(raft::device_matrix_view<InputType, IndexType> const& in_matrix,
                        raft::device_matrix_view<OutputType, IndexType>& out_matrix,
                        cudaStream_t stream = 0)
{
    IndexType rows = in_matrix.extent(0);
    IndexType cols = in_matrix.extent(1);

    RAFT_EXPECTS(rows > 0 && cols > 0, "Empty matrix");
    RAFT_EXPECTS(rows == out_matrix.extent(0) && cols == out_matrix.extent(1), "Shape mismatch");

    const InputType* d_input = in_matrix.data_handle();
    OutputType* d_output = out_matrix.data_handle();

    thrust::device_vector<IndexType> keys(rows * cols);
    thrust::transform(
        thrust::make_counting_iterator<IndexType>(0),
        thrust::make_counting_iterator<IndexType>(rows * cols),
        keys.begin(),
        [=] __device__ (IndexType i) {
            return i / cols; 
        });

    // inclusive scan by key
    thrust::inclusive_scan_by_key(
        thrust::cuda::par.on(stream),
        keys.begin(), keys.end(),   // keys
        d_input,                    // input values
        d_output);                  // output
}

template <typename InputType, typename OutputType, typename IndexType>
void matrix_scan_cub(raft::device_matrix_view<InputType, IndexType> const &in_matrix,
                     raft::device_matrix_view<OutputType, IndexType> &out_matrix,
                     bool is_inclusive = true, 
                     cudaStream_t stream = 0) {
    IndexType rows = in_matrix.extent(0);
    IndexType cols = in_matrix.extent(1);
    IndexType total = rows * cols;

    assert(in_matrix.extent(0) == out_matrix.extent(0));
    assert(in_matrix.extent(1) == out_matrix.extent(1));

    InputType* d_in = in_matrix.data_handle();
    OutputType* d_out = out_matrix.data_handle();

    IndexType* d_keys;
    cudaMalloc(&d_keys, sizeof(IndexType) * total);

    // fill keys with: [0, 0, 0, ..., n, n, n]
    thrust::counting_iterator<IndexType> row_indices(0);
    thrust::for_each_n(
        thrust::cuda::par.on(stream),
        row_indices,
        rows,
        [=] __device__ (IndexType row) {
            IndexType base = row * cols;
            for (IndexType i = 0; i < cols; ++i)
                d_keys[base + i] = row;
        });

    // query temperary needed by cub scan
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    
    if (is_inclusive){
        cub::DeviceScan::InclusiveSumByKey(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_in,
            d_out,
            total,
            cub::Equality(),
            stream);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceScan::InclusiveSumByKey(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_in,
            d_out,
            total,
            cub::Equality(),
            stream);
    } else {
        cub::DeviceScan::ExclusiveSumByKey(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_in,
            d_out,
            total,
            cub::Equality(),
            stream);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceScan::ExclusiveSumByKey(
            d_temp_storage,
            temp_storage_bytes,
            d_keys,
            d_in,
            d_out,
            total,
            cub::Equality(),
            stream);
    }
    
    cudaFree(d_temp_storage);   
    cudaFree(d_keys);
}

template<typename ElementType, typename IndexType>
void fill(ElementType* array,
          ElementType value, 
          IndexType size)
{
    fill_kernel<<<(size + 255) / 256, 256>>>(array, value, size);
}

template <typename m_t, typename idx_t = int>
void sliceMatrix(const m_t* in,
                 idx_t n_rows,
                 idx_t n_cols,
                 m_t* out,
                 idx_t x1,
                 idx_t y1,
                 idx_t x2,
                 idx_t y2,
                 bool row_major)
{
  auto lda = row_major ? n_cols : n_rows;
  dim3 block(256);
  dim3 grid(((x2 - x1) * (y2 - y1) + block.x - 1) / block.x);
  if (row_major)
    slice<<<grid, block>>>(in, lda, out, y1, x1, y2, x2);
  else
    slice<<<grid, block>>>(in, lda, out, x1, y1, x2, y2);
}

template <typename m_t, typename idx_t, typename layout_t>
void slice(raft::resources const& handle,
           raft::device_matrix_view<m_t, idx_t, layout_t> const& in,
           raft::device_matrix_view<m_t, idx_t, layout_t> out,
           slice_coordinates<idx_t> coords)
{
  // todo: add parafilter expects to find the expected dimension semantics
  sliceMatrix(in.data_handle(),
                      in.extent(0),
                      in.extent(1),
                      out.data_handle(),
                      coords.row1,
                      coords.col1,
                      coords.row2,
                      coords.col2,
                      true);
}

template<typename ElementType, typename IndexType>
inline void matrix_add_with_weights(raft::device_resources const& dev_resources,
                                    raft::device_matrix_view<ElementType, IndexType> const& A,
                                    raft::device_matrix_view<ElementType, IndexType> const& B,
                                    raft::device_matrix_view<ElementType, IndexType> C,
                                    ElementType w1, 
                                    ElementType w2) 
{
    IndexType n_row_a = A.extent(0);
    IndexType n_dim_a = A.extent(1);

    IndexType n_row_b = B.extent(0);
    IndexType n_dim_b = B.extent(1);

    assert(n_row_a == n_row_b && n_dim_a == n_dim_b);

    int block_dim_x = (n_row_a + block_size_x - 1) / block_size_x;
    int block_dim_y = (n_dim_a + block_size_y - 1) / block_size_y;

    dim3 full_blocks_per_grid(block_dim_x, block_dim_y);
    dim3 full_threads_per_block(block_size_x, block_size_y);

    matrix_weight_add_kernel<<<full_blocks_per_grid, full_threads_per_block>>>(A.data_handle(), B.data_handle(), C.data_handle(), 
        n_row_a, n_dim_a, w1, w2);
} 

// select elements of in an indices set from the input matrix and put it into the output matrix
template<typename ElementType, typename IndexType>
inline void select_elements(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<ElementType, IndexType> const& input_matrix,
                            raft::device_matrix_view<IndexType, IndexType> const& indices, 
                            raft::device_matrix_view<ElementType, IndexType> &output,
                            bool is_select_row = true, 
                            ElementType invalid_value = std::numeric_limits<ElementType>::max()) 
{
    IndexType n_row_o = indices.extent(0);
    IndexType n_row_i = input_matrix.extent(0);
    
    IndexType n_dim_i = input_matrix.extent(1);
    IndexType n_dim_o = std::min(indices.extent(1), output.extent(1)); 

    if (!is_select_row) {
        dim3 blocks(32, 8);
        dim3 grids((n_dim_o + blocks.x - 1) / blocks.x, (n_row_o + blocks.y - 1) / blocks.y);
        select_elements_kernel<<<grids, blocks>>>(input_matrix.data_handle(), 
                indices.data_handle(), output.data_handle(), n_row_o, n_row_i, n_dim_o, n_dim_i, invalid_value);
    }
    else {
        dim3 blocks(8, 8, 8);
        dim3 grids((n_row_o + blocks.x - 1) / blocks.x, (n_dim_o + blocks.y - 1) / blocks.y, (n_dim_i + blocks.z - 1) / blocks.z);
        select_elements_row_kernel<<<grids, blocks>>>(input_matrix.data_handle(), 
                indices.data_handle(), output.data_handle(), n_row_o, n_row_i, n_dim_o, n_dim_i, invalid_value);
    }

    checkCUDAErrorWithLine("select elements kernel failde:");
}

// ond dimensional version wrapper for matrix select elements
template<typename ElementType, typename IndexType>
inline void select_elements(raft::device_resources const& dev_resources,
                            raft::device_vector_view<ElementType, IndexType> const& input_vector,
                            raft::device_vector_view<IndexType, IndexType> const& indices_vector, 
                            raft::device_vector_view<ElementType, IndexType> &output_vector,
                            ElementType invalid_value = std::numeric_limits<ElementType>::max()
                            ) 
{
    auto input_matrix = raft::make_device_matrix_view<ElementType, IndexType>(input_vector.data_handle(), 1, input_vector.extent(0));
    auto indices = raft::make_device_matrix_view<ElementType, IndexType>(indices_vector.data_handle(), 1, indices_vector.extent(0));
    auto output = raft::make_device_matrix_view<ElementType, IndexType>(output_vector.data_handle(), 1, output_vector.extent(0));

    select_elements(dev_resources, input_matrix, indices, output, false, invalid_value);
}

template<typename ElementType, typename IndexType>
inline void select_elements(raft::device_resources const& dev_resources,
                            raft::device_vector_view<ElementType, IndexType> const& input_vector,
                            raft::device_matrix_view<IndexType, IndexType> const& indices, 
                            raft::device_matrix_view<ElementType, IndexType> &output,
                            ElementType invalid_value = std::numeric_limits<ElementType>::max()
                            ) 
{
    auto input_matrix = raft::make_device_matrix_view<ElementType, IndexType>(input_vector.data_handle(), input_vector.extent(0), 1);

    select_elements(dev_resources, input_matrix, indices, output, true, invalid_value);
}



template <typename T, typename IndexType>
void transpose_matrix(const T* input, T* output, IndexType rows, IndexType cols) {
    // Define block and grid sizes
    if (cols == 1)  {
        cudaMemcpy(output, input, rows * cols * sizeof(T), cudaMemcpyDeviceToDevice);
        return ;
    }
    dim3 blockDim(1, 1024);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    transpose_kernel<T, IndexType><<<gridDim, blockDim>>>(input, output, rows, cols);

    // Synchronize the stream (optional, for error checking)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

template <typename ElementType, typename IndexType>
inline ElementType array_min_max_reduce(ElementType* in_array_device,
    IndexType n_elements,
    bool is_min = true) {
    // Constants
    constexpr IndexType block_size = 512; // Adjust based on GPU architecture
    IndexType threads_cnt = n_elements;

    // Compute initial number of blocks
    IndexType block_cnt = (threads_cnt + 2 * block_size - 1) / (2 * block_size);
    IndexType remaining = n_elements;

    // Allocate temporary memory on the device
    ElementType* sums_device = nullptr;
    cudaMalloc(&sums_device, block_cnt * sizeof(ElementType));
    ElementType* in_array_temp = nullptr;
    cudaMalloc(&in_array_temp, n_elements * sizeof(ElementType));

    // Copy the input array to temporary memory
    cudaMemcpy(in_array_temp, in_array_device, n_elements * sizeof(ElementType), cudaMemcpyDeviceToDevice);

    ElementType final_result;

    // Iterative reduction
    while (remaining > 1) {
        // Launch reduction kernel
        min_max_reduce_kernel << <block_cnt, block_size >> > (in_array_temp, sums_device, remaining, is_min);
        cudaDeviceSynchronize();

        // Update remaining elements and block count
        remaining = block_cnt;
        block_cnt = (remaining + 2 * block_size - 1) / (2 * block_size);

        // Swap input and output arrays for the next iteration if needed
        if (remaining > 1) {
            std::swap(in_array_temp, sums_device);
        }
    }

    // Copy the final result back to the host
    cudaMemcpy(&final_result, sums_device, sizeof(ElementType), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(sums_device);
    cudaFree(in_array_temp);

    return final_result;
}

template <typename ElementType, typename IndexType>
void matrix_reduce_sum(ElementType* d_in, ElementType* d_out, IndexType R, IndexType C)
{
    thrust::device_vector<IndexType> d_offsets(R+1);
    thrust::sequence(d_offsets.begin(), d_offsets.end(), (IndexType)0, C);
    
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_bytes,
        d_in,
        d_out,
        R,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_offsets.data()) + 1
    );
    cudaMalloc(&d_temp, temp_bytes);

    cub::DeviceSegmentedReduce::Sum(
        d_temp, temp_bytes,
        d_in,
        d_out,
        R,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_offsets.data()) + 1
    );

    cudaFree(d_temp);
}

template<typename InputType, 
         typename BinaryOp>
InputType device_reduce(
    InputType* d_A,
    size_t N,
    InputType init,
    BinaryOp op)
{
    thrust::device_ptr<InputType> ptr_A(d_A);
    return thrust::reduce(
        thrust::device, 
        ptr_A,          
        ptr_A + N,      
        init,           
        op            
    );
}


template <typename KeyType, typename CountsType>
size_t deduplicate_sorted_keys(
    const KeyType* d_keys_in,           
    size_t num_keys,                     
    KeyType* d_unique_keys_out,          
    CountsType* d_counts_out,                  
    thrust::equal_to<KeyType> eq = thrust::equal_to<KeyType>()  
) 
{
    auto ones = thrust::make_constant_iterator(1);
    
    thrust::device_ptr<const KeyType> keys_in_ptr(d_keys_in);
    thrust::device_ptr<KeyType> unique_out_ptr(d_unique_keys_out);
    thrust::device_ptr<CountsType> counts_out_ptr(d_counts_out);
    
    auto new_end = thrust::reduce_by_key(
        keys_in_ptr,            
        keys_in_ptr + num_keys, 
        ones,                  
        unique_out_ptr,      
        counts_out_ptr,       
        eq                    
    );
    size_t unique_count = new_end.first - unique_out_ptr;
    
    return unique_count;
}

template <typename T, typename Functor>
__global__ void matrix_map_kernel(
    T* matrx,   
    int n_row, 
    int n_dim,  
    Functor functor 
)
{
    
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row_idx >= n_row || col_idx >= n_dim) return;

    size_t linear_index = static_cast<size_t>(row_idx) * n_dim + static_cast<size_t>(col_idx);
    matrx[linear_index] = functor(row_idx, col_idx);
}

template<typename ElementType, typename IndexType, typename Functor>
void matrix_map_matrix(
    raft::device_matrix_view<ElementType, IndexType> matrix,
    Functor functor
)
{
    IndexType n_rows = matrix.extent(0);
    IndexType n_cols = matrix.extent(1);  

    dim3 block_dim(8, 64); 
    
    dim3 grid_dim(
        (n_rows + block_dim.x - 1) / block_dim.x,
        (n_cols + block_dim.y - 1) / block_dim.y
    );

    sequence_matrix_kernel<<<grid_dim, block_dim>>>(
        matrix.data_handle(), 
        n_rows, 
        n_cols, 
        functor
    );
}

template<typename T>
__device__ __host__ inline T invalid_index_value() {
    if constexpr (std::is_pointer<T>::value) {
        return nullptr;
    } else {
        return std::numeric_limits<T>::max();
    }
}

template<typename OffsetType, typename SizeType>
__global__ static void build_index_table_kernel(
        const OffsetType* __restrict__ offsets,
        const SizeType* __restrict__ prefixes,   // exclusive scan of lengths
        const SizeType* __restrict__ row_lenthes,
        OffsetType* __restrict__ index_table,
        uint64_t prefix_len, uint64_t queries, uint64_t tot_len) 
{
    uint64_t tid = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t q = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= tot_len || q >= queries) return;

    // binary search to find query id for this tid
    uint64_t l = 0, r = prefix_len;
    const SizeType* prefix = &prefixes[q * prefix_len]; 
    const OffsetType* offset = &offsets[q * prefix_len];
    int row_len = row_lenthes[q];
    if (tid >= row_len) {
        index_table[tot_len * q + tid] = invalid_index_value<OffsetType>();
        return ;
    }
    // todo: use a inline device function
    while (l < r) {
        int m = (l + r) / 2;
        if (prefix[m + 1] <= tid)
            l = m + 1;
        else
            r = m;
    }

    l = min(l, prefix_len - 1);
    if constexpr (std::is_pointer<OffsetType>::value) {
        if (offset[l] == nullptr) {
            index_table[tot_len * q + tid] = nullptr;
            return;
        }
    }
    int index_in_query = tid - prefix[l];
    index_table[tot_len * q + tid] = offset[l] + index_in_query;
}

template <typename T>
__global__ void rowwise_add_last_elements(
    const T* __restrict__ input,  // lengthes
    const T* __restrict__ scan,   // length_sums
    T* __restrict__ output,       // output (can be same as scan)
    uint64_t rows, uint64_t cols)
{
    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    T last_exclusive = scan[row * cols + cols - 1];
    T last_value = input[row * cols + cols - 1];
    output[row] = last_exclusive + last_value;
}

template<typename OffsetType, typename SizeType>
void calc_row_lengthes(
    raft::device_matrix_view<SizeType, uint64_t> const& lengthes, 
    raft::device_matrix_view<SizeType, uint64_t> const& length_sums, 
    raft::device_vector_view<OffsetType, uint64_t> &row_lengthes
) 
{
    uint64_t n_queries = lengthes.extent(0);
    uint64_t seg_counts = lengthes.extent(1);
    
    dim3 row_blocks(64);
    dim3 row_grids((n_queries + row_blocks.x - 1) / row_blocks.x);
    rowwise_add_last_elements<<<row_grids, row_blocks>>>(lengthes.data_handle(), length_sums.data_handle(), 
            row_lengthes.data_handle(), n_queries, seg_counts);
}

template<typename OffsetType, typename SizeType>
void build_offset_table(
    raft::device_matrix_view<SizeType, uint64_t> const& lengthes, 
    raft::device_matrix_view<SizeType, uint64_t> const& length_sums, 
    raft::device_matrix_view<OffsetType, uint64_t> const& starts,
    raft::device_matrix_view<OffsetType, uint64_t> &offsets_table, 
    std::optional<raft::device_vector_view<OffsetType, uint64_t>> row_lengthes_optional = std::nullopt
) 
{
    uint64_t n_queries = lengthes.extent(0);
    uint64_t seg_counts = lengthes.extent(1);

    raft::device_vector_view<SizeType, uint64_t> row_lengthes{};
    if (!row_lengthes_optional.has_value()) { 
        row_lengthes = parafilter_mmr::make_device_vector_view<SizeType, uint64_t>(n_queries);
        calc_row_lengthes(lengthes, length_sums, row_lengthes);
    }
    else row_lengthes = *row_lengthes_optional;
    cudaDeviceSynchronize();
    auto table_len = thrust::reduce(
        thrust::device_pointer_cast(row_lengthes.data_handle()), 
        thrust::device_pointer_cast(row_lengthes.data_handle()) + n_queries,
        std::numeric_limits<SizeType>::min(),
        thrust::maximum<SizeType>() 
    );
    offsets_table = parafilter_mmr::make_device_matrix_view<OffsetType, uint64_t>(n_queries, table_len);

    dim3 blocks(4, 64);
    dim3 grids((n_queries + blocks.x - 1) / blocks.x, (table_len + blocks.y - 1) / blocks.y);
    build_index_table_kernel<<<grids, blocks>>>(starts.data_handle(), length_sums.data_handle(), row_lengthes.data_handle(), offsets_table.data_handle(), seg_counts, n_queries, table_len);
}

template<typename SrcType, typename OffsetType, typename SizeType>
void collect_segments_data(
    raft::device_resources const& dev_resources, 
    raft::device_matrix_view<SizeType, uint64_t> const& lengthes, 
    raft::device_matrix_view<OffsetType, uint64_t> const& starts, 
    raft::device_matrix_view<SrcType, uint64_t> const& src_data, 
    raft::device_matrix_view<SrcType, uint64_t> &dst_data,
    SrcType invalid_value = std::numeric_limits<SrcType>::max(), 
    std::optional<raft::device_matrix_view<OffsetType, uint64_t>> index_table_optional = std::nullopt
)
{
    uint64_t n_queries = lengthes.extent(0);
    uint64_t seg_counts = lengthes.extent(1);
    raft::device_matrix_view<OffsetType, uint64_t> index_table{};
    raft::device_matrix_view<SizeType, uint64_t> length_sums{};

    if (index_table_optional.has_value()) {
        index_table = *index_table_optional;
    } else {
        length_sums = parafilter_mmr::make_device_matrix_view<SizeType, uint64_t>(n_queries, seg_counts);
        matrix_scan_cub(lengthes, length_sums, false);
        build_offset_table(lengthes, length_sums, starts, index_table); 
    }

    uint64_t table_len = index_table.extent(1);
    dst_data = parafilter_mmr::make_device_matrix_view<SrcType, uint64_t>(n_queries, table_len);
    select_elements<SrcType, uint64_t>(dev_resources, src_data, index_table, dst_data, false, invalid_value);
}

template<typename InputType,  
         typename OutputType, 
         typename BinaryOp>
void custom_device_transform(
    InputType* d_A,
    InputType* d_B,
    OutputType* d_C,
    size_t N,
    BinaryOp op)
{
    thrust::device_ptr<InputType> ptr_A(d_A);
    thrust::device_ptr<InputType> ptr_B(d_B);
    thrust::device_ptr<OutputType> ptr_C(d_C);

    thrust::transform(
        thrust::device, 
        ptr_A,          
        ptr_A + N,      
        ptr_B,          
        ptr_C,          
        op             
    );
}
