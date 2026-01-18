#include "core/zorder.cuh"

__device__ uint64_t morton_encode(const int* coords, MortonConfig_t config) 
{
    uint64_t code = 0;
    for (int b = 0; b < config.max_bits; ++b) {
#pragma unroll 
        for (int i = 0; i < config.dims; ++i) {
            if (!((config.valid_map >> i) & 1)) continue;
            int bit = (coords[i] >> (config.max_bits - 1 - b)) & 1;
            code = (code << 1) | bit;
        }
    }
    return code;
}

__device__ void morton_decode(uint64_t code, int* coords, MortonConfig_t config) 
{
    for (int i = 0; i < config.dims; ++i) coords[i] = 0;

    int active_dims = __popc(config.valid_map);

    for (int b = 0; b < config.max_bits; ++b) {
        int j = 0; // index in active dims
#pragma unroll
        for (int i = 0; i < config.dims; ++i) {
            if (!((config.valid_map >> i) & 1)) continue;
            int shift = (config.max_bits - 1 - b) * active_dims + (active_dims - 1 - j);
            int bit = (code >> shift) & 1;
            coords[i] |= bit << (config.max_bits - 1 - b);
            j++;
        }
    }
}

__global__ void encode_points_kernel(const int* __restrict__ points, uint64_t* __restrict__ codes, int num_points, MortonConfig_t config) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    int coords[MAX_DIMS];
    for (int i = 0; i < config.dims; ++i) coords[i] = points[idx * config.dims + i];
    codes[idx] = morton_encode(coords, config);
}

__device__ bool point_in_box(const int* coord, const int* box, int dims) 
{
    for (int i = 0; i < dims; ++i) {
        if (coord[i] < box[2 * i] || coord[i] > box[2 * i + 1]) return false;
    }
    return true;
}

void encode_points(
    raft::device_matrix_view<int, uint64_t> const &points,
    raft::device_vector_view<uint64_t, uint64_t>& zcodes, 
    MortonConfig_t config
) 
{
    uint64_t n_points = points.extent(0);
    dim3 blocks(128);
    dim3 grids((n_points + blocks.x - 1) / blocks.x);

    encode_points_kernel<<<grids, blocks>>>(points.data_handle(), zcodes.data_handle(), n_points, config);
    checkCUDAError("encode points failed");
}

__device__ __forceinline__ int leading_step_diff(uint32_t a, uint32_t b, int max_bits) 
{
    return __clz((a ^ b) << (32 - max_bits));
}

__device__ uint64_t set_low_k_bits(uint64_t target, uint64_t source, int k) 
{
    uint64_t clear_mask = ~((1ULL << k) - 1);
    uint64_t cleared_target = target & clear_mask;
    
    uint64_t extract_mask = (1ULL << k) - 1;
    uint64_t extracted_source = source & extract_mask;
    
    return cleared_target | extracted_source;
}

__device__ bool getNextZValueBaseline(uint64_t cur, const int* box, uint64_t& next, MortonConfig_t config)
{

    int dims = config.dims;
    int max_bits = config.max_bits;
    uint64_t nisp = cur + 1;

    int coords[MAX_DIMS];
    morton_decode(nisp, coords, config);

    // Quick accept
    bool in = true;
    if (!point_in_box(coords, box, dims)) {
        in = false;
    }
    if (in) {
        next = nisp;
        return true;
    }

    // Prepare bit comparison context
    int flag[MAX_DIMS] = { 0 };              // -1: left, +1: right
    int outStep[MAX_DIMS];                     // first violating step
    int saveMin[MAX_DIMS];                     // first step can safely raise
    int saveMax[MAX_DIMS];                     // first step can safely lower
    int j = 0;
    int active_dims = __popc(config.valid_map);
    int valid_bits[MAX_DIMS];

#pragma unroll
    for (int i = 0; i < dims; ++i) {
        if (!((config.valid_map >> i) & 1)) continue;
        valid_bits[j++] = i;
        outStep[i] = max_bits;
        saveMin[i] = max_bits;
        saveMax[i] = max_bits;

        for (int s = 0; s < max_bits; ++s) {
            int shift = max_bits - 1 - s;
            int ql_bit = (box[2 * i] >> shift) & 1;
            int qh_bit = (box[2 * i + 1] >> shift) & 1;
            int c_bit = (coords[i] >> shift) & 1;

            // Detect left violation
            if (flag[i] == 0 && c_bit < ql_bit) {
                flag[i] = -1;
                outStep[i] = s;
                break;
            }
            if (c_bit > ql_bit) break;
        }

        if (flag[i] == 0) {
            // Detect right violation
            for (int s = 0; s < max_bits; ++s) {
                int shift = max_bits - 1 - s;
                int ql_bit = (box[2 * i] >> shift) & 1;
                int qh_bit = (box[2 * i + 1] >> shift) & 1;
                int c_bit = (coords[i] >> shift) & 1;

                if (c_bit > qh_bit) {
                    flag[i] = +1;
                    outStep[i] = s;
                    break;
                }
                if (c_bit < qh_bit) break;
            }
        }

        // saveMin (only if not left violating)
        if (flag[i] != -1) {
            for (int s = 0; s < max_bits; ++s) {
                int shift = max_bits - 1 - s;
                int ql_bit = (box[2 * i] >> shift) & 1;
                int c_bit = (coords[i] >> shift) & 1;
                if (c_bit > ql_bit) {
                    saveMin[i] = s;
                    break;
                }
            }
        }

        // saveMax (only if not right violating)
        if (flag[i] != +1) {
            for (int s = 0; s < max_bits; ++s) {
                int shift = max_bits - 1 - s;
                int qh_bit = (box[2 * i + 1] >> shift) & 1;
                int c_bit = (coords[i] >> shift) & 1;
                if (c_bit < qh_bit) {
                    saveMax[i] = s;
                    break;
                }
            }
        }
    }

    // Find first violating bit position
    int changeBP = 0;
    j = 0;
#pragma unroll
    for (int i = 0; i < dims; ++i) {
        if (!((config.valid_map >> i) & 1)) continue;
        if (outStep[i] < max_bits) {
            int bp = (max_bits - 1 - outStep[i]) * active_dims + (active_dims - 1 - j);
            if (bp > changeBP) {
                changeBP = bp;
            }
        }
        j++;
    }

    int d = active_dims - 1 - (changeBP % active_dims);
    d = valid_bits[d];
    if (flag[d] == +1) {
        // Try to find safe 0 → 1 bit
        int found = -1;
        for (int bp = changeBP + 1; bp < max_bits * active_dims; ++bp) {
            int dim = active_dims - 1 - (bp % active_dims);
            int t_dim = valid_bits[dim];
            if (bp <= (max_bits - 1 - saveMax[t_dim]) * active_dims + (active_dims - 1 - dim)) {
                if (((nisp >> bp) & 1ULL) == 0) {
                    found = bp;
                    break;
                }
            }
        }
        if (found == -1) return false;
        changeBP = found;
        d = active_dims - 1 - (changeBP % active_dims);
        d = valid_bits[d];
        saveMin[d] = max_bits - 1 - (changeBP / active_dims);
        flag[d] = 0;
    }


    // Fix lower bits
    int blocks = changeBP / active_dims;
    j = 0;
    // Fix lower bits
#pragma unroll
    for (int i = 0; i < dims; ++i) {
        if (!((config.valid_map >> i) & 1)) continue;
        if (flag[i] >= 0) {
            int bits = blocks;
            if (valid_bits[j] > d) bits++;
            if (changeBP <= (max_bits - 1 - saveMin[i]) * active_dims + (active_dims - 1 - j)) {
                coords[i] = set_low_k_bits(coords[i], 0, bits);
            }
            else {
                coords[i] = set_low_k_bits(coords[i], box[2 * i], bits);
            }
        }
        else {
            coords[i] = box[2 * i];
        }
        j++;
    }

    next = morton_encode(coords, config);
    next |= (1ULL << changeBP);
    return true;
}

__device__ static bool getNextZValueHPerf(uint64_t cur, const int* box, uint64_t& next, MortonConfig_t config)
{
    uint64_t nisp = cur + 1;
    next = nisp;
    int dims = config.dims;
    int max_bits = config.max_bits;
    int res = true;

    int coords[MAX_DIMS];
    morton_decode(nisp, coords, config);
    if (point_in_box(coords, box, dims)) return res;

    // Prepare bit comparison context
    int flag[MAX_DIMS] = { 0 };              // -1: left, +1: right
    int outStep[MAX_DIMS];                     // first violating step
    int saveMin[MAX_DIMS];                     // first step can safely raise
    int saveMax[MAX_DIMS];                     // first step can safely lower

#pragma unroll
    for (int i = 0; i < dims; i++) {
        outStep[i] = max_bits;
        saveMin[i] = max_bits;
        saveMax[i] = max_bits;

        int ql = box[2 * i];
        int qh = box[2 * i + 1];
        int coord = coords[i];

        int first_diff = leading_step_diff(ql, coord, max_bits);         // Leading zeros
        int is_left = (coord < ql);           // test for left broder
        flag[i] = is_left * -1;               // -1 or 0
        outStep[i] = is_left ? first_diff : max_bits;
        
        int bit_r = leading_step_diff(qh, coord, max_bits);
        int gt_r = (coord > qh);  
        flag[i] = (flag[i] == 0) ? (gt_r * +1) : flag[i];  // test right broder only if flag is zero
        outStep[i] = (flag[i] == +1) ? bit_r : outStep[i];

        int bit_m = leading_step_diff(ql, coord, max_bits);
        int gt_m = (coord > ql);
        saveMin[i] = ((flag[i] != -1) & gt_m) * bit_m + (((flag[i] != -1) & !gt_m) | flag[i] == -1) * saveMin[i];

        int bit_x = leading_step_diff(ql, coord, max_bits);
        int lt_x = (coord < qh);
        saveMax[i] = ((flag[i] != +1) & lt_x) * bit_x + (((flag[i] != +1) & !lt_x) | flag[i] == +1) * saveMax[i];
    }

    int changeBP = max_bits * dims;
#pragma unroll
    for (int i = 0; i < dims; ++i) {
        int is_valid = (outStep[i] < max_bits);
        int bp = (max_bits - 1 - outStep[i]) * dims + (dims - 1 - i);
        int masked_bp = is_valid * bp + (1 - is_valid) * (max_bits * dims); 

        changeBP = max(changeBP, masked_bp);
    }

    int d = dims - 1 - (changeBP % dims);
    int found_bp = dims * max_bits;
    int cd = d;

#pragma unroll
    for (int i = 0; i < dims; ++i) {
        int val = coords[i];
        int s_min = saveMax[i];

        uint32_t bits = (~val) & ((1u << max_bits) - 1); // ensure max bits
        int step_d = max_bits - 1 - (changeBP / dims); // step for changeBP

        int cut = (i < d) ? step_d + 1 : step_d;     // clear step > cut
        uint32_t cut_mask = ~((1u << (max_bits - cut)) - 1);  // clear step < cut

        // 3. mask for step > s_min
        uint32_t save_mask = (1u << (max_bits - s_min + 1)) - 1;

        // 4. merge 2 masks
        uint32_t valid_mask = bits & cut_mask & save_mask;

        // 5. find the first none-zero bit
        int s = __ffs(valid_mask);
        int valid = (s != 0);  // bool → 0 or 1
        int bp_candidate = (s - 1) * dims + (dims - 1 - i);
        int bp = valid ? bp_candidate : max_bits * dims;

        uint32_t updated_mask = -(uint32_t)(bp < found_bp); 
        found_bp = (updated_mask & bp) | (~updated_mask & found_bp);
        cd = (updated_mask & i) | (~updated_mask & cd);
    }

    res = res & (found_bp != max_bits * dims);
    int is_right_violation = (flag[d] == +1);

    d = is_right_violation * cd + (1 - is_right_violation) * d;
    changeBP = is_right_violation * found_bp + (1 - is_right_violation) * changeBP;

    int new_saveMin = max_bits - 1 - (changeBP / dims);
    saveMin[d] = is_right_violation * new_saveMin + (1 - is_right_violation) * saveMin[d];
    flag[d] = (1 - is_right_violation) * flag[d];
    nisp |= 1 << changeBP;

    int blocks = changeBP / dims;
    // Fix lower bits
#pragma unroll
    for (int i = 0; i < dims; ++i) {
        if (flag[i] >= 0) {
            int bits = blocks;
            if (i > d) bits--;
            if (changeBP < (max_bits - 1 - saveMin[i]) * dims + (dims - 1 - i)) {
                coords[i] = set_low_k_bits(coords[i], 0, bits);
            }
            else {
                coords[i] = set_low_k_bits(coords[i], box[2 * i], bits);
            }
        }
        else {
            coords[i] = box[2 * i];
        }
    }

    next = morton_encode(coords, config);
    next |= (1ULL << changeBP);
}

__device__ int lower_bound_device(const uint64_t* data, int le, int ri, uint64_t key) 
{
    int left = le, right = ri;
    while (left < right) {
        int mid = (left + right) / 2;
        if (data[mid] < key)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

__device__ uint64_t slide_out(uint64_t zc, int dim, const int* upper_bound, MortonConfig_t config)
{
    int coords[MAX_DIMS];
    morton_decode(zc, coords, config);

    if (coords[dim] > upper_bound[dim]) {
        return 0xFFFFFFFFFFFFFFFFull;
    }

    coords[dim] = upper_bound[dim] + 1;
    return morton_encode(coords, config);
}

__device__ uint64_t jump_out_gpu(
    uint64_t zc, uint64_t ze,
    const int* box,   // {lo, hi} per dim
    MortonConfig_t config
) 
{
    int dims = config.dims;
    int max_bits = config.max_bits;
    uint64_t z_min = ze;

    // Step 1: slide_out over dims
    int box_hi[MAX_DIMS];
#pragma unroll
    for (int d = 0; d < dims; ++d)
        box_hi[d] = box[2 * d + 1];

#pragma unroll
    for (int d = 0; d < dims; ++d) {
        uint64_t zout = slide_out(zc, d, box_hi, config);
        z_min = zout < z_min ? zout : z_min;
    }

    // Step 2: decode once zc → coords[]
    int coords[MAX_DIMS] = { 0 };
#pragma unroll
    for (int b = 0; b < max_bits; ++b) {
#pragma unroll
        for (int d = 0; d < dims; ++d) {
            int bit = (zc >> ((max_bits - 1 - b) * dims + (dims - 1 - d))) & 1;
            coords[d] |= (bit << (max_bits - 1 - b));
        }
    }

    // Step 3: simulate jump_bit_layer with O(1) update
    int coords_jump[MAX_DIMS];  

    for (int b = 0; b < dims * max_bits; ++b) {
        if (((zc >> b) & 1) != 0) continue;

        uint64_t mask = (1ULL << b);
        uint64_t zout = (zc | mask) & ~(mask - 1);

        if (zout >= z_min) break;

        // get dim of current morton bit
        int dim = dims - 1 - (b % dims);
        int step = max_bits - 1 - (b / dims);

        // update remaining dimernsion
#pragma unroll
        for (int d = 0; d < dims; ++d) {
            int mask0 = ~((1 << step + 1) - 1);
            int mask1 = ~((1 << (step)) - 1);
            int use_mask1 = -(d < dim); // all 1 if true, else 0
            int mask = (mask1 & use_mask1) | (mask0 & ~use_mask1);

            int cleared = coords[d] & mask;
            coords_jump[d] = (d == dim) ? (cleared | (1 << step)) : cleared;
        }

        // check whether it is out-of box
        int is_out = false;
#pragma unroll
        for (int d = 0; d < dims; ++d) {
            is_out |= (coords_jump[d] < box[2 * d] || coords_jump[d] > box[2 * d + 1]);
        }

        z_min = is_out * zout + (1 - is_out) * z_min;
    }

    return z_min;
}

struct statistic {
    int is_valid;
    int start;
    int count;
};

__global__ void zorder_range_query_kernel(
    const uint64_t* __restrict__ z_array_sorted,  // sorted zorder array
    const uint64_t* __restrict__ zcodes_starts, 
    const uint64_t* __restrict__ zcodes_lengthes,
    const uint64_t* __restrict__ candidates, 
    const uint64_t* __restrict__ query_end_points,  
    int n_candies, int n_splits, 
    const uint64_t* __restrict__ query_ranges,
    const int* __restrict__ query_boxes,
    int num_queries, int tot_clusters, 
    uint64_t* __restrict__ out_start, 
    uint64_t* __restrict__ out_len,
    MortonConfig_t config,
    statistic* seg_stats = nullptr, 
    int jump_count = 3 
    
) 
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.z * blockDim.z + threadIdx.z;
    
    int dims = config.dims;
    if (q >= num_queries || s >= n_splits || c >= n_candies) return;
    uint64_t linear_id = q * n_candies * n_splits + c * n_splits + s;

    int candi_cluster = candidates[c];
    if (candi_cluster >= tot_clusters) {
        out_start[linear_id] = std::numeric_limits<uint64_t>::max();
        out_len[linear_id] = 0;
        return ;
    }

    uint64_t len = zcodes_lengthes[candi_cluster];
    uint64_t start = zcodes_starts[candi_cluster];

    int lower = query_end_points[2 * linear_id];
    int lower_backup = lower;
    int upper = query_end_points[2 * linear_id + 1];

    if (lower == upper) {
        out_start[linear_id] = std::numeric_limits<uint64_t>::max();
        out_len[linear_id] = 0;

        return ;
    }

    const int* box = &query_boxes[q * 2 * dims];
    uint64_t zmin = z_array_sorted[lower];
    uint64_t zmax = z_array_sorted[upper - 1];

    uint64_t jump_value[5];

    int written = 0;
    uint64_t cur = zmin;
    uint64_t visited = 0;
    uint64_t valid = 0;
    int cur_count = 0;
    int is_valid = 1;

    while (cur_count < jump_count) {
        int idx = lower_bound_device(z_array_sorted, lower, upper, cur);
        if (idx >= upper) break;
        lower = idx;

        uint64_t zval = z_array_sorted[idx];
        int coord[MAX_DIMS];
        morton_decode(zval, coord, config);

        visited++;
        if (!point_in_box(coord, box, dims)) {
            uint64_t next_z = 0;
            bool valid = getNextZValueBaseline(zval, box, next_z, config);
            jump_value[cur_count] = next_z;
            if (!valid || next_z >= zmax) {
                is_valid = 0;
                cur_count++;
                break;
            }
            cur_count++;
            cur = next_z;
            continue;
        }

        valid++;
        cur_count++;
        break;
    }

    out_start[linear_id] =  lower;
;
    if (is_valid) {
        out_len[linear_id] = upper - lower;
    }
    else {
        out_len[linear_id] = 0;
    }

    if (seg_stats != nullptr) {
        int count = 0;
        int start = 0x3f3f3f3f;
        int coord[MAX_DIMS];
        for (int i = lower_backup; i < upper; i++) {
            cur = z_array_sorted[i];
            morton_decode(cur, coord, config);
            if (point_in_box(coord, box, dims)) {
                count++;
                if (start > i) start = i;
            }
        }
        seg_stats[linear_id].start = start;
        seg_stats[linear_id].count = count;
        seg_stats[linear_id].is_valid = is_valid;

        if (count > 0 && !is_valid) {
            printf("Jump failed find invalid value ! with start %lld, end %lld, Jump Vaue %lld, %lld, %lld, Jump count %d, box: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", 
                zmin, zmax, jump_value[0], jump_value[1], jump_value[2], cur_count, box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], box[8], box[9], box[10], box[11]);
        }
    }
    
}

__global__ static void get_query_end_points_kernel(
    const uint64_t* sorted_zcode_array, const uint64_t* zcodes_array_offsets, const uint64_t* zcodes_array_lengthes, 
    const uint64_t* query_ranges, const uint64_t* candi_arrays, uint64_t* out_end_points, uint64_t* out_counts, int n_queries, int n_candies, int n_splits, 
    int tot_clusters
)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.z * blockDim.z + threadIdx.z;

    if (q >= n_queries || c >= n_candies || s >= n_splits) return;

    uint64_t linear_id = q * n_candies * n_splits + c * n_splits + s;

    uint64_t zmin = query_ranges[q * 2];
    uint64_t zmax = query_ranges[q * 2 + 1];

    uint64_t candi_id = candi_arrays[q * n_candies + c];
    if (candi_id >= tot_clusters) {
        out_end_points[2 * linear_id] = std::numeric_limits<uint64_t>::max();
        out_end_points[2 * linear_id + 1] = std::numeric_limits<uint64_t>::max();
        return;
    } 
    
    uint64_t start = zcodes_array_offsets[candi_id];
    uint64_t size = zcodes_array_lengthes[candi_id];

    if (size == 0) {
        out_end_points[2 * linear_id] = std::numeric_limits<int>::max();
        out_end_points[2 * linear_id + 1] = std::numeric_limits<int>::max();
        out_counts[linear_id] = 0;
        return ;
    }

    int lower = lower_bound_device(sorted_zcode_array + start, 0, size, zmin);
    int upper = lower_bound_device(sorted_zcode_array + start, 0, size, zmax + 1);
    assert(lower < size && upper <= size);

    uint64_t len = upper - lower;
    n_splits = min(len, (uint64_t)n_splits);
    
    uint64_t _q = len / n_splits;
    uint64_t r = len % n_splits;
    if (s < n_splits) {
        lower += s * _q + min((uint64_t)s, r);
        upper = lower + _q + (s < r ? 1 : 0);
    }
    else {
        out_end_points[2 * linear_id] = std::numeric_limits<int>::max();
        out_end_points[2 * linear_id + 1] = std::numeric_limits<int>::max();
        out_counts[linear_id] = 0;
        return ;
    }

    out_end_points[2 * linear_id] = lower + start;
    out_end_points[2 * linear_id + 1] = upper + start;

    out_counts[linear_id] = upper - lower;
}

__global__ static void get_query_end_points_no_split_kernel(
    const uint64_t* sorted_zcode_array, const uint64_t* zcodes_array_offsets, const uint64_t* zcodes_array_lengthes, 
    const uint64_t* query_ranges, const uint64_t* candi_arrays, uint64_t* out_end_points, uint64_t* out_counts, int n_queries, int n_candies,
    int tot_clusters
) 
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= n_queries || c >= n_candies) return;

    uint64_t zmin = query_ranges[q * 2];
    uint64_t zmax = query_ranges[q * 2 + 1];

    uint64_t candi_id = candi_arrays[q * n_candies + c];
    uint64_t linear_id = q * n_candies + c;
    if (candi_id >= tot_clusters) {
        out_end_points[linear_id] = std::numeric_limits<uint64_t>::max();
        out_counts[linear_id] = 0;
        return;
    } 
    uint64_t start = zcodes_array_offsets[candi_id];
    uint64_t size = zcodes_array_lengthes[candi_id]; 

    int lower = lower_bound_device(sorted_zcode_array + start, 0, size, zmin);
    int upper = lower_bound_device(sorted_zcode_array + start, 0, size, zmax + 1);

    out_end_points[linear_id] = lower + start;

    out_counts[linear_id] = upper - lower;
}

static void get_query_end_points(
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array, // A second level zcode array splited by offsets and lengthes array 
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array_offsets, 
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array_lengthes, 
    raft::device_matrix_view<uint64_t, uint64_t> const& query_ranges, 
    raft::device_matrix_view<uint64_t, uint64_t> const& candi_arrays,
    raft::device_matrix_view<uint64_t, uint64_t>& out_end_points,   
    raft::device_matrix_view<uint64_t, uint64_t>& out_counts,   
    int n_splits, 
    bool no_splits = false
)
{
    int n_queries = query_ranges.extent(0);
    int n_candies = candi_arrays.extent(1);
    int tot_clusters = zcodes_array_offsets.extent(0);
    
    if (!no_splits) {
        dim3 blocks(16, 1, 32);
        dim3 grids((n_queries + blocks.x - 1) / blocks.x, 
        (n_candies + blocks.y - 1) / blocks.y, (n_splits + blocks.z - 1) / blocks.z);
        get_query_end_points_kernel<<<grids, blocks>>>(
            zcodes_array.data_handle(), zcodes_array_offsets.data_handle(),
            zcodes_array_lengthes.data_handle(), query_ranges.data_handle(), 
            candi_arrays.data_handle(), out_end_points.data_handle(), 
            out_counts.data_handle(), n_queries, n_candies, n_splits, tot_clusters
        );
    }
    else {
        dim3 blocks(64, 4);
        dim3 grids((n_queries + blocks.x - 1) / blocks.x, 
            (n_candies + blocks.y - 1) / blocks.y);
        get_query_end_points_no_split_kernel<<<grids, blocks>>>(
            zcodes_array.data_handle(), zcodes_array_offsets.data_handle(),
            zcodes_array_lengthes.data_handle(), query_ranges.data_handle(), 
            candi_arrays.data_handle(), out_end_points.data_handle(), 
            out_counts.data_handle(), n_queries, n_candies, tot_clusters
        );
    }
    checkCUDAErrorWithLine("get zcode end points failed");
}

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
)
{
    uint64_t n_queries = query_ranges.extent(0);
    int n_candies = candi_arrays.extent(1);
    int n_splits = out_sizes.extent(1) / n_candies;
    uint64_t tot_cluster = zcodes_array_offsets.extent(0);
    
    auto end_points = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, 2 * n_candies * n_splits);
    auto counts = parafilter_mmr::make_device_matrix_view<uint64_t, uint64_t>(n_queries, n_candies * n_splits);

    get_query_end_points(
        zcodes_arrays, 
        zcodes_array_offsets, 
        zcodes_array_lengthes, 
        query_ranges,
        candi_arrays,
        end_points, 
        counts,
        n_splits
    );

    

    dim3 blocks(16, 1, 32);
    dim3 grids((n_queries + blocks.x - 1) / blocks.x, 
        (n_candies + blocks.y - 1) / blocks.y, (n_splits + blocks.z - 1) / blocks.z);

    zorder_range_query_kernel<<<grids, blocks>>>(
        zcodes_arrays.data_handle(), 
        zcodes_array_offsets.data_handle(),
        zcodes_array_lengthes.data_handle(), 
        candi_arrays.data_handle(), 
        end_points.data_handle(), 
        n_candies, n_splits, 
        query_ranges.data_handle(), 
        query_boxes.data_handle(), 
        n_queries, tot_cluster, 
        out_starts.data_handle(), 
        out_sizes.data_handle(), 
        config
    );
}

void search_zorder_ranges(
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_arrays, // A second level zcode array splited by offsets and lengthes array 
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array_offsets, 
    raft::device_vector_view<uint64_t, uint64_t> const& zcodes_array_lengthes, 
    raft::device_matrix_view<uint64_t, uint64_t> const& query_ranges, 
    raft::device_matrix_view<int, uint64_t> const& query_boxes, 
    raft::device_matrix_view<uint64_t, uint64_t> const& candi_arrays,  
    raft::device_matrix_view<uint64_t, uint64_t> &end_points, 
    raft::device_matrix_view<uint64_t, uint64_t> &counts
)
{
    uint64_t n_queries = query_ranges.extent(0);
    int n_candies = candi_arrays.extent(1);

    get_query_end_points(
        zcodes_arrays, 
        zcodes_array_offsets, 
        zcodes_array_lengthes, 
        query_ranges,
        candi_arrays,
        end_points, 
        counts,
        1,
        true
    );
}