#pragma once
#include <cuda_runtime.h>
#include <utils/debugging_utils.cuh>

class Timer {
  public:
  Timer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  void start_timer()
  {
    cudaEventRecord(start);
  }
  void stop_timer()
  {
    cudaEventRecord(stop);
  }

  float get_time()
  {
    float millisec = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisec, start, stop);

    checkCUDAError("get kernel perf info failed in get_time");
    return millisec;
  }

  private:
  cudaEvent_t start;
  cudaEvent_t stop;
};
  
  //fixme: fake run use macro is not simple to control the, 
  //call in any levels, try to find better way 
  // todo: Add cudaDeviceSyncronize for kernel break-down anylyze.
#define parafilterPerfLogWraper(func, time) \
  if (break_down) { \
    global_timer.start_timer(); \
    {func ;} \
    global_timer.stop_timer(); \
    float elapsed = global_timer.get_time(); \
    time += elapsed; \
    LOG(INFO) << __func__ << " " << __LINE__ << " elapsed with time: " << elapsed / 1000.f << "s"; \
  } \
  else { \
    {func ;} \
  } 

inline void get_current_device_mem_info(uint64_t &available, uint64_t &total) {
  int id;
  cudaGetDevice(&id);
  cudaMemGetInfo(&available, &total);
}

inline void print_memory_usage(const std::string& prefix = "")
{
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << "\n";
        return;
    }

    size_t used_bytes = total_bytes - free_bytes;

    auto formatBytes = [](size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB"};
        double size = static_cast<double>(bytes);
        int unit = 0;
        while (size > 1024.0 && unit < 3) {
            size /= 1024.0;
            ++unit;
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return oss.str();
    };

    if (!prefix.empty())
        std::cout << prefix << " ";

    LOG(TRACE) << "GPU Memory: used = " << formatBytes(used_bytes)
              << ", free = " << formatBytes(free_bytes)
              << ", total = " << formatBytes(total_bytes);
}

inline uint32_t get_cur_device_maxi_threads() {
  int id;
  cudaGetDevice(&id);
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, id);
  return devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor;
}
  