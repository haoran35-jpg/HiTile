#include "four_stage.hpp"
#include <cstdio>

using namespace four_stage;

// Minimal kernel: run the 4 "phases" + one schedule step check.
__global__ void kernel_four_stage(int* out) {
  using T = tile<Mem::Shared, Arch::Hopper, 64, 64>;
  using A = assign_to_thread<Mem::Global, Mem::Shared, Arch::Hopper, 64, 64>;
  using S = swizzle<Mem::Shared, Arch::Hopper>;
  using P = pipeline<Arch::Hopper, 2>;
  using Sch = P::k_schedule;

  int phase = 0;
  phase += 1;
  phase += (A::block_threads == 128 ? 1 : 0);
  phase += (S::use_swizzle ? 1 : 0);
  phase += P::stages;
  // schedule: step 2 -> load buf 0, compute buf 1, k_compute 1
  phase += (Sch::load_buffer(2) == 0 && Sch::compute_buffer(2) == 1 ? 1 : 0);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    *out = phase;
}

int main() {
  int* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(int));
  kernel_four_stage<<<1, 128>>>(d_out);
  int h_out = 0;
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  // 1+1+1+2 + 1 (schedule check) = 6
  const int expected = 6;
  if (h_out != expected) {
    printf("FAIL: got %d, expected %d\n", h_out, expected);
    return 1;
  }
  printf("PASS: four_stage template test (value=%d)\n", h_out);
  return 0;
}
