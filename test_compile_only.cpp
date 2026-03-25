// Host-only test: instantiate the 4-stage templates (no CUDA required).
// Compile: g++ -std=c++17 -I. -o test_compile_only test_compile_only.cpp
#include "four_stage.hpp"
#include <cassert>
#include <cstdio>

using namespace four_stage;

int main() {
  using T = tile<Mem::Shared, Arch::Hopper, 64, 64>;
  assert(T::rows == 64 && T::cols == 64 && T::mem == Mem::Shared);

  using A = assign_to_thread<Mem::Global, Mem::Shared, Arch::Hopper, 64, 64>;
  assert(A::block_threads == 128);

  using S = swizzle<Mem::Shared, Arch::Hopper>;
  assert(S::use_swizzle == true);

  using P = pipeline<Arch::Hopper, 2>;
  assert(P::stages == 2);

  reg_tile<32, 16> r;
  (void)r;
  shared_tile<64, 64> s;
  (void)s;
  global_tile<128, 128> g;
  (void)g;

  printf("PASS: four_stage template instantiation (host-only)\n");
  return 0;
}
