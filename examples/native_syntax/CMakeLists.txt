# Example: using real CUDA <<<>>> syntax
# cuda_sim_add_sources auto-preprocesses main.cpp to replace <<<>>> with _launch()
add_executable(native_syntax_example)
cuda_sim_add_kernel(native_syntax_example kernel.ptx)
cuda_sim_add_sources(native_syntax_example main.cpp)
