# ncu / nsys helper notes
#
# Nsight Compute (per-kernel):
#   ncu --set full -o profiling/ncu/flash \
#       python benchmarks/benchmark_latency.py --seq 1024 --impls flash
#
# Useful metrics to dump into the README:
#   sm__throughput.avg.pct_of_peak_sustained_elapsed
#   dram__throughput.avg.pct_of_peak_sustained_elapsed
#   l1tex__data_bank_conflicts_pipe_lsu.sum
#   sm__sass_l1tex_pipe_lsu_mem_shared_op_ld.sum
#   sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
#   launch__registers_per_thread
#   launch__shared_mem_per_block_dynamic
#
# Nsight Systems (timeline / launch overhead):
#   nsys profile -o profiling/nsys/flash \
#       python benchmarks/benchmark_latency.py --seq 256 --impls naive flash
