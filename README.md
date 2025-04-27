# Benchmarking Tropical Tensor Networks

This repository benchmarks the performance of tropical tensor network contractions on GPUs. The high-level interface is provided by [GenericTensorNetworks.jl](https://github.com/QuEraComputing/GenericTensorNetworks.jl), while the contraction is executed using [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl), which leverages [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU acceleration. To specifically enhance semiring matrix multiplication, we have implemented [CuTropicalGEMM.jl](https://github.com/TensorBFS/CuTropicalGEMM.jl), which utilizes a C-CUDA backend, [TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda).

The benchmark data includes 10 networks with a space complexity of $2^{31}$, and an additional 10 networks with a space complexity of $2^{32}$, which are theoretically solvable on an A100 GPU. These networks are stored in the `networks/` folder, containing both the graph and the corresponding network's contraction order, such as `networks/sc31/graph_1.dot` and `networks/sc31/eincode_1.json`. These networks can be easily loaded in both Julia and Python; refer to the code in `jcode/benchmark_tropical.jl` and `pcode/torch_load.py`.

The tensors in these networks are straightforward: each edge of the graph corresponds to a rank-2 tensor, represented by
```math
\begin{pmatrix}
  0 & 0 \\
  0 & -\infty
\end{pmatrix}
```
and each vertex of the graph corresponds to a rank-1 tensor, given by
```math
\begin{pmatrix}
  0 \\
  1
\end{pmatrix}
```
These tensors are contracted under the max-plus tropical semiring.

The contraction result represents the size of the maximal independent set of the graph. The contraction complexities of the networks are documented in `data/complexity_sc31.csv` and `data/complexity_sc32.csv`, where `sc` denotes space complexity and `tc` denotes time complexity.

We benchmark these networks with `sc = 31` on an A800 GPU using both standard algebra and tropical algebra. The results are available in `data/benchmark_julia_real_sc31.csv` and `data/benchmark_julia_tropical_sc31.csv`. The latter file also provides the exact results for the maximal independent set problem derived from the network contraction. In this case the time cost of tropical algebra is about three times larger that standard algebra.
