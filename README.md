# Benchmark for Tropical Tensor Network

In this repo we benchmark the performance of contraction of tropical tensor networks on GPU.
The high level interface is [GenericTensorNetworks.jl](https://github.com/QuEraComputing/GenericTensorNetworks.jl), the contraction is done by [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl), which uses [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU acceleration.
Specially, to accelerate semiring matrix multiplication, we implement the [CuTropicalGEMM.jl](https://github.com/TensorBFS/CuTropicalGEMM.jl), which has a C-CUDA backend [TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda).

The data used for benchmark including 10 networks with space complexity of $2^{31}$, and I also provided 10 networks with space complexity of $2^{32}$, which are theoretical solvable on A100 GPU.
The benchmarked networks are stored in `networks/` folder, including the graph and the corresponding network's contraction order, for example, `networks/sc31/graph_1.dot` and `networks/sc31/eincode_1.json`.
One can easliy load these networks in both julia or python, please see the code in `jcode/benchmark_tropical.jl` and `pcode/torch_load.py`.

Tensors of these networks are simple, each edge of the graph corresponds to a rank-2 tensor, given by
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
under the max-plus tropical semiring.
The result of the contraction is the size of the maximal independent set of the graph.
The contraction complexity of the networks are extracted in `data/complexity_sc31.csv` and `data/complexity_sc32.csv`, `sc` for space complexity and `tc` for time complexity.

We benchmark these networks with `sc = 31` on A800 GPU, under both normal algebra and tropical algebra, the results are shown in `data/benchmark_julia_real_sc31.csv` and `data/benchmark_julia_tropical_sc31.csv`.
The latter also includes the exact results of the maximal independent set problem given by the contraction of the network.