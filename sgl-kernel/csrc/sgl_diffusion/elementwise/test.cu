
#include "timestep_embedding.cu"

TORCH_LIBRARY_FRAGMENT(my_sglang_ops, m) {
  m.def("timestep_embedding(Tensor input, Tensor output, int dim, int max_period) -> Tensor");
  m.impl("timestep_embedding", torch::kCUDA, &timestep_embedding);
}
