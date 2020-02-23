#include <torch/extension.h>
#include <iostream>
#include <vector>


torch::Tensor valid (torch::Tensor states) {
    auto statesd = states.accessor<unsigned char,2>();
    statesd[1][1] = 1;
    return states;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("valid", &valid, "valid");
}