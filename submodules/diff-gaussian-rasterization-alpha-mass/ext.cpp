#include <torch/extension.h>
#include "alpha_mass_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("accumulate_alpha_mass", &AccumulateAlphaMassCUDA);
}
