from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "diff-gaussian-rasterization"

setup(
    name="diff_gaussian_rasterization_alpha_mass",
    packages=["diff_gaussian_rasterization_alpha_mass"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_alpha_mass._C",
            sources=[
                str(BASE / "cuda_rasterizer" / "rasterizer_impl.cu"),
                str(BASE / "cuda_rasterizer" / "forward.cu"),
                str(BASE / "cuda_rasterizer" / "backward.cu"),
                str(HERE / "alpha_mass_points.cu"),
                str(HERE / "ext.cpp"),
            ],
            include_dirs=[str(BASE), str(BASE / "third_party" / "glm")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
