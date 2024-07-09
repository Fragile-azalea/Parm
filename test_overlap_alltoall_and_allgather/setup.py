"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import os
import sys
import platform as pf

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import IS_HIP_EXTENSION


root_path = os.path.dirname(sys.argv[0])
root_path = root_path if root_path else "."
root_path = os.path.abspath(root_path)

os.chdir(root_path)


def install():
    ext_libs = []
    ext_args = (
        [
            "-Wno-sign-compare",
            "-Wno-unused-but-set-variable",
            "-Wno-terminate",
            "-Wno-unused-function",
            "-Wno-strict-aliasing",
        ]
        if pf.system() == "Linux"
        else []
    )

    ext_libs += ["cuda", "nvrtc", "nccl"]
    ext_args += ["-DUSE_GPU", "-DUSE_NCCL"]

    setup(
        name="overlap_test",
        version="0.1",
        license="Apache-2.0",
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Environment :: GPU",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache-2.0 License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        packages=find_packages(),
        python_requires=">=3.9, <4",
        install_requires=[],
        zip_safe=False,
        ext_modules=[
            CUDAExtension(
                "overlap_test",
                sources=[
                    "./src/custom_kernel.cpp",
                ],
                library_dirs=["/usr/local/cuda/lib64/stubs"],
                libraries=ext_libs,
                extra_compile_args={"cxx": ext_args},
            )
        ],
        cmdclass={
            "build_ext": BuildExtension,
        },
    )


install()
