from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

extension_mod = Pybind11Extension(
    "arbors.cpp.quantized_convex_matching",
    ["src/arbors/cpp/quantizedConvexMatching.cpp"],
    extra_compile_args=["-I/usr/include/boost"],
    language="c++",
)

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ARBORS",
    version="0.3.0",
    author="Sarah Walling-Bell",
    author_email="sarah.wallingbell@alleninstitute.org",
    description="Python package for quantification of tree similarity",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    include_package_data=True,
    package_data={"arbors": ["data/*"]},
    ext_modules=[extension_mod],
    cmdclass={"build_ext": build_ext},
)
