from setuptools import setup, find_packages 
from pybind11.setup_helpers import Pybind11Extension, build_ext

extension_mod = Pybind11Extension(
    "tree_comparison.cpp.quantized_convex_matching",
    ['tree_comparison/cpp/quantizedConvexMatching.cpp'],
    # extra_compile_args=['-I/usr/include/boost'],
    extra_compile_args=['-I//allen/programs/celltypes/workgroups/mousecelltypes/SarahWB/boost'],
    language='c++'
)

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    name='tree_comparison',
    version='0.2.0',    
    author='Sarah Walling-Bell',
    author_email='sarah.wallingbell@alleninstitute.org',
    description='Python package for quantification of tree similarity',
    packages=find_packages(),
    install_requires=required,
    include_package_data=True,
    package_data={"tree_comparison": ["data/*"]},
    ext_modules=[extension_mod], 
    cmdclass={"build_ext": build_ext},
)
