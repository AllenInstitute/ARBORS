from setuptools import setup, find_packages 
from pybind11.setup_helpers import Pybind11Extension, build_ext

extension_mod = Pybind11Extension(
    "tree_comparison.cpp.quantized_convex_matching",
    ['tree_comparison/cpp/quantizedConvexMatching.cpp'],
    extra_compile_args=['-I/usr/include/boost'],
    language='c++'
)

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    name='tree_comparison',
    version='0.1.beta',    
    author='Matt Mallory',
    author_email='matt.mallory@alleninstitute.org',
    description='Python package for quantification of tree similarity',
    packages=find_packages(),
    install_requires=required,
    entry_points={
            "console_scripts": [
                "tree_compare = tree_comparison.run_tree_compare:console_script",
                "tree_compare_test = tree_comparison.test_tree_compare:console_script",
            ]
        },
    include_package_data=True,
    ext_modules=[extension_mod], 
    cmdclass={"build_ext": build_ext},
)
