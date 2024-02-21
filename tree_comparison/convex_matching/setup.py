from setuptools import setup, Extension

setup(
    name='quantized_convex_matching', 
    version='0.1',
    author='Sarah Walling-Bell',
    author_email='sarah.wallingbell@alleninstitute.com',
    description='Python wrapper for quantizedConvexMatching function',
    ext_modules=[
        Extension('quantized_convex_matching', ['quantizedConvexMatching.cpp'],  
                  include_dirs=['/home/sarah.wallingbell/anaconda3/envs/tree_comp_dev_new/lib/python3.9/site-packages/pybind11/include'],
                  extra_compile_args=['-I/usr/include/boost'],
                  language='c++')
    ],
    install_requires=['pybind11']
)
