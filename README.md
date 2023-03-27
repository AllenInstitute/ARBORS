# tree comparison

This package supports the quantitative comparison of tree topology. This is a beta version release that only
contains some features, and is actively being developed. Currently, we only support a length based similarity function
with a max depth parameter=1. We are actively developing code to support other input 
parameters.

Installation instructions
=========================
WIP  

conda create -n tree_compare_env python=3.9  
conda activate tree_compare_env    
pip install git+https://github.com/AllenInstitute/tree_comparison.git

NOTE:  
The above was tested and works on Windows. If the above fails, it's suggested to create a virtual environment with python 3.9 and install lap
from source (instead of PyPi) using the "Install from source" instructions here https://github.com/gatagat/lap.  
Once installed, then re-try:  
pip install git+https://github.com/AllenInstitute/tree_comparison.git



Scripts
=======
After installation the following console scripts will be available to run from the command line of your environment. To see detailed instructions on each script type the name of the SCRIPT_NAME --help

tree_compare
----------------------------
can be run in the following configurations:
1. give two swc files and compute the tree similarity  
   $ tree_compare  
    --input_swc_file_1 some/path/to/file1.swc  
    --input_swc_file_2 some/path/to/file2.swc  
    --output_file path/to/file_1_file_2_results.csv  
    --similarity_functions length  
    --max_depth 1  

  
2. give a directory of swc files and compute the tree similarity for all pairs of cells  
   $ tree_compare  
    --input_swc_dir path/to/directory/  
    --output_file path/to/results.csv  
    --similarity_functions length  
    --max_depth 1

tree_compare_test
------------------------------
test code to ensure repo is running properly. Currently only runs length similarity function
and max_depth=1. Ensures that a self to self similarity is equal to zero.






