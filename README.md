# tree comparison

This package supports the quantitative comparison of tree topology. This is a beta version release that only
contains some features, and is actively being developed. Currently, we only support a length based similarity function
with a max depth parameter =1. We are actively developing code to support other input 
parameters.

Installation instructions
=========================
conda create -n tree_compare_env python=3.9
pip install git+link_to_repo

Scripts
=======
After installation the following console scripts will be available to run from the command line of your environment. To see detailed instructions on each script type the name of the SCRIPT_NAME --help

tree_compare
----------------------------
can be run in the following configurations:
1. give two swc files and compute the tree similarity
2. give a directory of swc files and comput the tree similariy for all pairs of cells

tree_compare_test (TODO)
------------------------------
test code to ensure repo is running properly






