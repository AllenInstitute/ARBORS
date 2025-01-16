# tree comparison

This package supports the quantitative comparison of tree topology.

# Installation instructions
Install Boost: https://www.boost.org/

Setup a conda environment with the proper requirements and clone the repo
```bash
conda create -n tree_compare_env python=3.9  
conda activate tree_compare_env    
pip install pybind11    
# git clone git@github.com:AllenInstitute/tree_comparison.git
git clone git@github.com:sarahwallingbell/tree_comparison.git
```

Ensure correct path to Boost in setup.py
```bash
extra_compile_args=['-I/path/to/boost/install']
```

Pip install tree comparison 
```bash
cd tree_comparison
python setup.py build_ext --inplace
pip install . 
```

NOTE:
The above was tested and works on Linux.

# Scripts
After installation the following console script will be available to run from the command line of your environment. To see detailed instructions on each script type the name of the SCRIPT_NAME --help

## tree-compare 
Script to take two swc files and compute the tree similarity. 

# Explanation of some keyword arguments
This package uses a similarity function to find optimal node matching between two trees, and calculate the resulting similarity. 

**similarity_function**: which similarity function to use in tree node matching. 
- **length**: compare the length of the matched edges.  
- **convex**: compare the length *and orientation* of the matched edges. Takes tree topology into account.
    
**max_depth**: what depth of subtrees to match nodes between. 
- **1**: match subtrees to the child depth. 
- **2**: match subtrees to the grandchild depth. Robust to slight variance in branch labelling. 

**compartments**: which compartments of the tree to compare within.

**orientations**: what orientations (degrees) to compare the trees at if similarity_funtion is convex. 

**valid_set_dict**: a pre-computed dictionary of valid grandchild depth subtree node matches when using max_depth 2. 

# Example Usage
Compare the basal dendrites of two swc files using the convex similarity function and max depth 2 at four rotations.
```bash 
tree-compare 
--swc_1_path path/to/file1.swc 
--swc_2_path path/to/file2.swc 
--output_dir path/to/save/results 
--similarity_function convex 
--max_depth 2 
--compartments 3 
--orientations 0 90 180 270 
--valid_set_dict path/to/valid_set.json
```

# Statement of Support
This code is an important part of the internal Allen Institute code base and we are actively using and maintaining it. Issues are encouraged, but because this tool is so central to our mission pull requests might not be accepted if they conflict with our existing plans.



