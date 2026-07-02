# ARBORS

**Algorithm for Recursive Branch ORganization Similarity**

*A Python package for quantitative comparison of tree topology.*

<!-- TODO: add a workflow figure -->

<!-- ![ARBORS overview](docs/images/arbors_overview.png) -->

ARBORS computes similarity scores between trees using recursive node matching algorithms that account for branching topology and geometry. It supports trees represented as SWC files and is particularly well suited for comparing reconstructed neuronal morphologies.

---

## Installation

ARBORS has been tested on **Linux**.

#### 1. Install Boost

Install the Boost C++ headers using your preferred package manager or from: https://www.boost.org/

#### 2. Create a Python environment

```bash
conda create -n arbors_env python=3.9
conda activate arbors_env
pip install pybind11
```

#### 3. Clone the repository

```bash
git clone git@github.com:AllenInstitute/ARBORS.git
cd ARBORS
```

#### 4. Configure Boost


Before building, update the Boost include path in `setup.py` with the location of your Boost headers:

```python
extra_compile_args=['-I/path/to/boost/install']
```

#### 5. Build and install

```bash
python setup.py build_ext --inplace
pip install .
```

---

## Quick Start

Compare the basal dendrites of two SWC files using the convex similarity metric.

```bash
arbors \
    --swc_1_path path/to/file1.swc \
    --swc_2_path path/to/file2.swc \
    --output_dir path/to/output \
    --similarity_function convex \
    --max_depth 1 \
    --compartments 3 \
    --orientations 0 \
```

Results are written to the specified output directory.


---

## Key Parameters

| Parameter             | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| `similarity_function` | Similarity metric used for node matching (`length` or `convex`).                |
| `max_depth`           | Depth of subtree context used during matching (`1` or `2`).                     |
| `compartments`        | SWC compartment(s) to compare.                                                  |
| `orientations`        | Rotation angles (degrees) evaluated when using the `convex` metric.             |
| `valid_set_dict`      | Precomputed subtree matches for comparisons with `max_depth=2`. |

---



## Support

ARBORS is under active development.
