from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    name='tree_comparison',
    version='0.1.beta',
    packages=find_packages(),
    install_requires=required,
    entry_points={
            "console_scripts": [
                "tree_compare = tree_comparison.run_tree_compare:console_script",
                "tree_compare_test = tree_comparison.test_tree_compare:console_script",
            ]
        },
    include_package_data=True,
    author='Matt Mallory',
    author_email='matt.mallory@alleninstitute.org',
    description='Python package for quantification of tree similarity'
)
