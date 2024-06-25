from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Designing the trapped electron setups"

setup(
    name="PaulTrapAnalysis",
    version="1.0.0",
    description="Toolkit for developing and analyzing RF Paul traps",
    long_description=description,
    author="Qian Yu, Shuqi Xu, Andris Huang",
    keywords=["Quantum Information Processing", "RF Engineering", "Paul Traps"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "future",
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "matplotlib",
        'tqdm',
        #'optuna',
        'plotly',
        'scikit-learn',
        #'sphericart',
        'pyvista',
        'xarray',
        'cvxpy',
        'sympy',
        'cvxopt'
        #'tensorflow'
    ],
    package_data = {
        "edes": ["utils/*.py"]
    },
    setup_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    scripts=[
        
    ],
)
