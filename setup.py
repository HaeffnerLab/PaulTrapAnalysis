from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Analyzing RF Paul traps"

setup(
    name="PaulTrapAnalysis",
    version="2.0.0",
    description="Toolkit for developing and analyzing RF Paul traps",
    long_description=description,
    author="Shuqi Xu, Qian Yu, Andris Huang",
    keywords=["Quantum Information Processing", "RF Engineering", "Paul Traps"],
    url="https://github.com/Andris-Huang/PaulTrapAnalysis",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "future",
        "numpy",
        "scipy",
        "pandas",
        "setuptools", 
        #"setuptools-scm",
        "matplotlib",
        'tqdm',
        #'optuna',
        'plotly',
        'scikit-learn',
        #'sphericart',
        'pyvista==0.40.1',
        'xarray',
        'cvxpy',
        'sympy',
        'cvxopt',
        #'tensorflow'
        "bem @ https://github.com/Andris-Huang/bem/tarball/master"
    ],
    package_data = {
        "PaulTrapAnalysis": ["components/*.py", "functions/*.py"]
    },
    setup_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    scripts=[
        "PaulTrapAnalysis/scripts/download",
        "PaulTrapAnalysis/scripts/upload"
    ],
)
