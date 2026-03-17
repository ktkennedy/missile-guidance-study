from setuptools import setup, find_packages

setup(
    name="missile-guidance-study",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.5",
    ],
    extras_require={
        "mpc": ["casadi>=3.5"],
        "ml": ["scikit-learn>=1.0"],
        "notebook": ["jupyter>=1.0", "ipykernel>=6.0"],
        "all": ["casadi>=3.5", "scikit-learn>=1.0", "jupyter>=1.0", "ipykernel>=6.0"],
    },
)
