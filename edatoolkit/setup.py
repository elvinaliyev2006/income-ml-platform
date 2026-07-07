from setuptools import setup

setup(
    name="edatoolkit",
    version="0.2.0",
    author="Elvin Aliyev",
    description="A professional OOP-based EDA toolkit with statistical tests",
    package_dir={"": "."},
    packages=["edatoolkit"],
    install_requires=[
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "scipy"
    ],
    python_requires=">=3.7",
)