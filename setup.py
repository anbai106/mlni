import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlni",
    version="0.0.6",
    author="junhao.wen",
    author_email="junhao.wen89@email.com",
    description="Machine Learning in NeuroImaging for various tasks, e.g., regression, classification and clustering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anbai106/mlni",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'mlni = mlni.main:main',
        ],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
