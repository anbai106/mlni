import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyHYDRA",
    version="1.0.7",
    author="junhao.wen",
    author_email="junhao.wen89@email.com",
    description="A fast python implementation of HYDRA for classification and clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anbai106/pyhydra",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'pyhydra = pyhydra.main:main',
        ],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
