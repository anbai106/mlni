import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyhydra",
    version="1.0.0",
    author="junhao.wen",
    author_email="junhao.wen89@email.com",
    description="A fast python implementation of HYDRA for classification and clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anbai106/pyhydra",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)

