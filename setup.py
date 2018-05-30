import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geneticocolor",
    version="0.0.1",
    author="Beatriz Hernandez Perez",
    author_email="beatriz.hpz@gmail.com",
    description="Package for genetic generation of opposite colors for 2D point graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geneticocolor",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)