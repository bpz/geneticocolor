# Genetic Opposite Color (geneticocolor)

Geneticocolor is a python package for color sets generation.

## Install
Using shell just write the following line:
```
$ pip install geneticocolor
```
## How to use
Once installed, you can import and use the library in any python file.
```
import geneticocolor.color_generator as generator
import matplotlib.pyplot as plt

colors = generator.generate(x, y, point_classes)
plt.scatter(x, y, colors)

```
## More about this project

Geneticocolor is a python library that I developed when I did my final master degree project. This is the final library produced and published in [PyPI](https://pypi.org/project/geneticocolor/) and the developing repository is private (where all tests are implemented).

The goal of this library is producing a set of colors that can be used for different points in a graph, being each color different enough of the closer ones to improve visibility. A genetic algorithm is used to produce the colors.
