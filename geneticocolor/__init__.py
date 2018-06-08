# -*- coding: utf-8 -*-

"""
Geneticocolor
========
Geneticocolor is a Python package for color sets generation in 2D plots, based in oRGB color space.
Source::
    https://github.com/bpz/geneticocolor
Simple example
--------------
Generate colors for a set of points and plot them using matplotlib:
    >>> import geneticocolor.color_generator as generator
    >>> import matplotlib.pyplot as plt
    ...
    >>> colors = generator.generate(x, y, point_classes)
    >>> plt.scatter(x, y, colors)
License
-------
GNU General Public License v3.0
"""

name = "geneticocolor"
