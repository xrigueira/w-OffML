"""Datasets.

This module contains a collection of datasets for multiple tasks: classification, regression, etc.
The data corresponds to popular datasets and are conveniently wrapped to easily iterate over
the data in a stream fashion. All datasets have fixed size. Please refer to `river.synth` if you
are interested in infinite synthetic data generators.

"""
from .labeled import labeled_901
from .labeled import labeled_902
from .labeled import labeled_904
from .labeled import labeled_905
from .labeled import labeled_906
from .labeled import labeled_907
from .labeled import labeled_910
from .labeled import labeled_916

__all__ = [
    "labeled_901",
    "labeled_902",
    "labeled_904",
    "labeled_905",
    "labeled_906",
    "labeled_907",
    "labeled_910",
    "labeled_916"
]
