.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/fsscore.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/fsscore
    .. image:: https://readthedocs.org/projects/fsscore/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://fsscore.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/fsscore/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/fsscore
    .. image:: https://img.shields.io/pypi/v/fsscore.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/fsscore/
    .. image:: https://img.shields.io/conda/vn/conda-forge/fsscore.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/fsscore
    .. image:: https://pepy.tech/badge/fsscore/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/fsscore
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/fsscore

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

============
fsscore
============


    Synthetic complexity score augmented with human intuition.


A longer description of your project goes here...

Installation
============
::

    git clone https://github.com/schwallergroup/fsscore.git
    conda create -n fsscore python=3.10
    conda activate fsscore
    cd fsscore
    pip install -e .

.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd fsscore
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/

Note
====

This project has been set up using PyScaffold 4.4.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
