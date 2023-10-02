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
FSscore
============


    Synthetic feasibility score augmented with human knowledge and intuition.


The Focused Synthesizability score (FSscore) learns to rank structures based on binary preferences using a graph attention network. First, a baseline trained on an extensive set of reactant-product pairs is established that subsequently is fine-tuned with expert human feedback on a chemical space of interest.

Installation
============
::

    git clone https://github.com/schwallergroup/fsscore.git
    cd fsscore
    conda env create -f env.yml
    conda activate fsscore
    pip install -e .

.. _pyscaffold-notes:

Data
====
Training and testing data as well as trained models can be downloaded from figshare: https://figshare.com/s/2db88a98f73e22af6868
Please download the ``data`` and ``models`` folders and place them in the root directory of the repository.

Usage
=====

Scoring molecules
-----------------

To score molecules, use the ``score.py`` script. The script takes SMILES as input and outputs a CSV file with the scores. The script can be run as follows::

    python score.py --model_path <path_to_model_file> --data_path <path_to_csv_file> --compound_cols <SMILES_column> --save_filepath <path_to_save_file> --featurizer graph_2D --batch_size 128

The following arguments are used:
- ``--model_path``: Path to the model file. If no model path is provided the pre-trained graph-based model is used per default.
- ``--data_path``: Path to the CSV file with SMILES to score.
- ``--compound_cols``: Name of the column containing the SMILES.
- ``--save_filepath``: Path to save the CSV file with the scores.
- ``--featurizer``: Featurization method to use. The default is ``graph_2D``.
- ``--batch_size``: Batch size to use for scoring. The default is 128.

Fine-tuning
-----------

To fine-tune a model, use the ``finetuning.py`` script. The script takes a CSV file with SMILES as input and outputs a trained model. The script can be run as follows::

    python finetuning.py --data_path <path_to_finetuning_data> --featurizer graph_2D --compound_cols smiles_i smiles_j --rating_col target --save_dir <path_to_save_dir> --batch_size 4 --val_size 5 --n_epochs 20 --lr 0.0001 --datapoints 50 --track_improvement --track_pretest --earlystopping

The following arguments are used:
- ``--model_path``: Path to the model file. If no model path is provided the pre-trained graph-based model (GGLGGL) is used per default.
- ``--data_path``: Path to the CSV file with the fine-tuning data in two columns of SMILES and a column of binary preference labels.
- ``--featurizer``: Featurization method to use. The default is ``graph_2D``.
- ``--compound_cols``: Name of the columns containing the SMILES of the opposite pairs.
- ``--rating_col``: Name of the column containing the binary preference. 0 indicates that the molecule in the first column is harder to synthesize while 1 indicates that the moelcule in the second column is harder.
- ``--save_dir``: Directory to save the model in.
- ``--batch_size``: Batch size to use for training.
- ``--val_size``: Number (int) of fraction (float) of validation samples to use.
- ``--n_epochs``: Number of epochs to train for. Default is 20.
- ``--lr``: Learning rate to use. Default is 0.0001.
- ``--datapoints``: Number of data points to use for fine-tuning (leave out for production). Default is None, which results in the use of the whole dataset.
- ``--track_improvement``: Whether to track the improvement on the validation set. Defaults to True.
- ``--track_pretest``: Whether to track the performance on the pre-training test set. Defaults to True.
- ``--earlystopping``: Whether to use early stopping. Defaults to True.

Training a baseline model
-------------------------

To train a model, use the ``train.py`` script. The script takes a CSV file with SMILES as input and outputs a trained model. The script can be run as follows::

    python train.py --save_dir <path_to_save_dir> --featurizer graph_2D --n_epochs 250 --val_size 0.01 --batch_size 128 --arrange_layers GGLGGL --graph_encoder GNN --reload_interval 10

The following arguments are used (the same as described in the paper):
- ``--save_dir``: Directory to save the model in.
- ``--featurizer``: Featurization method to use. The default is ``graph_2D``.
- ``--n_epochs``: Number of epochs to train for.
- ``--val_size``: Fraction (float) of validation samples to use. Set to 0 to not use a validation set.
- ``--batch_size``: Batch size to use for training.
- ``--arrange_layers``: Arrangement of the graph attention layers. The default is ``GGLGGL``.
- ``--graph_encoder``: Graph encoder to use. The default is ``GNN``.
- ``--reload_interval``: Interval at which to save the model.

If you want to train a model with a fingerprint representation, do the following::
- ``--featurizer``: Select from ``morgan``, ``morgan_count``, ``morgan_chiral`` or ``morgan_chiral_count``
- ``--use_fp``: Set to True

App: FSscore
============

This repository contains a streamlit app that can be run locally. To run the app, use the following command::

    streamlit run streamlit_app/run.py

This will open a browser window with the app. Currently, only the labeling process is implemented. We are working on adding fine-tuning and scoring functionalities.
The app should be run locally as files are written and saved. For deployment, please refer to the streamlit documentation.


Making Changes & Contributing
=============================

This project uses pre-commit_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd fsscore
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Note
====

This project has been set up using PyScaffold 4.4.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
