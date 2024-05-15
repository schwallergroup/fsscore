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

.. .. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
..     :alt: Project generated with PyScaffold
..     :target: https://pyscaffold.org/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :alt: License: MIT
    :target: LICENSE.txt
.. image:: https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC
    :alt: Powered by RDKit
    :target: https://www.rdkit.org/


============
FSscore
============


    Focused synthesizability score augmented with human knowledge and intuition.


The Focused Synthesizability score (FSscore) learns to rank structures based on binary preferences using a graph attention network. First, a baseline trained on an extensive set of reactant-product pairs is established that subsequently is fine-tuned with on a chemical space of interest using binary preferences from human experts or carefully designed labels.

Installation
============
::

    git clone https://github.com/schwallergroup/fsscore.git
    cd fsscore
    conda create -n fsscore python=3.10
    conda activate FSscore
    pip install -r requirements.txt
    pip install -e .

This method was tested and developed on CUDA-enabled GPUs (linux OS).
.. _pyscaffold-notes:

Data
====
Trained models can be downloaded from figshare: https://figshare.com/s/2db88a98f73e22af6868
Please download the ``models`` folder and place it in the root directory of the repository. The best graph-based model (GGLGGL) is already included in this folder.

The code for the data processing is made available in this repository (``data_processing``). Please change the paths in the bash script accordingly. These include the directory where you want to store the data (the default is recommended) and set the path to the additional data paths. Subsequently, run the following command:
::

    cd data_processing
    ./process_data.sh

This will create the necessary data files for pre-training.

Usage
=====

Scoring molecules
-----------------

The following code shows an example on how to easily score molecules in python  This examples uses the graph-based implementation of the FSscore.

    .. code-block:: python

        from fsscore.score import Scorer
        from fsscore.models.ranknet import LitRankNet
        from fsscore.utils.paths import PRETRAIN_MODEL_PATH

        # 1) load pre-trained model or choose path to own model
        model = LitRankNet.load_from_checkpoint(PRETRAIN_MODEL_PATH)

        # 2) initialize scorer
        scorer = Scorer(model=model)

        # 3) predict scores given a list of SMILES
        scores = scorer.score(smiles)

To score molecules using the command line, use the ``score.py`` script. The script takes SMILES as input and outputs a CSV file with the scores. The script can be run as follows::

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

This command uses the training data used in our manuscript. To input your own data provide the path to ``--data_path`` and specifz the collumn names for the SMILES (``--compound_cols``) and the binary preference labels (``--rating_col``).

If you want to train a model with a fingerprint representation, do the following::
    - ``--featurizer``: Select from ``morgan``, ``morgan_count``, ``morgan_chiral`` or ``morgan_chiral_count``
    - ``--use_fp``: Set to True

App: FSscore
============

This repository contains a streamlit app that can be run locally. To run the app, use the following command::

    streamlit run streamlit_app/run.py

This will open a browser window with the app. Currently, only the labeling process is implemented. We are working on adding fine-tuning and scoring functionalities.
The app should be run locally as files are written and saved. For deployment, please refer to the streamlit documentation.
