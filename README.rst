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

.. _pyscaffold-notes:

Data
====
Training and testing data as well as trained models can be downloaded from figshare: https://figshare.com/s/2db88a98f73e22af6868
Please download the ```data``` and ```models``` folders and place them in the root directory of the repository.

Usage
=====

Scoring molecules
-----------------

To score molecules, use the ```score.py``` script. The script takes SMILES as input and outputs a CSV file with the scores. The script can be run as follows::

    python score.py --model_path <path_to_model_file> --data_path <path_to_csv_file> --compound_cols <SMILES_column> --save_filepath <path_to_save_file> --featurizer "graph_2D --batch_size 128

If no model path is provided the pre-trained graph-based model is used per default. The data path should point to a CSV file with a column containing SMILES. The column name can be specified with the ```--compound_cols``` argument. The ```--featurizer``` argument specifies the featurization method to use. The default is ```graph_2D```. The ```--batch_size``` argument specifies the batch size to use for scoring. The default is 128.

Fine-tuning
-----------

To fine-tune a model, use the ```finetuning.py``` script. The script takes a CSV file with SMILES as input and outputs a trained model. The script can be run as follows::

    python finetuning.py --data_path <path_to_finetuning_data> --featurizer graph_2D --compound_cols smiles_i smiles_j --rating_col target --save_dir <path_to_save_dir> --batch_size 4 --val_size 5 --n_epochs 20 --lr 0.0001 --datapoints 50 --track_improvement --track_pretest --earlystopping

This fine-tunes the best pre-trained model (graph, GGLGGL) based in a CSV containing two columns of SMILES and a column of binary preference labels. For production, ```val_size``` can be set to 0. The ```--data_path``` argument specifies the path to the CSV file with the fine-tuning data. The ```--featurizer``` argument specifies the featurization method to use. The default is ```graph_2D```. The ```--compound_cols``` argument specifies the columns containing the SMILES of the opposite pairs. The ```--rating_col``` argument specifies the column containing the binary preference. 0 indicates that the molecule in the first column is harder to synthesize while 1 indicates that the moelcule in the second column is harder. The ```--save_dir``` argument specifies the directory to save the model in. The ```--batch_size``` argument specifies the batch size to use for training. The default is 4. The ```--val_size``` argument specifies the number of validation samples to use. The default is 5. The ```--n_epochs``` argument specifies the number of epochs to train for. The default is 20. The ```--lr``` argument specifies the learning rate to use. The default is 0.0001. The ```--datapoints``` argument specifies the number of data points to use for fine-tuning (leave out for production). The default is None, making use of the whole dataset. The ```--track_improvement``` argument specifies whether to track the improvement on the validation set. The default is True. The ```--track_pretest``` argument specifies whether to track the performance on the pre-training test set. The default is True. The ```--earlystopping``` argument specifies whether to use early stopping. The default is True.

Training a model
---------------

To train a model, use the ```train.py``` script. The script takes a CSV file with SMILES as input and outputs a trained model. The script can be run as follows::

    python train.py --save_dir <path_to_save_dir> --featurizer graph_2D --n_epochs 250 --val_size 0.01 --batch_size 128 --arrange_layers GGLGGL --graph_encoder GNN --reload_interval 10

This trains a graph-based model in the same fashion as described. Please make sure you have the training data downloaded. The ```--save_dir``` argument specifies the directory to save the model in. The ```--featurizer``` argument specifies the featurization method to use. The default is ```graph_2D```. The ```--n_epochs``` argument specifies the number of epochs to train for. The default is 250. The ```--val_size``` argument specifies the fraction of the data to use for validation. The default is 0.01. The ```--batch_size``` argument specifies the batch size to use for training. The default is 128. The ```--arrange_layers``` argument specifies the arrangement of the graph attention layers. The default is ```GGLGGL```. The ```--graph_encoder``` argument specifies the graph encoder to use. The default is ```GNN```. The ```--reload_interval``` argument specifies the interval at which to save the model.

App: FSscore
============

This repository contains a streamlit app that can be run locally. To run the app, use the following command::

    streamlit run streamlit_app/run.py

This will open a browser window with the app. Currently, only the labeling process is implemented. We are working on adding fine-tuning and scoring functionalities.
The app should be run locally as files are written and saved. For deployment, please refer to the streamlit documentation.


Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
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
