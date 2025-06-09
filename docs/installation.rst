Installation and Configuration
==========================

This guide will help you install and configure the AI-Tornado-Chaser project.

Prerequisites
-----------

Before you begin, ensure you have the following installed:

* Python 3.8 or higher
* Git
* At least 8GB RAM (recommended for model training)
* Disk space: at least 5GB free
* CUDA toolkit (optional, for GPU support)

Core Dependencies
---------------

The project uses the following libraries:

* TensorFlow >= 2.11.0
* Keras >= 2.11.0
* XGBoost >= 1.7.0
* scikit-learn >= 1.2.0
* pandas >= 1.5.0
* numpy >= 1.23.0
* matplotlib >= 3.6.0
* sentence-transformers >= 2.2.0
* faiss-cpu >= 1.7.0 (or faiss-gpu for GPU support)

Documentation Dependencies
----------------------

For building documentation:

* sphinx >= 7.1.2
* sphinx-rtd-theme >= 1.3.0
* nbsphinx >= 0.9.2
* sphinx-autodoc-typehints >= 1.24.0
* Other dependencies listed in docs/requirements.txt

Installation Steps
---------------

1. Clone the Repository
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/yourusername/AI-Tornado-Chaser.git
   cd AI-Tornado-Chaser

2. Create Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m venv venv
   
   # Sur Windows
   venv\Scripts\activate
   
   # Sur Linux/Mac
   source venv/bin/activate

3. Installer les dépendances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -r requirements.txt

4. Télécharger les données
^^^^^^^^^^^^^^^^^^^^^^^^^^

Les données d'ouragans doivent être téléchargées depuis le CMA Tropical Cyclone Data Center :

.. code-block:: bash

   python scripts/download_data.py

5. Configuration initiale
^^^^^^^^^^^^^^^^^^^^^^^^^

Créez un fichier de configuration :

.. code-block:: bash

   cp config/config.example.yaml config/config.yaml

Modifiez les paramètres selon vos besoins dans ``config/config.yaml``.

Documentation Build
----------------

To build the documentation locally:

1. Install Documentation Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd docs
   pip install -r requirements.txt

2. Build Documentation
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # On Windows
   .\make.bat html
   
   # On Unix/MacOS
   make html

The documentation will be available at `docs/build/html/index.html`.

ReadTheDocs Configuration
---------------------

To configure documentation with ReadTheDocs:

1. Create a `.readthedocs.yaml` file in the project root:

.. code-block:: yaml

   version: 2
   
   build:
     os: ubuntu-22.04
     tools:
       python: "3.10"
   
   sphinx:
     configuration: docs/conf.py
   
   python:
     install:
       - requirements: docs/requirements.txt
     - pdf
     - htmlzip
   
   python:
     install:
       - requirements: docs/requirements.txt
       - requirements: requirements.txt

2. Créer ``docs/requirements.txt`` :

.. code-block:: text

   sphinx>=4.0.0
   sphinx-rtd-theme>=1.0.0
   sphinx-autodoc-typehints>=1.12.0

Vérification de l'installation
------------------------------

Pour vérifier que l'installation s'est déroulée correctement :

.. code-block:: bash

   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"
   python scripts/test_installation.py

Résolution des problèmes courants
----------------------------------

**Erreur CUDA**
   Si vous rencontrez des erreurs liées à CUDA, installez la version CPU de TensorFlow :
   
   .. code-block:: bash
   
      pip install tensorflow-cpu

**Problème de mémoire**
   Pour les systèmes avec peu de RAM, ajustez la taille des batches dans la configuration :
   
   .. code-block:: yaml
   
      model:
        batch_size: 4  # Réduire si nécessaire

**Erreur FAISS**
   Sur certains systèmes, installez faiss via conda :
   
   .. code-block:: bash
   
      conda install -c conda-forge faiss-cpu

Variables d'environnement
-------------------------

Définissez les variables d'environnement suivantes (optionnel) :

.. code-block:: bash

   export TORNADO_DATA_PATH=/path/to/data
   export TORNADO_MODEL_PATH=/path/to/models
   export TORNADO_LOG_LEVEL=INFO