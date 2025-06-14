Installation et Configuration
=============================

Ce guide vous aidera à installer et configurer le projet **AI-Tornado-Chaser**.

Prérequis
---------

Avant de commencer, assurez-vous d’avoir installé :

- Python 3.8 ou version supérieure  
- Git  
- Au moins 8 Go de RAM (recommandé pour l'entraînement du modèle)  
- Espace disque : minimum 5 Go libres  
- CUDA toolkit (optionnel, pour le support GPU)

Dépendances principales
-----------------------

Le projet utilise les bibliothèques suivantes :

- TensorFlow >= 2.11.0  
- Keras >= 2.11.0  
- XGBoost >= 1.7.0  
- scikit-learn >= 1.2.0  
- pandas >= 1.5.0  
- numpy >= 1.23.0  
- matplotlib >= 3.6.0  
- sentence-transformers >= 2.2.0  
- faiss-cpu >= 1.7.0 (ou faiss-gpu pour le support GPU)

Dépendances pour la documentation
---------------------------------

Pour générer la documentation :

- sphinx >= 7.1.2  
- sphinx-rtd-theme >= 1.3.0  
- nbsphinx >= 0.9.2  
- sphinx-autodoc-typehints >= 1.24.0  
- Autres dépendances listées dans ``docs/requirements.txt``

Étapes d'installation
---------------------

1. Cloner le dépôt
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/yourusername/AI-Tornado-Chaser.git
   cd AI-Tornado-Chaser

2. Créer un environnement virtuel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m venv venv

   # Sous Windows
   venv\Scripts\activate

   # Sous Linux/Mac
   source venv/bin/activate

3. Installer les dépendances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -r requirements.txt

4. Télécharger les données
^^^^^^^^^^^^^^^^^^^^^^^^^^

Les données doivent être récupérées depuis le **CMA Tropical Cyclone Data Center** :

.. code-block:: bash

   python scripts/download_data.py

5. Configuration initiale
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cp config/config.example.yaml config/config.yaml

Modifiez les paramètres dans ``config/config.yaml`` selon vos besoins.

Génération de la documentation
------------------------------

1. Installer les dépendances de documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd docs
   pip install -r requirements.txt

2. Générer la documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Sous Windows
   .\make.bat html

   # Sous Unix/MacOS
   make html

La documentation sera disponible dans ``docs/build/html/index.html``.

Configuration ReadTheDocs
-------------------------

1. Créer un fichier ``.readthedocs.yaml`` à la racine du projet :

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
       - requirements: requirements.txt

2. Créer le fichier ``docs/requirements.txt`` :

.. code-block:: text

   sphinx>=4.0.0
   sphinx-rtd-theme>=1.0.0
   sphinx-autodoc-typehints>=1.12.0

Vérification de l’installation
------------------------------

Pour vérifier que tout fonctionne :

.. code-block:: bash

   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"
   python scripts/test_installation.py

Dépannage
---------

**Erreur CUDA**  
Installez la version CPU de TensorFlow :

.. code-block:: bash

   pip install tensorflow-cpu

**Problème de mémoire**  
Réduisez la taille du batch dans le fichier de configuration :

.. code-block:: yaml

   model:
     batch_size: 4

**Erreur FAISS**  
Sur certains systèmes, installez FAISS via conda :

.. code-block:: bash

   conda install -c conda-forge faiss-cpu

Variables d’environnement
-------------------------

Définissez ces variables d’environnement (optionnel) :

.. code-block:: bash

   export TORNADO_DATA_PATH=/chemin/vers/les/données
   export TORNADO_MODEL_PATH=/chemin/vers/les/models
   export TORNADO_LOG_LEVEL=INFO
