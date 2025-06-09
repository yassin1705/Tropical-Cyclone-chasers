=================
Deployment Guide
=================

This guide explains how to deploy the AI-Tornado-Chaser project in both local and production environments.

Local Deployment
---------------

1. **Cloner le dépôt** :

.. code-block:: bash

   git clone https://github.com/username/ai-tornado-chaser.git
   cd ai-tornado-chaser

2. **Créer un environnement virtuel** :

.. code-block:: bash

   python -m venv env
   source env/bin/activate

3. **Installer les dépendances** :

.. code-block:: bash

   pip install -r requirements.txt

4. **Lancer l'application** :

.. code-block:: bash

   python app.py

**Déploiement avec Docker**
----------------------------

1. **Construire l'image Docker** :

.. code-block:: bash

   docker build -t tornado-app .

2. **Lancer le conteneur Docker** :

.. code-block:: bash

   docker run -d -p 5000:5000 tornado-app

**Déploiement avec ReadTheDocs**
--------------------------------

- Assurez-vous que `index.rst`, `conf.py`, et `.readthedocs.yaml` sont configurés.
- Activez le projet sur votre espace ReadTheDocs.
