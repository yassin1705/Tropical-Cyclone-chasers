=========================
Troubleshooting Guide
=========================

This guide helps resolve common issues encountered during installation, deployment, and execution of the AI-Tornado-Chaser project.

Model Training Issues
------------------

**Error: CUDA Out of Memory**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Insufficient GPU memory for the batch size  
**Solution:**

.. code-block:: python

    # Reduce batch size
    model.fit(X_train, y_train, 
             batch_size=32,  # Try smaller batch size
             epochs=10)

**Error: Model Not Converging**
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Learning rate or model architecture issues  
**Solution:**

1. Adjust learning rate:

   .. code-block:: python

       optimizer = Adam(learning_rate=0.001)
       model.compile(optimizer=optimizer, loss='mse')

2. Check data normalization:

   .. code-block:: python

       scaler = MinMaxScaler()
       X_train = scaler.fit_transform(X_train)

**Erreur : `ModuleNotFoundError`**
----------------------------------

**Cause :** Dépendance manquante.  
**Solution :**

.. code-block:: bash

   pip install -r requirements.txt

**Erreur : `Index not found for chatbot`**
------------------------------------------

**Cause :** Le fichier d'index `faq.index` est manquant.  
**Solution :** Exécutez le script pour générer l'index à partir des fichiers `.csv`.

**Erreur Docker : Port déjà utilisé**
-------------------------------------

**Solution :** Changez le port d'exécution :

.. code-block:: bash

   docker run -p 5001:5000 tornado-app

**Erreur TensorFlow version**
-----------------------------

**Solution :**

.. code-block:: bash

   pip install tensorflow==2.11.0

**Problème de configuration ReadTheDocs**
-----------------------------------------

- Vérifiez que `conf.py` et `index.rst` sont bien configurés dans votre dépôt ReadTheDocs.
- Si des erreurs persistent, consultez les logs de ReadTheDocs.
