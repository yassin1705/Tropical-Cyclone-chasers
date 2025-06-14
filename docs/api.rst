Référence de l'API
==================

Cette section fournit une documentation détaillée des points de terminaison (endpoints) et de l’utilisation de l’API d’**AI-Tornado-Chaser**.

Points de terminaison REST API
------------------------------

Prédiction de trajectoire
~~~~~~~~~~~~~~~~~~~~~~~~~~

**POST** http://localhost:500/predict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prédit les futures positions d'une tornade à partir de données de trajectoire historiques.

**Format de la requête :**

.. code-block:: json

    {
        "coordinates": [
            [latitude1, longitude1],
            [latitude2, longitude2],
            [latitude3, longitude3],
            [latitude4, longitude4],
            [latitude5, longitude5]
        ]
    }

- `coordinates` : Liste de 5 paires [latitude, longitude]  
- `horizon` : Nombre de positions futures à prédire (valeur par défaut : 10)

**Format de la réponse :**

.. code-block:: json

    {
        "predictions": [
            [pred_lat1, pred_lon1],
            [pred_lat2, pred_lon2],
            ...
            [pred_lat10, pred_lon10]
        ]
    }

**Exemple d’utilisation :**

.. code-block:: python

    import requests
    import json

    api_url = "http://localhost:500/predict"
    data = {
        "coordinates": [
            [57, 1399],
            [59, 1393],
            [63, 1387],
            [67, 1380],
            [72, 1373]
        ],
        "horizon": 10
    }

    response = requests.post(api_url, json=data)
    predictions = response.json()

Interaction avec le chatbot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**POST** /api/v1/get_instruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fournit des réponses intelligentes aux questions des utilisateurs concernant la sécurité en cas de tornade et des informations générales.

**Format de la requête :**

.. code-block:: json

    {
        "query": "Que dois-je faire en cas d’alerte tornade ?"
    }

**Format de la réponse :**

.. code-block:: json

    {
        "response": "Trouvez immédiatement un abri..."
    }

Gestion des erreurs
-------------------

L’API utilise les codes de statut HTTP standards :

- 200 : Requête réussie  
- 400 : Requête invalide (entrée incorrecte)  
- 404 : Ressource non trouvée  
- 500 : Erreur serveur interne  

Les réponses d'erreur incluent des messages détaillés :

.. code-block:: json

    {
        "error": "Format d’entrée invalide",
        "detail": "Les coordonnées doivent être une liste de 5 paires lat-long",
        "code": "INVALID_INPUT"
    }

Limitation de débit
-------------------

Les points de terminaison de l’API sont soumis à des limitations :

- API de prédiction : 100 requêtes par minute  
- API chatbot : 60 requêtes par minute  

Les en-têtes de limitation de débit sont inclus dans les réponses :

.. code-block:: text

    X-RateLimit-Limit: 100
    X-RateLimit-Remaining: 95
    X-RateLimit-Reset: 1623240000

Versionnement du modèle
-----------------------

L’API prend en charge plusieurs versions de modèle :

- Spécifiez la version dans l’URL : `/api/v1/predict`  
- Ou utilisez l’en-tête : `X-Model-Version: 2.0`  
- La version stable la plus récente est utilisée par défaut

Formats de données
------------------

Tous les points de terminaison de l’API :

- Acceptent et retournent des données en **JSON**
- Utilisent des horodatages **UTC**
- Prennent en charge la compression **GZIP**
- Gèrent les requêtes **CORS** pour les clients web
