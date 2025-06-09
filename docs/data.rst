Données et Prétraitement
========================

Cette section décrit en détail les données utilisées dans le projet AI Tornado Chaser, leur structure, et les étapes de prétraitement appliquées.

Vue d'ensemble des données
--------------------------

Le dataset principal provient du **CMA Tropical Cyclone Data Center** et contient des informations détaillées sur les cyclones tropicaux.

Statistiques générales
^^^^^^^^^^^^^^^^^^^^^^

* **Nombre d'ouragans** : 2,489
* **Durée moyenne des ouragans** : 174 heures
* **Période couverte** : 1949-2023 (74 ans)
* **Résolution temporelle** : 1 enregistrement toutes les 6 heures
* **Format des coordonnées** : Latitude (0.1°N), Longitude (0.1°E)

Structure des données
---------------------

Champs de données principaux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chaque enregistrement dans le dataset contient les champs suivants :

.. list-table:: Structure des données
   :widths: 20 20 60
   :header-rows: 1

   * - Champ
     - Type
     - Description
   * - YYYYMMDDHH
     - String
     - Date et heure en UTC (année-mois-jour-heure)
   * - LAT
     - Float
     - Latitude en degrés (0.1°N de précision)
   * - LONG
     - Float
     - Longitude en degrés (0.1°E de précision)
   * - PRES
     - Integer
     - Pression minimale (hPa) près du centre du cyclone
   * - WND
     - Float
     - Vitesse maximale soutenue des vents (m/s)
   * - Intensité
     - Integer
     - Catégorie d'intensité selon la norme chinoise

Classification d'intensité
^^^^^^^^^^^^^^^^^^^^^^^^^^

Le système utilise la norme chinoise pour la classification des cyclones tropicaux :

.. list-table:: Catégories d'intensité
   :widths: 10 30 30 30
   :header-rows: 1

   * - Code
     - Catégorie
     - Vitesse des vents (m/s)
     - Description
   * - 0
     - Inconnu/Faible
     - < 10.8
     - Plus faible qu'une dépression tropicale
   * - 1
     - Dépression Tropicale (TD)
     - 10.8-17.1
     - Formation initiale du cyclone
   * - 2
     - Tempête Tropicale (TS)
     - 17.2-24.4
     - Cyclone organisé avec rotation
   * - 3
     - Tempête Tropicale Sévère (STS)
     - 24.5-32.6
     - Intensification significative
   * - 4
     - Typhon (TY)
     - 32.7-41.4
     - Cyclone mature et dangereux
   * - 5
     - Typhon Sévère (STY)
     - 41.5-50.9
     - Cyclone très intense
   * - 6
     - Super Typhon (SuperTY)
     - ≥ 51.0
     - Cyclone de catégorie maximale
   * - 9
     - Cyclone Extratropical (ET)
     - Variable
     - Phase de transition

Prétraitement des données
-------------------------

Le prétraitement est crucial pour la performance des modèles. Voici les étapes principales :

1. Nettoyage des données
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Suppression des valeurs manquantes
   df = df.dropna(subset=['LAT', 'LONG', 'PRES', 'WND'])
   
   # Filtrage des valeurs aberrantes
   df = df[(df['LAT'] >= -90) & (df['LAT'] <= 90)]
   df = df[(df['LONG'] >= -180) & (df['LONG'] <= 180)]
   df = df[df['PRES'] > 0]
   df = df[df['WND'] >= 0]

2. Normalisation
^^^^^^^^^^^^^^^^

Deux stratégies de normalisation ont été testées :

**Normalisation globale (adoptée)**

.. code-block:: python

   from sklearn.preprocessing import MinMaxScaler
   
   scaler_global = MinMaxScaler()
   features_normalized = scaler_global.fit_transform(df[['LAT', 'LONG', 'PRES', 'WND']])

**Normalisation par trajectoire**

.. code-block:: python

   # Normalisation individuelle pour chaque ouragan
   for hurricane_id in df['hurricane_id'].unique():
       hurricane_data = df[df['hurricane_id'] == hurricane_id]
       scaler_local = MinMaxScaler()
       normalized_data = scaler_local.fit_transform(hurricane_data[features])

**Résultats de comparaison** :

* Normalisation globale : MSE = 36.2153
* Normalisation par trajectoire : MSE = 35.4074
* Amélioration : 2.23%

La normalisation globale a été retenue pour sa simplicité d'implémentation en production.

3. Création des séquences
^^^^^^^^^^^^^^^^^^^^^^^^^

Les données temporelles sont transformées en séquences pour l'entraînement des modèles :

.. code-block:: python

   def create_sequences(data, seq_length=5):
       """
       Crée des séquences temporelles pour l'entraînement
       """
       X, y = [], []
       for i in range(len(data) - seq_length):
           X.append(data[i:(i + seq_length)])
           y.append(data[i + seq_length])
       return np.array(X), np.array(y)

**Optimisation de la longueur de séquence** :

.. list-table:: Impact de la longueur de séquence
   :widths: 25 25 25 25
   :header-rows: 1

   * - Longueur
     - MSE
     - Ouragans ignorés
     - Recommandation
   * - 3
     - 45.2
     - 12
     - Trop court
   * - 5
     - 36.8
     - 24
     - **Optimal**
   * - 10
     - 52.1
     - 156
     - Perte de données
   * - 15
     - 73.6
     - 578
     - Non recommandé

4. Division train/validation/test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Division chronologique pour respecter la nature temporelle
   train_size = int(0.7 * len(data))
   val_size = int(0.15 * len(data))
   
   train_data = data[:train_size]
   val_data = data[train_size:train_size + val_size]
   test_data = data[train_size + val_size:]

Qualité des données
-------------------

Analyse de la qualité
^^^^^^^^^^^^^^^^^^^^^

* **Données manquantes** : < 1% du dataset total
* **Valeurs aberrantes** : 0.3% identifiées et supprimées
* **Cohérence temporelle** : 99.8% des séquences respectent l'ordre chronologique
* **Couverture géographique** : Océan Pacifique occidental principalement

Validation de la qualité
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Tests de validation des données
   def validate_data_quality(df):
       """
       Valide la qualité des données d'ouragan
       """
       checks = {
           'no_missing_values': df.isnull().sum().sum() == 0,
           'valid_coordinates': (df['LAT'].between(-90, 90)).all() and 
                              (df['LONG'].between(-180, 180)).all(),
           'positive_pressure': (df['PRES'] > 0).all(),
           'valid_wind_speed': (df['WND'] >= 0).all(),
           'temporal_consistency': is_temporally_ordered(df)
       }
       return checks

Utilisation des données
-----------------------

Les données prétraitées sont utilisées pour :

1. **Entraînement des modèles de trajectoire** : Prédiction de latitude/longitude
2. **Prédiction de durée** : Estimation de la durée de vie des ouragans
3. **Validation des modèles** : Tests de performance et généralisation
4. **Visualisation** : Création de cartes de trajectoires

Format de sortie
^^^^^^^^^^^^^^^^

Les données finales sont sauvegardées au format suivant :

.. code-block:: json

   {
     "sequences": [
       {
         "hurricane_id": "2023001",
         "sequence": [[lat1, lon1, pres1, wind1], ...],
         "target": [lat_next, lon_next],
         "metadata": {
           "start_date": "2023-05-15T00:00:00Z",
           "intensity_category": 4,
           "duration_hours": 96
         }
       }
     ]
   }