Développement des Modèles
=========================

Cette section détaille le développement, l'entraînement et l'optimisation des modèles d'apprentissage automatique utilisés dans AI Tornado Chaser.

Vue d'ensemble des modèles
---------------------------

Le système utilise deux types principaux de modèles :

1. **Modèles de prédiction de trajectoire** : BiLSTM, LSTM, GRU, CNN+LSTM
2. **Modèle de prédiction de durée** : XGBoost Regressor

Chaque modèle a été rigoureusement testé et optimisé pour obtenir les meilleures performances.

Benchmark des optimiseurs
--------------------------

L'optimiseur joue un rôle crucial dans la convergence et les performances du modèle. Une évaluation comparative a été menée sur plusieurs optimiseurs.

Optimiseurs testés
^^^^^^^^^^^^^^^^^^

.. list-table:: Comparaison des optimiseurs
   :widths: 20 25 25 30
   :header-rows: 1

   * - Optimiseur
     - MAE (échelle réelle)
     - MSE (échelle réelle)
     - Statut
   * - **Adam**
     - **4.2693**
     - **41.3673**
     - **Retenu**
   * - RMSprop
     - 9.0301
     - 154.0797
     - Écarté
   * - Adagrad
     - 25.8452
     - 1275.2479
     - Performance faible
   * - Nadam
     - 4.9257
     - 46.6625
     - Bon mais inférieur à Adam
   * - Ftrl
     - 155.6094
     - 34883.6641
     - Performance très faible
   * - SGD
     - 26.5162
     - 1355.8380
     - Convergence lente

**Conclusion** : L'optimiseur **Adam** a été sélectionné pour tous les modèles grâce à ses performances supérieures avec une MAE de 4.27 et une MSE de 41.37.

Stratégies de prédiction
-------------------------

Prédiction conjointe vs séparée
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Une analyse comparative a été menée pour déterminer s'il est préférable de prédire latitude et longitude conjointement ou séparément.

**Approche conjointe** :
* Un seul modèle avec deux sorties (latitude, longitude)
* Capture les corrélations spatiales
* Architecture plus complexe

**Approche séparée** :
* Deux modèles distincts
* Simplification de l'architecture
* Risque de perte de cohérence spatiale

.. code-block:: python

   # Modèle conjoint
   class JointTrajectoryModel(nn.Module):
       def __init__(self, input_size, hidden_size):
           super().__init__()
           self.bilstm = nn.LSTM(input_size, hidden_size, 
                                bidirectional=True, batch_first=True)
           self.fc = nn.Linear(hidden_size * 2, 2)  # 2 sorties: lat, lon
       
       def forward(self, x):
           lstm_out, _ = self.bilstm(x)
           return self.fc(lstm_out[:, -1, :])  # [lat, lon]

**Résultats** :

.. list-table:: Comparaison des approches
   :widths: 30 35 35
   :header-rows: 1

   * - Métrique
     - Modèles séparés
     - Modèle conjoint
   * - MAE Latitude
     - 5.2
     - **4.8**
   * - MAE Longitude
     - 6.1
     - **5.3**
   * - Cohérence spatiale
     - Faible
     - **Élevée**
   * - Temps d'entraînement
     - 2x plus long
     - **Standard**

**Conclusion** : Le modèle conjoint a été retenu pour capturer les corrélations spatiales et améliorer la cohérence des prédictions.

Benchmark des architectures de modèles
---------------------------------------

Six architectures différentes ont été évaluées dans des conditions identiques.

Paramètres d'entraînement communs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Longueur de séquence** : 5
* **Nombre d'époques** : 10
* **Taille de batch** : 16
* **Seed aléatoire** : 42 (pour la reproductibilité)

Architectures testées
^^^^^^^^^^^^^^^^^^^^^

**1. LSTM (Long Short-Term Memory)**

.. code-block:: python

   def build_lstm_model(input_shape):
       model = Sequential([
           LSTM(64, return_sequences=False, input_shape=input_shape),
           Dropout(0.2),
           Dense(32, activation='relu'),
           Dense(2)  # lat, lon
       ])
       return model

**2. GRU (Gated Recurrent Unit)**

.. code-block:: python

   def build_gru_model(input_shape):
       model = Sequential([
           GRU(64, return_sequences=False, input_shape=input_shape),
           Dropout(0.2),
           Dense(32, activation='relu'),
           Dense(2)
       ])
       return model

**3. BiLSTM (Bidirectional LSTM)**

.. code-block:: python

   def build_bilstm_model(input_shape):
       model = Sequential([
           Bidirectional(LSTM(32, return_sequences=False), input_shape=input_shape),
           Dropout(0.2),
           Dense(32, activation='relu'),
           Dense(2)
       ])
       return model

**4. CNN+LSTM**

.. code-block:: python

   def build_cnn_lstm_model(input_shape):
       model = Sequential([
           Conv1D(64, 3, activation='relu', input_shape=input_shape),
           MaxPooling1D(2),
           LSTM(32, return_sequences=False),
           Dropout(0.2),
           Dense(32, activation='relu'),
           Dense(2)
       ])
       return model

**5. MLP (Multi-Layer Perceptron)**

.. code-block:: python

   def build_mlp_model(input_shape):
       model = Sequential([
           Flatten(input_shape=input_shape),
           Dense(128, activation='relu'),
           Dropout(0.2),
           Dense(64, activation='relu'),
           Dropout(0.2),
           Dense(32, activation='relu'),
           Dense(2)
       ])
       return model

**6. Conv1D**

.. code-block:: python

   def build_conv1d_model(input_shape):
       model = Sequential([
           Conv1D(64, 3, activation='relu', input_shape=input_shape),
           Conv1D(32, 3, activation='relu'),
           GlobalMaxPooling1D(),
           Dense(32, activation='relu'),
           Dense(2)
       ])
       return model

Résultats du benchmark
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Performance des modèles (MSE)
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Modèle
     - MSE
     - MAE
     - Temps (min)
     - Rang
   * - **BiLSTM**
     - **67.47**
     - **5.48**
     - **5**
     - **1**
   * - LSTM
     - 89.32
     - 6.21
     - 4
     - 2
   * - GRU
     - 95.18
     - 6.87
     - 4
     - 3
   * - CNN+LSTM
     - 124.56
     - 8.12
     - 6
     - 4
   * - Conv1D
     - 178.92
     - 9.45
     - 3
     - 5
   * - MLP
     - 756.23
     - 18.67
     - 2
     - 6

**Analyse des résultats** :

* **BiLSTM** : Meilleure performance grâce à la capacité bidirectionnelle
* **MLP** : Performance la plus faible, confirmant l'importance de la nature séquentielle
* **CNN+LSTM** : Bon compromis mais plus complexe à entraîner

Optimisation de l'architecture
-------------------------------

Profondeur du modèle
^^^^^^^^^^^^^^^^^^^^

Une comparaison entre modèle léger et modèle profond a été effectuée.

**Modèle léger** :

.. code-block:: python

   model_light = Sequential([
       Bidirectional(LSTM(32)),
       Dense(16, activation='relu'),
       Dense(2)
   ])

**Performance** : MAE = 5.4763, MSE = 67.4690

**Modèle profond** :

.. code-block:: python

   model_deep = Sequential([
       Bidirectional(LSTM(64, return_sequences=True)),
       Dropout(0.3),
       Bidirectional(LSTM(32, return_sequences=True)),
       Dropout(0.3),
       Bidirectional(LSTM(16)),
       Dropout(0.2),
       Dense(64, activation='relu'),
       Dropout(0.2),
       Dense(32, activation='relu'),
       Dense(2)
   ])

**Performance initiale** : MAE = 26.0142, MSE = 1118.1473 (surapprentissage)
**Après suppression dropout** : MAE = 8.3381, MSE = 136.6769

**Conclusion** : Le modèle léger est plus efficace, avec une dégradation de 102.63% pour le modèle profond et un temps d'exécution 2.6x plus long.

Fonctions de perte
------------------

Comparaison MSE vs Distance Euclidienne
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**MSE (Mean Squared Error)** :

.. code-block:: python

   def mse_loss(y_true, y_pred):
       return tf.reduce_mean(tf.square(y_true - y_pred))

**Distance Euclidienne** :

.. code-block:: python

   def euclidean_loss(y_true, y_pred):
       return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1)))

**Résultats** :

* **MSE** : 67.4690
* **Distance Euclidienne** : 65.3996
* **Amélioration** : 3.07%

La fonction de perte euclidienne a été adoptée pour sa meilleure performance géospatiale.

Optimisation des hyperparamètres
---------------------------------

Taille de batch
^^^^^^^^^^^^^^^

.. list-table:: Impact de la taille de batch
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Batch Size
     - MSE (MSE Loss)
     - MSE (Euclidean)
     - Temps (min)
     - Recommandation
   * - **8**
     - **35.21**
     - **34.89**
     - **6**
     - **Optimal**
   * - 16
     - 67.47
     - 65.40
     - 5
     - Bon
   * - 32
     - 98.34
     - 89.23
     - 4
     - Acceptable
   * - 64
     - 156.78
     - 134.56
     - 3
     - Non recommandé

**Observations** :
* Batch size 8 : Meilleure précision mais temps légèrement plus long
* Performance dégradée avec l'augmentation du batch size
* Compromis optimal entre précision et temps d'entraînement

Longueur de séquence
^^^^^^^^^^^^^^^^^^^^

.. list-table:: Impact de la longueur de séquence
   :widths: 25 20 25 30
   :header-rows: 1

   * - Séq. Length
     - MSE
     - Ouragans ignorés
     - Impact
   * - 3
     - 45.2
     - 12
     - Trop court, perte d'information
   * - **5**
     - **36.8**
     - **24**
     - **Équilibre optimal**
   * - 10
     - 52.1
     - 156
     - Perte significative de données
   * - 15
     - 73.6
     - 578
     - Non viable (23% des données perdues)

**Conclusion** : Longueur de séquence = 5 retenue comme optimale.

Stratégies de normalisation
---------------------------

Deux approches de normalisation ont été comparées :

Normalisation globale
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sklearn.preprocessing import MinMaxScaler

   # Une seule normalisation pour tout le dataset
   scaler_global = MinMaxScaler()
   data_normalized = scaler_global.fit_transform(full_dataset)

**Avantages** :
* Simplicité d'implémentation
* Cohérence entre entraînement et inférence
* Aucun stockage de scalers multiples

Normalisation par trajectoire
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Normalisation individuelle pour chaque ouragan
   scalers = {}
   for hurricane_id in unique_hurricanes:
       scaler = MinMaxScaler()
       hurricane_data = data[data['id'] == hurricane_id]
       scalers[hurricane_id] = scaler.fit(hurricane_data)

**Avantages** :
* Préservation des variations locales
* Meilleure adaptation aux trajectoires atypiques

**Inconvénients** :
* Complexité de stockage et de gestion
* Difficulté pour les nouvelles trajectoires

**Résultats comparatifs** :

* **Normalisation globale** : MSE = 36.2153
* **Normalisation par trajectoire** : MSE = 35.4074
* **Amélioration** : 2.23%

**Verdict** : Normalisation globale retenue pour sa simplicité malgré la légère perte de performance.

Analyse des courbes d'apprentissage
-----------------------------------

L'analyse des courbes d'apprentissage révèle les phases d'entraînement :

Phases d'apprentissage
^^^^^^^^^^^^^^^^^^^^^^

**Phase initiale (0-20 époques)** :
* Chute rapide de l'erreur d'entraînement
* Diminution de l'erreur de validation
* Apprentissage actif des patterns

**Phase de stabilisation (20-150 époques)** :
* Convergence lente vers l'asymptote
* Erreur d'entraînement : ~0.0025
* Erreur de validation : ~0.0038-0.0040

**Détection du surapprentissage** :
* Écart stable entre entraînement et validation
* Pas de divergence significative
* Arrêt optimal à 50 époques

.. code-block:: python

   from tensorflow.keras.callbacks import EarlyStopping

   early_stopping = EarlyStopping(
       monitor='val_loss',
       patience=10,
       restore_best_weights=True,
       verbose=1
   )

Modèle de prédiction de durée
-----------------------------

Architecture XGBoost
^^^^^^^^^^^^^^^^^^^^^

Pour la prédiction de la durée des ouragans, un modèle XGBoost a été développé.

**Variables d'entrée** :
* Intensité initiale
* Latitude de formation
* Longitude de formation
* Pression atmosphérique
* Vitesse des vents
* Mois de formation

.. code-block:: python

   import xgboost as xgb
   from sklearn.model_selection import GridSearchCV

   # Configuration du modèle
   xgb_params = {
       'n_estimators': [100, 200, 300],
       'max_depth': [3, 4, 6, 8],
       'learning_rate': [0.01, 0.1, 0.2],
       'subsample': [0.8, 0.9, 1.0],
       'colsample_bytree': [0.8, 0.9, 1.0]
   }

   # Grid search avec validation croisée
   grid_search = GridSearchCV(
       xgb.XGBRegressor(random_state=42),
       xgb_params,
       cv=5,
       scoring='neg_mean_squared_error',
       n_jobs=-1
   )

   grid_search.fit(X_train, y_train)

**Performance finale** :
* **RMSE** : 12.0442 heures
* **MAE** : 9.5165 heures
* **Erreur moyenne** : ~2.25 jours

**Correction de biais** : Un biais correctif de +2 jours (8 séquences) a été appliqué pour compenser la sous-estimation systématique.

Validation et métriques
-----------------------

Métriques d'évaluation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from sklearn.metrics import mean_absolute_error, mean_squared_error

   def evaluate_model(y_true, y_pred):
       """Évalue les performances du modèle"""
       mae = mean_absolute_error(y_true, y_pred)
       mse = mean_squared_error(y_true, y_pred)
       rmse = np.sqrt(mse)
       
       # Métriques géospatiales
       distances = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
       mean_distance_error = np.mean(distances)
       
       return {
           'MAE': mae,
           'MSE': mse,
           'RMSE': rmse,
           'Mean_Distance_Error': mean_distance_error
       }

**Résultats finaux du modèle optimal** :

* **Architecture** : BiLSTM
* **Optimiseur** : Adam
* **Fonction de perte** : Distance Euclidienne
* **Batch size** : 8
* **Longueur de séquence** : 5
* **MAE** : 5.48
* **MSE** : 67.47
* **Temps d'entraînement** : 5 minutes