AI-Tornado-Chaser Documentation
============================

Welcome to the AI-Tornado-Chaser documentation. This comprehensive system combines advanced machine learning with intelligent assistance for natural disaster management, specifically focusing on tornado trajectory prediction and safety guidance.

Project Overview
-------------

AI-Tornado-Chaser is an advanced system that:

* Predicts tornado trajectories using deep learning
* Provides real-time safety guidance through an intelligent chatbot
* Analyzes historical data from 2,489 tornadoes (1949-2023)
* Implements state-of-the-art machine learning models
* Offers REST API for real-time predictions

Key Features
----------

1. **Trajectory Prediction**
   * BiLSTM-based sequence prediction
   * Multi-step forecasting
   * High accuracy positioning

2. **Intelligent Chatbot**
   * NLP-powered responses
   * Real-time safety advice
   * Context-aware assistance

3. **Data Analysis**
   * Historical pattern recognition
   * Trend analysis
   * Interactive visualizations

4. **API Integration**
   * RESTful endpoints
   * Real-time predictions
   * Easy integration options

Caractéristiques principales
----------------------------

* **Prédiction de trajectoires** : Modèles d'apprentissage profond pour prédire les coordonnées futures des ouragans
* **Prédiction de durée** : Estimation de la durée de vie des ouragans basée sur leurs caractéristiques initiales
* **Chatbot d'assistance** : Système intelligent fournissant des conseils de sécurité en temps réel
* **Analyse comparative** : Évaluation de multiples architectures de modèles (LSTM, BiLSTM, CNN+LSTM, etc.)
* **Optimisation avancée** : Tests de différents optimiseurs, tailles de batch, et stratégies de normalisation

Technologies utilisées
----------------------

* **Apprentissage automatique** : TensorFlow/Keras, XGBoost
* **Traitement du langage naturel** : SentenceTransformers, FAISS
* **Visualisation** : Matplotlib, visualisations de trajectoires
* **Source de données** : CMA Tropical Cyclone Data Center

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   installation
   quickstart
   api
   models
   data_processing
   visualization
   architecture
   deploiement
   notebooks
   chatbot
   depannage
   contributing
   changelog

Indices et tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`