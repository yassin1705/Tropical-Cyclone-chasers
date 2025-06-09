==============================
Intelligent Chatbot System
==============================

The intelligent chatbot module is designed to provide quick and relevant assistance to citizens during hurricane emergencies.

**Chatbot Architecture**
----------------------------
Le chatbot utilise un modèle NLP basé sur `all-MiniLM-L6-v2` pour encoder les questions des utilisateurs en vecteurs. Ces vecteurs sont comparés à un index pré-existant généré à partir d'une base de données de questions-réponses sur les ouragans, pour fournir une réponse pertinente et immédiate.

1. **Encodage NLP** : Utilisation de `SentenceTransformers` pour encoder la question en vecteur.
2. **Recherche avec FAISS** : `FAISS` est utilisé pour retrouver les questions les plus similaires dans l'index.
3. **Réponse en Temps Réel** : Le chatbot renvoie la réponse la plus proche en utilisant les données indexées.

**Exemple de Code**
-------------------

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   import faiss
   import numpy as np

   model = SentenceTransformer("all-MiniLM-L6-v2")
   question = "How to prepare for a hurricane?"
   embedding = model.encode([question])
   index = faiss.read_index("faq.index")
   _, I = index.search(np.array(embedding), k=1)
