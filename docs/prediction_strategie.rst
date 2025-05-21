Prediction Strategy
We evaluated two approaches for predicting cyclone movement:
Unified Model (Combined Latitude/Longitude)

Captures spatial correlations between coordinates
Allows for more coherent joint prediction
Can be more complex to train
May suffer from difficult-to-dissociate non-linear interactions

Separate Models (Individual Latitude and Longitude)

Each model focuses on a single axis of variation
Simplifies architecture and accelerates training
Risk of losing joint dependence between latitude and longitude
May introduce spatial inconsistencies
