======================
Analysis and Visualization
======================

The AI-Tornado-Chaser provides comprehensive tools for analyzing and visualizing tornado data and predictions.

Data Analysis Tools
-----------------

Exploratory Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

The project includes tools for analyzing tornado patterns:

.. code-block:: python

    # Analyze cyclone distribution
    df_unique_cyclone['month'] = df_unique_cyclone['datetime'].str[4:6].astype(int)
    monthly_distribution = df_unique_cyclone['month'].value_counts().sort_index()

Visualization Components
---------------------

Trajectory Visualization
~~~~~~~~~~~~~~~~~~~~~

1. Actual vs Predicted Paths:

   .. code-block:: python

       plt.scatter(y_test, predictions, alpha=0.6)
       plt.xlabel('Actual Position')
       plt.ylabel('Predicted Position')
       plt.title('Actual vs Predicted Tornado Positions')
       plt.grid(True)

2. Time Series Analysis:

   .. code-block:: python

       def plot_trajectory(actual, predicted, title="Tornado Trajectory"):
           plt.figure(figsize=(10, 6))
           plt.plot(actual[:, 0], actual[:, 1], 'b-', label='Actual Path')
           plt.plot(predicted[:, 0], predicted[:, 1], 'r--', label='Predicted Path')
           plt.xlabel('Longitude')
           plt.ylabel('Latitude')
           plt.title(title)
           plt.legend()
           plt.grid(True)

Performance Metrics
----------------

Model Evaluation Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Loss Curves:

   .. code-block:: python

       def plot_training_history(history):
           plt.figure(figsize=(12, 4))
           
           plt.subplot(1, 2, 1)
           plt.plot(history.history['loss'], label='Training Loss')
           plt.plot(history.history['val_loss'], label='Validation Loss')
           plt.title('Model Loss')
           plt.xlabel('Epoch')
           plt.ylabel('Loss')
           plt.legend()
           
           plt.subplot(1, 2, 2)
           plt.plot(history.history['mae'], label='Training MAE')
           plt.plot(history.history['val_mae'], label='Validation MAE')
           plt.title('Model MAE')
           plt.xlabel('Epoch')
           plt.ylabel('MAE')
           plt.legend()

2. Error Distribution:

   .. code-block:: python

       def plot_error_distribution(y_true, y_pred):
           errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
           plt.figure(figsize=(10, 6))
           plt.hist(errors, bins=50)
           plt.xlabel('Prediction Error (km)')
           plt.ylabel('Frequency')
           plt.title('Distribution of Prediction Errors')

Interactive Visualizations
-----------------------

Jupyter Notebook Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

The project provides interactive visualizations in Jupyter notebooks:

1. Real-time prediction visualization
2. Model comparison plots
3. Parameter tuning analysis

Dashboard Components
-----------------

The visualization system includes dashboard elements for:

1. Real-time Monitoring
   
   * Current predictions
   * Model performance metrics
   * System status

2. Historical Analysis
   
   * Past trajectory comparisons
   * Error analysis
   * Trend visualization

Export Capabilities
----------------

The visualization system supports multiple export formats:

1. Static Formats
   
   * PNG
   * PDF
   * SVG

2. Interactive Formats
   
   * HTML
   * Jupyter notebooks
   * Interactive dashboards
