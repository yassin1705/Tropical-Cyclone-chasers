
Normalization Strategies
========================

Normalization Strategy
We compared two primary normalization approaches:
Per-Trajectory Normalization
Advantages:

Values are consistently scaled within a local interval, facilitating learning
Variations within each trajectory are well-highlighted

Disadvantages:

During inference with an incomplete trajectory, final min/max values are unknown
Different scales between trajectories may hinder model generalization

Global Normalization
Advantages:

Consistent parameters (min, max, μ, σ) used in both training and inference
Better stability during deployment

Disadvantages:

Small-amplitude trajectories are "flattened" within a large interval
Model may take longer to converge if amplitudes vary significantly

Conclusion: Our testing showed that per-trajectory normalization yielded superior results for this specific prediction task.
