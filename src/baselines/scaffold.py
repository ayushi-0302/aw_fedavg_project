# src/baselines/aw_fedavg.py
import numpy as np

class AdaptiveWeightedFedAvg:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha  # weight for reliability
        self.beta = beta    # weight for data heterogeneity

    def aggregate(self, client_updates, reliabilities, heterogeneities):
        """
        Args:
            client_updates: list of numpy arrays (model weights or gradients)
            reliabilities: list of float, reliability of each client
            heterogeneities: list of float, data heterogeneity score for each client
        Returns:
            aggregated_model: numpy array
        """

        # Normalise factors
        reliabilities = np.array(reliabilities)
        heterogeneities = np.array(heterogeneities)
        reliabilities /= np.sum(reliabilities)
        heterogeneities /= np.sum(heterogeneities)

        # Combined adaptive weight
        combined_weights = self.alpha * reliabilities + self.beta * (1 - heterogeneities)
        combined_weights /= np.sum(combined_weights)

        # Weighted sum of updates
        aggregated_model = np.sum(
            [w * update for w, update in zip(combined_weights, client_updates)],
            axis=0
        )

        return aggregated_model
