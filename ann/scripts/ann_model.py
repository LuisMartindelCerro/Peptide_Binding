##  ANN Model
import numpy as np
from scipy.stats import pearsonr
from utility import (
    encode as _encode,
    sigmoid as _sigmoid,
    sigmoid_prime as _sigmoid_prime,
    forward as _forward,
    back_prop as _back_prop,
    feed_forward_network as _feed_forward_network,
    save_network as _save_network,
    load_network as _load_network
)

class ANN:
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=1,
        learning_rate=0.01,
        epochs=100,
        early_stopping=False,
        random_state=None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize weights and X template once
        self.w_h, self.w_o, X_template = _feed_forward_network(
            self.input_dim, self.hidden_dim, self.output_dim
        )
        self._X_template = X_template

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the network on (x_train, y_train). Optionally use (x_val, y_val) for early stopping.
        """
        dj_dw_h = np.zeros_like(self.w_h)
        dj_dw_o = np.zeros_like(self.w_o)

        best_weights = (self.w_h.copy(), self.w_o.copy())
        best_val_err = np.inf

        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(len(x_train))
            X_shuf = x_train[perm]
            y_shuf = y_train[perm]

            for x_vec, y_true in zip(X_shuf, y_shuf):
                # Clone template state for each example
                X_state = self._X_template.copy()
                X_state[0, :self.input_dim] = x_vec
                X_state[0, self.input_dim] = 1.0  # bias

                _forward(X_state, self.w_h, self.w_o)
                _back_prop(X_state, y_true, self.w_h, self.w_o, dj_dw_h, dj_dw_o)

                self.w_h -= self.learning_rate * dj_dw_h
                self.w_o -= self.learning_rate * dj_dw_o

            if x_val is not None and y_val is not None and self.early_stopping:
                preds_val = self.predict(x_val)
                val_err = np.mean((preds_val - y_val)**2)
                if val_err < best_val_err:
                    best_val_err = val_err
                    best_weights = (self.w_h.copy(), self.w_o.copy())
                else:
                    self.w_h, self.w_o = best_weights
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self

    def predict(self, x):
        """
        Compute network outputs for input matrix x (shape [n_samples, input_dim]).
        Returns array of shape [n_samples,].
        """
        preds = []
        for x_vec in x:
            X_state = self._X_template.copy()
            X_state[0, :self.input_dim] = x_vec
            X_state[0, self.input_dim] = 1.0  # bias

            _forward(X_state, self.w_h, self.w_o)
            preds.append(X_state[2, 0])
        return np.array(preds)

    def score(self, x, y_true, metric='pcc'):
        """
        Compute performance metric on (x, y_true).
        Supported metrics: 'pcc' (Pearson), 'mse'.
        """
        preds = self.predict(x)
        if metric == 'pcc':
            return pearsonr(y_true, preds)[0]
        elif metric == 'mse':
            return np.mean((preds - y_true)**2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def save(self, filename, **metadata):
        """
        Save network weights and optional metadata using professor's loader.
        metadata keys: lpcc, lerr, tpcc, terr, epochs.
        """
        _save_network(
            filename,
            self.w_h,
            self.w_o,
            metadata.get('lpcc', 0),
            metadata.get('lerr', 0),
            metadata.get('tpcc', 0),
            metadata.get('terr', 0),
            metadata.get('epochs', self.epochs)
        )

    @classmethod
    def load(cls, filename, learning_rate=0.01, epochs=100,
             early_stopping=False, random_state=None):
        """
        Load a saved network from file and return an ANN instance.
        """
        w_h, w_o = _load_network(filename)
        input_dim = w_h.shape[0] - 1
        hidden_dim = w_h.shape[1]
        output_dim = w_o.shape[1]

        ann = cls(
            input_dim, hidden_dim, output_dim,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping=early_stopping,
            random_state=random_state
        )
        ann.w_h = w_h
        ann.w_o = w_o
        return ann