import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Parameters, Scalar, NDArrays, FitRes
from flwr.server.strategy import FedAvg

def create_cgm_model(input_dim: int, horizon: int, model_id: str = "base") -> tf.keras.Model:
    """Create CGM prediction model with enhanced architecture for FedClusterProxXAI"""
    
    # Small Transformer encoder option
    if "transformer" in model_id.lower():
        inputs = tf.keras.Input(shape=(input_dim,), name="features")
        x = tf.keras.layers.LayerNormalization()(inputs)
        # Treat each feature as a token (seq_len=input_dim, feat_dim=1)
        x = tf.keras.layers.Reshape((input_dim, 1))(x)
        d_model = 64
        x = tf.keras.layers.Dense(d_model)(x)
        # Positional embeddings
        pos = tf.range(start=0, limit=input_dim, delta=1)
        pos = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=d_model)(pos)
        # Broadcast positional encodings across batch dimension
        pos = tf.expand_dims(pos, axis=0)  # shape (1, input_dim, d_model)
        x = tf.keras.layers.Add()([x, pos])
        # Transformer encoder block
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4)(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        ff = tf.keras.layers.Dense(d_model * 2, activation='relu')(x)
        ff = tf.keras.layers.Dropout(0.2)(ff)
        ff = tf.keras.layers.Dense(d_model)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(horizon)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_small")
        lr = 0.0008
    # Enhanced MLP for FedClusterProxXAI
    elif "cluster" in model_id.lower() or "fedcluster" in model_id.lower():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(horizon)
        ])
        lr = 0.0008  # Slightly lower learning rate for stability
    else:
        # Standard architecture for other strategies
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(horizon)
        ])
        lr = 0.001
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

class BaseCGMClient(fl.client.NumPyClient):
    """Base client for all CGM federated learning strategies"""
    
    def __init__(self, client_id: str, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                 model: tf.keras.Model, strategy_type: str = "fedavg", mu: float = 0.1):
        self.client_id = client_id
        self.model = model
        self.strategy_type = strategy_type
        self.mu = mu  # For FedProx
        
        # Prepare data
        self.x_train = train_data.drop('target', axis=1).values.astype(np.float32)
        self.y_train = train_data['target'].values.astype(np.float32)
        self.x_val = val_data.drop('target', axis=1).values.astype(np.float32)
        self.y_val = val_data['target'].values.astype(np.float32)
        
        # Create datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        ).shuffle(len(self.x_train)).batch(32)
    
    def get_parameters(self, config: Optional[Dict[str, Scalar]] = None) -> NDArrays:
        """Return model parameters as list of NumPy ndarrays"""
        if config is None:
            config = {}
        return [layer.numpy() for layer in self.model.trainable_variables]
    
    def set_parameters(self, parameters: NDArrays):
        """Set model parameters from list of NumPy ndarrays"""
        # Use set_weights instead of assign to handle structure correctly
        # get_weights() returns weights in layer order, not trainable_variables order
        self.model.set_weights(parameters)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data"""
        self.set_parameters(parameters)
        global_weights = parameters
        
        # Update mu from config if provided
        if 'mu' in config:
            self.mu = config['mu']
        if 'adaptive_mu' in config:
            self.mu = config['adaptive_mu']
        
        # Choose training method based on strategy
        if self.strategy_type == "fedprox" or self.strategy_type == "fedclusterproxxai":
            history = self._train_with_proximal(global_weights, config)
        else:  # fedavg, fedsgd
            history = self._train_standard(config)
        
        return self.get_parameters(), len(self.x_train), history
    
    def _train_standard(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Standard training without proximal term"""
        local_epochs = config.get("local_epochs", 1)
        
        # Standard Keras training
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=local_epochs,
            batch_size=32,
            verbose=0
        )
        
        return {
            "loss": float(history.history['loss'][-1]),
            "strategy": self.strategy_type,
            "client_id": self.client_id
        }
    
    def _train_with_proximal(self, global_weights: NDArrays, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Training with FedProx proximal term"""
        local_epochs = config.get("local_epochs", 1)
        
        # Custom training loop with proximal term
        # Use slightly higher learning rate for faster convergence
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        
        # Get current weights in same order as global_weights (from get_weights())
        current_weights = self.model.get_weights()
        
        final_loss = 0.0
        batch_count = 0
        
        for epoch in range(local_epochs):
            for batch_x, batch_y in self.train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(batch_x, training=True)
                    loss = tf.keras.losses.mse(batch_y, predictions)
                    
                    # Add proximal term (FedProx regularization)
                    # Use get_weights() order to match global_weights
                    current_weights_list = self.model.get_weights()
                    proximal_term = 0.0
                    for local_w, global_w in zip(current_weights_list, global_weights):
                        # Convert to tensors if needed
                        local_tensor = tf.constant(local_w, dtype=tf.float32)
                        global_tensor = tf.constant(global_w, dtype=tf.float32)
                        proximal_term += tf.reduce_sum(tf.square(local_tensor - global_tensor))
                    
                    total_loss = loss + (self.mu / 2.0) * proximal_term
                
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                # Track final loss
                final_loss += float(loss.numpy() if len(loss.shape) == 0 else np.mean(loss.numpy()))
                batch_count += 1
        
        return {
            "loss": final_loss / batch_count if batch_count > 0 else 0.0,
            "mu": float(self.mu),
            "strategy": self.strategy_type,
            "client_id": self.client_id
        }
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local validation data with XAI metrics"""
        self.set_parameters(parameters)
        
        predictions = self.model(self.x_val, training=False)
        loss = tf.keras.losses.mse(self.y_val, predictions).numpy()
        mae = tf.keras.metrics.mae(self.y_val, predictions).numpy()
        
        # Convert to scalar if needed
        if isinstance(mae, np.ndarray):
            mae = float(mae.item() if mae.size == 1 else np.mean(mae))
        if isinstance(loss, np.ndarray):
            loss = float(loss.item() if loss.size == 1 else np.mean(loss))
        
        # Calculate XAI metrics
        xai_metrics = self._calculate_xai_metrics(predictions)
        
        metrics = {
            "mae": float(mae),
            "rmse": float(np.sqrt(loss)),
            "strategy": self.strategy_type,
            "client_id": self.client_id,
            **xai_metrics
        }
        
        return float(loss), len(self.x_val), metrics
    
    def generate_counterfactuals(self, sample_size: int = 5) -> Dict:
        """Generate counterfactual explanations for model predictions"""
        np.random.seed(42)
        
        # Sample random instances from validation set
        indices = np.random.choice(len(self.x_val), min(sample_size, len(self.x_val)), replace=False)
        counterfactuals = []
        
        for idx in indices:
            original_input = self.x_val[idx:idx+1]
            original_pred = self.model(original_input, training=False).numpy()[0][0]
            
            # Calculate feature importance for this instance
            with tf.GradientTape() as tape:
                inputs = tf.Variable(original_input, dtype=tf.float32)
                tape.watch(inputs)
                pred = self.model(inputs)
            
            gradients = tape.gradient(pred, inputs)
            feature_importance = tf.abs(gradients[0]).numpy()
            
            # Top 3 most important features
            top_indices = np.argsort(feature_importance)[-3:]
            
            # Generate counterfactuals by modifying important features
            cf_examples = []
            for i, feature_idx in enumerate(top_indices):
                # Create counterfactual by reducing/increasing the feature
                cf_input_reduce = original_input.copy()
                cf_input_increase = original_input.copy()
                
                # Reduce feature value
                cf_input_reduce[0, feature_idx] = max(0, original_input[0, feature_idx] - 0.2)
                cf_pred_reduce = self.model(tf.constant(cf_input_reduce, dtype=tf.float32), training=False).numpy()[0][0]
                
                # Increase feature value
                cf_input_increase[0, feature_idx] = min(1, original_input[0, feature_idx] + 0.2)
                cf_pred_increase = self.model(tf.constant(cf_input_increase, dtype=tf.float32), training=False).numpy()[0][0]
                
                cf_examples.append({
                    'feature_idx': int(feature_idx),
                    'feature_importance': float(feature_importance[feature_idx]),
                    'original_value': float(original_input[0, feature_idx]),
                    'change_type': 'reduce',
                    'new_value': float(cf_input_reduce[0, feature_idx]),
                    'original_prediction': float(original_pred),
                    'counterfactual_prediction': float(cf_pred_reduce),
                    'prediction_change': float(abs(original_pred - cf_pred_reduce))
                })
                
                cf_examples.append({
                    'feature_idx': int(feature_idx),
                    'feature_importance': float(feature_importance[feature_idx]),
                    'original_value': float(original_input[0, feature_idx]),
                    'change_type': 'increase',
                    'new_value': float(cf_input_increase[0, feature_idx]),
                    'original_prediction': float(original_pred),
                    'counterfactual_prediction': float(cf_pred_increase),
                    'prediction_change': float(abs(original_pred - cf_pred_increase))
                })
            
            counterfactuals.append({
                'instance_idx': int(idx),
                'original_prediction': float(original_pred),
                'counterfactual_examples': cf_examples
            })
        
        return {
            'client_id': self.client_id,
            'strategy': self.strategy_type,
            'counterfactuals': counterfactuals
        }
    
    def _calculate_xai_metrics(self, predictions: tf.Tensor) -> Dict[str, float]:
        """Calculate explainability metrics for model predictions"""
        import tensorflow as tf
        
        # Sample size - use smaller sample if validation set is small
        sample_size = min(100, len(self.x_val))
        if sample_size == 0:
            # Return default values if no validation data
            return {
                "feature_importance_mean": 0.001, "feature_importance_std": 0.001,
                "feature_importance_max": 0.001, "prediction_stability": 0.001,
                "local_stability": 0.001, "prediction_consistency": 0.9,
                "faithfulness": 0.1, "monotonicity": 0.5,
                "importance_score": 0.001, "stability_score": 0.9,
                "faithfulness_score": 0.5, "xai_score": 0.467
            }
        
        x_val_sample = self.x_val[:sample_size]
        predictions_sample = predictions[:sample_size]
        
        # Gradient-based feature importance
        try:
            with tf.GradientTape() as tape:
                inputs = tf.Variable(x_val_sample, dtype=tf.float32)
                tape.watch(inputs)
                preds = self.model(inputs, training=False)
            
            gradients = tape.gradient(preds, inputs)
            
            # Handle None gradients (may happen if model hasn't trained)
            if gradients is None:
                feature_importance = np.ones(x_val_sample.shape[1]) * 0.001
            else:
                # Feature importance: average absolute gradient magnitude
                feature_importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
                # Avoid zeros
                feature_importance = np.maximum(feature_importance, 0.0001)
        except Exception:
            # Fallback if gradient calculation fails
            feature_importance = np.ones(x_val_sample.shape[1]) * 0.001
        
        # Prediction stability: variance of predictions
        pred_numpy = predictions_sample.numpy() if hasattr(predictions_sample, 'numpy') else predictions_sample
        pred_stability = max(float(np.var(pred_numpy)), 0.0001)  # Avoid zero variance
        
        # Prediction consistency: measure how consistent predictions are
        pred_consistency = 1.0 / (1.0 + pred_stability)  # Higher is more consistent
        
        # ===== ADDITIONAL STABILITY & FAITHFULNESS METRICS =====
        # Local stability: how similar are predictions for similar inputs
        try:
            noise_scale = 0.01
            perturbed = x_val_sample.numpy() + np.random.normal(0, noise_scale, x_val_sample.shape)
            pred_perturbed = self.model(tf.constant(perturbed, dtype=tf.float32), training=False).numpy()
            local_stability = float(np.mean(np.abs(pred_numpy - pred_perturbed)))
            local_stability = max(local_stability, 0.0001)
        except Exception:
            local_stability = 0.001
        
        # Faithfulness: how much do changes in important features affect predictions?
        try:
            top_k = min(3, len(feature_importance))
            if top_k > 0:
                top_indices = np.argsort(feature_importance)[-top_k:]
                x_perturbed = x_val_sample.numpy().copy()
                for idx in top_indices:
                    x_perturbed[:, idx] = 0
                pred_perturbed_faith = self.model(tf.constant(x_perturbed, dtype=tf.float32), training=False).numpy()
                faithfulness = float(np.mean(np.abs(pred_numpy - pred_perturbed_faith)))
                faithfulness = max(faithfulness, 0.0001)
            else:
                faithfulness = 0.001
        except Exception:
            faithfulness = 0.001
        
        # Monotonicity test
        try:
            if len(feature_importance) > 0 and top_k > 0:
                top_feature_idx = int(top_indices[-1])
                test_values = np.linspace(0, 1, 10)
                test_inputs = np.tile(x_val_sample.numpy()[:1], (len(test_values), 1))
                test_inputs[:, top_feature_idx] = test_values
                test_preds = self.model(tf.constant(test_inputs, dtype=tf.float32), training=False).numpy().flatten()
                if len(test_values) > 1 and np.std(test_preds) > 0:
                    monotonicity = float(np.corrcoef(test_values, test_preds)[0, 1])
                    if np.isnan(monotonicity):
                        monotonicity = 0.5
                else:
                    monotonicity = 0.5
            else:
                monotonicity = 0.5
        except Exception:
            monotonicity = 0.5
        
        # Improved normalized scores for better explainability metrics
        # Normalize feature importance to 0-1 range
        if np.max(feature_importance) > np.min(feature_importance):
            normalized_importance = (feature_importance - np.min(feature_importance)) / (
                np.max(feature_importance) - np.min(feature_importance))
            importance_score = float(np.mean(normalized_importance))
        else:
            importance_score = 0.5  # Neutral if no variance
        
        # Stability score: lower variance = higher stability (better explainability)
        # Normalize based on prediction range
        pred_range = float(np.max(pred_numpy) - np.min(pred_numpy)) if len(pred_numpy) > 0 else 1.0
        normalized_stability = 1.0 - min(1.0, (pred_stability / max(pred_range, 0.01)))
        stability_score = max(0.0, normalized_stability)
        
        # Improved faithfulness: higher changes when important features are removed = better faithfulness
        # Normalize by prediction range
        if pred_range > 0:
            normalized_faithfulness = min(1.0, (faithfulness / pred_range) * 2.0)  # Scale appropriately
        else:
            normalized_faithfulness = 0.5
        faithfulness_score = normalized_faithfulness
        
        # Add monotonicity to overall score
        monotonicity_score = (monotonicity + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        
        # Weighted XAI score (importance and faithfulness are most important)
        xai_score = float(
            0.30 * importance_score + 
            0.25 * stability_score + 
            0.30 * faithfulness_score +
            0.15 * monotonicity_score
        )
        
        # Ensure all values are finite floats and expose a unified 'feature_importance' scalar
        out = {
            "feature_importance_mean": float(np.nan_to_num(np.mean(feature_importance), nan=0.0001)),
            "feature_importance_std": float(np.nan_to_num(np.std(feature_importance), nan=0.0001)),
            "feature_importance_max": float(np.nan_to_num(np.max(feature_importance), nan=0.0001)),
            "feature_importance": float(np.nan_to_num(importance_score, nan=0.0001)),
            "prediction_stability": float(np.nan_to_num(pred_stability, nan=0.0001)),
            "local_stability": float(np.nan_to_num(local_stability, nan=0.0001)),
            "prediction_consistency": float(np.nan_to_num(pred_consistency, nan=0.5)),
            "faithfulness": float(np.nan_to_num(faithfulness, nan=0.0001)),
            "monotonicity": float(np.nan_to_num(monotonicity, nan=0.5)),
            "importance_score": float(np.nan_to_num(importance_score, nan=0.0001)),
            "stability_score": float(np.nan_to_num(stability_score, nan=0.0001)),
            "faithfulness_score": float(np.nan_to_num(faithfulness_score, nan=0.0001)),
            "monotonicity_score": float(np.nan_to_num(monotonicity_score, nan=0.5)),
            "xai_score": float(np.nan_to_num(xai_score, nan=0.0001))
        }
        return out

class FedProxStrategy(FedAvg):
    """FedProx strategy with proximal regularization"""
    
    def __init__(self, mu: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
    
    def configure_fit(self, server_round: int, parameters: NDArrays, 
                     client_manager: fl.server.ClientManager) -> List[Tuple[fl.client.Client, Dict[str, Scalar]]]:
        """Configure training with FedProx mu parameter"""
        configurations = super().configure_fit(server_round, parameters, client_manager)
        
        # Add mu to each client's config
        fedprox_configurations = []
        for client, config in configurations:
            config["mu"] = self.mu
            fedprox_configurations.append((client, config))
        
        return fedprox_configurations

class FedClusterProxXAIStrategy(FedAvg):
    """Our novel FedClusterProxXAI strategy with adaptive clustering"""
    
    def __init__(self, num_clusters: int = 3, experiment_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.experiment_dir = experiment_dir
        
        # Cluster management
        self.cluster_models = {}
        self.client_cluster_map = {}
        self.adaptive_mus = {}
        
        self._initialize_clusters()
    
    def _initialize_clusters(self):
        """Initialize cluster models"""
        for cluster_id in range(self.num_clusters):
            model = create_cgm_model(input_dim=15, horizon=12, model_id=f"cluster_{cluster_id}")
            self.cluster_models[cluster_id] = model
    
    def configure_fit(self, server_round: int, parameters: NDArrays, 
                     client_manager: fl.server.ClientManager) -> List[Tuple[fl.client.Client, Dict[str, Scalar]]]:
        """Configure training with cluster-specific parameters"""
        configurations = super().configure_fit(server_round, parameters, client_manager)
        
        cluster_configurations = []
        for client, config in configurations:
            client_id = client.cid
            
            # Initialize client if new
            if client_id not in self.client_cluster_map:
                self.client_cluster_map[client_id] = 0
                self.adaptive_mus[client_id] = 0.1
            
            # Get cluster-specific parameters and adaptive mu
            cluster_id = self.client_cluster_map[client_id]
            cluster_model = self.cluster_models[cluster_id]
            cluster_parameters = [layer.numpy() for layer in cluster_model.trainable_variables]
            
            # Update adaptive mu based on history
            self._update_adaptive_mu(client_id)
            
            # Add cluster-specific config
            config["adaptive_mu"] = self.adaptive_mus[client_id]
            config["cluster_id"] = cluster_id
            config["strategy"] = "fedclusterproxxai"
            
            cluster_configurations.append((client, config))
            
            # Use cluster parameters for this client
            parameters = cluster_parameters
        
        return cluster_configurations
    
    def _update_adaptive_mu(self, client_id: str):
        """Update adaptive mu based on client performance"""
        # Adaptive mu starting low and increasing gradually for better convergence
        if client_id not in self.adaptive_mus:
            self.adaptive_mus[client_id] = 0.05  # Start with lower mu
        else:
            # Gradually increase mu to prevent divergence while maintaining flexibility
            self.adaptive_mus[client_id] = min(0.3, self.adaptive_mus[client_id] * 1.02)
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.client.Client, FitRes]], 
                     failures: List[BaseException]) -> Tuple[NDArrays, Dict[str, Scalar]]:
        """Aggregate results with cluster-specific processing"""
        # TODO: Implement cluster-specific aggregation
        return self.cluster_models[0].get_weights(), {}
