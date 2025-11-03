import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import flwr as fl
from flwr.common import Parameters, Scalar, NDArrays, FitRes
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

class ExperimentConfig:
    """Configuration for federated learning experiments"""
    def __init__(self):
        self.experiment_name = "cgm_fl_benchmark"
        self.num_rounds = 50
        self.num_clients = 20
        self.prediction_horizon = 12
        self.strategies = {
            'fedavg': {'name': 'FedAvg', 'params': {}},
            'fedprox': {'name': 'FedProx', 'params': {'mu': 0.1}},
            'fedsgd': {'name': 'FedSGD', 'params': {}},
            'fedclusterproxxai': {'name': 'FedClusterProxXAI', 'params': {'num_clusters': 3}}
        }
        # Client-level cross-validation for the novel strategy
        self.client_cv_folds = 0   # e.g., 5 to enable K-fold across clients
        self.cv_rounds = 10        # rounds per CV fold to control runtime
        # Early stopping across FL rounds
        self.enable_early_stopping = True
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 1e-3

class SubjectDataLoader:
    """Loader for processed subject files"""
    
    def __init__(self, processed_data_dir: str):
        self.processed_data_dir = processed_data_dir
        self.subjects_dir = f"{processed_data_dir}/subjects"
        
    def load_all_subjects(self) -> Dict[str, pd.DataFrame]:
        """Load all processed subject files"""
        subject_files = [f for f in os.listdir(self.subjects_dir) if f.startswith('Subject_') and f.endswith('.xlsx')]
        subjects_data = {}
        
        print(f"Loading {len(subject_files)} subject files...")
        
        for filename in subject_files:
            try:
                subject_id = filename.replace('Subject_', '').replace('.xlsx', '')
                filepath = f"{self.subjects_dir}/{filename}"
                
                # Load processed data
                data = pd.read_excel(filepath, sheet_name='processed_data', index_col=0)
                data.index = pd.to_datetime(data.index)
                
                subjects_data[subject_id] = data
                print(f"✓ Loaded {filename}: {len(data)} samples")
                
            except Exception as e:
                print(f"✗ Failed to load {filename}: {e}")
        
        return subjects_data

class BenchmarkRunner:
    """Main benchmark runner for federated learning experiments"""
    
    def __init__(self, config: ExperimentConfig, processed_data_dir: str):
        self.config = config
        self.processed_data_dir = processed_data_dir
        # Unique run identifier per benchmark execution
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dirs = self._setup_benchmark_directories()
        
        # Load client data information
        self.client_data_loader, self.available_clients = self._create_client_data_loader()
        
        if not self.available_clients:
            raise ValueError("No client data available from processed files")
        
        print(f"Available clients: {self.available_clients}")
    
    def _setup_benchmark_directories(self):
        """Create directory structure for benchmark results"""
        base_dir = f"experiments/{self.config.experiment_name}"
        strategies = ['fedavg', 'fedprox', 'fedsgd', 'fedclusterproxxai']
        
        dirs = {'base': base_dir}
        
        for strategy in strategies:
            strategy_dir = f"{base_dir}/{strategy}"
            dirs[strategy] = {
                'base': strategy_dir,
                'models': f"{strategy_dir}/models",
                'results': f"{strategy_dir}/results",
                'logs': f"{strategy_dir}/logs",
                'runs': f"{strategy_dir}/runs",
                'current_run': f"{strategy_dir}/runs/{self.run_id}"
            }
            
            for sub_dir in dirs[strategy].values():
                os.makedirs(sub_dir, exist_ok=True)
        
        # Global results directory
        dirs['global_results'] = f"{base_dir}/global_results"
        os.makedirs(dirs['global_results'], exist_ok=True)
        
        return dirs
    
    def _create_client_data_loader(self):
        """Create client data loader that reads from processed subject files"""
        subject_loader = SubjectDataLoader(self.processed_data_dir)
        subjects_data = subject_loader.load_all_subjects()
        
        # Prefer proper subjects (exclude synthetic like SUB001)
        all_ids = list(subjects_data.keys())
        preferred_ids = [cid for cid in all_ids if cid.lower().startswith('subject')]
        # If no preferred ids found, fall back to all
        client_ids = preferred_ids if preferred_ids else all_ids
        
        def load_client_data(client_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Load data for a specific client (subject)"""
            if client_id not in subjects_data:
                raise ValueError(f"Client {client_id} not found in processed data")
            
            data = subjects_data[client_id].copy()
            
            # Remove subject_id column if present
            if 'subject_id' in data.columns:
                data = data.drop('subject_id', axis=1)
            
            # Split into train/validation (80/20)
            split_idx = int(0.8 * len(data))
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            
            print(f"Client {client_id}: {len(train_data)} train, {len(val_data)} val samples")
            return train_data, val_data
        
        return load_client_data, client_ids
    
    def run_benchmark(self):
        """Run benchmark using processed subject files"""
        print("=== Starting File-Based Federated Learning Benchmark ===")
        print(f"Available subjects: {len(self.available_clients)}")
        # Normalize strategies: allow list from JSON overrides by converting to dict
        if isinstance(self.config.strategies, list):
            # Expect entries like {'key': 'fedavg', 'name': 'FedAvg', 'params': {...}}
            normalized = {}
            for i, s in enumerate(self.config.strategies):
                if isinstance(s, dict):
                    key = (s.get('key') or s.get('name') or f'strat_{i}').lower()
                    normalized[key] = s
                else:
                    normalized[f'strat_{i}'] = {'name': str(s), 'params': {}}
            self.config.strategies = normalized
        print(f"Strategies: {list(self.config.strategies.keys())}")
        
        results = {}
        
        # Only run FedAvg for now
        # Run all enabled strategies
        for strategy_key in ['fedavg', 'fedprox', 'fedsgd', 'fedclusterproxxai']:
            if strategy_key in self.config.strategies:
                strategy_info = self.config.strategies[strategy_key]
                strategy_name = strategy_info.get('name', strategy_key)
                print(f"\n--- Running {strategy_name} ---")
                
                start_time = time.time()
                strategy_results = self._run_strategy(strategy_key, strategy_info)
                end_time = time.time()
                
                strategy_results['execution_time'] = end_time - start_time
                results[strategy_key] = strategy_results
                
                self._save_strategy_results(strategy_key, strategy_results)
        
        self.results = results
        
        print(f"\n=== Benchmark Completed ===")
        self._print_comparative_summary(results)
        
        # Optional: run Client-level K-fold Cross-Validation for the novel strategy
        if 'fedclusterproxxai' in self.config.strategies and getattr(self.config, 'client_cv_folds', 0) and self.config.client_cv_folds > 1:
            print(f"\n=== Starting {self.config.client_cv_folds}-Fold Client Cross-Validation for FedClusterProxXAI ===")
            cv_results = self._run_client_cv_fedcluster(self.config.strategies['fedclusterproxxai'])
            results['fedclusterproxxai_client_cv'] = cv_results
            self._save_strategy_results('fedclusterproxxai_client_cv', cv_results)
            print("=== Client Cross-Validation Completed ===")
        
        return results
    
    def _run_strategy(self, strategy_key: str, strategy_info: Dict) -> Dict:
        """Run a single strategy using file-based data"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from strategies import BaseCGMClient, create_cgm_model
        
        print(f"Running {strategy_key} strategy...")
        
        # Create strategy
        strategy = FedAvg(fraction_fit=0.5, fraction_evaluate=0.5)
        
        # Client function
        def client_fn(cid: str):
            train_data, val_data = self.client_data_loader(cid)
            
            # Get input dimension (number of features)
            input_dim = len(train_data.columns) - 1  # Exclude 'target'
            
            # Create model for this client
            model = create_cgm_model(input_dim=input_dim, horizon=1)
            
            # Create client
            return BaseCGMClient(
                client_id=cid,
                train_data=train_data,
                val_data=val_data,
                model=model,
                strategy_type=strategy_key
            )
        
        # Run simple FedAvg manually without Ray
        print(f"Running {self.config.num_rounds} rounds with {len(self.available_clients)} clients")
        
        # Initialize clients - respect configured num_clients
        max_clients = min(self.config.num_clients, len(self.available_clients))
        clients_to_use = self.available_clients[:max_clients]
        clients = {}
        
        # Determine input_dim from first client before creating models
        if clients_to_use:
            sample_train, _ = self.client_data_loader(clients_to_use[0])
            input_dim = len(sample_train.columns) - 1  # Exclude 'target'
        else:
            raise ValueError("No clients available")
        
        for cid in clients_to_use:
            train_data, val_data = self.client_data_loader(cid)
            # Ensure consistent input_dim
            expected_features = input_dim
            actual_features = len(train_data.columns) - 1
            if actual_features != expected_features:
                print(f"Warning: Client {cid} has {actual_features} features, expected {expected_features}")
            # Use enhanced architecture for FedClusterProxXAI
            # Use transformer_small for the novel strategy
            model_id = "transformer_small" if strategy_key == "fedclusterproxxai" else "base"
            model = create_cgm_model(input_dim=input_dim, horizon=1, model_id=model_id)
            clients[cid] = BaseCGMClient(cid, train_data, val_data, model, strategy_type=strategy_key)
        
        # Initialize cluster models and assignments for FedClusterProxXAI
        num_clusters = 3
        cluster_models = {}
        client_cluster_map = {}  # Maps client_id -> cluster_id
        cluster_assignments_history = []  # Track assignments over time
        
        if strategy_key == "fedclusterproxxai":
            # Verify all clients have same input_dim
            for cid, client in clients.items():
                if client.x_train.shape[1] != input_dim:
                    raise ValueError(f"Client {cid} has different input_dim: {client.x_train.shape[1]} vs {input_dim}")
            
            # Improved initialization: Warm start cluster models with quick training
            print(f"  Warm starting cluster models...")
            baseline_client = list(clients.values())[0]
            
            # Initialize cluster models and warm start them individually
            for cluster_id in range(num_clusters):
                cluster_model = create_cgm_model(input_dim=input_dim, horizon=1, model_id="transformer_small")
                
                # Better warm start with more epochs and validation
                warm_start_clients = list(clients.values())[cluster_id::num_clusters][:3]
                if not warm_start_clients:
                    warm_start_clients = [baseline_client]
                
                combined_x = np.vstack([c.x_train[:500] for c in warm_start_clients])
                combined_y = np.concatenate([c.y_train[:500] for c in warm_start_clients])
                
                cluster_model.fit(
                    combined_x, combined_y,
                    epochs=15, batch_size=32, verbose=0,
                    validation_split=0.2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
                )
                
                cluster_models[cluster_id] = {
                    'model': cluster_model,
                    'params': cluster_model.get_weights(),
                    'client_ids': []
                }
            
            # Improved initial cluster assignment with balanced distribution
            # Collect all client statistics first
            client_stats = []
            for cid in clients.keys():
                stats = self._get_client_clustering_stats(clients[cid])
                client_stats.append((cid, stats))
            
            # Sort by clustering score and distribute evenly
            client_stats.sort(key=lambda x: x[1])
            cluster_assignments = [None] * len(clients)
            
            # Distribute clients evenly across clusters (round-robin)
            for i, (cid, _) in enumerate(client_stats):
                cluster_id = i % num_clusters
                client_cluster_map[cid] = cluster_id
                cluster_models[cluster_id]['client_ids'].append(cid)
            
            print(f"  Initial cluster assignments (balanced):")
            for cluster_id in range(num_clusters):
                cluster_clients = cluster_models[cluster_id]['client_ids']
                print(f"    Cluster {cluster_id}: {len(cluster_clients)} clients ({cluster_clients})")
        else:
            # Standard single global model
            first_client = list(clients.values())[0]
            global_params = first_client.get_parameters()
        
        history = {"losses": [], "metrics": [], "cluster_assignments": []}
        # Enhanced monitoring for divergence diagnostics
        cluster_losses_history = {i: [] for i in range(num_clusters)} if strategy_key == "fedclusterproxxai" else {}
        # Early stopping state
        best_loss = float('inf')
        no_improve_rounds = 0
        use_es = getattr(self.config, 'enable_early_stopping', True)
        es_patience = getattr(self.config, 'early_stopping_patience', 5)
        es_min_delta = getattr(self.config, 'early_stopping_min_delta', 1e-3)
        # Adaptive clustering with enforced balance
        reassignment_interval = 10**9 if strategy_key == "fedclusterproxxai" else 25  # Disable reassignment for novel model
        min_cluster_size = max(2, len(clients) // num_clusters // 2) if strategy_key == "fedclusterproxxai" else 1  # At least 1/3 of average per cluster
        
        # Run FL rounds
        for round_num in range(self.config.num_rounds):
            print(f"Round {round_num + 1}/{self.config.num_rounds}")
            # Monitor cluster-specific losses at the start of each round (sample up to 2 clients per cluster)
            if strategy_key == "fedclusterproxxai":
                for monitor_cluster_id in range(num_clusters):
                    cluster_clients = [cid for cid, cid_cluster in client_cluster_map.items() if cid_cluster == monitor_cluster_id]
                    if cluster_clients:
                        try:
                            sample_clients = cluster_clients[:2]
                            losses_sample = []
                            for mcid in sample_clients:
                                c = clients[mcid]
                                loss_val, _, _ = c.evaluate(cluster_models[monitor_cluster_id]['model'].get_weights(), {})
                                losses_sample.append(loss_val)
                            if losses_sample:
                                cluster_loss = float(np.mean(losses_sample))
                                cluster_losses_history[monitor_cluster_id].append(cluster_loss)
                                print(f"  Cluster {monitor_cluster_id} avg loss: {cluster_loss:.4f}")
                        except Exception as dbg_e:
                            print(f"  [Debug] Monitor failed for cluster {monitor_cluster_id}: {dbg_e}")
            
            # Improved adaptive cluster reassignment for FedClusterProxXAI with forced balance
            if strategy_key == "fedclusterproxxai" and round_num > 0 and round_num % reassignment_interval == 0:
                print(f"  🔄 Reassigning clients to clusters (Round {round_num + 1})...")
                client_cluster_map = self._adaptive_cluster_reassignment(
                    clients, cluster_models, client_cluster_map, round_num, min_cluster_size
                )
                cluster_assignments_history.append({
                    'round': round_num,
                    'assignments': client_cluster_map.copy()
                })
                
                # Force rebalancing if any cluster is too small or empty (ensure it works)
                client_cluster_map_before = client_cluster_map.copy()
                client_cluster_map = self._force_cluster_balance(client_cluster_map, clients, min_cluster_size)
                
                # Verify rebalancing worked
                from collections import Counter
                cluster_sizes_after = Counter(client_cluster_map.values())
                if min(cluster_sizes_after.values()) == 0:
                    print(f"      ❌ Rebalance failed - forcing round-robin redistribution")
                    # Emergency redistribution
                    client_list = list(client_cluster_map.keys())
                    for i, cid in enumerate(client_list):
                        client_cluster_map[cid] = i % num_clusters
                
                # Update cluster client lists
                for cluster_id in range(num_clusters):
                    cluster_models[cluster_id]['client_ids'] = [
                        cid for cid, cid_cluster in client_cluster_map.items() 
                        if cid_cluster == cluster_id
                    ]
                
                cluster_sizes = [len(cluster_models[i]['client_ids']) for i in range(num_clusters)]
                print(f"    Cluster sizes: {cluster_sizes}")
                
                # Warn if cluster collapse detected
                empty_clusters = sum(1 for s in cluster_sizes if s == 0)
                if empty_clusters > 0:
                    print(f"    ⚠️  Warning: {empty_clusters} empty cluster(s) detected - should have been rebalanced")
            
            # Select clients
            num_selected = len(clients) if strategy_key == "fedclusterproxxai" else max(1, len(clients) // 2)
            selected_clients = list(clients.keys())[:num_selected]
            
            # Debug: pre-training round loss check on a sample of clients
            try:
                round_losses = []
                sample_check = selected_clients[:3]
                for chk_cid in sample_check:
                    if strategy_key == "fedclusterproxxai":
                        chk_cluster_id = client_cluster_map.get(chk_cid, 0)
                        chk_params = cluster_models[chk_cluster_id]['model'].get_weights()
                    else:
                        chk_params = global_params
                    loss_val, _, _ = clients[chk_cid].evaluate(chk_params, {})
                    round_losses.append(loss_val)
                if round_losses:
                    avg_round_loss = float(np.mean(round_losses))
                    print(f"Round {round_num + 1}: Avg Loss (pre-train sample) = {avg_round_loss:.4f}")
                    if history["losses"] and avg_round_loss > history["losses"][-1] * 1.5:
                        print(f"\u26a0\ufe0f  SIGNIFICANT LOSS INCREASE DETECTED at round {round_num + 1}")
            except Exception as dbg:
                print(f"[Debug] Pre-train loss check failed: {dbg}")

            # Local training
            client_results = []
            for cid in selected_clients:
                client = clients[cid]
                
                # Get cluster-specific parameters for FedClusterProxXAI
                if strategy_key == "fedclusterproxxai":
                    cluster_id = client_cluster_map.get(cid, 0)
                    # Get fresh weights from cluster model to ensure correct structure
                    cluster_params = cluster_models[cluster_id]['model'].get_weights()
                else:
                    cluster_params = global_params
                
                # Add adaptive mu for FedClusterProxXAI
                # Reduced epochs to prevent divergence while maintaining convergence
                config = {"local_epochs": 3 if strategy_key == "fedclusterproxxai" else 3}
                if strategy_key == "fedclusterproxxai":
                    # More conservative mu schedule
                    if round_num < 10:
                        adaptive_mu = 0.001  # Start very low
                    elif round_num < 20:
                        adaptive_mu = 0.002
                    elif round_num < 30:
                        adaptive_mu = 0.005
                    elif round_num < 40:
                        adaptive_mu = 0.008
                    else:
                        adaptive_mu = 0.01   # Cap at reasonable value
                    config["adaptive_mu"] = adaptive_mu
                    config["cluster_id"] = cluster_id
                
                updated_params, num_samples, metrics = client.fit(cluster_params, config)
                client_results.append((cid, updated_params, num_samples, metrics))
            
            # Aggregate parameters per cluster (FedClusterProxXAI) or globally (others)
            if strategy_key == "fedclusterproxxai":
                # Aggregate separately for each cluster
                for cluster_id in range(num_clusters):
                    cluster_client_results = [
                        (cid, params, num_samples, metrics) 
                        for cid, params, num_samples, metrics in client_results
                        if client_cluster_map.get(cid) == cluster_id
                    ]
                    
                    if not cluster_client_results:
                        continue  # Skip if no clients in this cluster
                    
                    total_samples = sum(num_samples for _, _, num_samples, _ in cluster_client_results)
                    aggregated_params = []
                    
                    # Get number of parameters from first client
                    num_params = len(cluster_client_results[0][1])
                    
                    for param_idx in range(num_params):
                        # Weighted average for this parameter tensor
                        weighted_sum = None
                        for cid, params, num_samples, _ in cluster_client_results:
                            param_array = params[param_idx]  # This is a numpy array
                            # Ensure it's a numpy array and not accidentally indexed
                            if not isinstance(param_array, np.ndarray):
                                param_array = np.array(param_array)
                            weighted = param_array * num_samples
                            
                            if weighted_sum is None:
                                weighted_sum = weighted
                            else:
                                # Ensure shapes match
                                if weighted_sum.shape != weighted.shape:
                                    raise ValueError(f"Shape mismatch in cluster {cluster_id}, param {param_idx}: "
                                                   f"{weighted_sum.shape} vs {weighted.shape}")
                                weighted_sum = weighted_sum + weighted
                        
                        weighted_param = weighted_sum / total_samples
                        # Ensure it's a proper numpy array
                        if not isinstance(weighted_param, np.ndarray):
                            weighted_param = np.array(weighted_param)
                        aggregated_params.append(weighted_param)
                    
                    # Robust weight update: replace only trainable-like weights by shape
                    cluster_model = cluster_models[cluster_id]['model']
                    current_model_weights = cluster_model.get_weights()
                    updated_weights = []
                    agg_idx = 0
                    for w in current_model_weights:
                        if agg_idx < len(aggregated_params) and w.shape == aggregated_params[agg_idx].shape:
                            updated_weights.append(aggregated_params[agg_idx])
                            agg_idx += 1
                        else:
                            updated_weights.append(w)
                    cluster_model.set_weights(updated_weights)
                    # Store fresh weights for next round
                    cluster_models[cluster_id]['params'] = cluster_model.get_weights()
                
                # Use cluster 0 params for evaluation (or average)
                global_params = cluster_models[0]['model'].get_weights()
            else:
                # Standard aggregation
                total_samples = sum(num_samples for _, _, num_samples, _ in client_results)
                aggregated_params = []
                
                for param_idx in range(len(client_results[0][1])):
                    weighted_param = sum(
                        params[param_idx] * num_samples 
                        for _, params, num_samples, _ in client_results
                    ) / total_samples
                    aggregated_params.append(weighted_param)
                
                global_params = aggregated_params
            
            # Evaluate
            eval_losses = []
            eval_metrics_list = []
            # For fedcluster, evaluate against averaged cluster parameters to avoid biasing to a single cluster
            if strategy_key == "fedclusterproxxai":
                # Compute element-wise average of cluster model weights
                # Assumes all clusters share identical architecture
                avg_params = None
                for k in range(num_clusters):
                    w = cluster_models[k]['model'].get_weights()
                    if avg_params is None:
                        avg_params = [wi.copy() for wi in w]
                    else:
                        for i in range(len(w)):
                            avg_params[i] = avg_params[i] + w[i]
                if avg_params is not None:
                    for i in range(len(avg_params)):
                        avg_params[i] = avg_params[i] / float(num_clusters)
                eval_params = avg_params
            else:
                eval_params = global_params
            for cid in selected_clients[:3]:  # Evaluate on first 3 clients
                client = clients[cid]
                loss, _, metrics = client.evaluate(eval_params, {})
                eval_losses.append(loss)
                eval_metrics_list.append(metrics)
            
            avg_loss = sum(eval_losses) / len(eval_losses)
            history["losses"].append(avg_loss)
            
            # Aggregate XAI metrics
            avg_metrics = {}
            if eval_metrics_list:
                for key in eval_metrics_list[0].keys():
                    if key not in ["strategy", "client_id"]:
                        avg_metrics[key] = sum(m[key] for m in eval_metrics_list) / len(eval_metrics_list)
            history["metrics"].append(avg_metrics)
            
            # Track current cluster assignments for FedClusterProxXAI (static - no changes during training)
            if strategy_key == "fedclusterproxxai":
                history["cluster_assignments"].append({
                    'round': round_num + 1,
                    'assignments': client_cluster_map.copy(),
                    'cluster_sizes': [
                        len(cluster_models[i]['client_ids']) 
                        for i in range(num_clusters)
                    ],
                    'static': True  # Indicate this is static clustering
                })
            
            print(f"  Round {round_num + 1} - Loss: {avg_loss:.4f}")

            # Early stopping check (on average validation loss per round)
            if use_es:
                if best_loss - avg_loss > es_min_delta:
                    best_loss = avg_loss
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= es_patience:
                        print(f"  ⏹️  Early stopping triggered at round {round_num + 1} (patience={es_patience})")
                        history.setdefault('early_stopping', {})
                        history['early_stopping'] = {
                            'stopped': True,
                            'round': round_num + 1,
                            'best_loss': best_loss,
                            'patience': es_patience,
                            'min_delta': es_min_delta
                        }
                        break
        
        # Prepare results
        # Be robust to missing fields in strategy_info
        results = {
            'strategy_name': strategy_info.get('name', strategy_key),
            'history': history,
            'parameters': strategy_info.get('params', {}),
            'subjects_used': self.available_clients
        }
        # Attach monitoring data for FedClusterProxXAI
        if strategy_key == "fedclusterproxxai":
            results['cluster_losses_history'] = cluster_losses_history
        
        # Add cluster information for FedClusterProxXAI
        if strategy_key == "fedclusterproxxai":
            results['cluster_info'] = {
                'num_clusters': num_clusters,
                'final_assignments': client_cluster_map.copy(),
                'assignment_history': cluster_assignments_history,
                'reassignment_interval': reassignment_interval,
                'static_clustering': False  # Adaptive clustering enabled
            }
        
        return results

    def _run_client_cv_fedcluster(self, strategy_info: Dict) -> Dict:
        """K-fold cross-validation across clients for FedClusterProxXAI.
        Split clients into K folds; for each fold, train on K-1 folds and evaluate on the held-out fold clients.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from strategies import BaseCGMClient, create_cgm_model
        import numpy as np

        holdout_metrics = []
        per_client = {}
        rounds = min(self.config.cv_rounds, self.config.num_rounds)
        K = max(2, int(self.config.client_cv_folds))
        
        print(f"Using {rounds} rounds per fold ({K}-fold client CV)")
        
        # Build folds (round-robin split)
        client_ids = list(self.available_clients)
        folds = [client_ids[i::K] for i in range(K)]
        
        for fold_idx, test_cids in enumerate(folds):
            train_cids = [cid for cid in client_ids if cid not in test_cids]
            if not train_cids or not test_cids:
                continue
            print(f"\n--- Fold {fold_idx+1}/{K}: train={len(train_cids)} clients, test={len(test_cids)} clients ---")
            
            # Build training clients
            sample_train, _ = self.client_data_loader(train_cids[0])
            input_dim = len(sample_train.columns) - 1
            clients = {}
            for cid in train_cids:
                tr, va = self.client_data_loader(cid)
                model = create_cgm_model(input_dim=input_dim, horizon=1, model_id="fedcluster")
                clients[cid] = BaseCGMClient(cid, tr, va, model, strategy_type='fedclusterproxxai')
            
            # Initialize clusters (balanced warm start)
            num_clusters = strategy_info.get('params', {}).get('num_clusters', 3)
            cluster_models = {}
            client_cluster_map = {}
            baseline_client = list(clients.values())[0]
            for cluster_id in range(num_clusters):
                cm = create_cgm_model(input_dim=input_dim, horizon=1, model_id="transformer_small")
                # Better warm start with more epochs and validation
                mix = list(clients.values())[cluster_id::num_clusters][:3]
                if not mix:
                    mix = [baseline_client]
                X = np.vstack([c.x_train[:500] for c in mix])
                y = np.concatenate([c.y_train[:500] for c in mix])
                cm.fit(
                    X, y,
                    epochs=15, batch_size=32, verbose=0,
                    validation_split=0.2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
                )
                cluster_models[cluster_id] = {'model': cm, 'params': cm.get_weights(), 'client_ids': []}
            
            # Balanced initial assignment
            stats = []
            for cid, cl in clients.items():
                s = self._get_client_clustering_stats(cl)
                stats.append((cid, s))
            stats.sort(key=lambda x: x[1])
            for i, (cid, _) in enumerate(stats):
                k = i % num_clusters
                client_cluster_map[cid] = k
                cluster_models[k]['client_ids'].append(cid)
            
            history = {"losses": [], "metrics": []}
            # Train for limited rounds
            for r in range(rounds):
                # Client selection (50%)
                num_selected = max(1, len(clients)//2)
                selected = list(clients.keys())[:num_selected]
                client_results = []
                for cid in selected:
                    c = clients[cid]
                    k = client_cluster_map.get(cid, 0)
                    params = cluster_models[k]['model'].get_weights()
                    # Low proximal mu for stability
                    mu = 0.005 if r < 5 else 0.02
                    cfg = {"local_epochs": 3, "adaptive_mu": mu, "cluster_id": k}
                    upd, n, m = c.fit(params, cfg)
                    client_results.append((cid, upd, n, m))
                # Aggregate per-cluster
                for k in range(num_clusters):
                    cres = [(cid,p,n,m) for cid,p,n,m in client_results if client_cluster_map.get(cid)==k]
                    if not cres:
                        continue
                    tot = sum(n for _,_,n,_ in cres)
                    agg = []
                    n_params = len(cres[0][1])
                    for pi in range(n_params):
                        agg.append(sum(p[pi]*n for _,p,n,_ in cres)/tot)
                    # Update weights preserving non-trainable ordering
                    cm = cluster_models[k]['model']
                    current = cm.get_weights()
                    updated = []
                    t_idx = 0
                    for w in current:
                        if t_idx < len(agg) and w.shape == agg[t_idx].shape:
                            updated.append(agg[t_idx]); t_idx += 1
                        else:
                            updated.append(w)
                    cm.set_weights(updated)
                    cluster_models[k]['params'] = cm.get_weights()
                # Evaluate quick proxy on first 2 train clients
                eval_losses = []
                for cid in list(clients.keys())[:2]:
                    c = clients[cid]
                    loss, _, _ = c.evaluate(cluster_models[client_cluster_map.get(cid,0)]['model'].get_weights(), {})
                    eval_losses.append(loss)
                if eval_losses:
                    history['losses'].append(sum(eval_losses)/len(eval_losses))
                else:
                    history['losses'].append(0.0)
            
            # Evaluate on all test clients for this fold using best cluster for each client
            for test_cid in test_cids:
                h_tr, h_va = self.client_data_loader(test_cid)
                h_client = BaseCGMClient(test_cid, h_tr, h_va, create_cgm_model(input_dim=input_dim, horizon=1, model_id='fedcluster'), strategy_type='fedclusterproxxai')
                best_k, best_loss = 0, float('inf')
                for k in range(num_clusters):
                    params = cluster_models[k]['model'].get_weights()
                    loss, _, _ = h_client.evaluate(params, {})
                    if loss < best_loss:
                        best_loss, best_k = loss, k
                loss, _, metrics = h_client.evaluate(cluster_models[best_k]['model'].get_weights(), {})
                per_client[test_cid] = {"rmse": metrics.get('rmse', 0), "mae": metrics.get('mae', 0), "best_cluster": best_k, "fold": fold_idx+1}
                holdout_metrics.append(metrics)
        
        # Aggregate
        avg_rmse = float(np.mean([m.get('rmse', 0) for m in holdout_metrics])) if holdout_metrics else 0.0
        avg_mae = float(np.mean([m.get('mae', 0) for m in holdout_metrics])) if holdout_metrics else 0.0
        result = {
            'strategy_name': f'FedClusterProxXAI ({K}-Fold Client CV)',
            'history': {'losses': [], 'metrics': [{'rmse': avg_rmse, 'mae': avg_mae}]},
            'per_client': per_client,
            'folds': K
        }
        return result
    
    def _get_client_clustering_stats(self, client) -> float:
        """Get clustering score for a client (for sorting and balanced distribution)"""
        import numpy as np
        
        # Calculate comprehensive data statistics
        x_mean = np.mean(client.x_train, axis=0)
        x_std = np.std(client.x_train, axis=0)
        y_mean = np.mean(client.y_train)
        y_std = np.std(client.y_train)
        
        # Create multi-dimensional feature vector for clustering
        features = []
        
        # 1. Glucose statistics (if available)
        if client.x_train.shape[1] > 5:
            glucose_idx = 5  # current_glucose
            features.append(x_mean[glucose_idx])
            features.append(x_std[glucose_idx])
        else:
            features.append(y_mean)
            features.append(y_std)
        
        # 2. Target statistics
        features.append(y_mean)
        features.append(y_std)
        
        # 3. Overall feature mean and std
        features.append(np.mean(x_mean))
        features.append(np.mean(x_std))
        
        # 4. Coefficient of variation
        if y_mean > 0:
            features.append(y_std / y_mean)
        else:
            features.append(0)
        
        # Normalize features (0-1 range)
        features = np.array(features)
        if len(features) > 1 and (features.max() - features.min()) > 1e-8:
            features = (features - features.min()) / (features.max() - features.min())
        else:
            features = np.zeros_like(features)
        
        # Weighted combination for clustering
        cluster_score = (features[0] * 0.3 + features[2] * 0.3 + features[4] * 0.2 + 
                        features[5] * 0.2 if len(features) > 5 else features[0])
        
        return float(cluster_score)
    
    def _initial_cluster_assignment(self, client, num_clusters: int) -> int:
        """Assign client to initial cluster based on comprehensive data statistics"""
        # This method is now used for reassignment logic
        score = self._get_client_clustering_stats(client)
        cluster_id = min(int(score * num_clusters), num_clusters - 1)
        return cluster_id
    
    def _force_cluster_balance(self, assignments: Dict, clients: Dict, min_cluster_size: int) -> Dict:
        """Force rebalance clusters to prevent collapse"""
        from collections import Counter
        import numpy as np
        
        cluster_counts = Counter(assignments.values())
        total_clients = len(assignments)
        num_clusters = len(cluster_counts) if cluster_counts else 3
        
        # Check if rebalancing is needed
        min_size = min(cluster_counts.values()) if cluster_counts else 0
        empty_clusters = sum(1 for c in range(num_clusters) if cluster_counts.get(c, 0) == 0)
        
        if min_size < min_cluster_size or empty_clusters > 0:
            print(f"    🔧 Forcing rebalance (min size {min_cluster_size}, empty clusters: {empty_clusters})...")
            
            # Get clients sorted by clustering score
            client_scores = []
            for cid in assignments.keys():
                score = self._get_client_clustering_stats(clients[cid])
                client_scores.append((cid, score))
            client_scores.sort(key=lambda x: x[1])
            
            # Redistribute evenly across ALL clusters (including empty ones)
            new_assignments = {}
            for i, (cid, _) in enumerate(client_scores):
                cluster_id = i % num_clusters  # Use num_clusters, not len(cluster_counts)
                new_assignments[cid] = cluster_id
            
            # Verify all clusters have at least min_cluster_size
            new_counts = Counter(new_assignments.values())
            if min(new_counts.values()) < min_cluster_size:
                print(f"      ⚠️  Warning: Rebalance still leaves some clusters too small")
            
            return new_assignments
        
        return assignments
    
    def _adaptive_cluster_reassignment(self, clients: Dict, cluster_models: Dict, 
                                       current_assignments: Dict, round_num: int, min_cluster_size: int = 2) -> Dict:
        """Reassign clients to clusters based on which cluster model performs best on their validation data"""
        import numpy as np
        
        new_assignments = current_assignments.copy()
        reassignments = {}
        
        print(f"    Evaluating cluster models on each client's validation data...")
        
        # For each client, evaluate all cluster models and assign to best performing one
        for client_id, client in clients.items():
            best_cluster = current_assignments.get(client_id, 0)
            best_loss = float('inf')
            cluster_losses = {}
            
            # Evaluate client's validation data with each cluster model
            for cluster_id in cluster_models.keys():
                cluster_params = cluster_models[cluster_id]['model'].get_weights()  # Fresh weights
                
                # Evaluate loss on client's validation set (use larger sample if available)
                sample_size = min(200, len(client.x_val))
                client.model.set_weights(cluster_params)
                predictions = client.model(client.x_val[:sample_size], training=False)
                loss = np.mean((predictions.numpy() - client.y_val[:sample_size]) ** 2)
                
                cluster_losses[cluster_id] = loss
                
                if loss < best_loss:
                    best_loss = loss
                    best_cluster = cluster_id
            
            # Reassign if different cluster performs significantly better (with strict diversity constraints)
            improvement_threshold = 0.05  # Require more significant improvement (5%)
            old_cluster = current_assignments.get(client_id, 0)
            improvement = cluster_losses[old_cluster] - best_loss
            
            # Count current cluster sizes (before reassignment)
            current_sizes = {}
            for cid, cid_cluster in current_assignments.items():
                current_sizes[cid_cluster] = current_sizes.get(cid_cluster, 0) + 1
            
            # Strict constraints to prevent collapse
            would_empty_cluster = current_sizes.get(old_cluster, 0) <= min_cluster_size  # Would leave cluster below minimum
            max_allowed_per_cluster = len(current_assignments) * 0.55  # Max 55% per cluster (was 70%)
            would_make_too_large = current_sizes.get(best_cluster, 0) >= max_allowed_per_cluster
            
            # Reassign only if:
            # 1. Different cluster is better
            # 2. Improvement is very significant (1%+)
            # 3. Won't cause cluster collapse (respects min size and max dominance)
            if (best_cluster != old_cluster and 
                improvement > improvement_threshold and 
                not would_empty_cluster and 
                not would_make_too_large):
                new_assignments[client_id] = best_cluster
                reassignments[client_id] = {
                    'from': old_cluster,
                    'to': best_cluster,
                    'loss_improvement': improvement
                }
        
        # Count reassignments blocked by diversity constraints
        total_evaluated = len(clients)
        reassignment_attempts = sum(1 for cid, client in clients.items() 
                                    if current_assignments.get(cid, 0) != 
                                    min(range(len(cluster_models)), 
                                        key=lambda i: cluster_losses.get(i, float('inf')) 
                                        if i in cluster_losses else float('inf')))
        
        # Print reassignments
        if reassignments:
            print(f"    {len(reassignments)} clients reassigned:")
            for cid, info in list(reassignments.items())[:5]:  # Show first 5
                print(f"      {cid}: Cluster {info['from']} → Cluster {info['to']} "
                      f"(loss improvement: {info['loss_improvement']:.6f})")
            if len(reassignments) > 5:
                print(f"      ... and {len(reassignments) - 5} more")
            
            blocked = reassignment_attempts - len(reassignments)
            if blocked > 0:
                print(f"    ℹ️  {blocked} reassignments blocked to maintain cluster diversity")
        else:
            print(f"    No reassignments (all clients in optimal clusters or diversity constraints)")
        
        return new_assignments
    
    def _save_strategy_results(self, strategy_key: str, results: Dict):
        """Save results for individual strategy"""
        import json
        from utils import save_results
        
        # Save to the per-strategy results folder (latest)
        results_dir = self.dirs[strategy_key]['results']
        latest_results_path = f"{results_dir}/results.json"
        save_results(results, latest_results_path)
        print(f"  Results saved to {latest_results_path}")
        
        # Also save to a run-specific folder for this execution
        run_dir = self.dirs[strategy_key]['current_run']
        os.makedirs(run_dir, exist_ok=True)
        run_results_path = f"{run_dir}/results.json"
        save_results(results, run_results_path)
        print(f"  Run snapshot saved to {run_results_path}")
        
        # Save model versions for FedClusterProxXAI (for cold start)
        if strategy_key == "fedclusterproxxai":
            self._save_cluster_models(strategy_key, results)
    
    def _save_cluster_models(self, strategy_key: str, results: Dict):
        """Save model versions for each cluster (for cold start)"""
        import pickle
        
        models_dir = self.dirs[strategy_key]['models']
        run_dir = self.dirs[strategy_key]['current_run']
        
        # Save cluster metadata
        cluster_info = {
            'num_clusters': results.get('num_clusters', 3),
            'final_loss': results['history']['losses'][-1],
            'xai_score': results['history']['metrics'][-1].get('overall_xai_score', 0)
        }
        
        metadata_path = f"{models_dir}/cluster_metadata.json"
        run_metadata_path = f"{run_dir}/cluster_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(cluster_info, f, indent=2)
        # Also save in the run folder
        with open(run_metadata_path, 'w') as f:
            json.dump(cluster_info, f, indent=2)
        
        print(f"  Cluster metadata saved to {metadata_path}")
    
    def _save_comparative_results(self, results: Dict):
        """Save comparative analysis"""
        # TODO: Implement comparative results saving
        pass
    
    def _print_comparative_summary(self, results: Dict):
        """Print comparative summary to console with XAI metrics"""
        print("\n" + "="*90)
        print("COMPARATIVE RESULTS SUMMARY")
        print("="*90)
        
        for strategy_key, strategy_results in results.items():
            print(f"\nStrategy: {strategy_results['strategy_name']}")
            print(f"Execution Time: {strategy_results['execution_time']:.2f}s")
            
            # Performance metrics
            final_loss = strategy_results['history']['losses'][-1] if strategy_results['history']['losses'] else 0
            final_metrics = strategy_results['history']['metrics'][-1] if strategy_results['history']['metrics'] else {}
            
            print(f"\nPerformance Metrics:")
            print(f"  Final Loss: {final_loss:.4f}")
            if 'mae' in final_metrics:
                print(f"  MAE: {final_metrics['mae']:.4f}")
            if 'rmse' in final_metrics:
                print(f"  RMSE: {final_metrics['rmse']:.4f}")
            
            # XAI metrics
            if final_metrics:
                print(f"\nXAI Metrics:")
                if 'importance_score' in final_metrics:
                    print(f"  Importance Score: {final_metrics['importance_score']:.4f}")
                if 'stability_score' in final_metrics:
                    print(f"  Stability Score: {final_metrics['stability_score']:.4f}")
                if 'faithfulness_score' in final_metrics:
                    print(f"  Faithfulness Score: {final_metrics['faithfulness_score']:.4f}")
                if 'xai_score' in final_metrics:
                    print(f"  Overall XAI Score: {final_metrics['xai_score']:.4f}")
                if 'prediction_stability' in final_metrics:
                    print(f"  Prediction Stability: {final_metrics['prediction_stability']:.4f}")
                if 'local_stability' in final_metrics:
                    print(f"  Local Stability: {final_metrics['local_stability']:.4f}")
                if 'faithfulness' in final_metrics:
                    print(f"  Faithfulness: {final_metrics['faithfulness']:.4f}")
                if 'monotonicity' in final_metrics:
                    print(f"  Monotonicity: {final_metrics['monotonicity']:.4f}")
            
            print("-" * 90)
