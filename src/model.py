import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime

# Scikit-learn imports
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    TimeSeriesSplit, KFold
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

# XGBoost and LightGBM (install if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    """
    Advanced model training with multiple algorithms, hyperparameter tuning, and validation
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = float('inf')
        self.training_history = {}
        
    def get_model_configs(self) -> Dict[str, Dict]:
        """Get configurations for different models"""
        configs = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'search_type': 'random'  # 'grid' or 'random'
            },
            
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'search_type': 'random'
            },
            
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'search_type': 'random'
            },
            
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
                },
                'search_type': 'grid'
            },
            
            'lasso': {
                'model': Lasso(random_state=self.random_state, max_iter=2000),
                'param_grid': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'selection': ['cyclic', 'random']
                },
                'search_type': 'grid'
            },
            
            'elastic_net': {
                'model': ElasticNet(random_state=self.random_state, max_iter=2000),
                'param_grid': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                'search_type': 'grid'
            },
            
            'svr': {
                'model': SVR(),
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                },
                'search_type': 'random'
            },
            
            'knn': {
                'model': KNeighborsRegressor(n_jobs=-1),
                'param_grid': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                'search_type': 'grid'
            },
            
            'mlp': {
                'model': MLPRegressor(random_state=self.random_state, max_iter=500),
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate': ['constant', 'adaptive'],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                'search_type': 'random'
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'search_type': 'random'
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            configs['lightgbm'] = {
                'model': lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'search_type': 'random'
            }
        
        return configs
    
    def load_data(self, features_path: str, labels_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load features and labels from CSV files with validation"""
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path)
        
        # Convert to numpy array and ensure 1D
        y = y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()
        
        print(f"Data loaded - Features: {X.shape}, Labels: {y.shape}")
        
        # Basic validation
        if len(X) != len(y):
            raise ValueError(f"Mismatch in samples: X has {len(X)}, y has {len(y)}")
        
        if X.isnull().any().any():
            print("Warning: Found NaN values in features")
        
        if np.isnan(y).any():
            print("Warning: Found NaN values in labels")
        
        return X, y
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: np.ndarray,
                      cv_folds: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict:
        """Comprehensive model evaluation using cross-validation"""
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                  scoring=scoring, n_jobs=-1)
        
        # Fit model to get predictions for additional metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate comprehensive metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        
        # Custom RUL-specific metrics
        early_predictions = np.sum(y_pred > y)  # Predictions that are too optimistic
        late_predictions = np.sum(y_pred < y)   # Predictions that are too pessimistic
        
        results = {
            'cv_mean': -cv_scores.mean(),  # Convert back from negative MSE
            'cv_std': cv_scores.std(),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'early_predictions': early_predictions,
            'late_predictions': late_predictions,
            'cv_scores': cv_scores.tolist()
        }
        
        return results
    
    def hyperparameter_tuning(self, model_name: str, model_config: Dict,
                             X: pd.DataFrame, y: np.ndarray,
                             cv_folds: int = 5, n_iter: int = 50,
                             scoring: str = 'neg_mean_squared_error') -> Tuple[Any, Dict]:
        """Perform hyperparameter tuning using GridSearch or RandomizedSearch"""
        
        print(f"Tuning hyperparameters for {model_name}...")
        
        model = model_config['model']
        param_grid = model_config['param_grid']
        search_type = model_config.get('search_type', 'random')
        
        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv_folds, scoring=scoring,
                n_jobs=-1, verbose=1, return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_grid, cv=cv_folds, scoring=scoring,
                n_iter=n_iter, n_jobs=-1, verbose=1, 
                random_state=self.random_state, return_train_score=True
            )
        
        # Perform search
        try:
            search.fit(X, y)
            
            # Get results
            best_model = search.best_estimator_
            tuning_results = {
                'best_params': search.best_params_,
                'best_score': -search.best_score_,  # Convert from negative MSE
                'cv_results': search.cv_results_,
                'search_type': search_type
            }
            
            print(f"Best parameters for {model_name}: {search.best_params_}")
            print(f"Best CV score: {-search.best_score_:.4f}")
            
            return best_model, tuning_results
            
        except Exception as e:
            print(f"Error tuning {model_name}: {str(e)}")
            # Return default model if tuning fails
            model.fit(X, y)
            return model, {'error': str(e)}
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[np.ndarray] = None,
                          tune_hyperparameters: bool = True) -> Dict:
        """Train a single model with optional hyperparameter tuning"""
        
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*50}")
        
        model_configs = self.get_model_configs()
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = model_configs[model_name]
        start_time = datetime.now()
        
        try:
            if tune_hyperparameters:
                # Hyperparameter tuning
                best_model, tuning_results = self.hyperparameter_tuning(
                    model_name, model_config, X_train, y_train
                )
            else:
                # Use default parameters
                best_model = model_config['model']
                best_model.fit(X_train, y_train)
                tuning_results = {'default_params': True}
            
            # Evaluate on training set
            train_results = self.evaluate_model(best_model, X_train, y_train)
            
            # Evaluate on validation set if provided
            val_results = {}
            if X_val is not None and y_val is not None:
                y_val_pred = best_model.predict(X_val)
                val_results = {
                    'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                    'mae': mean_absolute_error(y_val, y_val_pred),
                    'r2': r2_score(y_val, y_val_pred),
                    'mape': mean_absolute_percentage_error(y_val, y_val_pred) * 100
                }
                print(f"Validation RMSE: {val_results['rmse']:.4f}")
                print(f"Validation MAE: {val_results['mae']:.4f}")
                print(f"Validation R²: {val_results['r2']:.4f}")
            
            # Training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model and results
            self.models[model_name] = best_model
            
            results = {
                'model_name': model_name,
                'train_results': train_results,
                'val_results': val_results,
                'tuning_results': tuning_results,
                'training_time_seconds': training_time,
                'training_samples': len(X_train),
                'features_used': X_train.shape[1]
            }
            
            self.results[model_name] = results
            
            # Update best model if this one is better
            current_score = val_results.get('rmse', train_results['rmse'])
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_model = model_name
            
            print(f"Training completed in {training_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return {'error': str(e), 'model_name': model_name}
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: np.ndarray,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[np.ndarray] = None,
                        models_to_train: Optional[List[str]] = None,
                        tune_hyperparameters: bool = True) -> Dict:
        """Train multiple models and compare performance"""
        
        if models_to_train is None:
            models_to_train = list(self.get_model_configs().keys())
        
        print(f"Training {len(models_to_train)} models: {', '.join(models_to_train)}")
        
        all_results = {}
        
        for model_name in models_to_train:
            try:
                results = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, tune_hyperparameters
                )
                all_results[model_name] = results
                
            except Exception as e:
                print(f"Failed to train {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        # Create comparison summary
        self.create_model_comparison_summary()
        
        return all_results
    
    def create_model_comparison_summary(self) -> pd.DataFrame:
        """Create a summary comparison of all trained models"""
        
        summary_data = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
                
            train_results = results['train_results']
            val_results = results.get('val_results', {})
            
            summary_data.append({
                'Model': model_name,
                'Train_RMSE': train_results['rmse'],
                'Train_MAE': train_results['mae'],
                'Train_R2': train_results['r2'],
                'Train_MAPE': train_results['mape'],
                'Val_RMSE': val_results.get('rmse', np.nan),
                'Val_MAE': val_results.get('mae', np.nan),
                'Val_R2': val_results.get('r2', np.nan),
                'Val_MAPE': val_results.get('mape', np.nan),
                'CV_Mean': train_results['cv_mean'],
                'CV_Std': train_results['cv_std'],
                'Training_Time': results['training_time_seconds'],
                'Early_Predictions': train_results['early_predictions'],
                'Late_Predictions': train_results['late_predictions']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            # Sort by validation RMSE if available, otherwise by CV mean
            sort_col = 'Val_RMSE' if 'Val_RMSE' in summary_df.columns and not summary_df['Val_RMSE'].isna().all() else 'CV_Mean'
            summary_df = summary_df.sort_values(sort_col)
            
            print(f"\n{'='*80}")
            print("MODEL COMPARISON SUMMARY")
            print(f"{'='*80}")
            print(summary_df.round(4).to_string(index=False))
            print(f"{'='*80}")
            
            if self.best_model:
                print(f"Best Model: {self.best_model} (Score: {self.best_score:.4f})")
        
        self.training_history['summary'] = summary_df
        return summary_df
    
    def plot_model_comparison(self, save_path: str = 'models/model_comparison.png') -> None:
        """Create visualization comparing model performance"""
        
        if not self.results:
            print("No models trained yet")
            return
        
        # Prepare data for plotting
        model_names = []
        train_rmse = []
        val_rmse = []
        train_r2 = []
        val_r2 = []
        training_times = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
                
            model_names.append(model_name)
            train_rmse.append(results['train_results']['rmse'])
            val_rmse.append(results['val_results'].get('rmse', np.nan))
            train_r2.append(results['train_results']['r2'])
            val_r2.append(results['val_results'].get('r2', np.nan))
            training_times.append(results['training_time_seconds'])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: RMSE Comparison
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[0,0].bar(x_pos - width/2, train_rmse, width, label='Train RMSE', alpha=0.8)
        if not all(np.isnan(val_rmse)):
            axes[0,0].bar(x_pos + width/2, val_rmse, width, label='Val RMSE', alpha=0.8)
        
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].set_title('RMSE Comparison')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(model_names, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: R² Comparison
        axes[0,1].bar(x_pos - width/2, train_r2, width, label='Train R²', alpha=0.8)
        if not all(np.isnan(val_r2)):
            axes[0,1].bar(x_pos + width/2, val_r2, width, label='Val R²', alpha=0.8)
        
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].set_title('R² Score Comparison')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(model_names, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Training Time
        axes[1,0].bar(model_names, training_times, alpha=0.8, color='orange')
        axes[1,0].set_xlabel('Models')
        axes[1,0].set_ylabel('Training Time (seconds)')
        axes[1,0].set_title('Training Time Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: RMSE vs Training Time (scatter)
        # Use validation RMSE if available, otherwise training RMSE
        plot_rmse = [val if not np.isnan(val) else train for val, train in zip(val_rmse, train_rmse)]
        
        axes[1,1].scatter(training_times, plot_rmse, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1,1].annotate(name, (training_times[i], plot_rmse[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1,1].set_xlabel('Training Time (seconds)')
        axes[1,1].set_ylabel('RMSE')
        axes[1,1].set_title('RMSE vs Training Time')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
        plt.show()
    
    def save_models(self, model_dir: str = 'models') -> None:
        """Save all trained models and results"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save best model separately
        if self.best_model and self.best_model in self.models:
            best_model_path = os.path.join(model_dir, 'best_model.pkl')
            joblib.dump(self.models[self.best_model], best_model_path)
            print(f"Saved best model ({self.best_model}) to {best_model_path}")
        
        # Save results and metadata
        results_path = os.path.join(model_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for model_name, results in self.results.items():
                json_results[model_name] = self._convert_to_json_serializable(results)
            
            json.dump({
                'results': json_results,
                'best_model': self.best_model,
                'best_score': float(self.best_score),
                'training_timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Training results saved to {results_path}")
        
        # Save model comparison summary
        if 'summary' in self.training_history:
            summary_path = os.path.join(model_dir, 'model_comparison_summary.csv')
            self.training_history['summary'].to_csv(summary_path, index=False)
            print(f"Model comparison summary saved to {summary_path}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def load_best_model(self, model_dir: str = 'models') -> Any:
        """Load the best trained model"""
        best_model_path = os.path.join(model_dir, 'best_model.pkl')
        
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found at {best_model_path}")
        
        model = joblib.load(best_model_path)
        print(f"Best model loaded from {best_model_path}")
        return model

def main():
    """Main training function"""
    
    print(f"{'='*60}")
    print("ADVANCED MODEL TRAINING PIPELINE")
    print(f"{'='*60}")
    
    # Configuration
    RANDOM_STATE = 42
    TUNE_HYPERPARAMETERS = True
    MODELS_TO_TRAIN = None  # None means train all available models
    
    # Paths
    X_train_path = 'data/X_train.csv'
    y_train_path = 'data/y_train.csv'
    X_val_path = 'data/X_val.csv'
    y_val_path = 'data/y_val.csv'
    model_dir = 'models'
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(random_state=RANDOM_STATE)
    
    # Load training data
    print("Loading training data...")
    X_train, y_train = trainer.load_data(X_train_path, y_train_path)
    
    # Load validation data if available
    X_val, y_val = None, None
    if os.path.exists(X_val_path) and os.path.exists(y_val_path):
        print("Loading validation data...")
        X_val, y_val = trainer.load_data(X_val_path, y_val_path)
    else:
        print("Validation data not found. Using cross-validation only.")
    
    # Train all models
    print(f"\nStarting model training with hyperparameter tuning: {TUNE_HYPERPARAMETERS}")
    all_results = trainer.train_all_models(
        X_train, y_train, X_val, y_val,
        models_to_train=MODELS_TO_TRAIN,
        tune_hyperparameters=TUNE_HYPERPARAMETERS
    )
    
    # Create visualizations
    trainer.plot_model_comparison()
    
    # Save all models and results
    trainer.save_models(model_dir)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Models trained: {len([r for r in all_results.values() if 'error' not in r])}")
    print(f"Failed models: {len([r for r in all_results.values() if 'error' in r])}")
    if trainer.best_model:
        print(f"Best model: {trainer.best_model}")
        print(f"Best score: {trainer.best_score:.4f}")
    print(f"Results saved to: {model_dir}/")

if __name__ == "__main__":
    main()