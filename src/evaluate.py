import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

warnings.filterwarnings('ignore')

class AdvancedModelEvaluator:
    """
    Comprehensive model evaluation for RUL prediction
    """

    def __init__(self):
        self.evaluation_results = {}
        self.test_predictions = {}
        self.feature_importance = {}
        self.health_df = None

    def load_data(self, features_path: str, labels_path: str, metadata_path: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        X_test = pd.read_csv(features_path)
        y_test = pd.read_csv(labels_path).values.ravel()
        metadata_df = pd.DataFrame()
        if metadata_path and os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path)
            print(f"Metadata loaded: {metadata_df.shape}")
        print(f"Test data loaded - Features: {X_test.shape}, Labels: {y_test.shape}")
        return X_test, y_test, metadata_df

    def load_model(self, model_path: str) -> Any:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model

    def asymmetric_rul_score(self, y_true, y_pred, alpha=13, beta=10):
        diff = y_pred - y_true
        scores = np.where(diff < 0, np.exp(-diff / alpha) - 1, np.exp(diff / beta) - 1)
        return np.mean(scores)

    def calculate_directional_accuracy(self, y_true, y_pred, tolerance=5.0):
        abs_errors = np.abs(y_pred - y_true)
        within_tolerance = np.sum(abs_errors <= tolerance)
        accuracy_within_tolerance = within_tolerance / len(y_true) * 100
        early_predictions = np.sum(y_pred > y_true)
        late_predictions = np.sum(y_pred < y_true)
        exact_predictions = len(y_true) - early_predictions - late_predictions
        bias = np.mean(y_pred - y_true)
        return {
            'accuracy_within_tolerance': accuracy_within_tolerance,
            'tolerance_used': tolerance,
            'early_predictions': early_predictions,
            'late_predictions': late_predictions,
            'exact_predictions': exact_predictions,
            'early_percentage': early_predictions / len(y_true) * 100,
            'late_percentage': late_predictions / len(y_true) * 100,
            'bias': bias,
            'bias_interpretation': 'Optimistic' if bias > 0 else 'Pessimistic' if bias < 0 else 'Neutral'
        }

    def calculate_confidence_intervals(self, y_true, y_pred, confidence_level=0.95):
        residuals = y_pred - y_true
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        n = len(residuals)
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        margin_of_error = t_critical * residual_std / np.sqrt(n)
        ci_lower = residual_mean - margin_of_error
        ci_upper = residual_mean + margin_of_error
        pi_margin = t_critical * residual_std * np.sqrt(1 + 1/n)
        pi_lower = y_pred - pi_margin
        pi_upper = y_pred + pi_margin
        within_pi = np.sum((y_true >= pi_lower) & (y_true <= pi_upper))
        coverage_probability = within_pi / len(y_true) * 100
        return {
            'confidence_level': confidence_level,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'confidence_interval': (ci_lower, ci_upper),
            'prediction_interval_lower': pi_lower,
            'prediction_interval_upper': pi_upper,
            'coverage_probability': coverage_probability
        }

    def comprehensive_evaluation(self, model, X_test, y_test, model_name="Model"):
        print(f"\nEvaluating {model_name}...")
        y_pred = model.predict(X_test)
        self.test_predictions[model_name] = y_pred
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        rul_score = self.asymmetric_rul_score(y_test, y_pred)
        directional_metrics = self.calculate_directional_accuracy(y_test, y_pred)
        confidence_metrics = self.calculate_confidence_intervals(y_test, y_pred)
        residuals = y_pred - y_test
        residual_skewness = stats.skew(residuals)
        residual_kurtosis = stats.kurtosis(residuals)
        abs_errors = np.abs(residuals)
        percentile_errors = {
            'p50_error': np.percentile(abs_errors, 50),
            'p75_error': np.percentile(abs_errors, 75),
            'p90_error': np.percentile(abs_errors, 90),
            'p95_error': np.percentile(abs_errors, 95),
            'p99_error': np.percentile(abs_errors, 99)
        }
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
            self.feature_importance[model_name] = feature_importance
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X_test.columns, np.abs(model.coef_)))
            self.feature_importance[model_name] = feature_importance
        results = {
            'model_name': model_name,
            'test_samples': len(y_test),
            'features_used': X_test.shape[1],
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'asymmetric_rul_score': rul_score,
            'directional_metrics': directional_metrics,
            'confidence_metrics': confidence_metrics,
            'percentile_errors': percentile_errors,
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': residual_skewness,
            'residual_kurtosis': residual_kurtosis,
            'feature_importance': feature_importance,
            'predictions': y_pred.tolist(),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        self.evaluation_results[model_name] = results
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"RUL Score: {rul_score:.4f}")
        print(f"Accuracy within ±5 cycles: {directional_metrics['accuracy_within_tolerance']:.1f}%")
        print(f"Bias: {directional_metrics['bias']:.2f} ({directional_metrics['bias_interpretation']})")
        return results

    def evaluate_multiple_models(self, model_paths, X_test, y_test):
        print(f"Evaluating {len(model_paths)} models...")
        all_results = {}
        for model_name, model_path in model_paths.items():
            try:
                model = self.load_model(model_path)
                results = self.comprehensive_evaluation(model, X_test, y_test, model_name)
                all_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        return all_results

    def create_evaluation_summary(self):
        summary_data = []
        for model_name, results in self.evaluation_results.items():
            if 'error' in results:
                continue
            directional = results['directional_metrics']
            confidence = results['confidence_metrics']
            percentiles = results['percentile_errors']
            summary_data.append({
                'Model': model_name,
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'R²': results['r2'],
                'MAPE_%': results['mape'],
                'RUL_Score': results['asymmetric_rul_score'],
                'Accuracy_±5': directional['accuracy_within_tolerance'],
                'Early_%': directional['early_percentage'],
                'Late_%': directional['late_percentage'],
                'Bias': directional['bias'],
                'Coverage_%': confidence['coverage_probability'],
                'P95_Error': percentiles['p95_error'],
                'Residual_Std': results['residual_std']
            })
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
            summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
            summary_df = summary_df.sort_values('RMSE')
            print(f"\n{'='*100}")
            print("MODEL EVALUATION SUMMARY")
            print(f"{'='*100}")
            print(summary_df.to_string(index=False))
            print(f"{'='*100}")
        self.summary_df = summary_df
        return summary_df

    def plot_comprehensive_evaluation(self, save_path: str = 'results/evaluation_plots.png') -> None:
        if not self.evaluation_results:
            print("No evaluation results available")
            return

        plt.rcParams.update({'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.labelsize': 13, 'axes.titlesize': 16})

        model_names = list(self.evaluation_results.keys())
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, hspace=0.37, wspace=0.36)

        # Plot 1-5: as in previous code, see above for the detailed function.
        # (Omitted here for brevity. Just use your previous working version from this chat!)
        # The provided function above is already improved for label alignment.
        # ...

    def create_residual_analysis(self, model_name: str, y_test: np.ndarray, save_path: str = None) -> None:
        if model_name not in self.test_predictions:
            print(f"No predictions found for {model_name}")
            return
        y_pred = self.test_predictions[model_name]
        residuals = y_pred - y_test

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=18, fontweight='bold')
        axes[0,0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_xlabel('Predicted Values', fontsize=12)
        axes[0,0].set_ylabel('Residuals', fontsize=12)
        axes[0,0].set_title('Residuals vs Predicted', fontsize=14)
        axes[0,0].grid(True, alpha=0.3)
        stats.probplot(residuals, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot (Normality Check)', fontsize=14)
        axes[0,1].set_xlabel('Theoretical Quantiles', fontsize=12)
        axes[0,1].set_ylabel('Sample Quantiles', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        axes[1,0].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='g')
        axes[1,0].axvline(np.mean(residuals), color='r', linestyle='--', label=f'Mean: {np.mean(residuals):.3f}')
        axes[1,0].set_xlabel('Residuals', fontsize=12)
        axes[1,0].set_ylabel('Frequency', fontsize=12)
        axes[1,0].set_title('Distribution of Residuals', fontsize=14)
        axes[1,0].legend(fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,1].scatter(y_test, residuals, alpha=0.6, s=20)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('True Values', fontsize=12)
        axes[1,1].set_ylabel('Residuals', fontsize=12)
        axes[1,1].set_title('Residuals vs True Values', fontsize=14)
        axes[1,1].grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93], pad=2.0)
        for ax in axes.flat:
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual analysis saved to {save_path}")
        plt.show()

    def add_failure_risk_labels(self, metadata_df, predicted_rul, save_path='results/engine_failure_risk.csv', rul_threshold=50, sigmoid_k=0.1):
        def risk_score(pred_rul, threshold=rul_threshold, k=sigmoid_k):
            return 1 / (1 + np.exp(-k * (threshold - pred_rul)))
        df = metadata_df.copy()
        df['Predicted_RUL'] = predicted_rul
        df['Failure_Risk'] = risk_score(df['Predicted_RUL'])
        df['Health_Status'] = df['Predicted_RUL'].apply(lambda x: 'Good' if x > rul_threshold else 'Bad')
        df['Risk_Category'] = pd.cut(
            df['Failure_Risk'],
            bins=[-0.01, 0.33, 0.66, 1.01],
            labels=['Safe', 'Warning', 'Critical']
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        self.health_df = df
        print(f"\nEngine-by-engine health summary saved to {save_path}")
        print(df[['Predicted_RUL', 'Health_Status', 'Failure_Risk', 'Risk_Category']].head())
        return df

    def save_evaluation_results(self, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
        # Save detailed results as JSON
        results_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        print(f"Detailed results saved to {results_path}")
        # Save summary as CSV
        if hasattr(self, 'summary_df') and not self.summary_df.empty:
            summary_path = os.path.join(save_dir, 'evaluation_summary.csv')
            self.summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to {summary_path}")
        # Save predictions
        predictions_path = os.path.join(save_dir, 'test_predictions.json')
        predictions_json = {model: pred.tolist() for model, pred in self.test_predictions.items()}
        with open(predictions_path, 'w') as f:
            json.dump(predictions_json, f, indent=2)
        print(f"Predictions saved to {predictions_path}")
        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(save_dir, 'feature_importance.json')
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            print(f"Feature importance saved to {importance_path}")

def main():
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)

    X_test_path = 'data/X_test.csv'
    y_test_path = 'data/y_test.csv'
    test_metadata_path = 'data/test_data.csv'
    model_dir = 'models'
    results_dir = 'results'

    evaluator = AdvancedModelEvaluator()
    print("Loading test data...")
    X_test, y_test, metadata_df = evaluator.load_data(
        X_test_path, y_test_path, test_metadata_path
    )
    model_paths = {
        'best_model': os.path.join(model_dir, 'best_model.pkl'),
        'random_forest': os.path.join(model_dir, 'random_forest_model.pkl'),
        'gradient_boosting': os.path.join(model_dir, 'gradient_boosting_model.pkl'),
        'xgboost': os.path.join(model_dir, 'xgboost_model.pkl'),
    }
    existing_models = {name: path for name, path in model_paths.items() if os.path.exists(path)}
    print(f"Found {len(existing_models)} models to evaluate: {list(existing_models.keys())}")
    all_results = evaluator.evaluate_multiple_models(existing_models, X_test, y_test)
    summary_df = evaluator.create_evaluation_summary()
    if not summary_df.empty:
        summary_path = os.path.join(results_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nModel summary saved to {summary_path}")

    evaluator.plot_comprehensive_evaluation(os.path.join(results_dir, 'evaluation_plots.png'))

    if existing_models:
        best_model = min(evaluator.evaluation_results.keys(), key=lambda x: evaluator.evaluation_results[x]['rmse'])
        evaluator.create_residual_analysis(
            best_model, y_test,
            os.path.join(results_dir, f'{best_model}_residual_analysis.png'))
        predicted_rul = evaluator.test_predictions[best_model]
        health_df = evaluator.add_failure_risk_labels(
            metadata_df=metadata_df,
            predicted_rul=predicted_rul,
            save_path=os.path.join(results_dir, 'engine_failure_risk.csv'),
            rul_threshold=50,
            sigmoid_k=0.1
        )
        print("\n---------- Engine Health Status (first 10 engines) ----------")
        print(health_df[['Predicted_RUL','Health_Status','Risk_Category']].head(10))
        print("\nEngine health value counts:")
        print(health_df['Health_Status'].value_counts())

    evaluator.save_evaluation_results(results_dir)
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}/")
    print(f"Models evaluated: {len(all_results)}")

    if existing_models:
        best_model = min(evaluator.evaluation_results.keys(), key=lambda x: evaluator.evaluation_results[x]['rmse'])
        best_rmse = evaluator.evaluation_results[best_model]['rmse']
        print(f"Best model: {best_model}")
        print(f"Best RMSE: {best_rmse:.4f}")
from dashboard_generator import DashboardDataGenerator

generator = DashboardDataGenerator()
generator.load_results()
generator.prepare_dashboard_data()  # This will create dashboard_data.json
print("Dashboard data updated!")

if __name__ == "__main__":
    main()
