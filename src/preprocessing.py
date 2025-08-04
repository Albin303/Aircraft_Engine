import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy import stats
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline with feature engineering, selection, and scaling
    """

    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.scalers = {}
        self.feature_selectors = {}
        self.encoders = {}
        self.feature_importance_ = {}
        self.preprocessing_stats_ = {}

    def _get_default_config(self):
        return {
            'scaling_method': 'robust',  # StandardScaler, RobustScaler, or MinMaxScaler
            'feature_selection': True,
            'n_features_to_select': 20,
            'outlier_detection': True,
            'outlier_method': 'iqr',  # 'iqr' or 'zscore'
            'outlier_threshold': 3.0,
            'advanced_features': True,
            'pca_components': None,
            'rolling_windows': [3, 5, 10],
            'interaction_features': True,
            'polynomial_features': False,
            'polynomial_degree': 2,
        }

    def load_data(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Data loaded from {filepath} with shape {df.shape}")
        self._validate_data(df)
        return df

    def _validate_data(self, df):
        required_columns = ['Engine_ID', 'RUL']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            print(f"Warning: Empty columns found: {empty_cols}")
        print(f"Data validation passed. Columns: {len(df.columns)}, Rows: {len(df)}")

    def encode_health_state(self, df):
        if 'Health_State' not in df.columns:
            print("Health_State column not found, skipping encoding")
            return df
        mapping = {'critical': 0, 'degraded': 1, 'healthy': 2}
        unknown_vals = set(df['Health_State'].unique()) - set(mapping.keys())
        for val in unknown_vals:
            mapping[val] = 1  # Map unknown to 'degraded'
        df['Health_State'] = df['Health_State'].map(mapping)
        self.encoders['Health_State'] = mapping
        print(f"Health_State encoded with mapping: {mapping}")
        return df

    def advanced_feature_engineering(self, df):
        if not self.config['advanced_features']:
            return df
        df_enhanced = df.copy()
        grouped = df_enhanced.groupby('Engine_ID')
        sensors = ['HPT_outlet_temperature_R', 'HPC_outlet_temperature_R',
                   'Fan_stall_margin', 'LPC_outlet_temperature_R']

        for window in self.config['rolling_windows']:
            for col in sensors:
                if col in df_enhanced.columns:
                    df_enhanced[f'{col}_rolling_mean_{window}'] = grouped[col].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
                    df_enhanced[f'{col}_rolling_std_{window}'] = grouped[col].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
                    df_enhanced[f'{col}_rolling_max_{window}'] = grouped[col].rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
                    df_enhanced[f'{col}_rolling_min_{window}'] = grouped[col].rolling(window, min_periods=1).min().reset_index(level=0, drop=True)

        for col in sensors:
            if col in df_enhanced.columns:
                df_enhanced[f'{col}_trend_1st'] = grouped[col].diff().reset_index(level=0, drop=True)
                df_enhanced[f'{col}_trend_2nd'] = grouped[col].diff().diff().reset_index(level=0, drop=True)
                df_enhanced[f'{col}_ema'] = grouped[col].ewm(alpha=0.3).mean().reset_index(level=0, drop=True)
                df_enhanced[f'{col}_cumsum'] = grouped[col].cumsum().reset_index(level=0, drop=True)
                df_enhanced[f'{col}_cummax'] = grouped[col].cummax().reset_index(level=0, drop=True)

        if self.config['interaction_features']:
            if all(c in df_enhanced.columns for c in ['HPT_outlet_temperature_R', 'HPC_outlet_temperature_R']):
                df_enhanced['Temp_Ratio_HPT_HPC'] = df_enhanced['HPT_outlet_temperature_R'] / (df_enhanced['HPC_outlet_temperature_R'] + 1e-8)
                df_enhanced['Temp_Diff_HPT_HPC'] = df_enhanced['HPT_outlet_temperature_R'] - df_enhanced['HPC_outlet_temperature_R']
            if all(c in df_enhanced.columns for c in ['Press_Ratio_HPC', 'HPC_outlet_temperature_R']):
                df_enhanced['Press_Temp_Interaction'] = df_enhanced['Press_Ratio_HPC'] * df_enhanced['HPC_outlet_temperature_R']

        if 'Flight_cycle_number' in df_enhanced.columns:
            df_enhanced['Cycle_Squared'] = df_enhanced['Flight_cycle_number'] ** 2
            df_enhanced['Cycle_Log'] = np.log1p(df_enhanced['Flight_cycle_number'])
            df_enhanced['Engine_Max_Cycle'] = grouped['Flight_cycle_number'].transform('max')
            df_enhanced['Cycle_Percentage'] = df_enhanced['Flight_cycle_number'] / df_enhanced['Engine_Max_Cycle']

        for col in sensors:
            if col in df_enhanced.columns:
                df_enhanced[f'{col}_engine_mean'] = grouped[col].transform('mean')
                df_enhanced[f'{col}_engine_std'] = grouped[col].transform('std')
                df_enhanced[f'{col}_deviation_from_engine_mean'] = df_enhanced[col] - df_enhanced[f'{col}_engine_mean']

        print(f"Advanced feature engineering completed. New shape: {df_enhanced.shape}")
        return df_enhanced

    def handle_missing_values(self, df):
        df_imputed = df.copy()
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
        missing_stats = {}
        for col in numeric_columns:
            count = df_imputed[col].isnull().sum()
            if count > 0:
                missing_stats[col] = count
                if 'trend' in col.lower() or 'diff' in col.lower():
                    df_imputed[col] = df_imputed.groupby('Engine_ID')[col].fillna(method='ffill')
                    df_imputed[col] = df_imputed[col].fillna(0)
                else:
                    df_imputed[col] = df_imputed.groupby('Engine_ID')[col].transform(lambda x: x.fillna(x.median()))
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        self.preprocessing_stats_['missing_values'] = missing_stats
        total_missing = sum(missing_stats.values())
        print(f"Missing value imputation completed. Total values imputed: {total_missing}")
        return df_imputed

    def detect_and_handle_outliers(self, df, numeric_columns):
        if not self.config['outlier_detection']:
            return df
        df_clean = df.copy()
        outlier_stats = {}
        for col in numeric_columns:
            if col in df_clean.columns:
                if self.config['outlier_method'] == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                elif self.config['outlier_method'] == 'zscore':
                    z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                    outliers = z_scores > self.config['outlier_threshold']
                outlier_count = outliers.sum()
                outlier_stats[col] = outlier_count
                if outlier_count > 0:
                    if self.config['outlier_method'] == 'iqr':
                        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    else:
                        mean_val = df_clean[col].mean()
                        std_val = df_clean[col].std()
                        df_clean[col] = df_clean[col].clip(mean_val - 3*std_val, mean_val + 3*std_val)
        self.preprocessing_stats_['outliers'] = outlier_stats
        total_outliers = sum(outlier_stats.values())
        print(f"Outlier detection completed. Total outliers handled: {total_outliers}")
        return df_clean

    def select_features(self, X, y):
        if not self.config['feature_selection']:
            return X
        print("Starting feature selection...")
        selector_f = SelectKBest(score_func=f_regression, k=min(self.config['n_features_to_select'], X.shape[1]))
        selector_f.fit(X, y)
        selected_f = X.columns[selector_f.get_support()].tolist()
        selector_mi = SelectKBest(score_func=mutual_info_regression, k=min(self.config['n_features_to_select'], X.shape[1]))
        selector_mi.fit(X, y)
        selected_mi = X.columns[selector_mi.get_support()].tolist()
        combined_features = list(set(selected_f + selected_mi))
        if len(combined_features) > self.config['n_features_to_select']:
            f_scores = selector_f.scores_
            features_scores = list(zip(X.columns, f_scores))
            features_scores.sort(key=lambda x: x[1], reverse=True)
            combined_features = [f[0] for f in features_scores[:self.config['n_features_to_select']]]
        self.feature_importance_['f_regression'] = dict(zip(X.columns, selector_f.scores_))
        self.feature_importance_['mutual_info'] = dict(zip(X.columns, selector_mi.scores_))
        self.feature_selectors['combined'] = combined_features
        print(f"Feature selection completed. Selected {len(combined_features)} features.")
        print(f"Top features: {combined_features[:10]}")
        return X[combined_features]

    def scale_features(self, df, feature_cols, fit=True):
        df_scaled = df.copy()
        method = self.config['scaling_method']
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        if fit:
            df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
            self.scalers['feature_scaler'] = scaler
        else:
            if 'feature_scaler' not in self.scalers:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df_scaled[feature_cols] = self.scalers['feature_scaler'].transform(df_scaled[feature_cols])

        print(f"Features scaled with {method} scaler: {len(feature_cols)} features")
        return df_scaled

    def apply_pca(self, X, fit=True):
        if self.config['pca_components'] is None:
            return X
        if fit:
            pca = PCA(n_components=self.config['pca_components'], random_state=42)
            X_pca = pca.fit_transform(X)
            self.scalers['pca'] = pca
            cols = [f'PCA_Component_{i+1}' for i in range(self.config['pca_components'])]
            print(f"PCA applied. Reduced features from {X.shape[1]} to {self.config['pca_components']}")
            return pd.DataFrame(X_pca, columns=cols, index=X.index)
        else:
            if 'pca' not in self.scalers:
                raise ValueError("PCA not fitted. Call with fit=True first.")
            X_pca = self.scalers['pca'].transform(X)
            cols = [f'PCA_Component_{i+1}' for i in range(self.config['pca_components'])]
            return pd.DataFrame(X_pca, columns=cols, index=X.index)

    def generate_preprocessing_report(self, df_original, df_processed):
        print("\n" + "="*60)
        print("PREPROCESSING REPORT")
        print("="*60)
        print(f"Original data shape: {df_original.shape}")
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Features added: {df_processed.shape[1] - df_original.shape[1]}")
        if 'outliers' in self.preprocessing_stats_:
            print(f"Outliers handled: {sum(self.preprocessing_stats_['outliers'].values())}")
        if 'missing_values' in self.preprocessing_stats_:
            print(f"Missing values imputed: {sum(self.preprocessing_stats_['missing_values'].values())}")
        print("\nPreprocessing configuration:")
        for k, v in self.config.items():
            print(f"  {k}: {v}")
        print("="*60)

    def save_preprocessor(self, filepath):
        state = {
            'config': self.config,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'encoders': self.encoders,
            'feature_importance_': self.feature_importance_,
            'preprocessing_stats_': self.preprocessing_stats_,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(state, filepath)
        print(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath):
        state = joblib.load(filepath)
        self.config = state['config']
        self.scalers = state['scalers']
        self.feature_selectors = state['feature_selectors']
        self.encoders = state['encoders']
        self.feature_importance_ = state['feature_importance_']
        self.preprocessing_stats_ = state['preprocessing_stats_']
        print(f"Preprocessor loaded from {filepath}")

    def fit_transform(self, df):
        print("Starting complete preprocessing pipeline...")

        df_original = df.copy()
        df = self.encode_health_state(df)
        df = self.advanced_feature_engineering(df)
        df = self.handle_missing_values(df)

        if 'FD_set' in df.columns:
            df = df.drop(columns=['FD_set'])
            print("Dropped column 'FD_set' from data during fit_transform")

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Engine_ID', 'RUL', 'Health_State', 'Flight_cycle_number']
        numeric_columns = [col for col in numeric_columns if col not in exclude_cols]

        df = self.detect_and_handle_outliers(df, numeric_columns)
        if 'RUL' in df.columns:
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X_features = df[feature_cols]
            y_target = df['RUL']
            X_selected = self.select_features(X_features, y_target)
            selected_feature_names = X_selected.columns.tolist()
            keep_cols = exclude_cols + selected_feature_names
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols]
            df = self.scale_features(df, selected_feature_names, fit=True)
            if self.config['pca_components']:
                X_pca = self.apply_pca(df[selected_feature_names], fit=True)
                df = df[exclude_cols].join(X_pca)
        else:
            feature_cols = [col for col in numeric_columns if col in df.columns]
            df = self.scale_features(df, feature_cols, fit=True)

        self.generate_preprocessing_report(df_original, df)
        return df

    def transform(self, df):
        print("Transforming new data using fitted preprocessor...")
        if 'Health_State' in df.columns and 'Health_State' in self.encoders:
            df['Health_State'] = df['Health_State'].map(self.encoders['Health_State'])
        df = self.advanced_feature_engineering(df)
        df = self.handle_missing_values(df)

        if 'FD_set' in df.columns:
            df = df.drop(columns=['FD_set'])
            print("Dropped column 'FD_set' from data")

        if 'combined' in self.feature_selectors:
            exclude_cols = ['Engine_ID', 'RUL', 'Health_State', 'Flight_cycle_number']
            keep_cols = exclude_cols + self.feature_selectors['combined']
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols]
            df = self.scale_features(df, self.feature_selectors['combined'], fit=False)
            if 'pca' in self.scalers:
                X_pca = self.apply_pca(df[self.feature_selectors['combined']], fit=False)
                df = df[exclude_cols].join(X_pca)
        print(f"Data transformation completed. Final shape: {df.shape}")
        return df


def main():
    config = {
        'scaling_method': 'robust',
        'feature_selection': True,
        'n_features_to_select': 20,
        'outlier_detection': True,
        'outlier_method': 'iqr',
        'outlier_threshold': 3.0,
        'advanced_features': True,
        'pca_components': None,
        'rolling_windows': [3, 5, 10],
        'interaction_features': True,
        'polynomial_features': False,
        'polynomial_degree': 2,
    }

    input_path = 'data/cmapss_merged_curated.csv'
    output_path = 'data/advanced_processed_data.csv'
    preprocessor_path = 'models/preprocessor.pkl'

    preprocessor = AdvancedPreprocessor(config)
    df = preprocessor.load_data(input_path)
    df_processed = preprocessor.fit_transform(df)

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df_processed.to_csv(output_path, index=False)
    preprocessor.save_preprocessor(preprocessor_path)

    print(f"\nProcessed data saved to: {output_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print("\nPreprocessing pipeline completed successfully!")


if __name__ == '__main__':
    main()
