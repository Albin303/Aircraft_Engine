import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

class AdvancedDataSplitter:
    """
    Advanced data splitting with stratification and validation strategies
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.split_info = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the processed dataset with validation."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        print(f"Data loaded from {filepath} with shape {df.shape}")
        
        # Validate required columns
        required_columns = ['Engine_ID', 'RUL']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    
    def analyze_data_distribution(self, df: pd.DataFrame) -> None:
        """Analyze data distribution for better splitting strategy"""
        print("\n" + "="*50)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Engine statistics
        engine_stats = df.groupby('Engine_ID').agg({
            'RUL': ['min', 'max', 'mean', 'count'],
            'Flight_cycle_number': 'max' if 'Flight_cycle_number' in df.columns else 'count'
        }).round(2)
        
        print(f"Total engines: {df['Engine_ID'].nunique()}")
        print(f"Total samples: {len(df)}")
        print(f"Average samples per engine: {len(df) / df['Engine_ID'].nunique():.1f}")
        
        # RUL distribution
        print(f"\nRUL Statistics:")
        print(f"  Min RUL: {df['RUL'].min()}")
        print(f"  Max RUL: {df['RUL'].max()}")
        print(f"  Mean RUL: {df['RUL'].mean():.2f}")
        print(f"  Std RUL: {df['RUL'].std():.2f}")
        
        # Health state distribution if available
        if 'Health_State' in df.columns:
            health_dist = df['Health_State'].value_counts()
            print(f"\nHealth State Distribution:")
            for state, count in health_dist.items():
                print(f"  {state}: {count} ({count/len(df)*100:.1f}%)")
        
        self.split_info['data_stats'] = {
            'total_engines': df['Engine_ID'].nunique(),
            'total_samples': len(df),
            'avg_samples_per_engine': len(df) / df['Engine_ID'].nunique(),
            'rul_stats': df['RUL'].describe().to_dict()
        }
    
    def create_stratification_groups(self, df: pd.DataFrame, 
                                   method: str = 'rul_bins') -> pd.Series:
        """Create stratification groups for balanced splitting"""
        
        if method == 'rul_bins':
            # Create RUL-based stratification bins
            discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            rul_bins = discretizer.fit_transform(df[['RUL']]).ravel()
            
            # Combine with health state if available
            if 'Health_State' in df.columns:
                # Create combined stratification key
                strat_groups = df['Health_State'].astype(str) + '_' + rul_bins.astype(str)
            else:
                strat_groups = rul_bins.astype(str)
                
        elif method == 'engine_lifecycle':
            # Stratify based on engine lifecycle stage
            if 'Flight_cycle_number' in df.columns:
                engine_max_cycles = df.groupby('Engine_ID')['Flight_cycle_number'].max()
                df_with_max = df.merge(engine_max_cycles.rename('Max_Cycle'), 
                                     left_on='Engine_ID', right_index=True)
                df_with_max['Lifecycle_Stage'] = pd.cut(
                    df_with_max['Flight_cycle_number'] / df_with_max['Max_Cycle'],
                    bins=[0, 0.25, 0.5, 0.75, 1.0],
                    labels=['Early', 'Mid_Early', 'Mid_Late', 'Late']
                )
                strat_groups = df_with_max['Lifecycle_Stage'].astype(str)
            else:
                # Fallback to RUL bins
                discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                strat_groups = discretizer.fit_transform(df[['RUL']]).ravel().astype(str)
        
        else:
            raise ValueError(f"Unknown stratification method: {method}")
        
        return pd.Series(strat_groups, index=df.index)
    
    def split_data_by_engine_stratified(self, df: pd.DataFrame, 
                                      train_frac: float = 0.8,
                                      val_frac: float = 0.1,
                                      stratify_method: str = 'rul_bins',
                                      ensure_min_samples: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Advanced engine-based splitting with stratification and validation set
        """
        print(f"\nPerforming stratified engine-based split...")
        print(f"Train: {train_frac}, Validation: {val_frac}, Test: {1-train_frac-val_frac}")
        
        # Get unique engines and their characteristics for stratification
        engine_summary = df.groupby('Engine_ID').agg({
            'RUL': ['min', 'max', 'mean'],
            'Health_State': 'first' if 'Health_State' in df.columns else lambda x: 'unknown'
        }).reset_index()
        
        # Flatten column names
        engine_summary.columns = ['Engine_ID', 'RUL_min', 'RUL_max', 'RUL_mean', 'Health_State']
        
        # Create stratification groups at engine level
        if stratify_method == 'rul_bins':
            discretizer = KBinsDiscretizer(n_bins=min(5, len(engine_summary)//10), 
                                         encode='ordinal', strategy='quantile')
            engine_rul_bins = discretizer.fit_transform(engine_summary[['RUL_mean']]).ravel()
            
            if 'Health_State' in df.columns:
                engine_strat = engine_summary['Health_State'].astype(str) + '_' + engine_rul_bins.astype(str)
            else:
                engine_strat = engine_rul_bins.astype(str)
        
        # First split: separate test set
        if val_frac > 0:
            train_val_engines, test_engines = train_test_split(
                engine_summary['Engine_ID'], 
                test_size=1-train_frac-val_frac,
                random_state=self.random_state,
                stratify=engine_strat
            )
            
            # Second split: separate train and validation
            train_engines, val_engines = train_test_split(
                train_val_engines,
                test_size=val_frac/(train_frac+val_frac),
                random_state=self.random_state,
                stratify=engine_strat[train_val_engines.index]
            )
            
        else:
            # Simple train/test split
            train_engines, test_engines = train_test_split(
                engine_summary['Engine_ID'],
                test_size=1-train_frac,
                random_state=self.random_state,
                stratify=engine_strat
            )
            val_engines = pd.Series([], dtype='int64')
        
        # Create dataframes
        train_df = df[df['Engine_ID'].isin(train_engines)].reset_index(drop=True)
        test_df = df[df['Engine_ID'].isin(test_engines)].reset_index(drop=True)
        
        if val_frac > 0:
            val_df = df[df['Engine_ID'].isin(val_engines)].reset_index(drop=True)
        else:
            val_df = pd.DataFrame()
        
        # Ensure minimum samples
        if len(train_df) < ensure_min_samples:
            print(f"Warning: Training set has only {len(train_df)} samples (minimum: {ensure_min_samples})")
        
        # Store split information
        self.split_info['train_engines'] = len(train_engines)
        self.split_info['val_engines'] = len(val_engines) if val_frac > 0 else 0
        self.split_info['test_engines'] = len(test_engines)
        self.split_info['train_samples'] = len(train_df)
        self.split_info['val_samples'] = len(val_df) if val_frac > 0 else 0
        self.split_info['test_samples'] = len(test_df)
        
        print(f"Split completed:")
        print(f"  Training: {len(train_engines)} engines, {len(train_df)} samples")
        if val_frac > 0:
            print(f"  Validation: {len(val_engines)} engines, {len(val_df)} samples")
        print(f"  Test: {len(test_engines)} engines, {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def validate_split_quality(self, train_df: pd.DataFrame, 
                             val_df: pd.DataFrame, 
                             test_df: pd.DataFrame) -> None:
        """Validate the quality of the data split"""
        print(f"\n" + "="*50)
        print("SPLIT QUALITY VALIDATION")
        print("="*50)
        
        datasets = [('Train', train_df), ('Test', test_df)]
        if not val_df.empty:
            datasets.insert(1, ('Validation', val_df))
        
        # Statistical comparison
        for name, df in datasets:
            print(f"\n{name} Set Statistics:")
            print(f"  Engines: {df['Engine_ID'].nunique()}")
            print(f"  Samples: {len(df)}")
            print(f"  RUL - Mean: {df['RUL'].mean():.2f}, Std: {df['RUL'].std():.2f}")
            print(f"  RUL - Min: {df['RUL'].min()}, Max: {df['RUL'].max()}")
            
            if 'Health_State' in df.columns:
                health_dist = df['Health_State'].value_counts(normalize=True) * 100
                print(f"  Health Distribution: {dict(health_dist.round(1))}")
        
        # Check for data leakage (same engines in different sets)
        train_engines = set(train_df['Engine_ID'].unique())
        test_engines = set(test_df['Engine_ID'].unique())
        
        if not val_df.empty:
            val_engines = set(val_df['Engine_ID'].unique())
            overlap_train_val = train_engines.intersection(val_engines)
            overlap_val_test = val_engines.intersection(test_engines)
            
            if overlap_train_val or overlap_val_test:
                print(f"\nWARNING: Data leakage detected!")
                if overlap_train_val:
                    print(f"  Train-Validation overlap: {len(overlap_train_val)} engines")
                if overlap_val_test:
                    print(f"  Validation-Test overlap: {len(overlap_val_test)} engines")
            else:
                print(f"\n✓ No data leakage detected between sets")
        
        overlap_train_test = train_engines.intersection(test_engines)
        if overlap_train_test:
            print(f"WARNING: Train-Test overlap: {len(overlap_train_test)} engines")
        else:
            print(f"✓ No data leakage between train and test sets")
    
    def separate_features_and_labels(self, df: pd.DataFrame, 
                                   label_col: str = 'RUL',
                                   exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Enhanced feature-label separation with automatic exclusion detection
        """
        if exclude_cols is None:
            exclude_cols = ['Engine_ID', 'FD_set', 'Flight_cycle_number']
        
        # Add label column to exclusions
        exclude_cols = list(set(exclude_cols + [label_col]))
        
        # Automatically detect non-feature columns
        auto_exclude = []
        for col in df.columns:
            if col.lower() in ['index', 'id', 'timestamp', 'date', 'time']:
                auto_exclude.append(col)
            elif df[col].dtype == 'object' and col not in [label_col]:
                auto_exclude.append(col)
        
        exclude_cols.extend(auto_exclude)
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns found after exclusions")
        
        X = df[feature_cols]
        y = df[label_col] if label_col in df.columns else pd.Series()
        
        print(f"Features extracted: {X.shape}")
        print(f"Labels extracted: {y.shape}")
        print(f"Excluded columns: {exclude_cols}")
        
        return X, y
    
    def create_time_series_splits(self, df: pd.DataFrame, 
                                n_splits: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time-series aware splits for temporal validation"""
        if 'Flight_cycle_number' not in df.columns:
            raise ValueError("Flight_cycle_number required for time series splits")
            
        splits = []
        engines = df['Engine_ID'].unique()
        
        for engine in engines:
            engine_data = df[df['Engine_ID'] == engine].sort_values('Flight_cycle_number')
            n_samples = len(engine_data)
            
            if n_samples < n_splits + 1:
                continue
                
            # Create progressive splits (growing train set, fixed test size)
            test_size = max(1, n_samples // (n_splits + 1))
            
            for i in range(n_splits):
                train_end = n_samples - (n_splits - i) * test_size
                test_start = train_end
                test_end = min(test_start + test_size, n_samples)
                
                if train_end > 0 and test_end > test_start:
                    train_indices = engine_data.index[:train_end]
                    test_indices = engine_data.index[test_start:test_end]
                    
                    splits.append((train_indices, test_indices))
        
        print(f"Created {len(splits)} time-series splits")
        return splits
    
    def plot_split_visualization(self, train_df: pd.DataFrame, 
                               val_df: pd.DataFrame, 
                               test_df: pd.DataFrame,
                               save_path: str = 'data/split_visualization.png') -> None:
        """Create visualization of the data split"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Split Analysis', fontsize=16, fontweight='bold')
        
        datasets = [('Train', train_df), ('Test', test_df)]
        if not val_df.empty:
            datasets.insert(1, ('Validation', val_df))
        
        colors = ['blue', 'green', 'red'][:len(datasets)]
        
        # Plot 1: RUL distribution
        for i, (name, df) in enumerate(datasets):
            axes[0,0].hist(df['RUL'], alpha=0.6, label=name, color=colors[i], bins=30)
        axes[0,0].set_xlabel('RUL')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('RUL Distribution by Split')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Sample count by set
        set_names = [name for name, _ in datasets]
        sample_counts = [len(df) for _, df in datasets]
        axes[0,1].bar(set_names, sample_counts, color=colors)
        axes[0,1].set_ylabel('Number of Samples')
        axes[0,1].set_title('Sample Count by Set')
        for i, count in enumerate(sample_counts):
            axes[0,1].text(i, count + max(sample_counts)*0.01, str(count), 
                          ha='center', va='bottom')
        
        # Plot 3: Engine count by set
        engine_counts = [df['Engine_ID'].nunique() for _, df in datasets]
        axes[1,0].bar(set_names, engine_counts, color=colors)
        axes[1,0].set_ylabel('Number of Engines')
        axes[1,0].set_title('Engine Count by Set')
        for i, count in enumerate(engine_counts):
            axes[1,0].text(i, count + max(engine_counts)*0.01, str(count), 
                          ha='center', va='bottom')
        
        # Plot 4: Health state distribution (if available)
        if 'Health_State' in train_df.columns:
            health_data = []
            for name, df in datasets:
                health_dist = df['Health_State'].value_counts(normalize=True)
                health_data.append(health_dist)
            
            health_df = pd.DataFrame(health_data, index=set_names).fillna(0)
            health_df.plot(kind='bar', ax=axes[1,1], stacked=True)
            axes[1,1].set_ylabel('Proportion')
            axes[1,1].set_title('Health State Distribution')
            axes[1,1].legend(title='Health State')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            # Plot average samples per engine
            avg_samples = [len(df)/df['Engine_ID'].nunique() for _, df in datasets]
            axes[1,1].bar(set_names, avg_samples, color=colors)
            axes[1,1].set_ylabel('Avg Samples per Engine')
            axes[1,1].set_title('Average Samples per Engine')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Split visualization saved to {save_path}")
        plt.show()
    
    def save_split_info(self, filepath: str) -> None:
        """Save split information and statistics"""
        import json
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.split_info, f, indent=2)
        print(f"Split information saved to {filepath}")

def main():
    """Main data splitting function"""
    
    # Configuration
    TRAIN_FRAC = 0.7
    VAL_FRAC = 0.15  # 15% for validation
    TEST_FRAC = 0.15  # 15% for test
    RANDOM_STATE = 42
    
    # Paths
    input_path = 'data/advanced_processed_data.csv'
    
    # Initialize splitter
    splitter = AdvancedDataSplitter(random_state=RANDOM_STATE)
    
    # Load and analyze data
    df = splitter.load_data(input_path)
    splitter.analyze_data_distribution(df)
    
    # Perform stratified splitting
    train_df, val_df, test_df = splitter.split_data_by_engine_stratified(
        df, 
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        stratify_method='rul_bins'
    )
    
    # Validate split quality
    splitter.validate_split_quality(train_df, val_df, test_df)
    
    # Separate features and labels
    X_train, y_train = splitter.separate_features_and_labels(train_df, label_col='RUL')
    X_test, y_test = splitter.separate_features_and_labels(test_df, label_col='RUL')
    
    if not val_df.empty:
        X_val, y_val = splitter.separate_features_and_labels(val_df, label_col='RUL')
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    
    # Save all splits
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    X_train.to_csv('data/X_train.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    if not val_df.empty:
        val_df.to_csv('data/val_data.csv', index=False)
        X_val.to_csv('data/X_val.csv', index=False)
        y_val.to_csv('data/y_val.csv', index=False)
        print("Validation set files saved")
    
    # Create visualization
    splitter.plot_split_visualization(train_df, val_df, test_df)
    
    # Save split information
    splitter.save_split_info('data/split_info.json')
    
    print(f"\n" + "="*60)
    print("DATA SPLITTING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Files saved:")
    print(f"  - Training data: data/train_data.csv ({len(train_df)} samples)")
    print(f"  - Training features: data/X_train.csv {X_train.shape}")
    print(f"  - Training labels: data/y_train.csv {y_train.shape}")
    if not val_df.empty:
        print(f"  - Validation data: data/val_data.csv ({len(val_df)} samples)")
        print(f"  - Validation features: data/X_val.csv {X_val.shape}")
        print(f"  - Validation labels: data/y_val.csv {y_val.shape}")
    print(f"  - Test data: data/test_data.csv ({len(test_df)} samples)")
    print(f"  - Test features: data/X_test.csv {X_test.shape}")
    print(f"  - Test labels: data/y_test.csv {y_test.shape}")
    print(f"  - Split visualization: data/split_visualization.png")
    print(f"  - Split info: data/split_info.json")

if __name__ == '__main__':
    main()