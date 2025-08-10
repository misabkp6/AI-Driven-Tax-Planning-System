import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   RandomizedSearchCV, KFold)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import logging
from datetime import datetime
from typing import Dict, Tuple, List

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1
MODEL_VERSION = "3.2.0"
L1_ALPHA = 0.01
L2_LAMBDA = 1.0

# Valid categories for categorical features
CATEGORICAL_FEATURES = ['MaritalStatus', 'EmploymentType']
VALID_CATEGORIES = {
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'EmploymentType': ['Private', 'Public', 'Self-Employed']
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

class TaxFeatureEngineer:
    """Handle feature engineering for tax prediction."""
    
    def __init__(self):
        self.numerical_features = [
            'AnnualIncome', 'Investments', 'Deductions',
            'HRA', 'OtherIncome', 'Age', 'NumDependents',
            'BusinessExpenses'
        ]
        self.categorical_features = CATEGORICAL_FEATURES
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features with reduced complexity."""
        df = df.copy()
        
        # Clean categorical features first
        for feature in self.categorical_features:
            df[feature] = df[feature].apply(
                lambda x: self._get_closest_category(x, feature)
            )
        
        # Remove leaked features
        leaked_cols = ['PreviousTaxPaid', 'TaxableIncome', 'UpdatedTaxableIncome']
        df = df.drop(columns=[col for col in leaked_cols if col in df.columns])
        
        # Base calculations with safe division
        income_base = np.maximum(df['AnnualIncome'], 1)  # Avoid division by zero
        
        # Financial ratios
        df['investment_rate'] = (df['Investments'] / 150000).clip(0, 1)  # 80C limit
        df['expense_ratio'] = (df['BusinessExpenses'] / income_base).clip(0, 1)
        df['deduction_ratio'] = (df['Deductions'] / income_base).clip(0, 1)
        df['hra_ratio'] = (df['HRA'] / income_base).clip(0, 1)
        
        # Income features
        df['total_income'] = df['AnnualIncome'] + df['OtherIncome']
        df['net_income'] = df['total_income'] - df['BusinessExpenses']
        
        # Tax bracket (simplified)
        df['tax_bracket'] = pd.cut(
            df['total_income'],
            bins=[0, 500000, 1000000, float('inf')],
            labels=[0, 1, 2]
        ).astype(float)
        
        # Demographic features
        df['age_group'] = (df['Age'] // 10).clip(2, 6)
        df['dependent_factor'] = (df['NumDependents'] / 3).clip(0, 1)
        
        return df
    
    @staticmethod
    def _get_closest_category(value: str, feature: str) -> str:
        """Map value to closest valid category."""
        valid_cats = VALID_CATEGORIES[feature]
        value = str(value).lower()
        
        if pd.isna(value) or value not in [cat.lower() for cat in valid_cats]:
            if feature == 'EmploymentType':
                if 'self' in value or 'own' in value:
                    return 'Self-Employed'
                elif 'gov' in value or 'public' in value:
                    return 'Public'
                else:
                    return 'Private'
            elif feature == 'MaritalStatus':
                if 'single' in value or 'unmarried' in value:
                    return 'Single'
                elif 'divorced' in value or 'separated' in value:
                    return 'Divorced'
                else:
                    return 'Married'
        
        # Find exact match (case-insensitive)
        for cat in valid_cats:
            if cat.lower() == value.lower():
                return cat
        
        return valid_cats[0]  # Default to first category if no match

class TaxModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self):
        self.feature_engineer = TaxFeatureEngineer()
        self.model = None
        self.preprocessor = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for modeling."""
        df = self.feature_engineer.create_features(df)
        X = df.drop(['FinalTax', 'UpdatedFinalTax'], axis=1, errors='ignore')
        y = df[['FinalTax', 'UpdatedFinalTax']]
        return X, y
    
    def create_pipeline(self, params: Dict = None) -> Pipeline:
        """Create model pipeline with regularization."""
        if params is None:
            params = {
                'n_estimators': 50,
                'learning_rate': 0.03,
                'max_depth': 3,
                'min_child_weight': 3,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': L1_ALPHA,
                'reg_lambda': L2_LAMBDA,
                'objective': 'reg:squarederror'
            }
        
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler()),
            ('power', PowerTransformer(standardize=True))
        ])
        
        # Create preprocessor with explicit categories
        categories = [VALID_CATEGORIES[feature] for feature in CATEGORICAL_FEATURES]
        
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.feature_engineer.numerical_features),
            ('cat', OneHotEncoder(
                drop='first',
                sparse_output=False,
                categories=categories
            ), CATEGORICAL_FEATURES)
        ])
        
        self.model = xgb.XGBRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
        
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', self.model)
        ])
    
    def cross_validate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """Perform k-fold cross validation."""
        cv_scores = {
            'r2': [],
            'rmse': []
        }
        
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train_cv = X.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_train_cv = y.iloc[train_idx]
            y_val_cv = y.iloc[val_idx]
            
            pipeline = self.create_pipeline()
            pipeline.fit(X_train_cv, y_train_cv)
            
            pred = pipeline.predict(X_val_cv)
            cv_scores['r2'].append(r2_score(y_val_cv, pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_cv, pred)))
            
            print(f"Fold {fold}: R² = {cv_scores['r2'][-1]:.4f}, RMSE = ₹{cv_scores['rmse'][-1]:,.2f}")
        
        return {
            'cv_r2_mean': np.mean(cv_scores['r2']),
            'cv_r2_std': np.std(cv_scores['r2']),
            'cv_rmse_mean': np.mean(cv_scores['rmse']),
            'cv_rmse_std': np.std(cv_scores['rmse'])
        }

def main():
    """Main training pipeline."""
    try:
        # Setup
        for dir_name in ['models', 'logs', 'visualizations']:
            os.makedirs(dir_name, exist_ok=True)
        
        # Load and prepare data
        print("\n📊 Loading data...")
        df = pd.read_csv('Dataset/Enhanced_Indian_Tax_Dataset.csv')
        
        trainer = TaxModelTrainer()
        X, y = trainer.prepare_data(df)
        
        # Split data
        print("🔄 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Cross-validation
        print("\n📊 Performing cross-validation...")
        cv_metrics = trainer.cross_validate(X_train, y_train)
        
        # Train final model
        print("\n🚀 Training final model...")
        pipeline = trainer.create_pipeline()
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'cv_metrics': cv_metrics
        }
        
        # Save model and metadata
        joblib.dump(pipeline, 'models/tax_model.pkl')
        
        metadata = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": MODEL_VERSION,
            "performance_metrics": metrics,
            "feature_names": list(X.columns),
            "categories": VALID_CATEGORIES,
            "hyperparameters": pipeline.named_steps['regressor'].get_params()
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Print results
        print("\n📊 Model Performance:")
        print(f"Cross-validation R²: {cv_metrics['cv_r2_mean']:.4f} ± {cv_metrics['cv_r2_std']:.4f}")
        print(f"Cross-validation RMSE: ₹{cv_metrics['cv_rmse_mean']:,.2f} ± ₹{cv_metrics['cv_rmse_std']:,.2f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Test RMSE: ₹{metrics['test_rmse']:,.2f}")
        print(f"Test MAE: ₹{metrics['test_mae']:,.2f}")
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()