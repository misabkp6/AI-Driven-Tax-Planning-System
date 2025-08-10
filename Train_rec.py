import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
import joblib
import json
import os
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

# Create required directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'dtype') and obj.dtype.kind in ['i', 'u', 'f']:
        return obj.item()
    return obj

def determine_strategy(row: pd.Series) -> str:
    """Determine tax strategy based on financial profile."""
    income = row['AnnualIncome']
    tax_burden = row.get('TaxBurdenRatio', 0)
    age = row['Age']
    investments = row.get('Investments', 0)
    
    # Updated logic with strict age requirements for Senior_Planning
    is_senior = age >= 60
    
    if is_senior:
        return 'Senior_Planning'
    elif income > 1500000:
        return 'High_Income_Strategy'
    elif income > 1000000:
        return 'Aggressive_Tax_Planning'
    elif tax_burden > 0.2:
        return '80C_Investment'
    else:
        return 'Standard_Planning'

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict, List]:
    """Prepare and validate data for training."""
    print("\nAvailable columns:", df.columns.tolist())
    
    df = df.copy()
    
    # Create financial ratios if TaxLiability exists
    if 'TaxLiability' in df.columns and 'AnnualIncome' in df.columns:
        df['TaxBurdenRatio'] = (df['TaxLiability'] / df['AnnualIncome']).fillna(0)
    else:
        df['TaxBurdenRatio'] = 0
    
    # Required features mapping (old_name: new_name)
    column_mapping = {
        'Income': 'AnnualIncome',
        'Annual_Income': 'AnnualIncome',
        'Investment': 'Investments',
        'Deduction': 'Deductions',
        'House_Rent_Allowance': 'HRA',
        'Employment': 'EmploymentType',
        'Employment_Type': 'EmploymentType',
        'Marital_Status': 'MaritalStatus'
    }
    
    # Rename columns if alternative names exist
    df = df.rename(columns=column_mapping)
    
    # Default values for missing columns
    default_values = {
        'AnnualIncome': df['NetTaxableIncome'].mean() if 'NetTaxableIncome' in df.columns else 500000,
        'Investments': 0,
        'Deductions': df['Deductions'].mean() if 'Deductions' in df.columns else 0,
        'HRA': df['HRA'].mean() if 'HRA' in df.columns else 0,
        'Age': df['Age'].mean() if 'Age' in df.columns else 35,
        'EmploymentType': 'Private',
        'MaritalStatus': 'Single'
    }
    
    # Create missing columns with defaults
    for col, default in default_values.items():
        if col not in df.columns:
            df[col] = default
            print(f"Created {col} with default value: {default}")
    
    # Handle categorical features
    le_dict = {}
    categorical_features = ['EmploymentType', 'MaritalStatus']
    
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = df[feature].fillna('Unknown').astype(str)
        df[feature] = le.fit_transform(df[feature])
        le_dict[feature] = {
            'classes': le.classes_.tolist(),
            'mappings': {str(k): int(v) for k, v in 
                        zip(le.classes_, le.transform(le.classes_))}
        }
    
    # Create strategy if not present or regenerate to ensure correct mapping
    if 'EffectiveStrategy' not in df.columns:
        print("Creating EffectiveStrategy classification")
        df['EffectiveStrategy'] = df.apply(determine_strategy, axis=1)
    else:
        # Verify no incorrect assignments like Senior_Planning for young people
        incorrect_senior = ((df['EffectiveStrategy'] == 'Senior_Planning') & (df['Age'] < 60)).sum()
        if incorrect_senior > 0:
            print(f"Found {incorrect_senior} records with incorrect Senior_Planning assignments")
            print("Regenerating strategy assignments to enforce rules")
            df['EffectiveStrategy'] = df.apply(determine_strategy, axis=1)
    
    # Select and validate features for training
    feature_columns = [
        'AnnualIncome', 'Investments', 'Deductions', 'HRA',
        'Age', 'EmploymentType', 'MaritalStatus'
    ]
    
    # Verify all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after preparation: {missing_cols}")
    
    X = df[feature_columns]
    y = df['EffectiveStrategy']
    
    # Store list of unique target classes for prediction
    target_classes = sorted(y.unique().tolist())
    print(f"Target classes: {target_classes}")
    
    print("\nFeature columns ready:", X.columns.tolist())
    print("Target variable ready:", y.name)
    
    # Print strategy distribution for verification
    strategy_counts = y.value_counts()
    print("\nStrategy distribution:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count} ({count/len(df)*100:.1f}%)")
    
    # Verify age distribution for senior strategy
    senior_age_min = df[df['EffectiveStrategy'] == 'Senior_Planning']['Age'].min() if 'Senior_Planning' in y.values else 'N/A'
    print(f"Minimum age for Senior_Planning: {senior_age_min}")
    
    return X, y, le_dict, target_classes

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train and evaluate the model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, n_jobs=-1)
    
    y_pred = pipeline.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame(
        pipeline.named_steps['classifier'].feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    metrics = {
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'classification_report': class_report,
        'feature_importance': feature_importance.to_dict()
    }
    
    # Return additional data for visualization
    return pipeline, metrics, X_test, y_test, X_train, y_train

def plot_roc_curve_history(model, X_train, y_train, X_test, y_test, target_classes):
    """Generate ROC curve history showing model improvement during training"""
    plt.figure(figsize=(12, 10))
    
    # Convert target classes to numerical for ROC curve calculation
    le = LabelEncoder()
    le.fit(target_classes)
    
    # Create a reduced training history with incremental models
    n_trees_options = [10, 25, 50, 100, 150, 200]  # Different ensemble sizes
    
    # Get classifier from pipeline
    rf_classifier = model.named_steps['classifier']
    base_params = rf_classifier.get_params()
    
    # Plot ROC curves for different training stages
    for n_trees in n_trees_options:
        # Create model with specific number of trees
        reduced_model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=base_params['max_depth'],
            min_samples_split=base_params['min_samples_split'],
            min_samples_leaf=base_params['min_samples_leaf'],
            random_state=base_params['random_state'],
            class_weight=base_params['class_weight'],
            n_jobs=1  # Use single job for consistent training
        )
        
        # Train on the same data
        reduced_model.fit(model.named_steps['scaler'].transform(X_train), y_train)
        
        # For multi-class, we compute macro-average ROC curve 
        y_score = reduced_model.predict_proba(model.named_steps['scaler'].transform(X_test))
        
        # Plot a ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        # Plot only a subset of classes for clarity (max 3)
        plot_classes = target_classes[:min(3, len(target_classes))]
        
        for i, class_name in enumerate(plot_classes):
            class_idx = list(le.classes_).index(class_name) if class_name in le.classes_ else -1
            if class_idx == -1:
                continue
                
            y_test_binary = (y_test == class_name).astype(int)
            
            try:
                # Calculate ROC curve and area
                fpr[class_name], tpr[class_name], _ = roc_curve(
                    y_test_binary, 
                    y_score[:, class_idx]
                )
                roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
                
                linestyle = '-' if n_trees == n_trees_options[-1] else ':'
                linewidth = 2 if n_trees == n_trees_options[-1] else 1
                alpha = 1.0 if n_trees == n_trees_options[-1] else 0.6 + 0.4 * (n_trees_options.index(n_trees) / len(n_trees_options))
                
                # Plot ROC curve for this class and n_trees
                plt.plot(
                    fpr[class_name], 
                    tpr[class_name],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=f"{class_name} (Trees={n_trees}, AUC={roc_auc[class_name]:.2f})"
                )
            except Exception as e:
                print(f"Error plotting ROC for {class_name} with {n_trees} trees: {e}")
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Across Model Training History')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/roc_curve_history.png', dpi=300)
    plt.close()
    
    print("Training history ROC curve saved")

def visualize_model_performance(pipeline, X_train, y_train, X_test, y_test, metrics, target_classes):
    """Generate and save performance visualizations."""
    print("\n📊 Generating performance visualizations...")
    
    # Create a directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Get predictions for test data
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # ----- Accuracy Bar Chart -----
    plt.figure(figsize=(10, 6))
    accuracy_data = [
        metrics['train_accuracy'], 
        metrics['test_accuracy'], 
        metrics['cv_accuracy_mean']
    ]
    accuracy_labels = ['Training', 'Test', 'Cross-Validation']
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    bars = plt.bar(accuracy_labels, accuracy_data, color=colors)
    plt.ylim(max(0.7, min(accuracy_data) - 0.05), 1.0)
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/accuracy_comparison.png', dpi=300)
    plt.close()
    
    # ----- Confusion Matrix -----
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_classes, 
                yticklabels=target_classes)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png', dpi=300)
    plt.close()
    
    # ----- Feature Importance -----
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame(list(metrics['feature_importance']['importance'].items()), 
                               columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importance', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300)
    plt.close()
    
    # ----- ROC Curves -----
    plt.figure(figsize=(12, 10))
    
    # Create a label encoder to ensure consistent mapping
    le = LabelEncoder()
    le.fit(target_classes)
    
    # For each class
    for i, class_name in enumerate(target_classes):
        # Prepare binary classification (one vs rest)
        y_test_binary = (y_test == class_name).astype(int)
        
        # Check if the class index is valid
        if i >= len(pipeline.named_steps['classifier'].classes_):
            print(f"Warning: Class index {i} for {class_name} is out of bounds")
            continue
            
        try:
            y_score = y_pred_proba[:, i]
            
            # Calculate ROC
            fpr, tpr, _ = roc_curve(y_test_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, 
                    label=f'ROC for {class_name} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Error plotting ROC for {class_name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-Class Classification', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/roc_curves.png', dpi=300)
    plt.close()
    
    # ----- Combined Dashboard -----
    plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_classes, 
                yticklabels=target_classes)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 2. Accuracy comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(accuracy_labels, accuracy_data, color=colors)
    plt.ylim(max(0.7, min(accuracy_data) - 0.05), 1.0)
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.ylabel('Accuracy Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11)
    
    # 3. Feature Importance
    plt.subplot(2, 2, 3)
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importance', fontsize=14)
    
    # 4. Class Distribution
    plt.subplot(2, 2, 4)
    class_distribution = y_test.value_counts().reindex(target_classes, fill_value=0)
    wedges, texts, autotexts = plt.pie(class_distribution, 
                                       labels=class_distribution.index,
                                       autopct='%1.1f%%',
                                       textprops={'fontsize': 9},
                                       colors=sns.color_palette('pastel', len(class_distribution)))
    plt.title('Class Distribution in Test Set', fontsize=14)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('visualizations/model_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----- Training History ROC Curve -----
    try:
        plot_roc_curve_history(pipeline, X_train, y_train, X_test, y_test, target_classes)
    except Exception as e:
        print(f"Error generating ROC curve history: {e}")
    
    print(f"All visualizations saved to 'visualizations/' directory")

def save_model(pipeline: Pipeline, metrics: Dict, le_dict: Dict, target_classes: List[str]) -> None:
    """Save model and metadata with proper type conversion."""
    # Save model
    joblib.dump(pipeline, "models/tax_strategy_model.pkl")
    
    # Convert model parameters
    model_params = pipeline.named_steps['classifier'].get_params()
    serializable_params = convert_to_serializable(model_params)
    
    # Create a mapping from strategy names to indices
    strategy_to_idx = {strategy: i for i, strategy in enumerate(target_classes)}
    
    # Prepare metadata
    metadata = {
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_version': '2.0.0',
        'feature_encodings': convert_to_serializable(le_dict),
        'performance_metrics': convert_to_serializable(metrics),
        'model_parameters': serializable_params,
        'target_classes': convert_to_serializable(target_classes),
        'strategy_mapping': convert_to_serializable(strategy_to_idx)
    }
    
    # Save metadata
    with open('models/strategy_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model and metadata saved with {len(target_classes)} strategy classes")
    print(f"Strategy-to-index mapping: {strategy_to_idx}")

def main() -> None:
    try:
        # Load data
        print("📊 Loading data...")
        df = pd.read_csv("Dataset/cleaned_tax_strategy_dataset.csv")
        print(f"Loaded {len(df)} records")
        
        # Prepare data
        print("\n🔄 Preparing data...")
        X, y, le_dict, target_classes = prepare_data(df)
        
        # Train model
        print("\n🚀 Training model...")
        pipeline, metrics, X_test, y_test, X_train, y_train = train_model(X, y)
        
        # Generate visualizations
        visualize_model_performance(pipeline, X_train, y_train, X_test, y_test, metrics, target_classes)
        
        # Save model and metadata
        print("\n💾 Saving model and metadata...")
        save_model(pipeline, metrics, le_dict, target_classes)
        
        # Print results
        print("\n📊 Model Performance:")
        print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Cross-validation Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print("\n📋 Classification Report:")
        print(metrics['classification_report'])
        
        # Print feature importance
        print("\n🔍 Feature Importance:")
        for feature, importance in metrics['feature_importance']['importance'].items():
            print(f"{feature}: {importance:.4f}")
        
        print("\n✅ Training completed successfully!")
        print(f"Visualizations are available in the 'visualizations/' directory")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()