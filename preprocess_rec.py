import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# File paths
DATASET_PATH = "Dataset/tax_strategy_dataset.csv"
CLEANED_DATA_PATH = "Dataset/cleaned_tax_strategy_dataset.csv"

def load_data():
    df = pd.read_csv(DATASET_PATH)
    return df

def preprocess_data(df):
    df.fillna(0, inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in ['EmploymentType', 'MaritalStatus', 'EffectiveStrategy']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Computed features
    df['NetTaxableIncome'] = df['AnnualIncome'] - (df['Deductions'] + df['HRA'])
    df['TaxLiability'] = df['NetTaxableIncome'] * 0.1
    df['TaxSavingPotential'] = (df['Deductions'] * 0.2) + (df['HRA'] * 0.1)

    # ✅ Additional computed features
    df['IsSeniorCitizen'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
    df['IsHighEarner'] = df['AnnualIncome'].apply(lambda x: 1 if x >= 1000000 else 0)
    df['TaxBurdenRatio'] = df['TaxLiability'] / df['AnnualIncome']

    # Save cleaned data
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print("✅ Data preprocessed & saved!")

    return df

def eda(df):
    print("\n📊 Data Summary:\n", df.describe())
    print("\n🔍 Null Values:\n", df.isnull().sum())

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Pairplot
    sns.pairplot(df[['AnnualIncome', 'Deductions', 'HRA', 'Age', 'TaxLiability']], diag_kind='kde')
    plt.suptitle("Pairplot of Financial Variables", y=1.02)
    plt.show()

    # Distribution plots
    for col in ['AnnualIncome', 'Deductions', 'TaxLiability', 'TaxSavingPotential', 'Age']:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # Countplots for categorical features
    for col in ['EmploymentType', 'MaritalStatus', 'EffectiveStrategy']:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col)
        plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Boxplot - Strategy vs Income
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='EffectiveStrategy', y='AnnualIncome', data=df)
    plt.title("Annual Income by Strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Annual Income")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    eda(df)
