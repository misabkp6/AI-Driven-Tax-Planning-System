TAX_CONFIG = {
    "model_params": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 2, 3],
        "gamma": [0, 0.1, 0.2],
        "colsample_bytree": [0.7, 0.8, 0.9]
    },
    "tax_limits": {
        "80C": 150000,
        "HRA": 0.4,  # 40% of basic salary
        "Medical": 25000,
        "NPS": 50000,
        "Home_Loan": 200000
    }
}