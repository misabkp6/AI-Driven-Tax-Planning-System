import os
import joblib
import json
import pandas as pd

def debug_model():
    print("==== Model Debugging ====")
    
    # Check if model file exists
    model_path = "models/tax_strategy_model.pkl"
    if os.path.exists(model_path):
        print(f"✅ Model file exists at {model_path}")
        try:
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully: {type(model)}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    else:
        print(f"❌ Model file NOT found at {model_path}")
    
    # Check if metadata exists
    metadata_path = "models/strategy_model_metadata.json"
    if os.path.exists(metadata_path):
        print(f"✅ Metadata file exists at {metadata_path}")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded successfully")
            
            # Check target classes
            if 'target_classes' in metadata:
                print(f"✅ Target classes found: {metadata['target_classes']}")
            else:
                print(f"❌ Target classes NOT found in metadata")
                
            # Check encodings
            if 'feature_encodings' in metadata:
                print(f"✅ Feature encodings found for: {list(metadata['feature_encodings'].keys())}")
            else:
                print(f"❌ Feature encodings NOT found in metadata")
        except Exception as e:
            print(f"❌ Error loading metadata: {str(e)}")
    else:
        print(f"❌ Metadata file NOT found at {metadata_path}")
    
    # Test prediction with sample data
    if os.path.exists(model_path):
        try:
            from predict import predict_tax_strategy
            
            test_input = {
                "AnnualIncome": 820000,
                "Investments": 150000,
                "Deductions": 50000,
                "HRA": 120000,
                "Age": 35,
                "EmploymentType": "Private",
                "MaritalStatus": "Married"
            }
            
            print("\n==== Testing Prediction ====")
            result = predict_tax_strategy(test_input)
            
            print(f"Result strategy: {result.get('recommended_strategy', 'None')}")
            print(f"Recommendations: {result.get('recommendations', [])}")
            
        except Exception as e:
            print(f"❌ Error with prediction: {str(e)}")

if __name__ == "__main__":
    debug_model()