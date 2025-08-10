import numpy as np
import pandas as pd
import joblib
import os
import logging
import json
import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

# Setup logging
os.makedirs('logs', exist_ok=True)
log_file = os.path.join('logs', f'tax_prediction_{datetime.datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Make sure the models directory exists
os.makedirs('models', exist_ok=True)

def load_model():
    """Load the trained model and metadata."""
    try:
        model_path = "models/tax_strategy_model.pkl"
        metadata_path = "models/strategy_model_metadata.json"
        
        # Load model
        model = joblib.load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            logging.info(f"Metadata loaded successfully with keys: {list(metadata.keys())}")
            if 'target_classes' in metadata:
                logging.info(f"Target classes: {metadata['target_classes']}")
                if 'strategy_mapping' in metadata:
                    logging.info(f"Strategy mapping: {metadata['strategy_mapping']}")
            
        return model, metadata
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise

def determine_strategy(data: Dict[str, Any]) -> str:
    """Determine tax strategy based on financial profile."""
    try:
        income = data.get('AnnualIncome', 0)
        tax_liability = data.get('TaxLiability', 0)
        age = data.get('Age', 30)
        
        # Calculate tax burden ratio
        tax_burden = tax_liability / income if income > 0 else 0
        
        # Basic strategies based on age and income
        if age < 30:
            if income < 500000:
                return '80C_Investment'
            elif income < 1000000:
                return 'Aggressive_Tax_Planning'
            else:
                return 'High_Income_Strategy'
        elif age < 50:
            if income < 750000:
                return 'Standard_Planning'
            else:
                return 'High_Income_Strategy'
        else:
            return 'Senior_Planning'
    
    except Exception as e:
        logging.error(f"Error determining strategy: {str(e)}")
        return 'Standard_Planning'  # Default fallback

def get_strategy_details(strategy: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Get details for a given strategy."""
    age = data.get('Age', 30)
    income = data.get('AnnualIncome', 0)
    investments = data.get('Investments', 0)
    
    strategies = {
        '80C_Investment': {
            'name': '80C Investment Strategy',
            'description': 'Focus on maximizing Section 80C deductions',
            'recommendations': [
                'Invest in ELSS mutual funds for tax benefits with equity exposure',
                'Consider PPF for long-term tax-free returns',
                'Term life insurance premiums qualify for Section 80C',
                'Fixed deposits with 5-year lock-in period (tax saver FDs)',
                'National Savings Certificate (NSC) for guaranteed returns'
            ],
            'potential_savings': '₹46,800'
        },
        'Aggressive_Tax_Planning': {
            'name': 'Aggressive Tax Planning',
            'description': 'Comprehensive approach for maximizing multiple tax benefits',
            'recommendations': [
                'Max out 80C investments (₹1,50,000)',
                'Health insurance for additional 80D benefits (up to ₹25,000)',
                'NPS contribution for 80CCD(1B) benefits (extra ₹50,000)',
                'Home loan interest deduction if applicable',
                'Consider income splitting with family members'
            ],
            'potential_savings': '₹78,000+'
        },
        'High_Income_Strategy': {
            'name': 'High Income Strategy',
            'description': 'Advanced tax planning for higher income brackets',
            'recommendations': [
                'Maximize all available deductions (80C, 80D, 80CCD)',
                'Consider tax-efficient investments like arbitrage funds',
                'Evaluate home loan options for interest deductions',
                'Optimize capital gains realization timing',
                'Consider charitable donations for 80G benefits'
            ],
            'potential_savings': '₹1,25,000+'
        },
        'Standard_Planning': {
            'name': 'Standard Tax Planning',
            'description': 'Balanced approach for optimal tax efficiency',
            'recommendations': [
                'Balanced portfolio of ELSS, PPF and tax-saving FDs',
                'Health insurance for family (80D benefits)',
                'NPS for additional deduction and retirement planning',
                'Claim HRA benefits optimally',
                'Maintain adequate documentation for all deductions'
            ],
            'potential_savings': '₹52,000+'
        },
        'Senior_Planning': {
            'name': 'Senior Citizen Tax Strategy',
            'description': 'Tax planning optimized for seniors',
            'recommendations': [
                'Senior Citizens Savings Scheme (SCSS) investment',
                'Pradhan Mantri Vaya Vandana Yojana (PMVVY)',
                'Higher health insurance deduction (up to ₹50,000)',
                'Deduction for medical treatment (Sec. 80DDB)',
                'Tax-free interest up to ₹50,000 (Section 80TTB)'
            ],
            'potential_savings': '₹65,000+'
        }
    }
    
    # Default fallback strategy
    if strategy not in strategies:
        strategy = 'Standard_Planning'
    
    result = strategies[strategy].copy()
    
    # Customize potential savings based on income
    if income > 1000000:
        potential_savings = float(result['potential_savings'].replace('₹', '').replace(',', '').replace('+', ''))
        result['potential_savings'] = f"₹{potential_savings * 1.5:,.0f}+"
    elif income < 500000:
        potential_savings = float(result['potential_savings'].replace('₹', '').replace(',', '').replace('+', ''))
        result['potential_savings'] = f"₹{potential_savings * 0.6:,.0f}+"
    
    # Age-specific adjustments for recommendations
    if age < 30:
        if 'PPF' in ' '.join(result['recommendations']):
            result['recommendations'].append('Start tax planning early for compounding benefits')
    
    return result

def customize_recommendations(strategy: str, data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Customize recommendations based on user's specific financial situation."""
    age = data.get('Age', 30)
    income = data.get('AnnualIncome', 0)
    investments_80c = data.get('Investments', 0)
    
    logging.info(f"Recommendations customized based on: Age={age}, Income={income}, 80C={investments_80c}")
    
    # Copy original result to avoid modifying the source
    updated_result = result.copy()
    
    # Special handling for specific strategies
    if strategy == '80C_Investment':
        # If 80C already maxed out, adjust recommendations
        if investments_80c >= 150000:
            # Update description
            updated_result['strategy_description'] = "Advanced tax planning (80C already maximized)"
            
            # Replace 80C recommendations with beyond-80C options
            new_recs = []
            # Add a ✅ checkmark to the first recommendation
            new_recs.append(f"✅ You've already maximized Section 80C (₹{investments_80c:,})")
            
            # Add recommendations that aren't about 80C investments
            for rec in result['recommendations']:
                if '80C' in rec or 'Section 80C' in rec or any(item in rec.lower() for item in ['elss', 'ppf', 'fixed deposit']):
                    # Skip recommendations about 80C investments
                    continue
                else:
                    new_recs.append(rec)
            
            # Add additional beyond-80C recommendations
            new_recs.extend([
                "Life insurance premiums",
                "Consider NPS for additional ₹50,000 deduction",
                "Health insurance premiums (Section 80D)",
                "Education loan interest deduction (if applicable)"
            ])
            
            # Replace recommendations
            updated_result['recommendations'] = new_recs
    
    elif strategy == 'Senior_Planning' and age < 60:
        # This shouldn't normally happen due to business rule validation,
        # but as an extra safety check, modify recommendations if it does
        updated_result['strategy_description'] += " (adapted for pre-senior)"
        
        # Replace senior-specific recommendations
        recs = updated_result['recommendations']
        senior_terms = ['senior', 'scss', 'pmvvy', '80ttb']
        
        # Filter out senior-specific recommendations
        filtered_recs = [r for r in recs if not any(term in r.lower() for term in senior_terms)]
        
        # Add appropriate alternatives
        filtered_recs.extend([
            "Prepare for retirement with NPS contributions",
            "Build a balanced tax-saving portfolio for the transition to retirement",
            "Consider long-term health insurance plans"
        ])
        
        updated_result['recommendations'] = filtered_recs
    
    return updated_result

def generate_comprehensive_tax_plan(data: Dict[str, Any], main_strategy: str, strategy_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive tax planning approach using multiple strategies.
    
    Args:
        data: User's financial data
        main_strategy: The primary strategy predicted by ML
        strategy_info: The details of the main strategy
    
    Returns:
        Enhanced strategy details with comprehensive planning approach
    """
    age = data.get('Age', 30)
    income = data.get('AnnualIncome', 0)
    investments = data.get('Investments', 0)
    deductions = data.get('Deductions', 0)
    hra = data.get('HRA', 0)
    
    # Copy the original strategy info to avoid modifying it
    comprehensive_strategy = strategy_info.copy()
    
    # 1. Enhance the description to be more comprehensive
    comprehensive_strategy['description'] = f"{strategy_info['description']} with a comprehensive approach across multiple tax areas."
    
    # 2. Create structured recommendations by categories
    categorized_recommendations = {
        "income_tax": [],
        "investments": [],
        "insurance": [],
        "housing": [],
        "retirement": [],
        "additional": []
    }
    
    # 3. Add core recommendations from the main strategy to appropriate categories
    for rec in strategy_info['recommendations']:
        if any(term in rec.lower() for term in ['80c', 'elss', 'ppf', 'nsc', 'investment']):
            categorized_recommendations['investments'].append(rec)
        elif any(term in rec.lower() for term in ['insurance', 'premium']):
            categorized_recommendations['insurance'].append(rec)
        elif any(term in rec.lower() for term in ['home', 'hra', 'rent', 'loan']):
            categorized_recommendations['housing'].append(rec)
        elif any(term in rec.lower() for term in ['nps', 'retirement', 'senior']):
            categorized_recommendations['retirement'].append(rec)
        else:
            categorized_recommendations['additional'].append(rec)
    
    # 4. Add income-specific recommendations
    if income < 500000:
        categorized_recommendations['income_tax'].append(
            "Utilize the tax rebate under Section 87A (₹12,500 for income up to ₹5 lakh)")
    elif income < 1000000:
        categorized_recommendations['income_tax'].append(
            "Consider salary restructuring to maximize tax-free allowances")
    else:
        categorized_recommendations['income_tax'].append(
            "Evaluate both old and new tax regimes to determine optimal choice")
        categorized_recommendations['income_tax'].append(
            "Consider investing through HUF (Hindu Undivided Family) for additional tax benefits")
    
    # 5. Add investment recommendations based on profile
    if investments < 150000:
        remaining = 150000 - investments
        categorized_recommendations['investments'].append(
            f"You can still invest ₹{remaining:,} under 80C to reach the maximum limit")
        
        if income > 1000000:
            categorized_recommendations['investments'].append(
                "Consider debt funds (held >3 years) for tax-efficient returns with indexation benefits")
    else:
        categorized_recommendations['investments'].append(
            "Look beyond 80C - consider ELSS SIPs for tax benefits with equity exposure")
    
    # Always recommend NPS for additional tax benefits
    if not any('nps' in rec.lower() for rec in strategy_info['recommendations']):
        categorized_recommendations['retirement'].append(
            "Contribute to NPS for additional ₹50,000 deduction under Sec 80CCD(1B)")
    
    # 6. Add housing recommendations
    if hra > 0:
        categorized_recommendations['housing'].append(
            "Ensure you're claiming optimal HRA exemption with proper rent receipts/agreements")
    else:
        categorized_recommendations['housing'].append(
            "Consider rented accommodation to benefit from HRA exemption")
    
    # Add home loan recommendation for higher income individuals without one
    if income > 750000 and not any('home loan' in rec.lower() for rec in strategy_info['recommendations']):
        categorized_recommendations['housing'].append(
            "Explore home loan options for principal (80C) and interest deduction (up to ₹2 lakh)")
    
    # 7. Add health insurance recommendations
    if not any('health insurance' in rec.lower() for rec in strategy_info['recommendations']):
        if age < 40:
            categorized_recommendations['insurance'].append(
                "Purchase health insurance (Sec 80D) - premium up to ₹25,000 deductible")
        else:
            categorized_recommendations['insurance'].append(
                "Purchase comprehensive health insurance with adequate coverage (Sec 80D)")
    
    if age > 45:
        categorized_recommendations['insurance'].append(
            "Consider adding health insurance for parents - additional ₹50,000 deduction for senior parents")
    
    # 8. Add additional deduction recommendations
    if deductions < 50000:
        categorized_recommendations['additional'].append(
            "Consider donations to approved charities for deductions under Section 80G")
    
    # 9. Build final consolidated recommendations list
    final_recommendations = []
    
    if categorized_recommendations['income_tax']:
        final_recommendations.append("📋 INCOME TAX PLANNING:")
        final_recommendations.extend(categorized_recommendations['income_tax'])
        final_recommendations.append("")
    
    if categorized_recommendations['investments']:
        final_recommendations.append("💰 INVESTMENT STRATEGY:")
        final_recommendations.extend(categorized_recommendations['investments'])
        final_recommendations.append("")
    
    if categorized_recommendations['insurance']:
        final_recommendations.append("🛡️ INSURANCE BENEFITS:")
        final_recommendations.extend(categorized_recommendations['insurance'])
        final_recommendations.append("")
    
    if categorized_recommendations['housing']:
        final_recommendations.append("🏠 HOUSING & HRA:")
        final_recommendations.extend(categorized_recommendations['housing'])
        final_recommendations.append("")
    
    if categorized_recommendations['retirement']:
        final_recommendations.append("🔮 RETIREMENT PLANNING:")
        final_recommendations.extend(categorized_recommendations['retirement'])
        final_recommendations.append("")
    
    if categorized_recommendations['additional']:
        final_recommendations.append("✨ ADDITIONAL OPPORTUNITIES:")
        final_recommendations.extend(categorized_recommendations['additional'])
    
    # 10. Update the recommendations in the strategy
    comprehensive_strategy['recommendations'] = final_recommendations
    
    # 11. Update name to reflect comprehensive approach
    comprehensive_strategy['name'] = f"Comprehensive {strategy_info['name']}"
    
    return comprehensive_strategy

def predict_tax_strategy(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict tax strategy using ML model or rule-based fallback.
    
    Args:
        data: Dictionary with financial data including AnnualIncome, Investments, etc.
        
    Returns:
        Dictionary with predicted strategy and details
    """
    logging.info(f"Predicting strategy for input: {data}")
    
    # Initialize tracking variables for ML usage monitoring
    prediction_method = "Unknown"
    rules_applied = []
    decision_path = []
    feature_importance = None
    ml_confidence = None
    
    # Get tax liability
    tax_liability = data.get('TaxLiability', 0)
    
    # Calculate effective rate
    annual_income = data.get('AnnualIncome', 1)  # Avoid division by zero
    effective_rate = (tax_liability / annual_income) * 100 if annual_income > 0 else 0
    
    try:
        # Step 1: Try to load ML model
        decision_path.append("Attempting to load ML model")
        model, metadata = load_model()
        
        # Step 2: Process input features
        decision_path.append("Processing input features for ML model")
        input_data = {
            'AnnualIncome': data.get('AnnualIncome', 0), 
            'Investments': data.get('Investments', 0),
            'Deductions': data.get('Deductions', 0),
            'HRA': data.get('HRA', 0),
            'Age': data.get('Age', 30),
            'EmploymentType': data.get('EmploymentType', 'Private'),
            'MaritalStatus': data.get('MaritalStatus', 'Single')
        }
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        logging.info(f"Making prediction with features: {input_df.to_dict('records')}")
        
        # Handle categorical features based on metadata
        feature_encodings = metadata.get('feature_encodings', {})
        for col, encoding in feature_encodings.items():
            if col in input_df.columns and isinstance(input_df[col][0], str):
                classes = encoding.get('classes', [])
                if classes:
                    logging.info(f"Encoding {col} = {input_df[col][0]} using available classes: {classes}")
                    if input_df[col][0] in classes:
                        input_df[col] = classes.index(input_df[col][0])
                    else:
                        # Default to most common class if not found
                        input_df[col] = 0
        
        logging.info(f"Feature data types: {input_df.dtypes.to_dict()}")
        
        # Step 3: Make prediction with ML model
        decision_path.append("Generating ML prediction")
        X = input_df
        
        # Get probabilities and prediction
        class_probabilities = model.predict_proba(X)[0]
        predicted_strategy_idx = np.argmax(class_probabilities)
        max_probability = float(np.max(class_probabilities))
        ml_confidence = max_probability * 100
        
        # Get feature importance if the model supports it
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X.columns
            feature_importance = [(name, float(importance)) for name, importance in zip(feature_names, importances)]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            feature_importance = feature_importance[:5]  # Top 5 features
        
        # Map prediction to strategy name
        target_classes = metadata.get('target_classes', [])
        if target_classes and isinstance(predicted_strategy_idx, (int, np.integer)):
            # If predicted_strategy_idx is in range
            if 0 <= predicted_strategy_idx < len(target_classes):
                predicted_strategy = target_classes[predicted_strategy_idx]
                logging.info(f"Raw model prediction: {predicted_strategy}")
                logging.info(f"Class probabilities: {class_probabilities}")
            else:
                # Handle out of range index
                predicted_strategy = "Standard_Planning"  # Default
                rules_applied.append("Model predicted invalid strategy index")
                logging.warning(f"Invalid strategy index {predicted_strategy_idx}, using default")
        else:
            # Direct string prediction or unknown mapping
            predicted_strategy = predicted_strategy_idx
            if isinstance(predicted_strategy, str):
                logging.info(f"Strategy prediction is already a string: {predicted_strategy}")
            else:
                predicted_strategy = "Standard_Planning"  # Default
                rules_applied.append("Unknown target class mapping")
                logging.warning(f"Unknown target class mapping, using default strategy")
        
        # If prediction is made, log the confidence
        logging.info(f"Model confidence: {max_probability:.4f}")
        
        # Step 4: Apply Business Rules Validation
        decision_path.append("Validating prediction with business rules")
        age = data.get('Age', 30)
        income = data.get('AnnualIncome', 0)
        
        # Rule 1: Senior Planning only for people 60+
        if predicted_strategy == 'Senior_Planning' and age < 60:
            logging.warning(f"Age validation override: {age} is too young for Senior_Planning")
            
            # Find next highest probability class that isn't Senior_Planning
            alternatives = [(i, prob) for i, prob in enumerate(class_probabilities) 
                            if target_classes[i] != 'Senior_Planning']
            if alternatives:
                # Sort by probability (highest first)
                alternatives.sort(key=lambda x: x[1], reverse=True)
                alternative_idx, alternative_prob = alternatives[0]
                predicted_strategy = target_classes[alternative_idx]
                rules_applied.append(f"Age validation: {age} is too young for Senior Planning, using {predicted_strategy} instead")
                decision_path.append(f"Overrode Senior Planning due to age < 60, using {predicted_strategy}")
                logging.info(f"Using alternative strategy {predicted_strategy} with confidence {alternative_prob:.4f}")
            else:
                predicted_strategy = 'Standard_Planning'
                rules_applied.append("Age validation: fallback to standard planning")
                decision_path.append("No valid alternative found, using Standard Planning")
        
        # Rule 2: High Income Strategy only for 10L+ income
        if predicted_strategy == 'High_Income_Strategy' and income < 1000000:
            if income < 700000:  # Significant mismatch
                predicted_strategy = 'Standard_Planning'
                rules_applied.append(f"Income validation: {income} is too low for High Income Strategy")
                decision_path.append("Income too low for High Income Strategy, using Standard Planning")
            else:
                # Close enough to threshold, keep prediction
                rules_applied.append(f"Income warning: {income} is slightly below High Income threshold")
                decision_path.append("Income slightly below High Income threshold, but keeping recommendation")
        
        # Step 5: Apply confidence threshold
        if max_probability < 0.5:  # Model is not confident
            logging.warning(f"Model confidence too low: {max_probability:.4f}, using rule-based fallback")
            rule_based_strategy = determine_strategy(data)
            predicted_strategy = rule_based_strategy
            rules_applied.append(f"Low confidence ({max_probability:.4f}), using rule-based prediction")
            decision_path.append("ML confidence below threshold, switched to rule-based logic")
            prediction_method = "Rule-Based"
        elif rules_applied:
            # ML made the prediction but rules adjusted it
            prediction_method = "Hybrid"
        else:
            # Pure ML prediction
            prediction_method = "ML"
        
        # Step 6: Get strategy details
        decision_path.append("Getting detailed strategy information")
        strategy_info = get_strategy_details(predicted_strategy, data)
        
        # Step 7: Customize recommendations
        decision_path.append("Customizing recommendations for user profile")
        strategy_info = customize_recommendations(predicted_strategy, data, strategy_info)
        
        # Step 8: Generate comprehensive tax planning approach
        decision_path.append("Generating comprehensive tax planning approach")
        strategy_info = generate_comprehensive_tax_plan(data, predicted_strategy, strategy_info)
        
        logging.info(f"Prediction successful: Strategy={strategy_info['name']}")
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}", exc_info=True)
        # Fallback to rule-based prediction
        predicted_strategy = determine_strategy(data)
        strategy_info = get_strategy_details(predicted_strategy, data)
        
        # Still try to generate comprehensive recommendations
        try:
            strategy_info = generate_comprehensive_tax_plan(data, predicted_strategy, strategy_info)
        except Exception as comp_error:
            logging.error(f"Error generating comprehensive plan: {str(comp_error)}")
        
        prediction_method = "Rule-Based"
        rules_applied.append(f"Error: {str(e)}")
        decision_path.append("ML prediction failed, used rule-based fallback")
        
        logging.info(f"Using rule-based fallback: Strategy={strategy_info['name']}")
    
    # Prepare final result
    result = {
        'tax_liability': tax_liability,
        'monthly_tax': tax_liability / 12 if tax_liability else 0,
        'effective_rate': effective_rate,
        'tax_savings': data.get('TaxSavings', 0),
        'recommended_strategy': strategy_info['name'],
        'strategy_description': strategy_info['description'],
        'recommendations': strategy_info['recommendations'],
        'potential_savings': strategy_info['potential_savings'],
        'prediction_method': prediction_method,
        'rules_applied': rules_applied,
        'decision_path': decision_path,
        'ml_confidence': ml_confidence
    }
    
    # Add feature importance if available
    if feature_importance:
        result['feature_importance'] = feature_importance
    
    return result

if __name__ == '__main__':
    sample_input = {
        'AnnualIncome': 820000,
        'Investments': 150000,
        'Deductions': 50000,
        'HRA': 120000,
        'Age': 35,
        'EmploymentType': 'Private',
        'MaritalStatus': 'Married',
        'TaxLiability': 13000.00,
        'TaxSavings': 242840.00
    }
    
    result = predict_tax_strategy(sample_input)
    print("\n🧠 Tax Strategy Prediction Results:")
    print(f"Recommended Strategy: {result['recommended_strategy']}")
    print(f"Description: {result['strategy_description']}")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"• {rec}")
    print(f"\nPotential Tax Savings: {result['potential_savings']}")
    print(f"\nPrediction Method: {result['prediction_method']}")
    if result.get('ml_confidence'):
        print(f"ML Confidence: {result['ml_confidence']:.1f}%")
    if result.get('rules_applied'):
        print("\nBusiness Rules Applied:")
        for rule in result['rules_applied']:
            print(f"• {rule}")