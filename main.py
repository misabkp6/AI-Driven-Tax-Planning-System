import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import os
import json
from datetime import datetime
import logging

class TaxPredictor:
    def __init__(self):
        # Set up logging
        self._setup_logging()
        
        # Define feature ranges with descriptions
        self.feature_ranges = {
            'AnnualIncome': {
                'range': (300000, 2000000),
                'description': 'Total annual income from all sources',
                'type': 'float'
            },
            'Investments': {
                'range': (0, None),
                'description': 'Total investments under Section 80C',
                'type': 'float'
            },
            'Deductions': {
                'range': (0, None),
                'description': 'Other tax deductions',
                'type': 'float'
            },
            'PreviousTaxPaid': {
                'range': (0, None),
                'description': 'Tax paid in previous assessment year',
                'type': 'float'
            },
            'EmploymentType': {
                'range': (1, 3),
                'description': '1: Salaried, 2: Self-employed, 3: Business',
                'type': 'int'
            },
            'HRA': {
                'range': (0, None),
                'description': 'House Rent Allowance claimed',
                'type': 'float'
            },
            'OtherIncome': {
                'range': (0, None),
                'description': 'Income from other sources',
                'type': 'float'
            },
            'Age': {
                'range': (18, 100),
                'description': 'Age of the taxpayer',
                'type': 'int'
            },
            'NumDependents': {
                'range': (0, 5),
                'description': 'Number of dependent family members',
                'type': 'int'
            },
            'BusinessExpenses': {
                'range': (0, None),
                'description': 'Total business-related expenses',
                'type': 'float'
            },
            'MaritalStatus': {
                'range': (1, 2),
                'description': '1: Single, 2: Married',
                'type': 'int'
            }
        }
        
        # Enhanced tax saving options
        self.tax_saving_options = {
            '80C': {
                'limit': 150000,
                'instruments': {
                    'PPF': {
                        'returns': '7.1%',
                        'lock_in': '15 years',
                        'risk': 'Low',
                        'liquidity': 'Low',
                        'min_investment': 500
                    },
                    'ELSS': {
                        'returns': '12-15%',
                        'lock_in': '3 years',
                        'risk': 'High',
                        'liquidity': 'Medium',
                        'min_investment': 500
                    },
                    'NPS': {
                        'returns': '8-10%',
                        'lock_in': 'Till retirement',
                        'risk': 'Medium',
                        'liquidity': 'Low',
                        'min_investment': 500
                    }
                }
            },
            '80D': {
                'limit': 25000,
                'additional_senior_citizen': 50000,
                'family_floater': True,
                'preventive_health_checkup': 5000
            },
            'HRA': {
                'metro_cities': 0.5,
                'other_cities': 0.4,
                'basic_salary_percent': 0.4
            },
            '80G': {
                'limit': '100% or 50%',
                'description': 'Donations to approved charitable institutions'
            }
        }
        
        # Load model and scaler
        self.model, self.scaler = self._load_model()
        
        # Initialize history
        self.prediction_history = []

    def _setup_logging(self) -> None:
        """Set up logging configuration"""
        logging.basicConfig(
            filename='tax_predictions.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_model(self) -> Tuple:
        """Load the trained model and scaler with enhanced error handling"""
        model_path = os.path.join('models', 'improved_tax_model.pkl')
        scaler_path = os.path.join('models', 'improved_scaler.pkl')
        
        try:
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    "Model files not found. Please ensure the following files exist:\n"
                    f"- {model_path}\n"
                    f"- {scaler_path}"
                )
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logging.info("Model and scaler loaded successfully")
            return model, scaler
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise Exception(f"Error loading model: {str(e)}")

    def _validate_input(self, data: Dict) -> Optional[str]:
        """Enhanced input validation with type checking"""
        for feature, value in data.items():
            if feature not in self.feature_ranges:
                return f"Invalid feature: {feature}"
            
            feature_info = self.feature_ranges[feature]
            min_val, max_val = feature_info['range']
            
            # Type validation
            try:
                if feature_info['type'] == 'int':
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                return f"{feature} must be a {feature_info['type']}"
            
            # Range validation
            if min_val is not None and value < min_val:
                return f"{feature} cannot be less than {min_val}"
            if max_val is not None and value > max_val:
                return f"{feature} cannot be more than {max_val}"
        return None

    def predict(self, input_data: Dict) -> Dict:
        """Make tax predictions with enhanced validation and logging"""
        try:
            # Input validation
            error = self._validate_input(input_data)
            if error:
                raise ValueError(error)

            # Feature engineering
            df = pd.DataFrame([input_data])
            scaled_data = self.scaler.transform(df)
            predictions = self.model.predict(scaled_data)
            
            # Store prediction in history
            prediction_result = {
                'timestamp': datetime.now().isoformat(),
                'input': input_data,
                'final_tax': float(predictions[0][0]),
                'updated_tax': float(predictions[0][1])
            }
            self.prediction_history.append(prediction_result)
            
            logging.info(f"Tax prediction made: {prediction_result}")
            return {
                'final_tax': prediction_result['final_tax'],
                'updated_tax': prediction_result['updated_tax']
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise

    def analyze_tax_savings(self, income_data: Dict) -> Dict:
        """Enhanced tax savings analysis with detailed recommendations"""
        try:
            tax_savings = {
                'current_tax': 0,
                'potential_savings': 0,
                'recommendations': [],
                'total_savings_potential': 0
            }
            
            # Calculate current tax
            current_tax = self.predict(income_data)
            tax_savings['current_tax'] = current_tax['final_tax']
            
            # Analyze 80C savings
            current_80c = income_data.get('Investments', 0)
            max_80c = self.tax_saving_options['80C']['limit']
            if current_80c < max_80c:
                potential_saving = max_80c - current_80c
                tax_savings['recommendations'].append({
                    'section': '80C',
                    'potential_saving': potential_saving,
                    'instruments': self.tax_saving_options['80C']['instruments'],
                    'priority': 'High'
                })
                tax_savings['total_savings_potential'] += potential_saving
            
            # Analyze HRA benefits for salaried employees
            if income_data['EmploymentType'] == 1:
                basic_salary = income_data['AnnualIncome'] * self.tax_saving_options['HRA']['basic_salary_percent']
                max_hra = basic_salary * self.tax_saving_options['HRA']['metro_cities']
                current_hra = income_data.get('HRA', 0)
                if current_hra < max_hra:
                    potential_hra_saving = max_hra - current_hra
                    tax_savings['recommendations'].append({
                        'section': 'HRA',
                        'potential_saving': potential_hra_saving,
                        'note': 'Consider declaring actual rent paid to maximize HRA benefit',
                        'priority': 'Medium'
                    })
                    tax_savings['total_savings_potential'] += potential_hra_saving
            
            # Add health insurance recommendations (80D)
            if 'health_insurance' not in income_data or income_data['health_insurance'] < self.tax_saving_options['80D']['limit']:
                potential_80d = self.tax_saving_options['80D']['limit']
                tax_savings['recommendations'].append({
                    'section': '80D',
                    'potential_saving': potential_80d,
                    'note': 'Consider health insurance for tax benefits and protection',
                    'priority': 'High'
                })
                tax_savings['total_savings_potential'] += potential_80d
            
            logging.info(f"Tax savings analysis completed: {tax_savings}")
            return tax_savings
            
        except Exception as e:
            logging.error(f"Tax savings analysis error: {str(e)}")
            raise

    def suggest_investments(self, income_data: Dict) -> List[Dict]:
        """Enhanced investment suggestions based on comprehensive profiling"""
        try:
            risk_profile = self._determine_risk_profile(
                income_data['AnnualIncome'],
                income_data['Age'],
                income_data['NumDependents']
            )
            
            investment_mapping = {
                'Conservative': [
                    {
                        'instrument': 'PPF',
                        'allocation': 0.5,
                        'reason': 'Safe, guaranteed returns',
                        'min_amount': 500,
                        'expected_returns': '7.1%'
                    },
                    {
                        'instrument': 'Tax-Saving FD',
                        'allocation': 0.3,
                        'reason': 'Fixed returns with tax benefits',
                        'min_amount': 1000,
                        'expected_returns': '5.5-6.5%'
                    },
                    {
                        'instrument': 'ELSS',
                        'allocation': 0.2,
                        'reason': 'Limited equity exposure',
                        'min_amount': 500,
                        'expected_returns': '12-15%'
                    }
                ],
                'Moderate': [
                    {
                        'instrument': 'ELSS',
                        'allocation': 0.4,
                        'reason': 'Growth with tax benefits',
                        'min_amount': 500,
                        'expected_returns': '12-15%'
                    },
                    {
                        'instrument': 'PPF',
                        'allocation': 0.3,
                        'reason': 'Stable long-term returns',
                        'min_amount': 500,
                        'expected_returns': '7.1%'
                    },
                    {
                        'instrument': 'NPS',
                        'allocation': 0.3,
                        'reason': 'Additional tax benefits',
                        'min_amount': 500,
                        'expected_returns': '8-10%'
                    }
                ],
                'Aggressive': [
                    {
                        'instrument': 'ELSS',
                        'allocation': 0.5,
                        'reason': 'High growth potential',
                        'min_amount': 500,
                        'expected_returns': '12-15%'
                    },
                    {
                        'instrument': 'NPS',
                        'allocation': 0.3,
                        'reason': 'Long-term wealth creation',
                        'min_amount': 500,
                        'expected_returns': '8-10%'
                    },
                    {
                        'instrument': 'PPF',
                        'allocation': 0.2,
                        'reason': 'Portfolio stability',
                        'min_amount': 500,
                        'expected_returns': '7.1%'
                    }
                ]
            }
            
            recommendations = investment_mapping[risk_profile]
            logging.info(f"Investment recommendations generated for {risk_profile} profile")
            return recommendations
            
        except Exception as e:
            logging.error(f"Investment suggestion error: {str(e)}")
            raise

    def _determine_risk_profile(self, annual_income: float, age: int, 
                              dependents: int) -> str:
        """Enhanced risk profile determination with multiple factors"""
        # Base score calculation
        score = 0
        
        # Income factor (0-40 points)
        if annual_income < 500000:
            score += 10
        elif annual_income < 1000000:
            score += 25
        else:
            score += 40
        
        # Age factor (0-40 points)
        if age < 30:
            score += 40
        elif age < 45:
            score += 30
        elif age < 60:
            score += 20
        else:
            score += 10
        
        # Dependents factor (0-20 points)
        score -= dependents * 4  # Reduce score for each dependent
        
        # Determine profile based on final score
        if score < 40:
            return 'Conservative'
        elif score < 70:
            return 'Moderate'
        return 'Aggressive'

    def predict_with_recommendations(self, input_data: Dict) -> Dict:
        """Comprehensive tax analysis with enhanced recommendations"""
        try:
            tax_prediction = self.predict(input_data)
            tax_savings = self.analyze_tax_savings(input_data)
            investment_recommendations = self.suggest_investments(input_data)
            
            result = {
                'tax_liability': tax_prediction,
                'potential_savings': tax_savings,
                'investment_recommendations': investment_recommendations,
                'summary': {
                    'current_tax': tax_prediction['final_tax'],
                    'potential_savings': tax_savings['total_savings_potential'],
                    'effective_tax_rate': (tax_prediction['final_tax'] / input_data['AnnualIncome']) * 100
                }
            }
            
            logging.info("Comprehensive tax analysis completed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Comprehensive analysis error: {str(e)}")
            raise

    def export_prediction_history(self, filename: str = 'prediction_history.json') -> None:
        """Export prediction history to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.prediction_history, f, indent=4)
            logging.info(f"Prediction history exported to {filename}")
        except Exception as e:
            logging.error(f"Error exporting prediction history: {str(e)}")
            raise

def main():
    try:
        predictor = TaxPredictor()
        
        # Sample data for testing
        sample_data = {
            'AnnualIncome': 1500000,
            'Investments': 100000,
            'Deductions': 150000,
            'PreviousTaxPaid': 180000,
            'EmploymentType': 1,
            'HRA': 300000,
            'OtherIncome': 50000,
            'Age': 35,
            'NumDependents': 2,
            'BusinessExpenses': 100000,
            'MaritalStatus': 1
        }
        
        # Get comprehensive analysis
        result = predictor.predict_with_recommendations(sample_data)
        
        # Print detailed analysis
        print("\n📊 Tax Planning Analysis")
        print("\n1. Tax Liability:")
        print(f"   ├─ Final Tax: ₹{result['tax_liability']['final_tax']:,.2f}")
        print(f"   ├─ Updated Tax: ₹{result['tax_liability']['updated_tax']:,.2f}")
        print(f"   └─ Effective Tax Rate: {result['summary']['effective_tax_rate']:.2f}%")
        
        print("\n2. Potential Tax Savings:")
        print(f"   ├─ Total Potential Savings: ₹{result['potential_savings']['total_savings_potential']:,.2f}")
        for rec in result['potential_savings']['recommendations']:
            print(f"   ├─ Section {rec['section']} (Priority: {rec['priority']})")
            print(f"   │  └─ Potential Saving: ₹{rec['potential_saving']:,.2f}")
        
        print("\n3. Investment Recommendations:")
        for inv in result['investment_recommendations']:
            print(f"   ├─ {inv['instrument']}")
            print(f"   │  ├─ Allocation: {inv['allocation']*100}%")
            print(f"   │  ├─ Expected Returns: {inv['expected_returns']}")
            print(f"   │  ├─ Minimum Investment: ₹{inv['min_amount']:,}")
            print(f"   │  └─ Reason: {inv['reason']}")
        
        # Export prediction history
        predictor.export_prediction_history()
        
    except FileNotFoundError as e:
        print(f"\n❌ Model Loading Error: {str(e)}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()