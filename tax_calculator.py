import numpy as np
from typing import Dict, List, Union

# Tax constants
TAX_SLABS = [
    (0, 300000, 0),         # Nil up to 3L
    (300000, 600000, 0.05), # 5% from 3L to 6L
    (600000, 900000, 0.10), # 10% from 6L to 9L
    (900000, 1200000, 0.15),# 15% from 9L to 12L
    (1200000, 1500000, 0.20),# 20% from 12L to 15L
    (1500000, float('inf'), 0.30) # 30% above 15L
]

DEDUCTION_LIMITS = {
    '80C': 150000,
    '80D': 25000,
    'HRA': 'dynamic',  # Based on salary and city
    'NPS': 50000
}

def calculate_tax(income: float, deductions: float) -> Dict[str, float]:
    """Calculate tax based on income and deductions."""
    taxable_income = max(0, income - deductions)
    tax = 0
    
    for lower, upper, rate in TAX_SLABS:
        if taxable_income > lower:
            slab_income = min(taxable_income - lower, upper - lower)
            tax += slab_income * rate
    
    # Add 4% health and education cess
    cess = tax * 0.04
    total_tax = tax + cess
    
    return {
        'taxable_income': taxable_income,
        'tax': tax,
        'cess': cess,
        'total_tax': total_tax
    }

def get_tax_saving_recommendations(
    income: float, 
    current_investments: float,
    age: int,
    marital_status: str
) -> Dict[str, List[str]]:
    """Generate personalized tax saving recommendations."""
    remaining_80c = max(0, DEDUCTION_LIMITS['80C'] - current_investments)
    tax_bracket = 0.3 if income > 1500000 else 0.2 if income > 1200000 else 0.15
    potential_savings = remaining_80c * tax_bracket
    
    recommendations = {
        'priority': [],
        'additional': []
    }
    
    # Age-based recommendations
    if age < 30:
        if remaining_80c > 0:
            recommendations['priority'].extend([
                "🎯 Young Investor Focus (Potential tax saving: ₹{:,.0f}):".format(potential_savings),
                "1. Start ELSS mutual funds SIP (equity with 3-year lock-in)",
                "2. Term insurance premium (protection + tax saving)",
                "3. PPF account (long-term tax-free savings)",
                f"• Can still invest ₹{remaining_80c:,.0f} under 80C"
            ])
    elif age < 45:
        if remaining_80c > 0:
            recommendations['priority'].extend([
                "🎯 Mid-Career Focus (Potential tax saving: ₹{:,.0f}):".format(potential_savings),
                "1. Maximize home loan benefits (principal under 80C)",
                "2. ELSS + PPF combination",
                "3. Family health insurance premium",
                f"• Can still invest ₹{remaining_80c:,.0f} under 80C"
            ])
    else:
        if remaining_80c > 0:
            recommendations['priority'].extend([
                "🎯 Senior Planning (Potential tax saving: ₹{:,.0f}):".format(potential_savings),
                "1. Senior Citizen Savings Scheme (if 60+)",
                "2. Safe 80C options: PPF, Tax-saving FDs",
                "3. Enhanced health insurance coverage",
                f"• Can still invest ₹{remaining_80c:,.0f} under 80C"
            ])
    
    # Income-based recommendations
    recommendations['additional'].extend([
        "\n💡 Additional Tax Saving Options:",
        "• Health Insurance (80D): Up to ₹25,000 deduction",
        "• NPS Additional: Extra ₹50,000 deduction",
        "• Home Loan Interest (24b): Up to ₹2L"
    ])
    
    # Marital status specific
    if marital_status.lower() == 'married':
        recommendations['additional'].extend([
            "\n👨‍👩‍👦 Family Planning:",
            "• Family floater health insurance",
            "• Split investments between spouses",
            "• Joint home loan benefits"
        ])
    
    return recommendations