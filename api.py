from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import pandas as pd
import numpy as np
from predict import predict_tax_strategy
import logging
import os
from datetime import datetime

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/api_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Create FastAPI app
app = FastAPI(
    title="Tax Strategy AI API",
    description="API for tax liability prediction and strategy recommendations",
    version="2.0.0",
)

class TaxInputModel(BaseModel):
    AnnualIncome: float = Field(..., title="Annual Income", description="Total annual income", gt=0)
    Investments: float = Field(0, title="Tax-saving Investments", description="Total investments under Section 80C")
    Deductions: float = Field(0, title="Additional Deductions", description="Other tax deductions")
    HRA: float = Field(0, title="HRA", description="House Rent Allowance")
    Age: int = Field(..., title="Age", description="Age of taxpayer", ge=18, le=100)
    EmploymentType: str = Field("Private", title="Employment Type", description="Type of employment")
    MaritalStatus: str = Field("Single", title="Marital Status", description="Marital status")
    TaxLiability: Optional[float] = Field(None, title="Tax Liability", description="Calculated tax liability")
    TaxSavings: Optional[float] = Field(None, title="Tax Savings", description="Estimated tax savings")

class TaxOutputModel(BaseModel):
    tax_liability: float
    monthly_tax: float
    effective_rate: float
    tax_savings: float
    recommended_strategy: str
    strategy_description: str
    recommendations: List[str]
    potential_savings: str
    error: Optional[str] = None

@app.post("/predict", response_model=TaxOutputModel)
async def predict(request: TaxInputModel):
    """
    Predict tax liability and recommend strategies
    """
    try:
        # Convert input model to dictionary
        input_data = request.dict()
        
        # If tax liability is not provided, calculate it here
        if input_data.get("TaxLiability") is None:
            # This is where you'd implement your tax calculation logic
            # For now, we'll use a simplified calculation
            taxable_income = max(0, input_data["AnnualIncome"] - input_data["Investments"] - 
                               input_data["Deductions"] - input_data["HRA"])
            
            # Simple progressive tax calculation (modify as per Indian tax rules)
            if taxable_income <= 250000:
                tax = 0
            elif taxable_income <= 500000:
                tax = (taxable_income - 250000) * 0.05
            elif taxable_income <= 750000:
                tax = 12500 + (taxable_income - 500000) * 0.10
            elif taxable_income <= 1000000:
                tax = 37500 + (taxable_income - 750000) * 0.15
            elif taxable_income <= 1250000:
                tax = 75000 + (taxable_income - 1000000) * 0.20
            elif taxable_income <= 1500000:
                tax = 125000 + (taxable_income - 1250000) * 0.25
            else:
                tax = 187500 + (taxable_income - 1500000) * 0.30
                
            # Add cess (4%)
            tax = tax * 1.04
            
            input_data["TaxLiability"] = tax
            
            # Calculate tax savings (simplified)
            standard_tax = input_data["AnnualIncome"] * 0.3 * 1.04  # 30% + 4% cess
            input_data["TaxSavings"] = max(0, standard_tax - tax)
        
        # Get strategy prediction
        result = predict_tax_strategy(input_data)
        
        return result
    
    except Exception as e:
        logging.error(f"API error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)