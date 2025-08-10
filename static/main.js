document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })

    // Form submission handler
    document.getElementById('taxForm').addEventListener('submit', async function(e) {
        e.preventDefault()
        
        // Show loading state
        const submitBtn = this.querySelector('button[type="submit"]')
        const spinner = submitBtn.querySelector('.spinner-border')
        submitBtn.disabled = true
        spinner.classList.remove('d-none')
        
        try {
            const formData = new FormData(this)
            const data = Object.fromEntries(formData.entries())
            
            // Convert string values to numbers where needed
            const numericFields = ['annualIncome', 'otherIncome', 'previousTax', 'age', 
                                 'dependents', 'investments', 'hra', 'deductions', 'businessExpenses']
            numericFields.forEach(field => {
                data[field] = parseFloat(data[field]) || 0
            })
            
            const response = await fetch('/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            
            const result = await response.json()
            if (!response.ok) throw new Error(result.error)
            
            displayResults(result)
            
        } catch (error) {
            console.error('Error:', error)
            alert('Failed to calculate tax: ' + error.message)
        } finally {
            // Reset loading state
            submitBtn.disabled = false
            spinner.classList.add('d-none')
        }
    })
})

function displayResults(data) {
    // Show results container
    document.getElementById('results').style.display = 'block'
    
    // Update metrics
    document.getElementById('predictedTax').textContent = 
        Number(data.predicted_tax).toLocaleString('en-IN')
    document.getElementById('potentialSavings').textContent = 
        Number(data.savings_potential).toLocaleString('en-IN')
    
    // Update progress bars and scores
    const healthScore = data.financial_health * 100
    const efficiencyScore = data.tax_efficiency * 100
    
    const healthScoreBar = document.getElementById('healthScoreBar')
    const efficiencyScoreBar = document.getElementById('efficiencyScoreBar')
    
    healthScoreBar.style.width = `${healthScore}%`
    efficiencyScoreBar.style.width = `${efficiencyScore}%`
    
    document.getElementById('healthScore').textContent = `${healthScore.toFixed(1)}%`
    document.getElementById('efficiencyScore').textContent = `${efficiencyScore.toFixed(1)}%`
    
    // Display recommendations
    const recomsContainer = document.getElementById('recommendations')
    recomsContainer.innerHTML = ''
    
    data.recommendations.forEach(rec => {
        const card = document.createElement('div')
        card.className = `recommendation-card ${rec.priority.toLowerCase()}-priority`
        card.innerHTML = `
            <h6 class="recommendation-title">${rec.category}</h6>
            <div class="recommendation-details">
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Potential Savings:</strong> ₹${Number(rec.potential_savings).toLocaleString('en-IN')}</p>
                        <p><strong>Required Investment:</strong> ₹${Number(rec.required_investment).toLocaleString('en-IN')}</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Priority:</strong> ${rec.priority}</p>
                        <p><strong>Tip:</strong> ${rec.description}</p>
                    </div>
                </div>
            </div>
        `
        recomsContainer.appendChild(card)
    })
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' })
}