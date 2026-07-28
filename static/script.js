document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!modelLoaded) {
        alert('Please train the model first!');
        return;
    }
    
    // Show loading state
    const submitBtn = e.target.querySelector('.btn-predict');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Analyzing...</span>';
    submitBtn.disabled = true;
    
    const formData = new FormData(e.target);
    const data = {};
    
    formData.forEach((value, key) => {
        data[key] = parseFloat(value);
    });
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResult(result);
            showNotification('Prediction completed successfully!', 'success');
        } else {
            showNotification('Error: ' + result.error, 'error');
        }
    } catch (error) {
        showNotification('Error making prediction: ' + error.message, 'error');
    } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
});

document.getElementById('prescriptionForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!modelLoaded) {
        alert('Please train the model first!');
        return;
    }

    const submitBtn = e.target.querySelector('.btn-predict');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Analyzing Prescription...</span>';
    submitBtn.disabled = true;

    const formData = new FormData(e.target);

    try {
        const response = await fetch('/predict-prescription', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            displayResult(result);
            showNotification('Prescription prediction completed successfully!', 'success');
        } else {
            showNotification('Error: ' + result.error, 'error');
        }
    } catch (error) {
        showNotification('Error processing prescription: ' + error.message, 'error');
    } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
});

let predictionChart = null;

// Train model button
const trainButton = document.getElementById('trainButton');
if (trainButton) {
    trainButton.addEventListener('click', async () => {
        const statusDiv = document.getElementById('trainingStatus');
        statusDiv.classList.remove('hidden');
        trainButton.disabled = true;
        
        try {
            const response = await fetch('/train', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                statusDiv.innerHTML = '<div class="loading-spinner"></div><p class="success">✓ ' + result.message + ' Reloading page...</p>';
                setTimeout(() => {
                    location.reload();
                }, 2000);
            } else {
                statusDiv.innerHTML = '<p class="error">✗ Training failed: ' + result.message + '</p>';
                trainButton.disabled = false;
            }
        } catch (error) {
            statusDiv.innerHTML = '<p class="error">✗ Error: ' + error.message + '</p>';
            trainButton.disabled = false;
        }
    });
}

document.querySelectorAll('.btn-reset').forEach((button) => {
    button.addEventListener('click', () => {
        document.getElementById('result').classList.add('hidden');
        if (predictionChart) {
            predictionChart.destroy();
            predictionChart = null;
        }
    });
});

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');
    
    const readmissionClass = result.readmission === 'Yes' ? 'readmission-yes' : 'readmission-no';
    const iconClass = result.readmission === 'Yes' ? 'fa-exclamation-circle' : 'fa-check-circle';
    const riskLevel = result.probability_yes * 100;
    
    // Destroy existing chart if any
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    // Build risk factors HTML
    let riskFactorsHTML = '';
    if (result.analysis && result.analysis.risk_factors && result.analysis.risk_factors.length > 0) {
        riskFactorsHTML = '<div class="analysis-section risk-factors-section">';
        riskFactorsHTML += '<h3><i class="fas fa-exclamation-triangle"></i> Identified Risk Factors</h3>';
        result.analysis.risk_factors.forEach(factor => {
            riskFactorsHTML += `
                <div class="risk-factor-card risk-severity-${factor.severity}">
                    <div class="factor-title">${factor.factor}</div>
                    <div class="factor-impact">${factor.impact}</div>
                </div>
            `;
        });
        riskFactorsHTML += '</div>';
    }
    
    // Build recommendations HTML
    let recommendationsHTML = '';
    if (result.analysis && result.analysis.recommendations && result.analysis.recommendations.length > 0) {
        recommendationsHTML = '<div class="analysis-section recommendations-section">';
        recommendationsHTML += '<h3><i class="fas fa-lightbulb"></i> Recommended Actions to Reduce Readmission</h3>';
        
        // Sort by priority
        const priorityOrder = { critical: 0, high: 1, medium: 2 };
        const sortedRecs = [...result.analysis.recommendations].sort((a, b) => 
            priorityOrder[a.priority] - priorityOrder[b.priority]
        );
        
        sortedRecs.forEach((rec, index) => {
            const priorityIcon = rec.priority === 'critical' ? 'fa-star' : 
                                rec.priority === 'high' ? 'fa-arrow-up' : 'fa-check';
            recommendationsHTML += `
                <div class="recommendation-card rec-priority-${rec.priority}">
                    <div class="rec-header">
                        <span class="rec-priority"><i class="fas ${priorityIcon}"></i> ${rec.priority.toUpperCase()}</span>
                        <h4>${rec.title}</h4>
                    </div>
                    <p class="rec-description">${rec.description}</p>
                </div>
            `;
        });
        recommendationsHTML += '</div>';
    }
    
    // Build risk summary HTML
    let riskSummaryHTML = '';
    if (result.analysis && result.analysis.risk_summary) {
        riskSummaryHTML = `
            <div class="analysis-section risk-summary-section">
                <div class="risk-summary-box">
                    <i class="fas fa-info-circle"></i>
                    <p>${result.analysis.risk_summary}</p>
                </div>
            </div>
        `;
    }

    let uploadDetailsHTML = '';
    if (result.source === 'prescription_upload') {
        const features = result.extracted_features || {};
        const meds = (result.detected_medications || []).length > 0
            ? result.detected_medications.join(', ')
            : 'No known medication names detected';

        uploadDetailsHTML = `
            <div class="analysis-section upload-summary-section">
                <h3><i class="fas fa-file-medical"></i> Prescription Extraction Summary</h3>
                <div class="upload-summary-grid">
                    <div><strong>Age:</strong> ${Number(features.age ?? 0).toFixed(0)}</div>
                    <div><strong>Hospital Stay:</strong> ${Number(features.time_in_hospital ?? 0).toFixed(0)} days</div>
                    <div><strong>Lab Procedures:</strong> ${Number(features.num_lab_procedures ?? 0).toFixed(0)}</div>
                    <div><strong>Medical Procedures:</strong> ${Number(features.num_procedures ?? 0).toFixed(0)}</div>
                    <div><strong>Medications:</strong> ${Number(features.num_medications ?? 0).toFixed(0)}</div>
                    <div><strong>Diagnoses:</strong> ${Number(features.number_diagnoses ?? 0).toFixed(0)}</div>
                </div>
                <p class="upload-med-list"><strong>Detected medications:</strong> ${meds}</p>
            </div>
        `;
    }
    
    resultContent.innerHTML = `
        <div class="prediction-box">
            <div class="prediction-result ${readmissionClass}">
                <div class="prediction-icon">
                    <i class="fas ${iconClass}"></i>
                </div>
                <h3>Readmission Prediction: <span>${result.readmission}</span></h3>
                <p style="font-size: 1.1rem; color: #64748b; margin-top: 0.5rem;">
                    ${result.readmission === 'Yes' ? 
                        'High risk of readmission detected. Consider follow-up care.' : 
                        'Low risk of readmission. Standard discharge protocol recommended.'}
                </p>
            </div>

            <div class="risk-meter">
                <h4 style="text-align: center; margin-bottom: 1rem; color: #64748b;">Risk Level</h4>
                <div class="risk-bar">
                    <div class="risk-indicator" style="left: ${riskLevel}%"></div>
                </div>
                <div class="risk-labels">
                    <span style="color: ${riskLevel < 33 ? 'var(--success)' : '#94a3b8'}; font-weight: ${riskLevel < 33 ? '600' : '400'};">Low Risk</span>
                    <span style="color: ${riskLevel >= 33 && riskLevel < 66 ? 'var(--warning)' : '#94a3b8'}; font-weight: ${riskLevel >= 33 && riskLevel < 66 ? '600' : '400'};">Moderate Risk</span>
                    <span style="color: ${riskLevel >= 66 ? 'var(--danger)' : '#94a3b8'}; font-weight: ${riskLevel >= 66 ? '600' : '400'};">High Risk</span>
                </div>
            </div>

            <div class="probability-grid">
                <div class="probability-card low-risk">
                    <h4>No Readmission</h4>
                    <div class="percentage">${(result.probability_no * 100).toFixed(1)}%</div>
                    <div style="font-size: 0.9rem; color: #64748b;">Confidence Level</div>
                </div>
                <div class="probability-card high-risk">
                    <h4>Readmission</h4>
                    <div class="percentage">${(result.probability_yes * 100).toFixed(1)}%</div>
                    <div style="font-size: 0.9rem; color: #64748b;">Confidence Level</div>
                </div>
            </div>

            <div class="chart-container">
                <canvas id="predictionChart"></canvas>
            </div>
            
            ${uploadDetailsHTML}
            ${riskSummaryHTML}
            ${riskFactorsHTML}
            ${recommendationsHTML}
        </div>
    `;
    
    resultDiv.classList.remove('hidden');
    
    // Scroll to results
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Create chart
    createPredictionChart(result);
}

function createPredictionChart(result) {
    const ctx = document.getElementById('predictionChart');
    
    predictionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['No Readmission', 'Readmission'],
            datasets: [{
                data: [result.probability_no * 100, result.probability_yes * 100],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(16, 185, 129, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 14,
                            family: 'Inter'
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(2) + '%';
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true
            }
        }
    });
}

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${type === 'success' ? 'var(--success)' : 'var(--danger)'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 1000;
        animation: slideInRight 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Add input validation with visual feedback
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        const min = parseFloat(e.target.min);
        const max = parseFloat(e.target.max);
        
        if (value < min || (max && value > max)) {
            e.target.style.borderColor = 'var(--danger)';
        } else {
            e.target.style.borderColor = 'var(--success)';
        }
    });
});
