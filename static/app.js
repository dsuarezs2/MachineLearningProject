/* ==========================================
   CardioInsight AI — Frontend Logic (static/app.js)
   ========================================== */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('patient-form');
    const welcomePanel = document.getElementById('welcome-panel');
    const loadingPanel = document.getElementById('loading-panel');
    const resultsPanel = document.getElementById('results-panel');
    
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugePercentage = document.getElementById('gauge-percentage');
    const riskBadge = document.getElementById('risk-badge');
    const diagnosisText = document.getElementById('diagnosis-text');
    const modelInfo = document.getElementById('model-info');
    const contributionsList = document.getElementById('contributions-list');
    
    let impactChart = null;

    // Helper functions to get form values
    const getFloatOrNull = (id) => {
        const val = document.getElementById(id).value;
        return val === '' ? null : parseFloat(val);
    };

    const getBoolVal = (id) => {
        return document.getElementById(id).value === 'true';
    };

    const getStrVal = (id) => {
        return document.getElementById(id).value;
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Basic frontend validation
        let isValid = true;
        const ageInput = document.getElementById('age');
        const ageGroup = ageInput.closest('.input-group');
        
        if (!ageInput.value || parseFloat(ageInput.value) <= 0 || parseFloat(ageInput.value) > 120) {
            ageGroup.classList.add('invalid');
            isValid = false;
        } else {
            ageGroup.classList.remove('invalid');
        }

        if (!isValid) {
            ageInput.focus();
            return;
        }

        // Show loader and hide content
        welcomePanel.classList.add('hidden');
        resultsPanel.classList.add('hidden');
        loadingPanel.classList.remove('hidden');

        // Gather patient data matching Pydantic model
        const patientData = {
            age: parseFloat(ageInput.value),
            sex: getStrVal('sex'),
            dataset: getStrVal('dataset'),
            cp: getStrVal('cp'),
            trestbps: getFloatOrNull('trestbps'),
            chol: getFloatOrNull('chol'),
            fbs: getBoolVal('fbs'),
            restecg: getStrVal('restecg'),
            thalch: getFloatOrNull('thalch'),
            exang: getBoolVal('exang'),
            oldpeak: getFloatOrNull('oldpeak'),
            slope: getStrVal('slope'),
            ca: getFloatOrNull('ca'),
            thal: getStrVal('thal'),
            model_name: getStrVal('model_name')
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(patientData)
            });

            if (!response.ok) {
                throw new Error('Error al conectar con la API de diagnóstico.');
            }

            const data = await response.json();
            
            // Hide loading panel and show results
            loadingPanel.classList.add('hidden');
            resultsPanel.classList.remove('hidden');

            renderResults(data);

        } catch (error) {
            loadingPanel.classList.add('hidden');
            welcomePanel.classList.remove('hidden');
            alert(error.message || 'Ocurrió un error inesperado al procesar la solicitud.');
        }
    });

    function renderResults(data) {
        const prob = data.probability;
        const pred = data.prediction;
        const riskLevel = data.risk_level;
        const contributions = data.contributions;
        const modelUsed = data.model_used;

        // 1. Animate SVG Gauge
        // Semi-circle path length: PI * R = 3.14159 * 40 = 125.6
        const pathLength = 125.6;
        const offset = pathLength * (1 - prob);
        gaugeFill.style.strokeDashoffset = offset;
        gaugePercentage.textContent = `${Math.round(prob * 100)}%`;

        // Update colors and badges based on risk level
        gaugeFill.className.baseVal = "gauge-fill"; // reset classes
        riskBadge.className = "risk-badge";
        
        if (riskLevel === "Bajo") {
            gaugeFill.classList.add("risk-low");
            riskBadge.classList.add("badge-low");
            riskBadge.textContent = "Riesgo Bajo";
            diagnosisText.textContent = "Sin Enfermedad Detectada";
            diagnosisText.style.color = "#10b981";
        } else if (riskLevel === "Moderado") {
            gaugeFill.classList.add("risk-moderate");
            riskBadge.classList.add("badge-moderate");
            riskBadge.textContent = "Riesgo Moderado";
            diagnosisText.textContent = "Probabilidad Moderada";
            diagnosisText.style.color = "#f59e0b";
        } else {
            gaugeFill.classList.add("risk-high");
            riskBadge.classList.add("badge-high");
            riskBadge.textContent = "Riesgo Alto";
            diagnosisText.textContent = "Presencia de Enfermedad Cardíaca";
            diagnosisText.style.color = "#ef4444";
        }

        // Update Model info
        const modelMap = {
            'rf': 'Random Forest (Bosque Aleatorio)',
            'lr': 'Regresión Logística',
            'dt': 'Árbol de Decisión'
        };
        modelInfo.textContent = `Modelo de predicción: ${modelMap[modelUsed] || modelUsed}`;

        // 2. Render Chart.js
        renderChart(contributions);

        // 3. Render detailed contributions list
        renderContributionsList(contributions);
    }

    function renderChart(contributions) {
        // Destroy existing chart to prevent layout overlapping
        if (impactChart) {
            impactChart.destroy();
        }

        // Limit chart to top 7 most impactful features for cleaner view
        const topContributions = contributions.slice(0, 7);

        const labels = topContributions.map(c => c.name);
        const dataValues = topContributions.map(c => c.impact * 100); // convert to percentage
        
        // Background colors: red for positive impact (increases risk), green for negative (reduces risk)
        const backgroundColors = dataValues.map(v => v >= 0 ? 'rgba(239, 68, 68, 0.85)' : 'rgba(16, 185, 129, 0.85)');
        const borderColors = dataValues.map(v => v >= 0 ? '#ef4444' : '#10b981');

        const ctx = document.getElementById('impact-chart').getContext('2d');
        impactChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Impacto en el riesgo (%)',
                    data: dataValues,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y', // horizontal bar chart
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                const sign = value >= 0 ? '+' : '';
                                return `Impacto: ${sign}${value.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.08)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            callback: function(value) {
                                const sign = value >= 0 ? '+' : '';
                                return `${sign}${value}%`;
                            }
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#f8fafc',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    }
                }
            }
        });
    }

    function renderContributionsList(contributions) {
        contributionsList.innerHTML = '';

        contributions.forEach(c => {
            const li = document.createElement('li');
            li.className = 'contribution-item';

            const nameDiv = document.createElement('div');
            nameDiv.className = 'feat-info';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'feat-name';
            nameSpan.textContent = c.name;
            
            const valSpan = document.createElement('span');
            valSpan.className = 'feat-val';
            valSpan.textContent = `Valor: ${c.value}`;
            
            nameDiv.appendChild(nameSpan);
            nameDiv.appendChild(valSpan);

            const impactDiv = document.createElement('div');
            impactDiv.className = 'feat-impact';

            const impactPct = c.impact * 100;
            const sign = impactPct >= 0 ? '+' : '';
            
            const pctSpan = document.createElement('span');
            pctSpan.textContent = `${sign}${impactPct.toFixed(1)}%`;

            const icon = document.createElement('i');
            if (impactPct > 0.5) {
                impactDiv.classList.add('impact-pos');
                icon.className = 'fa-solid fa-circle-arrow-up';
            } else if (impactPct < -0.5) {
                impactDiv.classList.add('impact-neg');
                icon.className = 'fa-solid fa-circle-arrow-down';
            } else {
                impactDiv.classList.add('impact-neutral');
                icon.className = 'fa-solid fa-minus';
                pctSpan.textContent = '0.0%';
            }

            impactDiv.appendChild(pctSpan);
            impactDiv.appendChild(icon);

            li.appendChild(nameDiv);
            li.appendChild(impactDiv);

            contributionsList.appendChild(li);
        });
    }
});
