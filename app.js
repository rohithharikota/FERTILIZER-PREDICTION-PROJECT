// ===== DOM REFERENCES =====
const navbar = document.getElementById('navbar');
const hamburger = document.getElementById('hamburger');
const navLinks = document.getElementById('navLinks');
const themeToggle = document.getElementById('themeToggle');
const sunIcon = document.getElementById('sunIcon');
const moonIcon = document.getElementById('moonIcon');
const predictForm = document.getElementById('predictForm');
const submitBtn = document.getElementById('submitBtn');
const btnText = document.getElementById('btnText');
const spinner = document.getElementById('spinner');
const resultsSection = document.getElementById('results');

// ===== API BASE URL =====
const API_BASE = window.location.origin;  // Flask serves on same origin

// ===== AREA CALCULATOR STATE =====
let currentPerAcreMin = null;   // kg per acre — minimum
let currentPerAcreMax = null;   // kg per acre — maximum
let currentDosageCrop = '';

// ===== THEME TOGGLE =====
function getPreferredTheme() {
    const stored = localStorage.getItem('agrisense-theme');
    if (stored) return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('agrisense-theme', theme);
    if (theme === 'dark') {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
    } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
    }
}

applyTheme(getPreferredTheme());

themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
});

// ===== NAVBAR SCROLL =====
window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
});

// ===== MOBILE MENU =====
hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('open');
    navLinks.classList.toggle('open');
});

// Close menu on link click
navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
        hamburger.classList.remove('open');
        navLinks.classList.remove('open');
    });
});

// ===== ACTIVE NAV LINK =====
const sections = document.querySelectorAll('section[id]');

function updateActiveNav() {
    const scrollY = window.scrollY + 100;
    sections.forEach(section => {
        const top = section.offsetTop;
        const height = section.offsetHeight;
        const id = section.getAttribute('id');
        const link = document.querySelector(`.nav-links a[href="#${id}"]`);
        if (link) {
            if (scrollY >= top && scrollY < top + height) {
                document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
                link.classList.add('active');
            }
        }
    });
}

window.addEventListener('scroll', updateActiveNav);

// ===== SCROLL REVEAL =====
const revealElements = document.querySelectorAll('.reveal');

const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
});

revealElements.forEach(el => revealObserver.observe(el));

// ===== FORM VALIDATION =====
const fields = [
    { id: 'nitrogen', min: 0, max: 200, type: 'number' },
    { id: 'phosphorus', min: 0, max: 200, type: 'number' },
    { id: 'potassium', min: 0, max: 200, type: 'number' },
    { id: 'humidity', min: 0, max: 100, type: 'number' },
    { id: 'moisture', min: 0, max: 100, type: 'number' },
    { id: 'temperature', min: -10, max: 60, type: 'number' },
    { id: 'soilType', type: 'select' },
    { id: 'cropType', type: 'select' },
];

function validateField(field) {
    const el = document.getElementById(field.id);
    const errorEl = document.getElementById(field.id + 'Error');
    let valid = true;

    if (field.type === 'select') {
        valid = el.value !== '';
    } else {
        const val = parseFloat(el.value);
        valid = el.value !== '' && !isNaN(val) && val >= field.min && val <= field.max;
    }

    el.classList.toggle('error', !valid);
    if (errorEl) errorEl.classList.toggle('show', !valid);
    return valid;
}

function validateAll() {
    let allValid = true;
    fields.forEach(f => {
        if (!validateField(f)) allValid = false;
    });
    return allValid;
}

// Clear errors on input
fields.forEach(f => {
    const el = document.getElementById(f.id);
    el.addEventListener('input', () => validateField(f));
    el.addEventListener('change', () => validateField(f));
});

// ===== REAL API PREDICTION =====
async function getPrediction(inputs) {
    const payload = {
        temperature: parseFloat(inputs.temperature),
        humidity: parseFloat(inputs.humidity),
        moisture: parseFloat(inputs.moisture),
        nitrogen: parseFloat(inputs.nitrogen),
        phosphorus: parseFloat(inputs.phosphorus),
        potassium: parseFloat(inputs.potassium),
        soil_type: inputs.soilType,
        crop_type: inputs.cropType
    };

    const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.success) {
        throw new Error(data.error || 'Prediction failed');
    }

    return data;
}

// ===== RENDER RESULTS =====
function renderResults(data) {
    const result = data.prediction;
    const input = data.input;

    // Main card
    document.getElementById('fertilizerName').textContent = result.fertilizer;
    document.getElementById('fertilizerType').textContent = result.type;
    document.getElementById('applicationRate').textContent = result.application_rate;
    document.getElementById('applicationMethod').textContent = result.application_method;
    document.getElementById('bestSeason').textContent = result.best_season;

    // Confidence
    document.getElementById('confidenceValue').textContent = result.confidence + '%';
    const bar = document.getElementById('confidenceBar');
    bar.style.width = '0%';
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            bar.style.width = result.confidence + '%';
        });
    });

    // NPK chart — show the user's input values
    document.getElementById('npkN').textContent = input.nitrogen;
    document.getElementById('npkP').textContent = input.phosphorus;
    document.getElementById('npkK').textContent = input.potassium;

    // Tips
    const tipsContainer = document.getElementById('tipsList');
    tipsContainer.innerHTML = result.tips.map(tip => `
    <div class="tip-item">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
      ${tip}
    </div>
  `).join('');

    // Dosage Recommendation
    if (result.dosage_recommendation) {
        const dosage = result.dosage_recommendation;
        document.getElementById('dosageCrop').textContent = `For ${dosage.crop} crop`;
        document.getElementById('dosagePerAcre').textContent = dosage.per_acre;
        document.getElementById('dosageTotal').textContent = dosage.total_per_season;
        document.getElementById('dosageSplits').textContent =
            dosage.splits === 1 ? '1 (Single application)' : `${dosage.splits} split doses`;
        document.getElementById('dosageSchedule').textContent = dosage.schedule;
        document.getElementById('dosageGeneral').textContent = dosage.general_rate;

        // Store per-acre range for area calculator
        currentDosageCrop = dosage.crop;
        const parsed = parsePerAcreRange(dosage.per_acre);
        currentPerAcreMin = parsed ? parsed.min : null;
        currentPerAcreMax = parsed ? parsed.max : null;

        // Reset area calculator UI
        document.getElementById('farmArea').value = '';
        document.getElementById('areaCalcResult').style.display = 'none';
        document.getElementById('areaCalcError').style.display = 'none';
    }

    // Expected Outcomes
    const outcomesContainer = document.getElementById('outcomesList');
    const outcomeConfig = {
        crop_yield: { icon: '🌾', iconClass: 'yield', valueClass: 'green' },
        cost_reduction: { icon: '💰', iconClass: 'cost', valueClass: 'gold' },
        environmental: { icon: '🌍', iconClass: 'env', valueClass: 'teal' },
        soil_health: { icon: '🪴', iconClass: 'soil', valueClass: 'brown' },
        decision_making: { icon: '🧠', iconClass: 'decide', valueClass: 'purple' },
    };

    if (result.expected_outcomes && Object.keys(result.expected_outcomes).length > 0) {
        outcomesContainer.innerHTML = Object.entries(result.expected_outcomes).map(([key, o]) => {
            const cfg = outcomeConfig[key] || { icon: '📊', iconClass: 'yield', valueClass: 'green' };
            return `
        <div class="outcome-item">
          <div class="outcome-icon ${cfg.iconClass}">${cfg.icon}</div>
          <div class="outcome-content">
            <div class="outcome-header">
              <span class="outcome-label">${o.label}</span>
              <span class="outcome-value ${cfg.valueClass}">${o.value}</span>
            </div>
            <div class="outcome-desc">${o.desc}</div>
          </div>
        </div>
      `;
        }).join('');
    }

    // Show results
    resultsSection.classList.add('show');

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// ===== SHOW ERROR MESSAGE =====
function showError(message) {
    // Create a temporary error banner
    const existing = document.querySelector('.error-banner');
    if (existing) existing.remove();

    const banner = document.createElement('div');
    banner.className = 'error-banner';
    banner.style.cssText = `
    position: fixed; top: 80px; left: 50%; transform: translateX(-50%);
    background: #e63946; color: white; padding: 1rem 2rem;
    border-radius: 10px; z-index: 9999; font-weight: 500;
    box-shadow: 0 4px 20px rgba(230,57,70,.3);
    animation: slideUp 0.4s ease;
    max-width: 90%; text-align: center;
  `;
    banner.textContent = message;
    document.body.appendChild(banner);

    setTimeout(() => {
        banner.style.opacity = '0';
        banner.style.transition = 'opacity 0.3s ease';
        setTimeout(() => banner.remove(), 300);
    }, 5000);
}

// ===== FORM SUBMIT =====
predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!validateAll()) return;

    // Loading state
    submitBtn.disabled = true;
    btnText.textContent = 'Analyzing with ML Model...';
    spinner.style.display = 'block';

    // Gather inputs
    const inputs = {};
    fields.forEach(f => {
        inputs[f.id] = document.getElementById(f.id).value;
    });

    try {
        const data = await getPrediction(inputs);
        renderResults(data);
    } catch (err) {
        console.error('Prediction failed:', err);
        showError(err.message || 'Failed to connect to the prediction server. Make sure the Flask server is running.');
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = 'Get Fertilizer Recommendation';
        spinner.style.display = 'none';
    }
});

// ===== SMOOTH SCROLL FOR ANCHOR LINKS =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(anchor.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// ===== AREA DOSAGE CALCULATOR =====

/**
 * Parse a per-acre string like "35–50 kg" or "35-50 kg" into {min, max} numbers.
 */
function parsePerAcreRange(str) {
    if (!str || str === 'N/A' || str === '—') return null;
    // Normalise en-dash and em-dash to hyphen, strip kg
    const clean = str.replace(/[–—]/g, '-').replace(/kg.*/i, '').trim();
    const parts = clean.split('-').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    if (parts.length === 2) return { min: parts[0], max: parts[1] };
    if (parts.length === 1) return { min: parts[0], max: parts[0] };
    return null;
}

/**
 * Convert the given area + unit to acres.
 * 1 hectare = 2.47105 acres
 * 1 sq ft   = 1/43560 acres
 */
function toAcres(value, unit) {
    if (unit === 'acre')    return value;
    if (unit === 'hectare') return value * 2.47105;
    if (unit === 'sqft')    return value / 43560;
    return value;
}

function unitLabel(unit) {
    if (unit === 'acre')    return 'acres';
    if (unit === 'hectare') return 'hectares';
    if (unit === 'sqft')    return 'sq. feet';
    return unit;
}

function calculateAreaDosage() {
    const errorEl  = document.getElementById('areaCalcError');
    const resultEl = document.getElementById('areaCalcResult');
    errorEl.style.display  = 'none';
    resultEl.style.display = 'none';

    if (currentPerAcreMin === null) {
        errorEl.textContent = 'Please run a prediction first to get dosage data.';
        errorEl.style.display = 'block';
        return;
    }

    const areaInput = parseFloat(document.getElementById('farmArea').value);
    const unit      = document.getElementById('areaUnit').value;

    if (isNaN(areaInput) || areaInput <= 0) {
        errorEl.textContent = 'Please enter a valid farm area greater than 0.';
        errorEl.style.display = 'block';
        return;
    }

    const acres  = toAcres(areaInput, unit);
    const minQty = (currentPerAcreMin * acres).toFixed(1);
    const maxQty = (currentPerAcreMax * acres).toFixed(1);

    document.getElementById('areaResultMin').textContent = `${minQty} kg`;
    document.getElementById('areaResultMax').textContent = `${maxQty} kg`;
    document.getElementById('areaResultNote').textContent =
        `For ${areaInput} ${unitLabel(unit)} (≈ ${acres.toFixed(2)} acres) of ${currentDosageCrop} crop. ` +
        `Based on ${currentPerAcreMin}–${currentPerAcreMax} kg per acre dosage rate.`;

    resultEl.style.display = 'block';
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
