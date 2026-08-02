// Frontend Logic for AI Resume Screening Agent
const API_BASE = 'http://localhost:8000';

// App State
let selectedFiles = [];
let isDemoMode = false;
let demoFileList = [];

// Templates mapping
const templates = {
    python: `We are looking for a Senior Python Developer with 5+ years of experience.
Core requirements:
- Deep experience building scalable backend APIs with FastAPI or Django.
- Strong knowledge of SQL databases, particularly PostgreSQL (query optimization, indexing).
- Solid knowledge of Docker containers and cloud hosting (AWS EC2, S3, RDS).
- Familiarity with Git, unit testing, and automated CI/CD pipelines.`,
    
    react: `We are seeking a Frontend React Engineer to build high-performance web dashboards.
Core requirements:
- Expert-level understanding of React (Hooks, Context, State Management) and TypeScript.
- Strong styling proficiency using Vanilla CSS, Tailwind CSS, or Sass.
- Experience migrating legacy webpack platforms to Vite for faster HMR builds.
- Dedicated to pixel-perfect UI execution and responsive mobile layouts.`,
    
    pm: `We are hiring a Technical Product Manager to oversee SaaS platform development.
Core requirements:
- 5+ years of experience leading cross-functional engineering teams.
- Extensive experience managing sprint backlogs and roadmaps using Jira and Confluence.
- Strong background in Agile methodologies (Scrum, Kanban, sprints).
- Proficient in SQL query analysis and visualizing metrics using Tableau or Amplitude.`
};

// DOM Elements
const badge = document.getElementById('agent-status-badge');
const form = document.getElementById('screener-form');
const jdInput = document.getElementById('job-description');
const fileInput = document.getElementById('file-input');
const dropzone = document.getElementById('dropzone');
const fileList = document.getElementById('file-list');
const fileListContainer = document.getElementById('file-list-container');
const fileCountText = document.getElementById('file-count');
const btnSubmit = document.getElementById('btn-submit-screen');

// Panels
const panelPlaceholder = document.getElementById('results-placeholder');
const panelLoading = document.getElementById('results-loading');
const panelError = document.getElementById('results-error');
const panelCandidates = document.getElementById('results-candidates');
const candidatesListContainer = document.getElementById('candidates-list');
const modeBadge = document.getElementById('results-mode-badge');
const countBadge = document.getElementById('results-count-badge');
const errText = document.getElementById('error-message-text');

// Loading steps
const stepParsing = document.getElementById('step-parsing');
const stepNlp = document.getElementById('step-nlp');
const stepLlm = document.getElementById('step-llm');

// Init check
window.addEventListener('DOMContentLoaded', async () => {
    await checkBackendStatus();
    updateSubmitButtonState();
});

// Check API status
async function checkBackendStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        
        if (data.status === 'ok') {
            if (data.openai_configured) {
                badge.className = 'status-badge active-hybrid';
                badge.innerHTML = '<span class="pulse-dot"></span> OpenAI Hybrid Mode Active';
            } else {
                badge.className = 'status-badge active-nlp';
                badge.innerHTML = '<span class="pulse-dot"></span> Classical NLP (Offline Mode)';
            }
        }
    } catch (err) {
        console.error('Backend offline', err);
        badge.className = 'status-badge error';
        badge.innerHTML = '<span class="pulse-dot" style="background-color:#ef4444"></span> Backend Offline';
    }
}

// Fill JD template
window.fillTemplate = function(role) {
    if (templates[role]) {
        jdInput.value = templates[role];
        updateSubmitButtonState();
    }
};

// Seed sample resumes on the server
window.seedDemoResumes = async function() {
    try {
        badge.className = 'status-badge loading';
        badge.innerHTML = '<span class="pulse-dot"></span> Seeding files...';
        
        const res = await fetch(`${API_BASE}/api/generate-samples`, { method: 'POST' });
        const data = await res.json();
        
        if (data.success) {
            isDemoMode = true;
            selectedFiles = []; // clear manual files
            demoFileList = data.files;
            
            // Render files list in UI
            renderFilesList(data.files, true);
            updateSubmitButtonState();
            
            // Check status again
            await checkBackendStatus();
        }
    } catch (err) {
        console.error(err);
        alert('Failed to connect to backend to seed files. Ensure FastAPI is running.');
        await checkBackendStatus();
    }
};

// Clear file list
window.clearSelectedFiles = function() {
    selectedFiles = [];
    isDemoMode = false;
    demoFileList = [];
    fileInput.value = '';
    renderFilesList([], false);
    updateSubmitButtonState();
};

// Drag and drop events
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        handleFilesSelected(e.dataTransfer.files);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFilesSelected(e.target.files);
    }
});

function handleFilesSelected(files) {
    isDemoMode = false;
    demoFileList = [];
    
    for (let file of files) {
        // Prevent duplicates
        if (!selectedFiles.some(f => f.name === file.name)) {
            selectedFiles.push(file);
        }
    }
    
    renderFilesList(selectedFiles, false);
    updateSubmitButtonState();
}

function renderFilesList(files, isDemo = false) {
    fileList.innerHTML = '';
    
    if (files.length === 0) {
        fileListContainer.style.display = 'none';
        return;
    }
    
    fileListContainer.style.display = 'flex';
    fileCountText.textContent = `${files.length} Files ${isDemo ? 'Seeded' : 'Selected'}`;
    
    files.forEach((file, index) => {
        const name = isDemo ? file : file.name;
        const sizeInfo = isDemo ? 'Demo Template' : formatBytes(file.size);
        
        const li = document.createElement('li');
        li.className = 'file-item';
        li.innerHTML = `
            <div class="file-info">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="file-icon"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                <span><strong>${name}</strong> (${sizeInfo})</span>
            </div>
            ${isDemo ? '' : `<button type="button" class="btn-remove-file" onclick="removeFile(${index})">&times;</button>`}
        `;
        fileList.appendChild(li);
    });
}

window.removeFile = function(index) {
    selectedFiles.splice(index, 1);
    renderFilesList(selectedFiles, false);
    updateSubmitButtonState();
};

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function updateSubmitButtonState() {
    const hasJd = jdInput.value.trim().length > 0;
    const hasFiles = selectedFiles.length > 0 || demoFileList.length > 0;
    btnSubmit.disabled = !(hasJd && hasFiles);
}

jdInput.addEventListener('input', updateSubmitButtonState);

// Form Submit - Trigger screening
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const jd = jdInput.value.trim();
    if (!jd) return;
    
    // Switch UI panels
    panelPlaceholder.style.display = 'none';
    panelCandidates.style.display = 'none';
    panelError.style.display = 'none';
    panelLoading.style.display = 'flex';
    
    // Loading stepper animation sequence
    setStepState('parsing', 'loading');
    setStepState('nlp', 'waiting');
    setStepState('llm', 'waiting');
    
    try {
        let data;
        
        if (isDemoMode) {
            // Seeding pipeline: trigger endpoint on server direct paths
            setStepState('parsing', 'completed');
            setStepState('nlp', 'loading');
            
            // Wait 600ms for visual stepper progression
            await new Promise(r => setTimeout(r, 600));
            setStepState('nlp', 'completed');
            setStepState('llm', 'loading');
            
            const formData = new FormData();
            formData.append('job_description', jd);
            
            const res = await fetch(`${API_BASE}/api/screen-samples`, {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) throw new Error(await res.text());
            data = await res.json();
            
        } else {
            // Upload pipeline: upload user files
            setStepState('parsing', 'loading');
            
            const formData = new FormData();
            formData.append('job_description', jd);
            
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            // Let parsing show for a moment
            await new Promise(r => setTimeout(r, 800));
            setStepState('parsing', 'completed');
            setStepState('nlp', 'loading');
            
            // Let NLP show for a moment
            await new Promise(r => setTimeout(r, 800));
            setStepState('nlp', 'completed');
            setStepState('llm', 'loading');
            
            const res = await fetch(`${API_BASE}/api/screen`, {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) throw new Error(await res.text());
            data = await res.json();
        }
        
        setStepState('llm', 'completed');
        
        // Short pause to appreciate progress
        await new Promise(r => setTimeout(r, 400));
        
        renderCandidates(data.candidates, data.mode);
        
    } catch (err) {
        console.error(err);
        panelLoading.style.display = 'none';
        panelError.style.display = 'flex';
        
        let message = 'Connection to the FastAPI server failed. Make sure your backend service is running.';
        if (err.message) {
            try {
                const parsed = JSON.parse(err.message);
                message = parsed.detail || message;
            } catch(e) {
                message = err.message;
            }
        }
        errText.textContent = message;
    }
});

function setStepState(stepName, state) {
    const el = document.getElementById(`step-${stepName}`);
    if (!el) return;
    
    const indicator = el.querySelector('.step-indicator');
    
    if (state === 'waiting') {
        el.className = 'stepper-step';
        indicator.innerHTML = `<span class="step-dot">${stepName === 'parsing' ? '1' : stepName === 'nlp' ? '2' : '3'}</span>`;
    } else if (state === 'loading') {
        el.className = 'stepper-step active';
        indicator.innerHTML = '<span class="spinner-small"></span>';
    } else if (state === 'completed') {
        el.className = 'stepper-step completed';
        indicator.innerHTML = '✓';
    }
}

function renderCandidates(candidates, mode) {
    panelLoading.style.display = 'none';
    panelCandidates.style.display = 'flex';
    candidatesListContainer.innerHTML = '';
    
    // Set Mode description
    if (mode === 'hybrid') {
        modeBadge.className = 'mode-badge';
        modeBadge.textContent = 'Hybrid scoring active (30% NLP / 70% GPT)';
    } else {
        modeBadge.className = 'mode-badge warning';
        modeBadge.style.backgroundColor = 'rgba(245, 158, 11, 0.08)';
        modeBadge.style.color = '#f59e0b';
        modeBadge.style.borderColor = 'rgba(245, 158, 11, 0.2)';
        modeBadge.textContent = 'Classical NLP Vector Search (offline)';
    }
    
    countBadge.textContent = `Found ${candidates.length} Candidates`;
    
    if (candidates.length === 0) {
        candidatesListContainer.innerHTML = '<div style="text-align:center; padding: 20px; color: var(--text-muted)">No valid resumes found.</div>';
        return;
    }
    
    candidates.forEach((cand, idx) => {
        const card = document.createElement('div');
        card.className = 'candidate-card';
        
        // Skills pills
        const skillsHtml = cand.skills.map(s => `<span class="skill-pill">${s}</span>`).join('');
        
        // Strengths, weaknesses, gaps lists
        const strengthsHtml = cand.strengths.map(s => `<li>${s}</li>`).join('');
        const weaknessesHtml = cand.weaknesses.map(w => `<li>${w}</li>`).join('');
        const gapsHtml = cand.gaps.map(g => `<li>${g}</li>`).join('');
        
        card.innerHTML = `
            <div class="candidate-summary" onclick="toggleCardAccordion(this.parentElement)">
                <div class="candidate-left">
                    <div class="rank-badge">${idx + 1}</div>
                    <div class="candidate-identity">
                        <h3>${cand.candidate_name}</h3>
                        <p>${cand.filename} • ${cand.education}</p>
                    </div>
                </div>
                <div class="candidate-right">
                    <div class="score-container">
                        <span class="hybrid-score-badge">${Math.round(cand.hybrid_score)}% Match</span>
                        <span class="sub-scores">NLP: ${Math.round(cand.nlp_score)}% | LLM: ${Math.round(cand.llm_score)}%</span>
                    </div>
                    <svg class="expand-chevron" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
                </div>
            </div>
            
            <div class="candidate-details">
                <div class="detail-section">
                    <h4>Identified Skills</h4>
                    <div class="skills-pills">${skillsHtml || 'No specific skills highlighted.'}</div>
                </div>
                
                <div class="detail-section">
                    <h4>Experience Profile</h4>
                    <p>${cand.experience || 'Not specified.'}</p>
                </div>
                
                <div class="detail-section">
                    <h4>Evaluation Summary</h4>
                    <p style="font-style: italic; color: #cbd5e1; border-left: 2px solid var(--primary); padding-left: 10px;">
                        "${cand.reasoning}"
                    </p>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px;">
                    ${strengthsHtml ? `
                    <div class="detail-section">
                        <h4>Strengths</h4>
                        <ul class="bullet-list strengths">${strengthsHtml}</ul>
                    </div>` : ''}
                    
                    ${weaknessesHtml ? `
                    <div class="detail-section">
                        <h4>Areas for Improvement</h4>
                        <ul class="bullet-list weaknesses">${weaknessesHtml}</ul>
                    </div>` : ''}
                    
                    ${gapsHtml ? `
                    <div class="detail-section">
                        <h4>Job Description Gaps</h4>
                        <ul class="bullet-list gaps">${gapsHtml}</ul>
                    </div>` : ''}
                </div>
            </div>
        `;
        
        candidatesListContainer.appendChild(card);
    });
}

window.toggleCardAccordion = function(cardElement) {
    const isOpen = cardElement.classList.contains('open');
    
    // Close other open cards
    document.querySelectorAll('.candidate-card.open').forEach(c => {
        if (c !== cardElement) c.classList.remove('open');
    });
    
    // Toggle active card
    if (isOpen) {
        cardElement.classList.remove('open');
    } else {
        cardElement.classList.add('open');
    }
};
