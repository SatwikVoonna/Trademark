/* ================================================================
   SciSearch — Frontend Logic
   ================================================================ */

// --- Particle Background Animation ---
(function initParticles() {
    const canvas = document.getElementById('particleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let particles = [];
    const PARTICLE_COUNT = 60;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createParticle() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            size: Math.random() * 2.5 + 0.5,
            speedX: (Math.random() - 0.5) * 0.3,
            speedY: (Math.random() - 0.5) * 0.3,
            opacity: Math.random() * 0.4 + 0.1,
            pulseSpeed: Math.random() * 0.02 + 0.005,
            pulseOffset: Math.random() * Math.PI * 2,
        };
    }

    function initParticleArray() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push(createParticle());
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const time = Date.now() * 0.001;

        particles.forEach(p => {
            p.x += p.speedX;
            p.y += p.speedY;

            // Wrap around
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            const pulse = Math.sin(time * p.pulseSpeed * 10 + p.pulseOffset) * 0.5 + 0.5;
            const alpha = p.opacity * (0.5 + pulse * 0.5);

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(192, 133, 82, ${alpha})`;
            ctx.fill();
        });

        // Draw subtle connections between close particles
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 120) {
                    const alpha = (1 - dist / 120) * 0.06;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(192, 133, 82, ${alpha})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(animate);
    }

    resize();
    initParticleArray();
    animate();

    window.addEventListener('resize', () => {
        resize();
        initParticleArray();
    });
})();


// --- Theme Logic ---
const themeToggle = document.getElementById('themeToggle');
const currentTheme = localStorage.getItem('theme') || 'light';

if (currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
}

themeToggle.addEventListener('click', () => {
    let theme = document.documentElement.getAttribute('data-theme');
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    }
});


// --- DOM Elements ---
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const loadingState = document.getElementById('loadingState');
const emptyState = document.getElementById('emptyState');
const resultsContainer = document.getElementById('resultsContainer');
const queryInfo = document.getElementById('queryInfo');
const chips = document.querySelectorAll('.chip');


// --- API Config ---
const API_BASE = window.location.origin;


// --- Search Logic ---
async function performSearch(query) {
    if (!query.trim()) return;

    // Show loading, hide others
    loadingState.style.display = 'block';
    emptyState.style.display = 'none';
    resultsContainer.innerHTML = '';
    queryInfo.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query.trim() }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        loadingState.style.display = 'none';

        if (!data.result || data.result.length === 0) {
            emptyState.style.display = 'block';
            return;
        }

        renderResults(data);
    } catch (error) {
        console.error('Search failed:', error);
        loadingState.style.display = 'none';
        emptyState.style.display = 'block';
        document.querySelector('.empty-text').textContent = 'Search failed';
        document.querySelector('.empty-hint').textContent = error.message || 'Please try again';
    }
}


// --- Render Results ---
function renderResults(data) {
    const results = data.result;

    results.forEach((result, index) => {
        const card = createResultCard(result, index, data.query);
        resultsContainer.appendChild(card);
    });

    // Show query info
    queryInfo.style.display = 'block';
    let infoHTML = `Showing ${results.length} results for "<strong>${escapeHTML(data.query)}</strong>"`;
    if (data.cache_hit) {
        infoHTML += `<span class="cache-badge">⚡ Cache Hit (${(data.similarity_score * 100).toFixed(1)}%)</span>`;
    }
    queryInfo.innerHTML = infoHTML;

    // Trigger score bar animations after cards appear
    setTimeout(() => {
        document.querySelectorAll('.score-bar-fill').forEach(bar => {
            bar.style.width = bar.dataset.score + '%';
        });
    }, 100);
}


function createResultCard(result, index, queryText) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.animationDelay = `${index * 0.12}s`;

    const classNum = result.class || 0;
    const chapter = result.chapter || 'Unknown Chapter';
    const passage = result.passage || '';
    const score = result.score || 0;
    const subject = result.subject || 'Science';
    const keywords = result.keywords || [];
    const pageNum = result.page || 1;
    const fileName = result.file || '';

    // URL to open PDF at specific page
    const pdfUrl = `/pdfs/${fileName}#page=${pageNum}`;

    // Highlight keywords in passage
    const highlightedPassage = highlightText(passage, keywords, queryText);

    // Subject tag class
    const subjectClass = subject.toLowerCase();
    const subjectIcon = getSubjectIcon(subject);

    card.innerHTML = `
        <div class="card-header">
            <span class="class-badge">CLASS ${classNum}</span>
            <div class="chapter-title-row">
                <a href="${pdfUrl}" target="_blank" class="chapter-title" title="Open PDF at page ${pageNum}">
                    ${escapeHTML(chapter)} <span class="link-icon">↗</span>
                </a>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; margin-top: -4px;">
                <span class="subject-tag ${subjectClass}">${subjectIcon} ${escapeHTML(subject)}</span>
                <span style="font-size: 0.65rem; color: var(--text-muted); font-family: 'Space Mono'; opacity: 0.7;">PAGE ${pageNum}</span>
            </div>
        </div>
        <div class="card-passage">${highlightedPassage}</div>
        <div class="score-bar-wrapper">
            <span class="score-label">MATCH</span>
            <div class="score-bar-track">
                <div class="score-bar-fill" data-score="${score}"></div>
            </div>
            <span class="score-value">${score.toFixed(1)}%</span>
        </div>
    `;

    return card;
}


// --- Utility Functions ---

function highlightText(text, keywords, queryText) {
    let escaped = escapeHTML(text);

    // Get all terms to highlight: keywords + query words
    const queryWords = queryText.toLowerCase()
        .split(/\s+/)
        .filter(w => w.length >= 3)
        .map(w => w.replace(/[^a-zA-Z0-9]/g, ''));

    const allTerms = [...new Set([...keywords, ...queryWords])].filter(Boolean);

    allTerms.forEach(term => {
        const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
        escaped = escaped.replace(regex, '<span class="highlight">$1</span>');
    });

    return escaped;
}

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function truncate(str, maxLen) {
    if (str.length <= maxLen) return str;
    return str.substring(0, maxLen) + '...';
}

function getSubjectIcon(subject) {
    const icons = {
        physics: '⚡',
        chemistry: '🧪',
        biology: '🧬',
        science: '🔬',
    };
    return icons[subject.toLowerCase()] || '📖';
}


// --- Event Listeners ---

// Search on Enter
searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        performSearch(searchInput.value);
    }
});

// Search on button click
searchBtn.addEventListener('click', () => {
    performSearch(searchInput.value);
});

// Chip clicks
chips.forEach(chip => {
    chip.addEventListener('click', () => {
        const query = chip.dataset.query;
        searchInput.value = query;
        performSearch(query);
    });
});

// Focus search on page load
searchInput.focus();
