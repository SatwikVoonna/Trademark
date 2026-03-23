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
    document.documentElement.classList.add('dark');
}

themeToggle.addEventListener('click', () => {
    let theme = document.documentElement.getAttribute('data-theme');
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'light');
        document.documentElement.classList.remove('dark');
        localStorage.setItem('theme', 'light');
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        document.documentElement.classList.add('dark');
        localStorage.setItem('theme', 'dark');
    }
});


// --- DOM Elements ---
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const loadingState = document.getElementById('loadingState');
const emptyState = document.getElementById('emptyState');
const initialState = document.getElementById('initialState');
const resultsContainer = document.getElementById('resultsContainer');
const queryInfo = document.getElementById('queryInfo');
const chips = document.querySelectorAll('.chip');
const resetFiltersBtn = document.getElementById('resetFiltersBtn');
const chapterFiltersContainer = document.getElementById('chapterFilters');


// --- API Config ---
const API_BASE = window.location.origin;


// --- Filter State & Logic ---
function getActiveFilters() {
    const filters = {};
    
    document.querySelectorAll('.filter-checkbox:checked').forEach(cb => {
        const cat = cb.dataset.category || cb.closest('[id$="Filters"]').id.replace('Filters','');
        let val = cb.value;
        if (cat === 'class') {
            val = parseInt(val);
        }
        if (!filters[cat]) {
            filters[cat] = [];
        }
        filters[cat].push(val);
    });

    // Only return categories that have at least one selection
    return filters;
}

function updateFilterCounts(facets) {
    if (!facets) return;

    // Update subjects and classes counts
    document.querySelectorAll('.filter-checkbox').forEach(cb => {
        const cat = cb.dataset.category || cb.closest('[id$="Filters"]').id.replace('Filters','');
        if (cat === 'subject' || cat === 'class') {
            const countSpan = cb.nextElementSibling.querySelector('.filter-count');
            if (countSpan) {
                const count = (facets[cat] && facets[cat][cb.value]) ? facets[cat][cb.value] : 0;
                countSpan.textContent = `(${count})`;
            }
        }
    });

    // Populate dynamic chapters if not populated yet, or update them
    if (facets.chapter && Object.keys(facets.chapter).length > 0) {
        // Keep checked chapters
        const checkedChapters = Array.from(document.querySelectorAll('#chapterFilters input:checked')).map(cb => cb.value);
        
        let html = '';
        Object.keys(facets.chapter).sort().forEach(chapter => {
            const isChecked = checkedChapters.includes(chapter) ? 'checked' : '';
            const count = facets.chapter[chapter];
            
            html += `
                <label class="flex items-center gap-3 p-2 rounded-xl hover:bg-[#f3e9dc] dark:hover:bg-copper/10 transition-colors cursor-pointer group">
                    <input type="checkbox" data-category="chapter" value="${escapeHTML(chapter)}" ${isChecked} class="filter-checkbox w-4 h-4 rounded border-copper text-copper focus:ring-copper cursor-pointer" />
                    <span class="text-[0.85rem] font-medium group-hover:translate-x-1 transition-transform dark:text-[#E8D5B0] ${isChecked ? 'text-[#d4af37] font-bold' : ''}">${escapeHTML(chapter)} <span class="text-xs opacity-50 ml-1 filter-count">(${count})</span></span>
                </label>
            `;
        });
        chapterFiltersContainer.innerHTML = html;

        // Re-attach event listeners to new dynamic checkboxes
        chapterFiltersContainer.querySelectorAll('.filter-checkbox').forEach(cb => {
            cb.addEventListener('change', () => performSearch(searchInput.value));
        });
    } else if (Object.keys(facets.chapter || {}).length === 0 && chapterFiltersContainer.innerHTML.includes('Perform a search')) {
         chapterFiltersContainer.innerHTML = '<p class="text-[0.8rem] opacity-50 italic px-2">No chapters match.</p>';
    }
}


// --- Search Logic ---
async function performSearch(query) {
    if (!query.trim()) return;

    // Show loading, hide others
    if (initialState) initialState.style.display = 'none';
    loadingState.style.display = 'block';
    emptyState.style.display = 'none';
    emptyState.innerHTML = '<p class="text-2xl font-bold font-heading text-[var(--text-primary)] mb-4">Discovery Failed</p>';
    resultsContainer.innerHTML = '';
    queryInfo.style.display = 'none';

    const filters = getActiveFilters();

    try {
        console.log('Searching for:', query, 'with filters:', filters);

        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query.trim(), filters: filters }),
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Server error ${response.status}: ${errText}`);
        }

        const data = await response.json();
        console.log('API response:', data);
        loadingState.style.display = 'none';

        if (data.facets) {
             updateFilterCounts(data.facets);
        }

        if (!data.result || data.result.length === 0) {
            emptyState.innerHTML = `
                <p class="text-2xl font-bold font-heading text-[var(--text-primary)] mb-2">No Results Found</p>
                <p class="text-[var(--text-muted)] mt-2 text-sm font-mono">Try a different query or clear your filters</p>`;
            emptyState.style.display = 'block';
            return;
        }

        renderResults(data);
    } catch (error) {
        console.error('Search failed:', error);
        loadingState.style.display = 'none';
        emptyState.innerHTML = `
            <p class="text-2xl font-bold font-heading text-[var(--text-primary)] mb-2">Discovery Failed</p>
            <p class="text-[var(--text-muted)] mt-2 text-sm font-mono">${error.message || 'Could not connect to the server. Is it running?'}</p>`;
        emptyState.style.display = 'block';
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
    queryInfo.style.display = 'flex';
    queryInfo.className = 'query-info mb-8 pb-4 border-b border-copper/20 flex flex-wrap items-center gap-3 text-sm font-medium animate-[fadeIn_0.5s_ease]';
    
    let infoHTML = `<span class="text-[var(--text-secondary)] uppercase tracking-widest text-xs">Found ${results.length} passages for "<strong class="text-[var(--text-primary)] font-bold">${escapeHTML(data.query)}</strong>"</span>`;
    
    // Add applied filter pills
    const activeFilters = getActiveFilters();
    Object.keys(activeFilters).forEach(cat => {
         if (activeFilters[cat].length > 0) {
             activeFilters[cat].forEach(val => {
                 infoHTML += `<span class="px-3 py-1 bg-copper/10 text-copper rounded-full border border-copper/30 text-[0.7rem] font-bold uppercase tracking-tighter">${cat}: ${val}</span>`;
             });
         }
    });

    if (data.cache_hit) {
        infoHTML += `<span class="ml-auto px-4 py-1.5 bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20 rounded-full font-bold text-[0.7rem] uppercase tracking-widest flex items-center gap-2 shadow-sm"><svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg> Cached</span>`;
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
    card.className = 'result-card glass-panel rounded-2xl p-8 transition-all duration-500 hover:-translate-y-2 hover:shadow-2xl hover:border-copper animate-[fadeInUp_0.6s_ease_backwards] relative overflow-hidden group';
    card.style.animationDelay = `${index * 0.15}s`;

    const classNum = result.class || 0;
    const chapter = result.chapter || 'Unknown Chapter';
    const passage = result.passage || '';
    const score = result.score || 0;
    const subject = result.subject || 'Science';
    const keywords = result.keywords || [];
    const pageNum = result.page || 1;
    const fileName = result.file || '';

    const pdfUrl = `/pdfs/${fileName}#page=${pageNum}`;
    const highlightedPassage = highlightText(passage, keywords, queryText);
    const subjectIcon = getSubjectIcon(subject);



    card.innerHTML = `
        <div class="flex flex-col gap-4 relative z-10">
            <div class="flex items-center justify-between">
                <span class="font-heading text-[0.7rem] font-black text-copper tracking-[3px] uppercase">CLASS ${classNum}</span>
                <span class="text-[0.65rem] font-bold text-[var(--text-muted)] font-mono opacity-50 uppercase tracking-widest bg-copper/5 px-3 py-1 rounded-full border border-copper/10">Page ${pageNum}</span>
            </div>
            
            <div class="my-1">
                <a href="${pdfUrl}" target="_blank" class="font-heading text-xl md:text-2xl font-black text-[var(--text-primary)] no-underline inline-block hover:text-copper transition-all duration-300 pb-1 border-b-2 border-transparent hover:border-copper group/link" title="Open Source PDF">
                   ${escapeHTML(chapter)} <span class="text-sm ml-2 opacity-30 group-hover/link:translate-x-1 group-hover/link:-translate-y-1 group-hover/link:opacity-100 transition-all inline-block">↗</span>
                </a>
            </div>

            <div class="flex items-center gap-4 flex-wrap mt-2">
                <span class="font-bold text-[0.75rem] text-[var(--dark-brown)] dark:text-white px-3 py-1.5 rounded-xl bg-copper/10 border border-copper/20 shadow-sm flex items-center gap-2 uppercase tracking-wider">${subjectIcon} ${escapeHTML(subject)}</span>

                
                <div class="ml-auto w-32 flex flex-col gap-1">
                    <div class="flex justify-between items-end">
                        <span class="text-[0.6rem] font-black text-copper tracking-[2px] uppercase opacity-60">PRECISION</span>
                        <span class="text-xs font-black text-copper font-mono">${score.toFixed(1)}%</span>
                    </div>
                    <div class="h-2 bg-copper/10 dark:bg-white/5 rounded-full overflow-hidden shadow-inner">
                        <div class="h-full bg-gradient-to-r from-copper to-brown transition-all duration-1000 ease-out score-bar-fill w-0" data-score="${score}"></div>
                    </div>
                </div>
            </div>

            <div class="text-[var(--text-secondary)] text-[1.05rem] leading-relaxed font-normal card-passage mt-3 border-l-4 border-copper/10 pl-5 dark:border-white/5 transition-colors group-hover:border-copper/40 italic">"${highlightedPassage}"</div>
        </div>
        <div class="absolute -right-12 -bottom-12 w-48 h-48 bg-copper/5 rounded-full blur-3xl group-hover:bg-copper/10 transition-colors duration-700"></div>
    `;

    return card;
}


// --- Utility Functions ---

function highlightText(text, keywords, queryText) {
    let escaped = escapeHTML(text);
    const queryWords = queryText.toLowerCase()
        .split(/\s+/)
        .filter(w => w.length >= 3)
        .map(w => w.replace(/[^a-zA-Z0-9]/g, ''));
    const allTerms = [...new Set([...keywords, ...queryWords])].filter(Boolean);

    allTerms.forEach(term => {
        const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
        escaped = escaped.replace(regex, '<span class="bg-copper/20 text-[var(--text-primary)] border-b-2 border-copper/60 font-bold px-0.5 rounded-sm transition-all hover:bg-copper hover:text-white">$1</span>');
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

function getSubjectIcon(subject) {
    const icons = { physics: '⚡', chemistry: '🧪', biology: '🧬', science: '🔬' };
    return icons[subject.toLowerCase()] || '📖';
}


// --- Event Listeners ---
searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') performSearch(searchInput.value); });
searchBtn.addEventListener('click', () => performSearch(searchInput.value));
chips.forEach(chip => { 
    chip.addEventListener('click', () => { 
        searchInput.value = chip.dataset.query; 
        performSearch(searchInput.value); 
    }); 
});

document.querySelectorAll('.filter-checkbox').forEach(cb => {
    cb.addEventListener('change', () => performSearch(searchInput.value));
});

resetFiltersBtn.addEventListener('click', () => {
    document.querySelectorAll('.filter-checkbox:checked').forEach(cb => { cb.checked = false; });
    if (searchInput.value.trim() !== '') {
        performSearch(searchInput.value);
    } else {
        chapterFiltersContainer.innerHTML = '<p class="text-[0.8rem] opacity-50 italic px-2">Perform a search to see chapters...</p>';
        document.querySelectorAll('.filter-count').forEach(span => span.textContent = '(0)');
    }
});

searchInput.focus();
