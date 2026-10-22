/**
 * NLP Text Intelligence Studio - Client Interactions Script
 */

document.addEventListener('DOMContentLoaded', () => {
    const rawTextInput = document.getElementById('rawTextInput');
    const wordCounter = document.getElementById('wordCounter');
    const charCounter = document.getElementById('charCounter');
    const clearTextBtn = document.getElementById('clearTextBtn');
    const presetBtns = document.querySelectorAll('.preset-btn');
    const copySummaryBtn = document.getElementById('copySummaryBtn');
    const analyzeForm = document.getElementById('analyzeForm');
    const submitBtn = document.getElementById('submitBtn');

    // 1. Text Metrics Counter
    function updateCounters() {
        if (!rawTextInput) return;
        const text = rawTextInput.value.trim();
        const chars = rawTextInput.value.length;
        const words = text ? text.split(/\s+/).filter(Boolean).length : 0;

        if (charCounter) charCounter.textContent = chars.toLocaleString();
        if (wordCounter) wordCounter.textContent = words.toLocaleString();
    }

    if (rawTextInput) {
        rawTextInput.addEventListener('input', updateCounters);
        updateCounters(); // Initialize on page load
    }

    // 2. Preset Buttons
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleText = btn.getAttribute('data-text');
            if (sampleText && rawTextInput) {
                rawTextInput.value = sampleText;
                updateCounters();
                rawTextInput.focus();

                // Highlight active button briefly
                presetBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }
        });
    });

    // 3. Clear Text Button
    if (clearTextBtn && rawTextInput) {
        clearTextBtn.addEventListener('click', () => {
            rawTextInput.value = '';
            updateCounters();
            rawTextInput.focus();
        });
    }

    // 4. Copy Summary to Clipboard
    if (copySummaryBtn) {
        copySummaryBtn.addEventListener('click', async () => {
            const summaryText = copySummaryBtn.getAttribute('data-copy');
            if (!summaryText) return;

            try {
                await navigator.clipboard.writeText(summaryText);
                const originalHtml = copySummaryBtn.innerHTML;
                copySummaryBtn.innerHTML = `
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <span>Copied!</span>
                `;
                copySummaryBtn.style.color = 'var(--pos)';
                copySummaryBtn.style.borderColor = 'var(--pos)';

                setTimeout(() => {
                    copySummaryBtn.innerHTML = originalHtml;
                    copySummaryBtn.style.color = '';
                    copySummaryBtn.style.borderColor = '';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        });
    }

    // 5. Submit Form Loading State
    if (analyzeForm && submitBtn) {
        analyzeForm.addEventListener('submit', (e) => {
            const formaction = e.submitter ? e.submitter.getAttribute('formaction') : null;
            // Only apply analyzing spinner to the main analyze button, not download_pdf
            if (!formaction || formaction === '/analyze') {
                submitBtn.disabled = true;
                submitBtn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="spin">
                        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
                    </svg>
                    <span>Analyzing Pipeline...</span>
                `;
            }
        });
    }
});
