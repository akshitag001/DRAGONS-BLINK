document.addEventListener('DOMContentLoaded', () => {
    
    // UI Elements
    const toggleFaces = document.getElementById('toggle-faces');
    const toggleCards = document.getElementById('toggle-cards');
    
    const valFps = document.getElementById('val-fps');
    const valFaces = document.getElementById('val-faces');
    const valCards = document.getElementById('val-cards');
    
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    // Handle toggles
    function sendToggle(setting, value) {
        fetch('/api/toggle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ setting, value })
        }).catch(err => console.error("Error sending toggle:", err));
    }

    toggleFaces.addEventListener('change', (e) => {
        sendToggle('blur_faces', e.target.checked);
    });

    toggleCards.addEventListener('change', (e) => {
        sendToggle('blur_cards', e.target.checked);
    });

    // Helper for smooth counting animation
    function animateValue(obj, start, end, duration) {
        if (start === end) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end;
            }
        };
        window.requestAnimationFrame(step);
    }

    // State tracking for animations
    let currentFaces = 0;
    let currentCards = 0;
    
    // Poll state from server
    async function fetchState() {
        try {
            const response = await fetch('/api/state');
            const data = await response.json();
            
            // Sync switches (only if user hasn't just clicked them to avoid jitter)
            if (document.activeElement !== toggleFaces) toggleFaces.checked = data.blur_faces;
            if (document.activeElement !== toggleCards) toggleCards.checked = data.blur_cards;
            
            // Update metrics
            valFps.innerText = data.fps.toFixed(1);
            
            if (data.total_faces !== currentFaces) {
                animateValue(valFaces, currentFaces, data.total_faces, 400);
                currentFaces = data.total_faces;
            }
            
            if (data.total_cards !== currentCards) {
                animateValue(valCards, currentCards, data.total_cards, 400);
                currentCards = data.total_cards;
            }
            
            // Update status indicator
            if (data.is_running) {
                statusDot.className = 'pulse-dot active';
                statusText.innerText = 'Pipeline Active';
            } else {
                statusDot.className = 'pulse-dot';
                statusText.innerText = 'Pipeline Stopped';
            }
            
        } catch (err) {
            console.error("Failed to fetch state:", err);
            statusDot.className = 'pulse-dot';
            statusText.innerText = 'Server Offline';
        }
    }

    // Initial fetch
    fetchState();
    
    // Poll every 500ms
    setInterval(fetchState, 500);
});
