/**
 * CopyCat Dashboard JavaScript
 * Handles API communication and UI updates
 */

// State
let updateInterval = null;
let isRunning = false;
let isPaused = false;

// API Base URL
const API_BASE = '';

// DOM Elements
const elements = {
    statusBadge: document.getElementById('status-badge'),
    statusText: document.querySelector('.status-text'),
    statusDot: document.querySelector('.status-dot'),
    totalPnl: document.getElementById('total-pnl'),
    totalPnlPct: document.getElementById('total-pnl-pct'),
    winRate: document.getElementById('win-rate'),
    sharpeRatio: document.getElementById('sharpe-ratio'),
    maxDrawdown: document.getElementById('max-drawdown'),
    tradesExecuted: document.getElementById('trades-executed'),
    copiedTraders: document.getElementById('copied-traders'),
    tradersList: document.getElementById('traders-list'),
    apiHealth: document.getElementById('api-health'),
    circuitBreaker: document.getElementById('circuit-breaker'),
    cycleCount: document.getElementById('cycle-count'),
    uptime: document.getElementById('uptime'),
    btnStart: document.getElementById('btn-start'),
    btnPause: document.getElementById('btn-pause'),
    btnResume: document.getElementById('btn-resume'),
    btnStop: document.getElementById('btn-stop'),
    toastContainer: document.getElementById('toast-container'),
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    startPolling();
    updateStatus();
});

/**
 * Start polling for updates
 */
function startPolling() {
    // Update every 2 seconds
    updateInterval = setInterval(async () => {
        await updateStatus();
        await updatePerformance();
        await updateHealth();
    }, 2000);
}

/**
 * Stop polling
 */
function stopPolling() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

/**
 * Update status display
 */
async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        if (data.success) {
            const status = data.status;
            isRunning = status.is_running;
            isPaused = status.is_paused;
            
            // Update status badge
            updateStatusBadge(status.is_running, status.is_paused);
            
            // Update traders count
            elements.copiedTraders.textContent = status.copied_traders;
            
            // Update button states
            updateButtonStates(status.is_running, status.is_paused);
            
            // Update uptime
            if (status.uptime_seconds > 0) {
                elements.uptime.textContent = formatDuration(status.uptime_seconds);
            }
        }
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

/**
 * Update status badge
 */
function updateStatusBadge(running, paused) {
    elements.statusDot.classList.remove('running', 'paused', 'stopped');
    
    if (!running) {
        elements.statusDot.classList.add('stopped');
        elements.statusText.textContent = 'Stopped';
    } else if (paused) {
        elements.statusDot.classList.add('paused');
        elements.statusText.textContent = 'Paused';
    } else {
        elements.statusDot.classList.add('running');
        elements.statusText.textContent = 'Running';
    }
}

/**
 * Update button states
 */
function updateButtonStates(running, paused) {
    elements.btnStart.disabled = running;
    elements.btnPause.disabled = !running || paused;
    elements.btnResume.disabled = !running || !paused;
    elements.btnStop.disabled = !running;
}

/**
 * Update performance metrics
 */
async function updatePerformance() {
    try {
        const response = await fetch(`${API_BASE}/api/performance`);
        const data = await response.json();
        
        if (data.success) {
            const metrics = data.metrics;
            
            // Update P&L
            elements.totalPnl.textContent = formatCurrency(metrics.total_pnl);
            elements.totalPnl.textContent.className = 'metric-value ' + (metrics.total_pnl >= 0 ? 'positive' : 'negative');
            
            const pnlPct = metrics.total_pnl_pct * 100;
            elements.totalPnlPct.textContent = formatPercent(pnlPct);
            elements.totalPnlPct.className = 'metric-change ' + (metrics.total_pnl_pct >= 0 ? 'positive' : 'negative');
            
            // Update other metrics
            elements.winRate.textContent = formatPercent(metrics.win_rate * 100);
            elements.sharpeRatio.textContent = metrics.sharpe_ratio.toFixed(2);
            elements.maxDrawdown.textContent = formatPercent(metrics.max_drawdown * 100);
            elements.tradesExecuted.textContent = metrics.trades_executed;
        }
    } catch (error) {
        console.error('Error fetching performance:', error);
    }
}

/**
 * Update health status
 */
async function updateHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        
        if (data.success) {
            const health = data.health;
            
            // API Health
            elements.apiHealth.textContent = health.api_healthy ? 'Healthy' : 'Unhealthy';
            elements.apiHealth.className = 'health-value ' + (health.api_healthy ? 'healthy' : 'unhealthy');
            
            // Circuit Breaker
            elements.circuitBreaker.textContent = health.circuit_breaker_open ? 'Open' : 'Closed';
            elements.circuitBreaker.className = 'health-value ' + (health.circuit_breaker_open ? 'warning' : 'healthy');
            
            // Cycle Count
            elements.cycleCount.textContent = health.cycle_count;
        }
    } catch (error) {
        console.error('Error fetching health:', error);
    }
}

/**
 * Update traders list
 */
async function updateTradersList() {
    try {
        const response = await fetch(`${API_BASE}/api/traders`);
        const data = await response.json();
        
        if (data.success) {
            if (data.count === 0) {
                elements.tradersList.innerHTML = '<div class="empty-state">No traders copied yet</div>';
            } else {
                elements.tradersList.innerHTML = data.traders.map(trader => `
                    <div class="trader-item">
                        <div class="trader-info">
                            <span class="trader-address">${formatAddress(trader.address)}</span>
                            <span class="trader-position">Position: ${formatCurrency(trader.position_size)}</span>
                        </div>
                        <div class="trader-actions">
                            <button class="btn btn-danger btn-small" onclick="removeTrader('${trader.address}')">
                                Remove
                            </button>
                        </div>
                    </div>
                `).join('');
            }
        }
    } catch (error) {
        console.error('Error fetching traders:', error);
    }
}

/**
 * Start orchestrator
 */
async function startOrchestrator() {
    try {
        const response = await fetch(`${API_BASE}/api/start`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('Orchestrator started', 'success');
            await updateStatus();
        } else {
            showToast(data.message || 'Failed to start', 'error');
        }
    } catch (error) {
        showToast('Error starting orchestrator', 'error');
    }
}

/**
 * Stop orchestrator
 */
async function stopOrchestrator() {
    try {
        const response = await fetch(`${API_BASE}/api/stop`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('Orchestrator stopped', 'success');
            await updateStatus();
        } else {
            showToast(data.message || 'Failed to stop', 'error');
        }
    } catch (error) {
        showToast('Error stopping orchestrator', 'error');
    }
}

/**
 * Pause orchestrator
 */
async function pauseOrchestrator() {
    try {
        const response = await fetch(`${API_BASE}/api/pause`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('Orchestrator paused', 'success');
            await updateStatus();
        } else {
            showToast(data.message || 'Failed to pause', 'error');
        }
    } catch (error) {
        showToast('Error pausing orchestrator', 'error');
    }
}

/**
 * Resume orchestrator
 */
async function resumeOrchestrator() {
    try {
        const response = await fetch(`${API_BASE}/api/resume`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('Orchestrator resumed', 'success');
            await updateStatus();
        } else {
            showToast(data.message || 'Failed to resume', 'error');
        }
    } catch (error) {
        showToast('Error resuming orchestrator', 'error');
    }
}

/**
 * Add trader
 */
async function addTrader() {
    const addressInput = document.getElementById('trader-address');
    const address = addressInput.value.trim();
    
    if (!address) {
        showToast('Please enter a trader address', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/traders/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ address }),
        });
        const data = await response.json();
        
        if (data.success) {
            showToast('Trader added successfully', 'success');
            addressInput.value = '';
            await updateTradersList();
        } else {
            showToast(data.details?.rejection_reasons?.join(', ') || 'Failed to add trader', 'error');
        }
    } catch (error) {
        showToast('Error adding trader', 'error');
    }
}

/**
 * Remove trader
 */
async function removeTrader(address) {
    try {
        const response = await fetch(`${API_BASE}/api/traders/remove`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ address }),
        });
        const data = await response.json();
        
        if (data.success) {
            showToast('Trader removed', 'success');
            await updateTradersList();
        } else {
            showToast('Failed to remove trader', 'error');
        }
    } catch (error) {
        showToast('Error removing trader', 'error');
    }
}

// Helper Functions

/**
 * Format currency
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);
}

/**
 * Format percent
 */
function formatPercent(value) {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
}

/**
 * Format address
 */
function formatAddress(address) {
    if (!address) return '';
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
}

/**
 * Format duration
 */
function formatDuration(seconds) {
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
    return `${Math.floor(seconds / 86400)}d`;
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    elements.toastContainer.appendChild(toast);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Enter key for adding trader
document.getElementById('trader-address')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        addTrader();
    }
});
