// DriveToQuizlet Dashboard - Enhanced JavaScript

class DashboardController {
    constructor() {
        this.state = {
            pendingFiles: [],
            lessons: [],
            isLoading: false,
            lastUpdate: null,
            autoRefresh: true
        };
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.initializeTooltips();
        await this.loadInitialData();
        this.startAutoRefresh();
        this.addKeyboardShortcuts();
    }

    setupEventListeners() {
        // Enhanced button listeners with loading states
        document.getElementById('scanDriveBtn')?.addEventListener('click', () => this.scanDrive());
        document.getElementById('approveAllBtn')?.addEventListener('click', () => this.approveAllFiles());
        document.getElementById('refreshLessonsBtn')?.addEventListener('click', () => this.loadLessons());
        
        // Tab switching with history API
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Real-time search functionality
        this.addSearchFunctionality();
        
        // Settings toggle
        this.addSettingsPanel();
    }

    addSearchFunctionality() {
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = 'Search lessons...';
        searchInput.className = 'px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent';
        
        const searchContainer = document.querySelector('#lessons-tab .flex.justify-between');
        if (searchContainer) {
            searchContainer.insertBefore(searchInput, searchContainer.lastElementChild);
            
            searchInput.addEventListener('input', (e) => {
                this.filterLessons(e.target.value);
            });
        }
    }

    filterLessons(query) {
        const lessons = document.querySelectorAll('#lessonsList > div');
        lessons.forEach(lesson => {
            const lessonName = lesson.querySelector('h3')?.textContent.toLowerCase() || '';
            const shouldShow = lessonName.includes(query.toLowerCase());
            lesson.style.display = shouldShow ? 'block' : 'none';
        });
    }

    addSettingsPanel() {
        const settingsBtn = document.createElement('button');
        settingsBtn.innerHTML = '<i class="fas fa-cog"></i>';
        settingsBtn.className = 'bg-gray-600 text-white px-3 py-2 rounded-lg hover:bg-gray-700 transition-colors';
        settingsBtn.title = 'Settings';
        
        const header = document.querySelector('header .flex.items-center.space-x-4');
        if (header) {
            header.appendChild(settingsBtn);
            settingsBtn.addEventListener('click', () => this.showSettingsModal());
        }
    }

    showSettingsModal() {
        const modal = this.createModal('Settings', this.getSettingsContent());
        document.body.appendChild(modal);
        
        // Auto refresh toggle
        modal.querySelector('#autoRefreshToggle')?.addEventListener('change', (e) => {
            this.state.autoRefresh = e.target.checked;
            if (this.state.autoRefresh) {
                this.startAutoRefresh();
            } else {
                this.stopAutoRefresh();
            }
        });
    }

    getSettingsContent() {
        return `
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <label for="autoRefreshToggle" class="text-sm font-medium text-gray-700">Auto Refresh</label>
                    <input type="checkbox" id="autoRefreshToggle" ${this.state.autoRefresh ? 'checked' : ''} 
                           class="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded">
                </div>
                <div class="flex items-center justify-between">
                    <label class="text-sm font-medium text-gray-700">Theme</label>
                    <select class="px-3 py-1 border border-gray-300 rounded">
                        <option value="light">Light</option>
                        <option value="dark">Dark</option>
                        <option value="auto">Auto</option>
                    </select>
                </div>
            </div>
        `;
    }

    createModal(title, content) {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 z-50 flex items-center justify-center modal-backdrop';
        modal.innerHTML = `
            <div class="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 modal-content">
                <div class="flex items-center justify-between p-6 border-b border-gray-200">
                    <h3 class="text-lg font-semibold text-gray-900">${title}</h3>
                    <button class="text-gray-400 hover:text-gray-600 modal-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="p-6">
                    ${content}
                </div>
                <div class="flex justify-end p-6 border-t border-gray-200">
                    <button class="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors modal-close">
                        Close
                    </button>
                </div>
            </div>
        `;

        // Close modal functionality
        modal.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', () => modal.remove());
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });

        return modal;
    }

    addKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + R: Refresh
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                this.loadInitialData();
            }
            
            // Ctrl/Cmd + S: Scan Drive
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.scanDrive();
            }
            
            // Tab navigation: 1, 2, 3
            if (e.key >= '1' && e.key <= '3') {
                const tabs = ['pending', 'lessons', 'activity'];
                this.switchTab(tabs[parseInt(e.key) - 1]);
            }
        });
    }

    switchTab(tabName) {
        // Update URL without page reload
        window.history.pushState({tab: tabName}, '', `#${tabName}`);
        
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            const isActive = btn.dataset.tab === tabName;
            btn.classList.toggle('active', isActive);
            btn.classList.toggle('border-purple-500', isActive);
            btn.classList.toggle('text-purple-600', isActive);
            btn.classList.toggle('border-transparent', !isActive);
            btn.classList.toggle('text-gray-500', !isActive);
        });
        
        // Show tab content with animation
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        
        const activeTab = document.getElementById(`${tabName}-tab`);
        if (activeTab) {
            activeTab.classList.remove('hidden');
            activeTab.classList.add('slide-in');
        }
    }

    async loadInitialData() {
        this.showLoading(true);
        try {
            await Promise.all([
                this.loadPendingFiles(),
                this.loadLessons()
            ]);
            this.state.lastUpdate = new Date();
            this.updateLastRefreshTime();
        } catch (error) {
            this.showNotification('Error loading data', 'error');
            console.error('Error loading initial data:', error);
        } finally {
            this.showLoading(false);
        }
    }

    async loadPendingFiles() {
        try {
            const response = await fetch('/api/pending-files');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.state.pendingFiles = data.files;
            this.renderPendingFiles();
            this.updateStats();
        } catch (error) {
            console.error('Error loading pending files:', error);
            this.showNotification('Error loading pending files', 'error');
        }
    }

    renderPendingFiles() {
        const container = document.getElementById('pendingFilesList');
        if (!container) return;
        
        if (this.state.pendingFiles.length === 0) {
            container.innerHTML = this.getEmptyState('check-circle', 'No files pending approval');
            return;
        }

        container.innerHTML = this.state.pendingFiles.map((file, index) => `
            <div class="border border-gray-200 rounded-lg p-4 card-hover slide-in" style="animation-delay: ${index * 0.1}s">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-4">
                        <div class="p-2 rounded-full ${this.getFileTypeClass(file.type)}">
                            <i class="fas ${this.getFileTypeIcon(file.type)} text-white"></i>
                        </div>
                        <div>
                            <h3 class="font-medium text-gray-900">${file.name}</h3>
                            <p class="text-sm text-gray-600">Lesson: ${file.lesson}</p>
                            <p class="text-xs text-gray-500">
                                <i class="fas fa-clock mr-1"></i>
                                ${this.formatDate(file.detected_date)}
                            </p>
                        </div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="status-badge status-pending status-indicator">Pending</span>
                        <button onclick="dashboard.approveFile('${file.id}')" 
                                class="bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700 transition-colors focus-ring">
                            <i class="fas fa-check mr-1"></i>Approve
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    }

    async loadLessons() {
        try {
            const response = await fetch('/api/lessons');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.state.lessons = data.lessons;
            this.renderLessons();
            this.updateStats();
        } catch (error) {
            console.error('Error loading lessons:', error);
            this.showNotification('Error loading lessons', 'error');
        }
    }

    renderLessons() {
        const container = document.getElementById('lessonsList');
        if (!container) return;
        
        if (this.state.lessons.length === 0) {
            container.innerHTML = this.getEmptyState('graduation-cap', 'No lessons found');
            return;
        }

        container.innerHTML = this.state.lessons.map((lesson, index) => `
            <div class="border border-gray-200 rounded-lg p-6 card-hover slide-in progress-card" style="animation-delay: ${index * 0.1}s">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-gray-900">${lesson.name}</h3>
                    <span class="status-badge status-indicator ${lesson.has_output ? 'status-complete' : 'status-pending'}">
                        ${lesson.has_output ? 'Ready' : 'In Progress'}
                    </span>
                </div>
                
                <div class="space-y-2 mb-4">
                    ${this.renderLessonFeatures(lesson)}
                </div>
                
                <div class="flex space-x-2">
                    ${lesson.has_output ? 
                        `<button onclick="dashboard.downloadLesson('${lesson.name}')" 
                                 class="flex-1 btn-gradient text-white px-3 py-2 rounded text-sm focus-ring">
                            <i class="fas fa-download mr-1"></i>Download TSV
                         </button>` :
                        `<button onclick="dashboard.processLesson('${lesson.name}')" 
                                 class="flex-1 bg-blue-600 text-white px-3 py-2 rounded text-sm hover:bg-blue-700 transition-colors focus-ring">
                            <i class="fas fa-cog mr-1"></i>Process
                         </button>`
                    }
                </div>
                
                <div class="mt-3 text-xs text-gray-500">
                    <i class="fas fa-clock mr-1"></i>
                    Last modified: ${this.formatDate(lesson.last_modified)}
                </div>
            </div>
        `).join('');
    }

    renderLessonFeatures(lesson) {
        const features = [
            { key: 'has_presentations', icon: 'presentation', label: 'Presentations' },
            { key: 'has_notes', icon: 'sticky-note', label: 'Notes' },
            { key: 'has_audio', icon: 'microphone', label: 'Audio' }
        ];
        
        return features.map(feature => `
            <div class="flex items-center text-sm">
                <i class="fas fa-${feature.icon} mr-2 ${lesson[feature.key] ? 'text-green-600' : 'text-gray-400'}"></i>
                <span class="${lesson[feature.key] ? 'text-gray-900' : 'text-gray-500'}">${feature.label}</span>
                ${lesson[feature.key] ? '<i class="fas fa-check text-green-500 ml-auto"></i>' : ''}
            </div>
        `).join('');
    }

    getEmptyState(icon, message) {
        return `
            <div class="text-center py-8">
                <i class="fas fa-${icon} text-4xl text-gray-400 mb-4"></i>
                <p class="text-gray-600">${message}</p>
            </div>
        `;
    }

    getFileTypeClass(type) {
        const classes = {
            'presentation': 'file-presentation',
            'notes': 'file-notes',
            'audio': 'file-audio'
        };
        return classes[type] || 'bg-gray-500';
    }

    getFileTypeIcon(type) {
        const icons = {
            'presentation': 'fa-presentation',
            'notes': 'fa-sticky-note',
            'audio': 'fa-microphone'
        };
        return icons[type] || 'fa-file';
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        
        if (diffHours < 1) return 'Just now';
        if (diffHours < 24) return `${diffHours}h ago`;
        
        return date.toLocaleDateString();
    }

    updateStats() {
        const stats = {
            pendingCount: this.state.pendingFiles.length,
            lessonCount: this.state.lessons.length,
            readyCount: this.state.lessons.filter(l => l.has_output).length
        };
        
        Object.entries(stats).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                this.animateCounter(element, parseInt(element.textContent) || 0, value);
            }
        });
    }

    animateCounter(element, from, to) {
        const duration = 500;
        const start = Date.now();
        
        const update = () => {
            const progress = Math.min((Date.now() - start) / duration, 1);
            const current = Math.floor(from + (to - from) * progress);
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        };
        
        update();
    }

    updateLastRefreshTime() {
        const lastScan = document.getElementById('lastScan');
        if (lastScan && this.state.lastUpdate) {
            lastScan.textContent = this.formatDate(this.state.lastUpdate.toISOString());
        }
    }

    // API Methods
    async scanDrive() {
        this.showLoading(true, 'Scanning Google Drive...');
        try {
            const response = await fetch('/api/scan-drive', { method: 'POST' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.showNotification(`Scan complete: ${data.results?.new_files || 0} new files found`, 'success');
            await this.loadPendingFiles();
        } catch (error) {
            this.showNotification('Error scanning drive', 'error');
            console.error('Scan error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    async approveFile(fileId) {
        try {
            const response = await fetch(`/api/approve-file/${fileId}`, { method: 'POST' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.showNotification(data.message, 'success');
            
            // Remove file from pending list with animation
            const fileElement = document.querySelector(`[onclick*="${fileId}"]`)?.closest('.border');
            if (fileElement) {
                fileElement.style.transform = 'translateX(100%)';
                fileElement.style.opacity = '0';
                setTimeout(() => this.loadPendingFiles(), 300);
            }
            
            await this.loadLessons();
        } catch (error) {
            this.showNotification('Error approving file', 'error');
            console.error('Approve error:', error);
        }
    }

    async approveAllFiles() {
        if (this.state.pendingFiles.length === 0) return;
        
        this.showLoading(true, 'Approving all files...');
        try {
            const promises = this.state.pendingFiles.map(file => 
                fetch(`/api/approve-file/${file.id}`, { method: 'POST' })
            );
            
            await Promise.all(promises);
            this.showNotification('All files approved successfully', 'success');
            await this.loadInitialData();
        } catch (error) {
            this.showNotification('Error approving files', 'error');
            console.error('Approve all error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    async processLesson(lessonName) {
        this.showLoading(true, `Processing lesson: ${lessonName}...`);
        try {
            const response = await fetch(`/api/process-lesson/${lessonName}`, { method: 'POST' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.showNotification(data.message, data.status === 'success' ? 'success' : 'error');
            await this.loadLessons();
        } catch (error) {
            this.showNotification('Error processing lesson', 'error');
            console.error('Process error:', error);
        } finally {
            this.showLoading(false);
        }
    }

    downloadLesson(lessonName) {
        window.open(`/api/download/${lessonName}`, '_blank');
        this.showNotification(`Download started for ${lessonName}`, 'info');
    }

    // UI Helper Methods
    showLoading(show, message = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const messageEl = overlay?.querySelector('span');
        
        if (overlay) {
            overlay.classList.toggle('hidden', !show);
            if (messageEl) messageEl.textContent = message;
        }
    }

    showNotification(message, type = 'success') {
        const toast = document.getElementById('notificationToast');
        const messageEl = document.getElementById('notificationMessage');
        const icon = toast?.querySelector('i');
        
        if (!toast || !messageEl) return;
        
        // Update content
        messageEl.textContent = message;
        
        // Update styling based on type
        const styles = {
            success: { bg: 'bg-green-600', icon: 'fa-check' },
            error: { bg: 'bg-red-600', icon: 'fa-exclamation-triangle' },
            info: { bg: 'bg-blue-600', icon: 'fa-info-circle' },
            warning: { bg: 'bg-yellow-600', icon: 'fa-exclamation-triangle' }
        };
        
        const style = styles[type] || styles.success;
        toast.className = `fixed top-4 right-4 ${style.bg} text-white px-6 py-3 rounded-lg shadow-lg transform translate-x-full transition-transform duration-300 z-50`;
        
        if (icon) {
            icon.className = `fas ${style.icon}`;
        }
        
        // Show and auto-hide
        toast.classList.remove('translate-x-full');
        setTimeout(() => toast.classList.add('translate-x-full'), 4000);
    }

    initializeTooltips() {
        // Simple tooltip implementation
        document.querySelectorAll('[title]').forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                const tooltip = document.createElement('div');
                tooltip.className = 'absolute bg-gray-800 text-white px-2 py-1 rounded text-xs z-50';
                tooltip.textContent = e.target.title;
                tooltip.style.top = e.target.offsetTop - 30 + 'px';
                tooltip.style.left = e.target.offsetLeft + 'px';
                e.target.parentNode.appendChild(tooltip);
                e.target.removeAttribute('title');
                e.target.dataset.originalTitle = tooltip.textContent;
            });
        });
    }

    startAutoRefresh() {
        this.stopAutoRefresh();
        if (this.state.autoRefresh) {
            this.refreshInterval = setInterval(() => {
                this.loadInitialData();
            }, 30000);
        }
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }
}

// Initialize dashboard when DOM is loaded
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new DashboardController();
});

// Handle browser back/forward buttons
window.addEventListener('popstate', (e) => {
    if (e.state?.tab) {
        dashboard.switchTab(e.state.tab);
    }
});