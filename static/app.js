/**
 * TripPlanner AI — Client-Side Application
 * Handles chat interactions, session/thread-based history, markdown rendering, tool traces, and budget status.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatContainer = document.getElementById('chatContainer');
    const messagesFeed = document.getElementById('messagesFeed');
    const heroWelcome = document.getElementById('heroWelcome');
    const chatForm = document.getElementById('chatForm');
    const promptInput = document.getElementById('promptInput');
    const sendBtn = document.getElementById('sendBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const historyList = document.getElementById('historyList');
    const openSidebarBtn = document.getElementById('openSidebarBtn');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const sidebar = document.getElementById('sidebar');

    // Load saved conversation threads
    let threads = JSON.parse(localStorage.getItem('tripPlannerThreads') || '[]');
    let currentThreadId = null;

    // Save threads to LocalStorage
    function saveThreads() {
        localStorage.setItem('tripPlannerThreads', JSON.stringify(threads));
    }

    // Generate a clean conversation title from the first prompt
    function generateThreadTitle(prompt) {
        const clean = prompt.trim().replace(/^plan a \d+-day trip (?:from [^ ]+ )?to /i, '');
        if (clean.length > 28) {
            return clean.slice(0, 26) + '...';
        }
        return clean.charAt(0).toUpperCase() + clean.slice(1);
    }

    // Render sidebar conversation list
    function renderHistory() {
        historyList.innerHTML = '';
        if (threads.length === 0) {
            const emptyEl = document.createElement('div');
            emptyEl.style.fontSize = '0.75rem';
            emptyEl.style.color = 'var(--text-muted)';
            emptyEl.style.padding = '8px 12px';
            emptyEl.textContent = 'No past trip plans yet';
            historyList.appendChild(emptyEl);
            return;
        }

        threads.slice().reverse().forEach((thread) => {
            const el = document.createElement('div');
            el.className = `history-item ${thread.id === currentThreadId ? 'active' : ''}`;
            
            el.innerHTML = `
                <div class="history-left">
                    <i class="fa-regular fa-compass thread-icon"></i>
                    <span class="history-title" title="${escapeHtml(thread.title)}">${escapeHtml(thread.title)}</span>
                </div>
                <button class="delete-thread-btn" title="Delete conversation">
                    <i class="fa-solid fa-trash"></i>
                </button>
            `;

            // Switch to thread on click
            el.addEventListener('click', (e) => {
                if (e.target.closest('.delete-thread-btn')) return;
                loadThread(thread.id);
            });

            // Delete thread handler
            const deleteBtn = el.querySelector('.delete-thread-btn');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteThread(thread.id);
            });

            historyList.appendChild(el);
        });
    }

    // Load an existing thread onto the screen
    function loadThread(threadId) {
        const thread = threads.find(t => t.id === threadId);
        if (!thread) return;

        currentThreadId = threadId;
        renderHistory();

        // Clear current feed and hide hero
        messagesFeed.innerHTML = '';
        heroWelcome.style.display = 'none';

        // Render all messages in this thread
        thread.messages.forEach(msg => {
            appendMessage(msg.role, msg.content, msg.traces, msg.budget, msg.elapsed_seconds, false);
        });

        scrollToBottom();
        promptInput.focus();

        // Close sidebar on mobile if open
        sidebar.classList.remove('open');
    }

    // Delete a thread
    function deleteThread(threadId) {
        threads = threads.filter(t => t.id !== threadId);
        saveThreads();

        if (currentThreadId === threadId) {
            startNewChat();
        } else {
            renderHistory();
        }
    }

    // Start a New Conversation
    function startNewChat() {
        currentThreadId = null;
        messagesFeed.innerHTML = '';
        heroWelcome.style.display = 'flex';
        promptInput.value = '';
        promptInput.style.height = 'auto';
        updateSendButtonState();
        renderHistory();
        promptInput.focus();
    }

    newChatBtn.addEventListener('click', startNewChat);

    // Auto-resize input textarea
    promptInput.addEventListener('input', () => {
        promptInput.style.height = 'auto';
        promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
        updateSendButtonState();
    });

    function updateSendButtonState() {
        sendBtn.disabled = !promptInput.value.trim();
    }

    // Sidebar Mobile Toggle
    if (openSidebarBtn) {
        openSidebarBtn.addEventListener('click', () => sidebar.classList.add('open'));
    }
    if (closeSidebarBtn) {
        closeSidebarBtn.addEventListener('click', () => sidebar.classList.remove('open'));
    }

    // Suggestion Cards click
    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            if (prompt) {
                promptInput.value = prompt;
                sendMessage(prompt);
            }
        });
    });

    // Form Submit
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = promptInput.value.trim();
        if (text) {
            sendMessage(text);
        }
    });

    // Enter to submit, Shift+Enter for newline
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const text = promptInput.value.trim();
            if (text) {
                sendMessage(text);
            }
        }
    });

    // Send Message Workflow
    async function sendMessage(userText) {
        // Hide Hero
        heroWelcome.style.display = 'none';
        
        // Reset Input
        promptInput.value = '';
        promptInput.style.height = 'auto';
        updateSendButtonState();

        // If this is a new conversation thread, initialize it
        if (!currentThreadId) {
            currentThreadId = 'thread_' + Date.now();
            const newThread = {
                id: currentThreadId,
                title: generateThreadTitle(userText),
                messages: [],
                createdAt: new Date().toISOString()
            };
            threads.push(newThread);
            saveThreads();
            renderHistory();
        }

        const currentThread = threads.find(t => t.id === currentThreadId);

        // 1. Append User Message
        appendMessage('user', userText);
        if (currentThread) {
            currentThread.messages.push({ role: 'user', content: userText });
            saveThreads();
        }

        // 2. Show Typing / Reasoning Indicator
        const loadingRow = appendTypingIndicator();
        scrollToBottom();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userText, thread_id: currentThreadId })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            
            // Remove loading indicator
            loadingRow.remove();

            // 3. Append Assistant Response with Tool Traces & Budget Info
            appendMessage('assistant', data.reply, data.traces, data.budget, data.elapsed_seconds);

            // Save assistant message to current thread
            if (currentThread) {
                currentThread.messages.push({
                    role: 'assistant',
                    content: data.reply,
                    traces: data.traces,
                    budget: data.budget,
                    elapsed_seconds: data.elapsed_seconds
                });
                saveThreads();
            }

        } catch (err) {
            loadingRow.remove();
            const errorMsg = `**Error processing request**: ${err.message}\n\nPlease check your LLM API connection or retry your request.`;
            appendMessage('assistant', errorMsg);
            if (currentThread) {
                currentThread.messages.push({ role: 'assistant', content: errorMsg });
                saveThreads();
            }
        }



        scrollToBottom();
    }

    function appendMessage(role, content, traces = [], budget = null, elapsedSec = null, animate = true) {
        const row = document.createElement('div');
        row.className = `message-row ${role}`;
        if (!animate) row.style.animation = 'none';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-compass"></i>';

        const wrapper = document.createElement('div');
        wrapper.className = 'message-content-wrapper';

        // Assistant Special Panels: Tool Traces & Budget Pill
        if (role === 'assistant') {
            // 1. Tool Execution Trace Accordion (if tools were called)
            if (traces && traces.length > 0) {
                const traceBox = document.createElement('div');
                traceBox.className = 'tool-trace-box';

                const summaryHeader = document.createElement('div');
                summaryHeader.className = 'trace-summary-header';
                summaryHeader.innerHTML = `
                    <div class="trace-title">
                        <i class="fa-solid fa-bolt"></i>
                        <span>Executed <strong>${traces.length}</strong> Live Tool${traces.length > 1 ? 's' : ''} ${elapsedSec ? `(${elapsedSec}s)` : ''}</span>
                    </div>
                    <i class="fa-solid fa-chevron-down trace-toggle-icon"></i>
                `;

                const detailsContainer = document.createElement('div');
                detailsContainer.className = 'trace-details';

                traces.forEach(t => {
                    const item = document.createElement('div');
                    item.className = 'trace-item';
                    item.innerHTML = `
                        <div class="trace-item-header">
                            <span class="trace-tool-name"><i class="fa-solid fa-gear"></i> ${escapeHtml(t.name)}</span>
                            <span class="trace-time">${escapeHtml(t.timestamp || '')}</span>
                        </div>
                        <div class="trace-json"><strong>Arguments:</strong> ${escapeHtml(JSON.stringify(t.args, null, 2))}\n\n<strong>Result:</strong> ${escapeHtml(typeof t.result === 'string' ? t.result : JSON.stringify(t.result, null, 2))}</div>
                    `;
                    detailsContainer.appendChild(item);
                });

                summaryHeader.addEventListener('click', () => {
                    traceBox.classList.toggle('expanded');
                });

                traceBox.appendChild(summaryHeader);
                traceBox.appendChild(detailsContainer);
                wrapper.appendChild(traceBox);
            }

            // 2. Budget Status Pill (if budget was parsed)
            if (budget && budget.user_budget !== null && budget.user_budget !== undefined) {
                const pill = document.createElement('div');
                const totalCost = budget.total_cost || 0;
                const userBudget = budget.user_budget || 0;
                const isWithin = totalCost <= userBudget;

                pill.className = `budget-status-pill ${isWithin ? 'within-budget' : 'over-budget'}`;
                pill.innerHTML = `
                    <i class="fa-solid ${isWithin ? 'fa-circle-check' : 'fa-triangle-exclamation'}"></i>
                    <span>Budget: $${userBudget.toFixed(2)} | Flight + Hotel: $${totalCost.toFixed(2)} (${isWithin ? `Saved $${(userBudget - totalCost).toFixed(2)}` : `Over $${(totalCost - userBudget).toFixed(2)}`})</span>
                `;
                wrapper.appendChild(pill);
            }
        }

        // Message Bubble with Markdown rendering
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = renderMarkdown(content);
        wrapper.appendChild(bubble);

        if (role === 'user') {
            row.appendChild(wrapper);
            row.appendChild(avatar);
        } else {
            row.appendChild(avatar);
            row.appendChild(wrapper);
        }

        messagesFeed.appendChild(row);
        return row;
    }

    function appendTypingIndicator() {
        const row = document.createElement('div');
        row.className = 'message-row assistant';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fa-solid fa-compass"></i>';

        const wrapper = document.createElement('div');
        wrapper.className = 'message-content-wrapper';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble typing-bubble';
        bubble.innerHTML = `
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 8px;">LangGraph agent researching live flights, hotels & weather...</span>
        `;

        wrapper.appendChild(bubble);
        row.appendChild(avatar);
        row.appendChild(wrapper);
        messagesFeed.appendChild(row);
        return row;
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * Client-side lightweight Markdown + Table Renderer
     */
    function renderMarkdown(md) {
        if (!md) return '';
        
        let html = md;

        // Escape raw HTML tags (except ones we intentionally generate)
        html = html.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // Headers
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

        // Horizontal Rules
        html = html.replace(/^---$/gim, '<hr>');

        // Bold & Italic
        html = html.replace(/\*\*\*(.*?)\*\*\*/gim, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

        // Blockquotes
        html = html.replace(/^&gt; (.*$)/gim, '<blockquote>$1</blockquote>');

        // Markdown Tables
        html = html.replace(/((?:\|[^\n]+\|\r?\n)+)/g, (match) => {
            const lines = match.trim().split('\n').filter(l => l.trim().startsWith('|'));
            if (lines.length < 2) return match;

            let tableHtml = '<table>';
            let isHeader = true;

            lines.forEach((line, index) => {
                if (line.includes('---')) {
                    isHeader = false;
                    return;
                }
                const cells = line.split('|').slice(1, -1).map(c => c.trim());
                if (isHeader && index === 0) {
                    tableHtml += '<thead><tr>' + cells.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
                } else {
                    tableHtml += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
                }
            });

            tableHtml += '</tbody></table>';
            return tableHtml;
        });

        // Unordered Lists
        html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gims, '<ul>$1</ul>');

        // Clean up redundant nested <ul>
        html = html.replace(/<\/ul>\s*<ul>/gim, '');

        // Paragraph linebreaks
        html = html.replace(/\n\n/gim, '<br><br>');

        return html;
    }

    // Initialize sidebar history
    renderHistory();
});
