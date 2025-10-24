// Lightweight frontend chatbot UI. Sends questions to /ask-rag and displays placeholder answers.
document.addEventListener('DOMContentLoaded', function(){
  const chatToggle = document.getElementById('chatToggle');
  const chatPanel = document.getElementById('chatPanel');
  const chatForm = document.getElementById('chatForm');
  const chatLog = document.getElementById('chatLog');
  const chatInput = document.getElementById('chatInput');

  if (chatToggle && chatPanel) {
    chatToggle.addEventListener('click', () => {
      const open = chatPanel.classList.toggle('open');
      chatPanel.setAttribute('aria-hidden', open ? 'false' : 'true');
      chatToggle.setAttribute('aria-pressed', open ? 'true' : 'false');
      if (open) chatInput.focus();
    });
  }

  function appendMessage(who, text) {
    const div = document.createElement('div');
    div.className = 'chat-msg ' + (who === 'user' ? 'chat-user' : 'chat-system');
    const time = document.createElement('div');
    time.className = 'chat-time';
    time.textContent = new Date().toLocaleTimeString();
    const txt = document.createElement('div');
    txt.className = 'chat-text';
    txt.textContent = text;
    div.appendChild(txt);
    div.appendChild(time);
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  if (chatForm) {
    chatForm.addEventListener('submit', async function(e){
      e.preventDefault();
      const q = chatInput.value.trim();
      if (!q) return;
      appendMessage('user', q);
      chatInput.value = '';
      appendMessage('system', 'Thinking...');

      try {
        // Get original text from hidden data attribute
        const resultData = document.getElementById('result-data');
        const originalText = resultData ? resultData.getAttribute('data-original-text') : '';
        
        const res = await fetch('/ask-rag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            question: q,
            text: originalText || ''
          })
        });
        const data = await res.json();
        // remove the 'Thinking...' placeholder (last system message)
        const placeholders = chatLog.querySelectorAll('.chat-system');
        if (placeholders.length) placeholders[placeholders.length-1].remove();
        if (data && data.answer) appendMessage('system', data.answer);
        else appendMessage('system', data?.error || 'No answer returned (placeholder).');
      } catch (err) {
        const placeholders = chatLog.querySelectorAll('.chat-system');
        if (placeholders.length) placeholders[placeholders.length-1].remove();
        appendMessage('system', 'Error contacting RAG endpoint. (placeholder)');
      }
    });
  }

  // add presets (quick questions)
  const presets = ['Which cures worked for fever?', 'Top cures for infections', 'Which cures failed?'];
  const presetRow = document.createElement('div');
  presetRow.style.display = 'flex';
  presetRow.style.gap = '6px';
  presetRow.style.padding = '8px';
  presets.forEach(p => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'btn btn-sm btn-outline-secondary'; b.textContent = p; b.style.flex = '1';
    b.addEventListener('click', function(){ chatInput.value = p; chatForm.dispatchEvent(new Event('submit', {cancelable:true})); });
    presetRow.appendChild(b);
  });
  if (chatPanel && chatPanel.querySelector('.chat-log')) chatPanel.insertBefore(presetRow, chatPanel.querySelector('.chat-log'));

  // close on Escape
  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape' && chatPanel && chatPanel.classList.contains('open')) {
      chatPanel.classList.remove('open');
      chatPanel.setAttribute('aria-hidden','true');
    }
  });
});
