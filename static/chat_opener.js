// Open Q&A Bot (chat widget)
function openChatBot() {
  const chatPanel = document.getElementById('chatPanel');
  const chatToggle = document.getElementById('chatToggle');
  const chatInput = document.getElementById('chatInput');
  
  if (chatPanel && chatToggle) {
    chatPanel.classList.add('open');
    chatPanel.setAttribute('aria-hidden', 'false');
    chatToggle.setAttribute('aria-pressed', 'true');
    
    // Focus on input for immediate typing
    if (chatInput) {
      setTimeout(function() { chatInput.focus(); }, 300);
    }
    
    // Add welcome message if chat is empty
    const chatLog = document.getElementById('chatLog');
    if (chatLog && chatLog.children.length === 0) {
      const welcomeMsg = document.createElement('div');
      welcomeMsg.className = 'chat-msg chat-system';
      const chatText = document.createElement('div');
      chatText.className = 'chat-text';
      chatText.innerHTML = '👋 Welcome to the Healer Bot! Ask me anything about the treatments in your scrolls. Try questions like:<br><br>• "Which cures worked for fever?"<br>• "What did Healer Anna use?"<br>• "Show me failed treatments"';
      const chatTime = document.createElement('div');
      chatTime.className = 'chat-time';
      chatTime.textContent = new Date().toLocaleTimeString();
      welcomeMsg.appendChild(chatText);
      welcomeMsg.appendChild(chatTime);
      chatLog.appendChild(welcomeMsg);
    }
  }
}
