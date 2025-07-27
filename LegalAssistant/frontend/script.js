// Store references to each page section for navigation
const pages = {
  chat: document.getElementById('page-chat'),
  resources: document.getElementById('page-resources'),
  settings: document.getElementById('page-settings'),
};

// Handle navigation between pages
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.target;

    // Hide all pages
    Object.values(pages).forEach(page => page.classList.add('hidden'));

    // Show selected page
    pages[target].classList.remove('hidden');
  });
});

// Handle sending a chat message
document.getElementById('sendBtn').addEventListener('click', async () => {
  const input = document.getElementById('userInput');
  const chatBox = document.getElementById('chat');
  const text = input.value.trim();

  if (!text) return;

  // Display user's message
  const userMsg = document.createElement('div');
  userMsg.textContent = `👤: ${text}`;
  chatBox.appendChild(userMsg);
  input.value = '';
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    // Send message to backend
    const response = await fetch('http://localhost:5000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })  // 💡 FIX: Use 'text' not 'message'
    });

    const data = await response.json();

    // Display bot's response
    if (data && data.response) {
      const botMsg = document.createElement('div');
      // Initialize an empty string
      let bot_response = "";

      // Append the answer with formatting
      bot_response += `\n🔎 Answer:\n${data.response.answer}`;

      // Append the sources
      // bot_response += `\n📄 Sources: ${data.response.sources}\n`;

      botMsg.textContent = `🤖: ${bot_response}`;
      chatBox.appendChild(botMsg);
      chatBox.scrollTop = chatBox.scrollHeight;
    } else {
      throw new Error("Invalid response format");
    }

  } catch (err) {
    console.error('Error:', err);
    const errorMsg = document.createElement('div');
    errorMsg.textContent = '❌ BOT: Failed to connect to server.';
    chatBox.appendChild(errorMsg);
  }
});

// Handle deleting all messages and data
document.getElementById('deleteBtn').addEventListener('click', () => {
  if (confirm("Delete all conversations and personal data?")) {
    document.getElementById('chat').innerHTML = '';
    alert("✅ Data deleted.");
  }
});
