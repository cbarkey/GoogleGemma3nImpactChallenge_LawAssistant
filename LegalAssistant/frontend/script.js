const pages = {
  chat: document.getElementById('page-chat'),
  resources: document.getElementById('page-resources'),
  settings: document.getElementById('page-settings'),
};

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.target;
    Object.values(pages).forEach(page => page.classList.add('hidden'));
    pages[target].classList.remove('hidden');
  });
});

document.getElementById('sendBtn').addEventListener('click', () => {
  const input = document.getElementById('userInput');
  const chatBox = document.getElementById('chat');
  const text = input.value.trim();
  if (text) {
    const msg = document.createElement('div');
    msg.textContent = `👤: ${text}`;
    chatBox.appendChild(msg);
    input.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});

document.getElementById('deleteBtn').addEventListener('click', () => {
  if (confirm("Delete all conversations and personal data?")) {
    document.getElementById('chat').innerHTML = '';
    alert("Data deleted.");
  }
});
