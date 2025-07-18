document.getElementById('sendBtn').addEventListener('click', async () => {
    const inputElement = document.getElementById('userInput');
    const message = inputElement.value.trim();
    if (!message) return;

    appendMessage(message, 'user');
    inputElement.value = '';

    try {
        const response = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        appendMessage(data.response, 'bot');
    } catch (err) {
        console.error('Error:', err);
        appendMessage('Failed to connect to server.', 'bot');
    }
});

function appendMessage(message, sender) {
    const chatBox = document.getElementById('chat');
    const messageDiv = document.createElement('div');
    messageDiv.className = sender;
    messageDiv.innerText = message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
