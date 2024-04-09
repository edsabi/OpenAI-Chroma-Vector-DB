document.addEventListener("DOMContentLoaded", function() {
  document.getElementById("submitButton").addEventListener("click", submitPrompt);
  document.getElementById("clearButton").addEventListener("click", clearPromptAndResponse);
  document.getElementById("promptInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent default form submission behavior
      submitPrompt();
    }
  });
});

function submitPrompt() {
  var prompt = document.getElementById("promptInput").value;
  fetchResponse(prompt);
  // Clear the input field after submission
  document.getElementById("promptInput").value = "";
}

function fetchResponse(prompt) {
  fetch('/response', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ prompt: prompt })
  })
  .then(response => response.json())
  .then(data => {
    displayResponse(prompt, data.result);
  })
  .catch(error => console.error('Error:', error));
}

function displayResponse(prompt, response) {
  var responseDiv = document.getElementById("response");
  // Show the prompt along with the response
  responseDiv.innerHTML = "<strong>Prompt:</strong><br>" + prompt + "<br><br>" +
                          "<strong>Response:</strong><br>" + response;
}

function clearPromptAndResponse() {
  // Clear the input field and the response area
  document.getElementById("promptInput").value = "";
  document.getElementById("response").innerHTML = "";
}
