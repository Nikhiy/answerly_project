document.addEventListener("DOMContentLoaded", async () => {
  const currentUrlDiv = document.getElementById("current-url");
  const statusDiv = document.getElementById("status");
  const answerDiv = document.getElementById("answer");
  const questionInput = document.getElementById("question");
  const askBtn = document.getElementById("askBtn");
  const loadingIndicator = document.querySelector('.loading'); // The loading indicator div

  // Get active tab URL and display it
  try {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    currentUrlDiv.textContent = tab.url || "URL not available";
  } catch {
    currentUrlDiv.textContent = "Failed to get current URL";
  }

  askBtn.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    answerDiv.textContent = "";
    statusDiv.textContent = "";

    if (!question) {
      statusDiv.textContent = "Please enter a question.";
      statusDiv.style.opacity = 1; // Ensure status is visible immediately
      return;
    }

    statusDiv.textContent = "Loading...";
    statusDiv.style.opacity = 1; // Ensure status is visible immediately

    // Show loading indicator
    loadingIndicator.style.display = "block";
    loadingIndicator.classList.add("loading-indicator"); // Start the pulsing animation

    try {
      // Use the URL shown in UI in case tab changed
      const currentUrl = currentUrlDiv.textContent;

      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, url: currentUrl }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();

      // Fade in the answer after receiving the response
      answerDiv.textContent = `Answer:\n${data.answer}\n\nSource:\n${data.source}`;
      answerDiv.style.opacity = 1; // Ensure the answer is visible immediately

      // Hide loading indicator
      loadingIndicator.style.display = "none";

      statusDiv.textContent = "Success!";
      statusDiv.style.opacity = 1; // Ensure status is visible
    } catch (error) {
      // Hide loading indicator on error
      loadingIndicator.style.display = "none";

      statusDiv.textContent = `Error: ${error.message}`;
      statusDiv.style.opacity = 1; // Ensure status is visible
    }
  });
});