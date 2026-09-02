const loginView = document.querySelector("#login-view");
const chatView = document.querySelector("#chat-view");
const loginForm = document.querySelector("#login-form");
const chatForm = document.querySelector("#chat-form");
const loginError = document.querySelector("#login-error");
const chatError = document.querySelector("#chat-error");
const messages = document.querySelector("#messages");
const question = document.querySelector("#question");
let token = sessionStorage.getItem("access_token");

function showChat() { loginView.classList.add("hidden"); chatView.classList.remove("hidden"); question.focus(); }
function showLogin() { chatView.classList.add("hidden"); loginView.classList.remove("hidden"); }
function addMessage(text, role) {
    const item = document.createElement("div");
    item.className = `message ${role}`;
    item.innerHTML = `<span class="avatar">${role === "assistant" ? "N" : "Y"}</span><div><p class="message-label">${role === "assistant" ? "NORTHSTAR" : "YOU"}</p><p></p></div>`;
    item.querySelector("p:last-child").textContent = text;
    messages.appendChild(item);
    item.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

loginForm.addEventListener("submit", async (event) => {
    event.preventDefault(); loginError.textContent = "";
    const body = Object.fromEntries(new FormData(loginForm));
    try {
        const response = await fetch("/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        if (!response.ok) throw new Error("Invalid username or password.");
        token = (await response.json()).access_token; sessionStorage.setItem("access_token", token); showChat();
    } catch (error) { loginError.textContent = error.message; }
});

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault(); chatError.textContent = "";
    const text = question.value.trim(); if (!text) return;
    addMessage(text, "user"); question.value = "";
    try {
        const response = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` }, body: JSON.stringify({ question: text, k: 4 }) });
        if (response.status === 401) { token = null; sessionStorage.removeItem("access_token"); showLogin(); throw new Error("Your session expired. Please sign in again."); }
        if (!response.ok) throw new Error("The assistant is temporarily unavailable.");
        addMessage((await response.json()).answer, "assistant");
    } catch (error) { chatError.textContent = error.message; }
});

document.querySelector("#logout-button").addEventListener("click", () => { token = null; sessionStorage.removeItem("access_token"); showLogin(); });
if (token) showChat();