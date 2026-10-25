(() => {
  const form = document.querySelector("#chat-form");
  const input = document.querySelector("#question-input");
  const list = document.querySelector("#message-list");
  const welcome = document.querySelector("#welcome-state");
  const sendButton = document.querySelector("#send-button");
  const modalBackdrop = document.querySelector("#modal-backdrop");
  const modalTitle = document.querySelector("#modal-title");
  const modalContent = document.querySelector("#modal-content");
  const sidebar = document.querySelector("#sidebar");
  const mobileScrim = document.querySelector("#mobile-scrim");
  const historyKey = "novashop-support-history-v1";
  let messages = [];
  let isLoading = false;
  let sessionId = newSessionId();
  let history = readHistory();

  function newSessionId() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function readHistory() {
    try {
      const stored = JSON.parse(localStorage.getItem(historyKey) || "[]");
      return Array.isArray(stored) ? stored : [];
    } catch {
      return [];
    }
  }

  function saveHistory() {
    if (!messages.length) return;
    const firstQuestion = messages.find((message) => message.role === "user")?.text || "New conversation";
    const session = {
      id: sessionId,
      title: firstQuestion.length > 58 ? `${firstQuestion.slice(0, 55)}…` : firstQuestion,
      updatedAt: new Date().toISOString(),
      messages,
    };
    history = [session, ...history.filter((item) => item.id !== sessionId)].slice(0, 30);
    try { localStorage.setItem(historyKey, JSON.stringify(history)); } catch { /* Private browsing may disable local storage. */ }
  }

  function addText(parent, tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text;
    parent.append(element);
    return element;
  }

  function renderMessages() {
    welcome.hidden = messages.length > 0;
    list.replaceChildren();
    for (const message of messages) {
      const row = document.createElement("article");
      row.className = `message ${message.role}`;
      const avatar = addText(row, "div", "message-avatar", message.role === "user" ? "Y" : "✦");
      avatar.setAttribute("aria-hidden", "true");
      const body = document.createElement("div");
      body.className = "message-body";
      const meta = document.createElement("div");
      meta.className = "message-meta";
      addText(meta, "strong", "", message.role === "user" ? "You" : "NovaShop AI");
      addText(meta, "span", "", message.time || "Just now");
      body.append(meta);

      if (message.role === "assistant") {
        const statusText = message.status === "answered" ? "✓ Answered"
          : message.status === "error" ? "! Couldn't connect"
            : "ⓘ Insufficient context";
        const statusClass = message.status === "answered" ? "message-status"
          : `message-status ${message.status === "error" ? "error" : "insufficient"}`;
        addText(body, "span", statusClass, statusText);
      }

      addText(body, "div", "message-bubble", message.text);
      if (message.role === "assistant" && message.sources?.length) {
        const sourceList = document.createElement("div");
        sourceList.className = "source-list";
        for (const source of message.sources) {
          const chip = document.createElement("div");
          chip.className = "source-chip";
          addText(chip, "span", "source-chip-icon", "▤");
          const label = source.section
            ? `${source.source || "Policy document"} · ${source.section}`
            : (source.source || "Policy document");
          addText(chip, "span", "", label);
          sourceList.append(chip);
        }
        body.append(sourceList);
      }
      row.append(body);
      list.append(row);
    }
    if (isLoading) {
      const row = document.createElement("article");
      row.className = "message assistant loading-message";
      addText(row, "div", "message-avatar", "✦");
      const body = document.createElement("div");
      body.className = "message-body";
      addText(body, "div", "message-meta", "NovaShop AI is checking the policies…");
      const bubble = document.createElement("div");
      bubble.className = "message-bubble typing-bubble";
      bubble.setAttribute("aria-label", "Assistant is typing");
      for (let index = 0; index < 3; index += 1) bubble.append(document.createElement("i"));
      body.append(bubble);
      row.append(body);
      list.append(row);
    }
    requestAnimationFrame(() => {
      const scroll = document.querySelector("#conversation-scroll");
      scroll.scrollTop = scroll.scrollHeight;
    });
  }

  function timestamp() {
    return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date());
  }

  async function submitQuestion(question) {
    const cleaned = question.trim();
    if (!cleaned || isLoading) return;
    messages.push({ role: "user", text: cleaned, time: timestamp() });
    isLoading = true;
    input.value = "";
    resizeInput();
    sendButton.disabled = true;
    renderMessages();
    saveHistory();
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ question: cleaned }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The support service is unavailable. Please try again.");
      messages.push({
        role: "assistant",
        text: payload.answer || "I couldn't find enough information in the available policy documents to answer that question.",
        status: payload.status === "answered" ? "answered" : "insufficient_context",
        sources: Array.isArray(payload.sources) ? payload.sources : [],
        time: timestamp(),
      });
    } catch (error) {
      messages.push({
        role: "assistant",
        text: error.message || "Something went wrong. Please try again.",
        status: "error",
        sources: [],
        time: timestamp(),
      });
    } finally {
      isLoading = false;
      sendButton.disabled = false;
      renderMessages();
      saveHistory();
      input.focus();
    }
  }

  function resizeInput() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
  }

  function openModal(title, kicker, contentBuilder) {
    modalTitle.textContent = title;
    document.querySelector("#modal-kicker").textContent = kicker;
    modalContent.replaceChildren();
    contentBuilder(modalContent);
    modalBackdrop.hidden = false;
    document.querySelector("#modal-close").focus();
  }

  function closeModal() {
    modalBackdrop.hidden = true;
  }

  function showHistory() {
    history = readHistory();
    openModal("Recent conversations", "CHAT HISTORY", (content) => {
      if (!history.length) {
        addText(content, "div", "history-empty", "Your conversations will appear here after your first question.");
        return;
      }
      history.forEach((session) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "history-row";
        addText(button, "span", "history-row-icon", "◷");
        const copy = document.createElement("span");
        copy.className = "history-row-copy";
        addText(copy, "strong", "", session.title || "Conversation");
        addText(copy, "small", "", new Date(session.updatedAt).toLocaleString());
        button.append(copy);
        button.addEventListener("click", () => {
          sessionId = session.id;
          messages = Array.isArray(session.messages) ? session.messages : [];
          closeModal();
          renderMessages();
          setActiveNav("chat");
        });
        content.append(button);
      });
    });
  }

  function showKnowledgeBase() {
    openModal("NovaShop policy library", "KNOWLEDGE BASE", (content) => {
      addText(content, "p", "knowledge-intro", "The assistant searches these policy documents to find relevant information before responding. If a topic is not covered, it will let you know.");
      const documents = [
        { name: "refund_policy.txt", detail: "Returns, eligibility, refunds, damaged or incorrect products, and refund status.", count: "6 policy sections" },
        { name: "shipping_policy.txt", detail: "Order processing, standard delivery, tracking, delays, addresses, and damaged shipments.", count: "6 policy sections" },
      ];
      documents.forEach((document) => {
        const row = document.createElement("div");
        row.className = "knowledge-doc";
        addText(row, "div", "knowledge-doc-icon", "▤");
        const copy = document.createElement("div");
        addText(copy, "strong", "", document.name);
        addText(copy, "p", "", document.detail);
        addText(copy, "p", "", document.count);
        row.append(copy);
        content.append(row);
      });
      addText(content, "div", "knowledge-footer", "Answers are generated only from retrieved policy excerpts. A close semantic match is not treated as proof that a policy supports an answer.");
    });
  }

  function showHowItWorks() {
    openModal("Answers grounded in policy", "HOW IT WORKS", (content) => {
      addText(content, "p", "knowledge-intro", "When you ask a question, NovaShop Support searches the policy library, checks the matching sections, and uses those excerpts to prepare a concise response.");
      [
        "Questions outside the known policy topics are marked Insufficient Context.",
        "Every answered response includes its source document and section.",
        "Cosine distance helps rank matches; it is not a confidence score.",
      ].forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "knowledge-doc";
        addText(row, "div", "knowledge-doc-icon", `${index + 1}`);
        addText(row, "p", "", item);
        content.append(row);
      });
    });
  }

  function setActiveNav(current) {
    document.querySelectorAll(".nav-item").forEach((item) => {
      const active = item.id === `nav-${current}`;
      item.classList.toggle("active", active);
      if (active) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });
  }

  function closeSidebar() {
    sidebar.classList.remove("open");
    mobileScrim.classList.remove("visible");
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuestion(input.value);
  });
  input.addEventListener("input", resizeInput);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  document.querySelectorAll(".suggestion-card").forEach((button) => {
    button.addEventListener("click", () => submitQuestion(button.dataset.question || ""));
  });
  document.querySelector("#new-chat").addEventListener("click", () => {
    saveHistory();
    messages = [];
    sessionId = newSessionId();
    renderMessages();
    setActiveNav("chat");
    closeSidebar();
    input.focus();
  });
  document.querySelector("#nav-chat").addEventListener("click", () => { closeModal(); setActiveNav("chat"); closeSidebar(); });
  document.querySelector("#nav-history").addEventListener("click", () => { showHistory(); closeSidebar(); });
  document.querySelector("#nav-knowledge").addEventListener("click", () => { showKnowledgeBase(); closeSidebar(); });
  document.querySelector("#view-sources").addEventListener("click", showKnowledgeBase);
  document.querySelector("#info-toggle").addEventListener("click", showHowItWorks);
  document.querySelector("#help-link").addEventListener("click", showHowItWorks);
  document.querySelector("#modal-close").addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", (event) => { if (event.target === modalBackdrop) closeModal(); });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") { closeModal(); closeSidebar(); } });
  document.querySelector("#mobile-menu").addEventListener("click", () => { sidebar.classList.add("open"); mobileScrim.classList.add("visible"); });
  mobileScrim.addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      document.querySelector("#new-chat").click();
    }
  });
  renderMessages();
})();
