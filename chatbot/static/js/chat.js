"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    chat: $("chat"),
    messages: $("messages"),
    empty: $("emptyState"),
    skeleton: $("historySkeleton"),
    form: $("composer"),
    input: $("input"),
    send: $("sendBtn"),
    clear: $("clearBtn"),
    model: $("modelName"),
    charCount: $("charCount"),
    settingsBtn: $("settingsBtn"),
    dialog: $("settingsDialog"),
    settingsForm: $("settingsForm"),
    systemPrompt: $("systemPrompt"),
    resetPrompt: $("resetPromptBtn"),
    toast: $("toast"),
  };

  const PROMPT_KEY = "chatbot.systemPrompt";
  const config = { maxMessageChars: 4000, maxSystemPromptChars: 2000, defaultSystemPrompt: "" };
  let controller = null; // AbortController for the in-flight request

  // ---------- storage (may be unavailable in private mode) ----------
  const storage = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch { /* ignore */ } },
    remove(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
  };

  // ---------- rendering ----------
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const icon = (paths) => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = paths; // static, trusted markup only
    return svg;
  };
  const ICONS = {
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/>',
    retry: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
  };

  /** Minimal, XSS-safe Markdown: fenced code, inline code, bold, paragraphs. No innerHTML. */
  function renderRich(container, text) {
    container.replaceChildren();
    text.split("```").forEach((part, i) => {
      if (i % 2 === 1) {
        const newline = part.indexOf("\n");
        const code = newline >= 0 ? part.slice(newline + 1) : part;
        const pre = el("pre");
        pre.append(el("code", "", code.replace(/\n$/, "")));
        container.append(pre);
        return;
      }
      for (const para of part.split(/\n{2,}/)) {
        if (!para.trim()) continue;
        const p = el("p");
        for (const token of para.replace(/^\n+|\n+$/g, "").split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/)) {
          if (!token) continue;
          if (token.startsWith("`") && token.endsWith("`") && token.length > 2) {
            p.append(el("code", "", token.slice(1, -1)));
          } else if (token.startsWith("**") && token.endsWith("**") && token.length > 4) {
            p.append(el("strong", "", token.slice(2, -2)));
          } else {
            p.append(document.createTextNode(token));
          }
        }
        container.append(p);
      }
    });
    if (!container.childNodes.length) container.append(el("p"));
  }

  function nearBottom() {
    const { scrollTop, scrollHeight, clientHeight } = els.chat;
    return scrollHeight - scrollTop - clientHeight < 120;
  }
  function scrollToBottom(force = false) {
    if (force || nearBottom()) els.chat.scrollTop = els.chat.scrollHeight;
  }

  function updateEmptyState() {
    els.empty.hidden = els.messages.children.length > 0;
  }

  function addMessage(role, text = "") {
    const item = el("li", `msg ${role}`);
    const avatar = el("div", "avatar", role === "user" ? "You" : "AI");
    const body = el("div", "msg-body");
    const bubble = el("div", "bubble");
    const tools = el("div", "msg-tools");
    body.append(bubble, tools);
    item.append(avatar, body);

    const msg = { item, bubble, tools, text };
    msg.setText = (value) => {
      msg.text = value;
      if (role === "user") bubble.textContent = value;
      else renderRich(bubble, value);
    };
    msg.setText(text);
    if (role === "assistant" && text) addCopyButton(msg);

    els.messages.append(item);
    updateEmptyState();
    scrollToBottom(true);
    return msg;
  }

  function toolButton(label, iconPaths, onClick) {
    const btn = el("button", "btn btn-ghost");
    btn.type = "button";
    btn.append(icon(iconPaths), el("span", "", label));
    btn.addEventListener("click", onClick);
    return btn;
  }

  function addCopyButton(msg) {
    const btn = toolButton("Copy", ICONS.copy, async () => {
      try {
        await navigator.clipboard.writeText(msg.text);
        btn.lastChild.textContent = "Copied";
        setTimeout(() => { btn.lastChild.textContent = "Copy"; }, 1500);
      } catch {
        toast("Clipboard is not available in this browser.");
      }
    });
    msg.tools.append(btn);
  }

  function addMeta(msg, text) {
    msg.tools.append(el("span", "msg-meta", text));
  }

  function showTyping(msg) {
    const dots = el("span", "typing");
    dots.setAttribute("aria-label", "Assistant is typing");
    dots.append(el("span"), el("span"), el("span"));
    msg.bubble.replaceChildren(dots);
  }

  let toastTimer;
  function toast(text) {
    els.toast.textContent = text;
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { els.toast.hidden = true; }, 2600);
  }

  // ---------- composer ----------
  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(els.input.scrollHeight, 200)}px`;
    const length = els.input.value.length;
    els.charCount.textContent = `${length} / ${config.maxMessageChars}`;
    els.charCount.classList.toggle("over", length > config.maxMessageChars);
  }

  function setStreaming(active) {
    els.send.classList.toggle("is-streaming", active);
    els.send.setAttribute("aria-label", active ? "Stop generating" : "Send message");
    els.clear.disabled = active;
    els.chat.setAttribute("aria-busy", String(active));
  }

  // ---------- networking ----------
  async function* readEvents(response) {
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += value;
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        let event = "message";
        let data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        yield { event, data: data ? JSON.parse(data) : {} };
      }
    }
  }

  function showError(userMsg, message, retryable) {
    const errorMsg = addMessage("error", message);
    if (retryable) {
      errorMsg.tools.append(toolButton("Retry", ICONS.retry, () => {
        errorMsg.item.remove();
        stream(userMsg);
      }));
    }
  }

  async function stream(userMsg) {
    const reply = addMessage("assistant");
    reply.item.classList.add("streaming");
    showTyping(reply);
    setStreaming(true);
    controller = new AbortController();

    let text = "";
    let finished = false;
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({
          message: userMsg.text,
          system_prompt: storage.get(PROMPT_KEY) || undefined,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        reply.item.remove();
        const retryable = response.status >= 500 || response.status === 429;
        showError(userMsg, body.error?.message || `Request failed (HTTP ${response.status}).`, retryable);
        return;
      }

      for await (const { event, data } of readEvents(response)) {
        if (event === "token") {
          text += data.text;
          reply.setText(text);
          scrollToBottom();
        } else if (event === "error") {
          if (!text) reply.item.remove();
          showError(userMsg, data.message, data.retryable);
          return;
        } else if (event === "done") {
          finished = true;
          addMeta(reply, `${(data.total_ms / 1000).toFixed(1)}s · first token ${data.ttft_ms}ms`);
        } else if (event === "meta" && data.dropped_messages > 0) {
          toast(`Older messages trimmed to fit the context window (${data.dropped_messages}).`);
        }
      }
      if (!finished) throw new TypeError("stream ended unexpectedly");
    } catch (err) {
      if (err.name === "AbortError") {
        if (text) addMeta(reply, "stopped");
        else reply.item.remove();
      } else {
        if (!text) reply.item.remove();
        showError(userMsg, "Connection to the server was lost. Check your network and retry.", true);
      }
    } finally {
      reply.item.classList.remove("streaming");
      if (text) addCopyButton(reply);
      controller = null;
      setStreaming(false);
      updateEmptyState();
    }
  }

  function submit(text) {
    const message = text.trim();
    if (!message || controller) return;
    if (message.length > config.maxMessageChars) {
      toast(`Messages are limited to ${config.maxMessageChars} characters.`);
      return;
    }
    els.input.value = "";
    autosize();
    stream(addMessage("user", message));
  }

  async function clearChat() {
    if (controller) return;
    try {
      const response = await fetch("/api/history", { method: "DELETE" });
      if (!response.ok) throw new Error(String(response.status));
      els.messages.replaceChildren();
      updateEmptyState();
      els.input.focus();
    } catch {
      toast("Could not clear the chat. Please try again.");
    }
  }

  async function init() {
    try {
      const [cfgRes, histRes] = await Promise.all([fetch("/api/config"), fetch("/api/history")]);
      const cfg = await cfgRes.json();
      config.maxMessageChars = cfg.max_message_chars;
      config.maxSystemPromptChars = cfg.max_system_prompt_chars;
      config.defaultSystemPrompt = cfg.default_system_prompt;
      els.model.textContent = cfg.model;
      els.systemPrompt.maxLength = config.maxSystemPromptChars;

      const { messages } = await histRes.json();
      for (const m of messages) addMessage(m.role, m.content);
    } catch {
      els.model.textContent = "offline";
      toast("Could not reach the server. Messages may fail until it is back.");
    } finally {
      els.skeleton.remove();
      els.chat.setAttribute("aria-busy", "false");
      updateEmptyState();
      autosize();
      scrollToBottom(true);
    }
  }

  // ---------- events ----------
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (controller) controller.abort();
    else submit(els.input.value);
  });

  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      els.form.requestSubmit();
    }
  });
  els.input.addEventListener("input", autosize);

  els.empty.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) submit(chip.textContent);
  });

  els.clear.addEventListener("click", clearChat);

  els.settingsBtn.addEventListener("click", () => {
    els.systemPrompt.value = storage.get(PROMPT_KEY) || config.defaultSystemPrompt;
    els.dialog.showModal();
  });
  els.resetPrompt.addEventListener("click", () => {
    els.systemPrompt.value = config.defaultSystemPrompt;
  });
  els.dialog.addEventListener("close", () => {
    if (els.dialog.returnValue !== "save") return;
    const value = els.systemPrompt.value.trim();
    if (!value || value === config.defaultSystemPrompt) storage.remove(PROMPT_KEY);
    else storage.set(PROMPT_KEY, value);
    toast("System prompt saved. It applies to your next message.");
  });

  init();
})();
