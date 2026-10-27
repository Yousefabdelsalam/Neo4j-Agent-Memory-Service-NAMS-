// MemoryRAG web UI. Plain JavaScript, no build step, no external libraries.
// It only talks to the REST API of this same server (see /docs for the API itself).
// All text coming from the server or the model is inserted with textContent, never innerHTML.

const $ = (selector) => document.querySelector(selector);
const chatEmpty = $("#chatEmpty"); // kept as a reference: clearing the message box detaches it

const SAMPLE_DOCUMENT = `# Project Atlas

Project Atlas is a satellite imaging platform built by the Geo team.
It processes radar images to detect flooding and sends alerts to regional emergency services.

## Status
- Launched pilot in the northern region in March.
- Next milestone: expand coverage to three more regions.
- Technology: radar imagery, a ChromaDB-backed search index and a weekly model retraining job.
`;

const state = {
  conversations: loadConversations(), // [{id, title}] kept in this browser only
  activeId: null,
  sending: false,
  insightsConv: null, // conversation the "behind the last answer" panel belongs to
  watchTimer: null,
};

// ---------- small helpers ----------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function loadConversations() {
  try {
    return JSON.parse(localStorage.getItem("memoryrag.conversations") || "[]");
  } catch (_) {
    return [];
  }
}

function saveConversations() {
  try {
    localStorage.setItem("memoryrag.conversations", JSON.stringify(state.conversations));
  } catch (_) {
    /* storage unavailable: the list just won't survive a reload */
  }
}

let toastTimer = null;
function toast(message, isError = false) {
  const box = $("#toast");
  box.textContent = message;
  box.className = "toast" + (isError ? " error" : "");
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (box.hidden = true), isError ? 7000 : 3500);
}

// Calls the API and turns error responses into readable Error messages.
async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (_) {
    throw new Error("Cannot reach the server. Is it running?");
  }
  let data = null;
  try {
    data = await response.json();
  } catch (_) {
    /* no JSON body */
  }
  if (!response.ok) {
    let detail = data && data.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; "); // FastAPI validation errors
    throw new Error(detail || `Request failed (HTTP ${response.status})`);
  }
  return data;
}

function scrollMessagesToEnd() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

// ---------- health chips ----------

async function refreshHealth() {
  const chips = $("#chips");
  chips.replaceChildren();
  const add = (text, level) => chips.append(el("span", `chip ${level}`, text));
  try {
    const h = await api("/health");
    add(h.nams.configured ? "NAMS configured" : "NAMS key missing", h.nams.configured ? "ok" : "bad");
    add(h.llm.configured ? `LLM: ${h.llm.model}` : "LLM key missing", h.llm.configured ? "ok" : "bad");
    add(`${h.chroma.chunks_indexed ?? 0} chunks indexed`, h.chroma.status === "ok" ? "ok" : "bad");
  } catch (error) {
    add("Server unreachable", "bad");
  }
}

// ---------- documents ----------

async function refreshDocuments() {
  try {
    const docs = await api("/documents");
    const list = $("#docList");
    list.replaceChildren();
    for (const doc of docs) {
      const item = el("li", "doc");
      item.append(el("span", "", doc.filename), el("span", "", `${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}`));
      item.title = doc.filename;
      list.append(item);
    }
    $("#docEmpty").hidden = docs.length > 0;
  } catch (error) {
    toast(error.message, true);
  }
}

async function uploadFile(file, documentId) {
  const form = new FormData();
  form.append("file", file);
  if (documentId) form.append("document_id", documentId);
  const result = await api("/documents/upload", { method: "POST", body: form });
  toast(`Indexed ${result.filename}: ${result.chunks_indexed} chunk${result.chunks_indexed === 1 ? "" : "s"}`);
}

async function uploadFiles(files) {
  for (const file of files) {
    try {
      await uploadFile(file);
    } catch (error) {
      toast(`${file.name}: ${error.message}`, true);
    }
  }
  refreshDocuments();
  refreshHealth();
}

// ---------- conversations ----------

function renderConversations() {
  const list = $("#convList");
  list.replaceChildren();
  for (const conv of state.conversations) {
    const item = el("li");
    const button = el("button", "conv" + (conv.id === state.activeId ? " active" : ""));
    button.type = "button";
    button.append(el("span", "", conv.title));
    const remove = el("span", "remove", "×");
    remove.title = "Remove from this list (the memory in NAMS is kept)";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      removeConversation(conv.id);
    });
    button.append(remove);
    button.addEventListener("click", () => selectConversation(conv.id));
    item.append(button);
    list.append(item);
  }
  $("#convEmpty").hidden = state.conversations.length > 0;
}

async function createConversation() {
  const title = `Conversation ${state.conversations.length + 1}`;
  const created = await api("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  state.conversations.push({ id: created.conversation_id, title: created.title });
  saveConversations();
  await selectConversation(created.conversation_id);
  return created.conversation_id;
}

function removeConversation(id) {
  state.conversations = state.conversations.filter((c) => c.id !== id);
  saveConversations();
  if (state.activeId === id) {
    state.activeId = null;
    showEmptyChat();
  }
  renderConversations();
}

function showEmptyChat() {
  $("#chatTitle").textContent = "Chat";
  $("#chatId").textContent = "";
  $("#messages").replaceChildren(chatEmpty);
  chatEmpty.hidden = false;
}

async function selectConversation(id) {
  state.activeId = id;
  const conv = state.conversations.find((c) => c.id === id);
  $("#chatTitle").textContent = conv ? conv.title : "Chat";
  $("#chatId").textContent = id;
  renderConversations();

  const box = $("#messages");
  box.replaceChildren();
  try {
    const data = await api(`/conversations/${encodeURIComponent(id)}/messages`);
    for (const message of data.messages) addMessage(message.role, message.content);
    if (data.messages.length === 0) {
      box.append(el("p", "muted small", "New conversation. Say something!"));
    }
  } catch (error) {
    toast(error.message, true);
  }
  scrollMessagesToEnd();
  $("#input").focus();
}

// ---------- chat ----------

function addMessage(role, text, extraClass = "") {
  chatEmpty.hidden = true;
  const wrapper = el("div", `msg ${role} ${extraClass}`.trim());
  wrapper.append(el("div", "bubble", text));
  $("#messages").append(wrapper);
  scrollMessagesToEnd();
  return wrapper;
}

function addTags(wrapper, data) {
  const tags = el("div", "tags");
  for (const source of data.document_sources) tags.append(el("span", "tag doc-tag", `Document: ${source}`));
  const count = data.memory_entities_used.length;
  if (count > 0) tags.append(el("span", "tag mem-tag", `${count} memory item${count === 1 ? "" : "s"}`));
  if (!data.messages_saved) tags.append(el("span", "tag", "Not saved to memory"));
  if (tags.children.length === 0) tags.append(el("span", "tag", "No documents or memory used"));
  wrapper.append(tags);
}

async function sendMessage(text) {
  if (state.sending) return;
  state.sending = true;
  $("#sendBtn").disabled = true;
  try {
    const conversationId = state.activeId || (await createConversation());
    addMessage("user", text);
    $("#input").value = "";
    const pending = addMessage("assistant", "Thinking…", "pending");

    try {
      const data = await api("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: conversationId, message: text }),
      });
      pending.remove();
      const answer = addMessage("assistant", data.answer);
      addTags(answer, data);
      renderInsights(data);
      watchExtraction(conversationId, data.extraction_status);
    } catch (error) {
      pending.remove();
      addMessage("assistant", error.message, "error");
    }
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.sending = false;
    $("#sendBtn").disabled = false;
    $("#input").focus();
  }
}

// ---------- "behind the last answer" panel ----------

const EXTRACTION_LABELS = {
  pending: ["Extracting entities in the background…", "warn pulse"],
  completed: ["Saved to long-term memory", "ok"],
  failed: ["Extraction failed", "bad"],
  unknown: ["Checking extraction…", "warn pulse"],
  not_checked: ["Not checked (messages not saved)", "bad"],
};

const MEMORY_LABELS = {
  ok: ["Found related memories", "ok"],
  empty: ["Nothing related found", ""],
  unavailable: ["Memory unavailable", "bad"],
};

function badge(label, level) {
  return el("span", `badge ${level}`.trim(), label);
}

function row(label, valueNode) {
  const r = el("div", "kv");
  r.append(el("span", "", label), valueNode);
  return r;
}

function entityCard(entity) {
  const card = el("div", "card");
  const title = el("div", "title");
  title.append(el("span", "", entity.name || "(unnamed)"));
  if (entity.type) title.append(el("span", "type", String(entity.type).toLowerCase()));
  card.append(title);
  if (entity.description) card.append(el("div", "desc", entity.description));
  return card;
}

function renderInsights(data) {
  state.insightsConv = data.conversation_id;
  const panel = $("#insights");
  panel.replaceChildren();

  const [memText, memLevel] = MEMORY_LABELS[data.memory_retrieval_status] || ["", ""];
  const [exText, exLevel] = EXTRACTION_LABELS[data.extraction_status] || ["", ""];

  panel.append(row("Memory lookup", badge(memText, memLevel)));
  panel.append(row("Messages saved", badge(data.messages_saved ? "Yes" : "No", data.messages_saved ? "ok" : "bad")));
  const extraction = row("Long-term memory", badge(exText, exLevel));
  extraction.id = "extractionRow";
  panel.append(extraction);

  const docs = el("div");
  docs.append(el("div", "subhead", "Documents used"));
  if (data.document_sources.length) {
    for (const name of data.document_sources) docs.append(el("div", "card", name));
  } else {
    docs.append(el("p", "muted small", "None matched this question."));
  }
  panel.append(docs);

  const memories = el("div");
  memories.append(el("div", "subhead", "Memories given to the model"));
  if (data.memory_entities_used.length) {
    for (const entity of data.memory_entities_used) memories.append(entityCard(entity));
  } else {
    memories.append(el("p", "muted small", "None."));
  }
  panel.append(memories);

  for (const warning of data.warnings || []) panel.append(el("div", "warnbox", warning));
}

function updateExtractionBadge(conversationId, extractionState) {
  const row = $("#extractionRow");
  if (!row || state.insightsConv !== conversationId) return;
  const [text, level] = EXTRACTION_LABELS[extractionState] || ["", ""];
  row.replaceChildren(el("span", "", "Long-term memory"), badge(text, level));
}

// NAMS extracts entities in the background, so poll its status for a while after each message.
function watchExtraction(conversationId, initialState) {
  clearTimeout(state.watchTimer);
  if (initialState === "completed" || initialState === "failed" || initialState === "not_checked") return;
  let attempts = 0;
  const tick = async () => {
    attempts += 1;
    try {
      const status = await api(`/conversations/${encodeURIComponent(conversationId)}/extraction-status`);
      updateExtractionBadge(conversationId, status.state);
      if (status.state === "completed") {
        toast("Saved to long-term memory. A new conversation can use it now.");
        return;
      }
      if (status.state === "failed" || (status.state === "unknown" && attempts >= 5)) return;
    } catch (_) {
      /* transient error: try again */
    }
    if (attempts < 40) state.watchTimer = setTimeout(tick, 3000);
  };
  state.watchTimer = setTimeout(tick, 2500);
}

// ---------- memory search ----------

async function searchMemory(query) {
  const results = $("#memoryResults");
  results.replaceChildren(el("p", "muted small", "Searching…"));
  try {
    const data = await api(`/memory/search?q=${encodeURIComponent(query)}&limit=8`);
    results.replaceChildren();
    for (const entity of data.results) results.append(entityCard(entity));
    if (data.note) results.append(el("p", "muted small", data.note));
  } catch (error) {
    results.replaceChildren(el("p", "warnbox", error.message));
  }
}

// ---------- wiring ----------

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("#input").value.trim();
  if (text) sendMessage(text);
});

$("#input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});

$("#newConvBtn").addEventListener("click", async () => {
  try {
    await createConversation();
  } catch (error) {
    toast(error.message, true);
  }
});

$("#memoryForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = $("#memoryQuery").value.trim();
  if (query) searchMemory(query);
});

const fileInput = $("#fileInput");
fileInput.addEventListener("change", () => {
  uploadFiles([...fileInput.files]);
  fileInput.value = "";
});

const dropzone = $("#dropzone");
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("over");
  uploadFiles([...event.dataTransfer.files]);
});

$("#sampleBtn").addEventListener("click", async () => {
  try {
    const file = new File([SAMPLE_DOCUMENT], "project-atlas.md", { type: "text/markdown" });
    await uploadFile(file, "sample-project-atlas"); // fixed id: clicking again replaces, never duplicates
    refreshDocuments();
    refreshHealth();
  } catch (error) {
    toast(error.message, true);
  }
});

const guide = $("#guide");
$("#guideBtn").addEventListener("click", () => guide.showModal());
for (const button of guide.querySelectorAll(".prompt")) {
  button.addEventListener("click", () => {
    $("#input").value = button.dataset.prompt || button.textContent;
    guide.close();
    $("#input").focus();
  });
}

renderConversations();
refreshHealth();
refreshDocuments();
if (state.conversations.length) selectConversation(state.conversations[state.conversations.length - 1].id);
