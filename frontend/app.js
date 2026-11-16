/*
 * app.js
 * The page's behaviour:
 *   - sends questions to POST /query and shows the answer with its badge and numbers
 *   - renders markdown and LaTeX maths in answers, safely
 *   - keeps the dashboard in sync with GET /stats, GET /history and GET /health
 *
 * Safety of answers (they come from a model, and may come from the shared cache):
 *   - raw HTML tags in an answer are shown as plain text, apart from a short list of harmless ones
 *   - the sanitiser then removes scripts, event handlers, images, media, frames, forms and styles
 */
(function () {
  "use strict";

  var MAX_QUERY_CHARS = 4000;
  var HISTORY_LIMIT = 200;

  var MODE_HINTS = {
    auto: "Checks the cache first, then the router picks the model.",
    small: "Forces the small model. The cache is skipped.",
    large: "Forces the large model. The cache is skipped."
  };

  var TIER_LABELS = { cache: "Cache hit", small: "Small model", large: "Large model" };

  // HTML tags that may stay in an answer as real tags (the sanitiser still checks them).
  // Any other tag-like text is shown as plain text instead of being built into page elements.
  var KEEP_TAGS = [
    "br", "b", "i", "u", "s", "em", "strong", "sup", "sub", "small", "mark", "del", "ins", "kbd",
    "p", "ul", "ol", "li", "hr", "blockquote", "code", "pre",
    "table", "thead", "tbody", "tr", "th", "td",
    "h1", "h2", "h3", "h4", "h5", "h6", "span", "div", "details", "summary"
  ];

  // Removed from answers for safety, even if they get past the step above.
  var FORBIDDEN_TAGS = [
    "img", "picture", "source", "video", "audio", "track", "iframe", "object", "embed",
    "form", "input", "button", "textarea", "select", "option", "style", "link", "meta", "base"
  ];
  var FORBIDDEN_ATTRS = ["style", "src", "srcset", "background", "poster", "formaction"];

  function byId(id) { return document.getElementById(id); }

  var els = {
    status: byId("status"),
    statusText: byId("statusText"),
    messages: byId("messages"),
    welcome: byId("welcome"),
    query: byId("query"),
    sendBtn: byId("sendBtn"),
    counter: byId("counter"),
    modeHint: byId("modeHint"),
    clearChatBtn: byId("clearChatBtn"),
    refreshBtn: byId("refreshBtn"),
    clearLogBtn: byId("clearLogBtn"),
    clearCacheBtn: byId("clearCacheBtn"),
    statRequests: byId("statRequests"),
    statRequestsSub: byId("statRequestsSub"),
    statHitRate: byId("statHitRate"),
    statHitRateSub: byId("statHitRateSub"),
    statSavings: byId("statSavings"),
    statSavingsSub: byId("statSavingsSub"),
    statCost: byId("statCost"),
    statCostSub: byId("statCostSub"),
    statLatency: byId("statLatency"),
    statMix: byId("statMix"),
    costNote: byId("costNote"),
    thresholds: byId("thresholds")
  };

  var welcomeNode = els.welcome;
  var costChart = null;
  var tierChart = null;
  var busy = false;

  /* ------------------------------------------------------------
     Small helpers
     ------------------------------------------------------------ */
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Costs are tiny (fractions of a cent), so show enough decimals to see them.
  function fmtUsd(value) {
    var v = Number(value) || 0;
    if (v === 0) { return "$0"; }
    if (Math.abs(v) < 0.01) { return "$" + v.toFixed(6); }
    return "$" + v.toFixed(4);
  }

  function fmtMs(value) {
    var v = Number(value) || 0;
    if (v >= 1000) { return (v / 1000).toFixed(2) + " s"; }
    return v.toFixed(0) + " ms";
  }

  function scrollToBottom() {
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  /* ------------------------------------------------------------
     Talking to the server
     ------------------------------------------------------------ */
  async function readError(response) {
    var detail = "";
    try {
      var body = await response.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        detail = body.detail.map(function (item) { return item.msg; }).join("; ");
      }
    } catch (error) {
      detail = "";
    }
    return detail || ("Request failed (HTTP " + response.status + ")");
  }

  async function requestJson(url, options) {
    var response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      throw new Error("Could not reach the server. Is it running?");
    }
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    return response.json();
  }

  /* ------------------------------------------------------------
     Rendering answers: markdown + LaTeX, sanitised
     ------------------------------------------------------------ */
  // Maths delimiters we support. Single dollar signs are NOT used, because
  // they would break ordinary text such as "it costs $4 and $5".
  var MATH_PATTERNS = [
    { re: /\$\$([\s\S]+?)\$\$/g, display: true },
    { re: /\\\[([\s\S]+?)\\\]/g, display: true },
    { re: /\\\(([\s\S]+?)\\\)/g, display: false }
  ];

  function renderMath(item) {
    if (!item) { return ""; }
    if (window.katex) {
      try {
        return window.katex.renderToString(item.tex, {
          displayMode: item.display,
          throwOnError: false
        });
      } catch (error) {
        /* fall through to plain text */
      }
    }
    return "<code>" + escapeHtml(item.tex) + "</code>";
  }

  // Turns tag-like text such as <img src=x> or <script> into visible characters,
  // unless the tag is on the short list of harmless ones.
  function showUnknownTagsAsText(text) {
    return text.replace(/<(\/?)([A-Za-z][A-Za-z0-9-]*)([^<>]*)>/g, function (match, slash, name, rest) {
      if (KEEP_TAGS.indexOf(name.toLowerCase()) !== -1) { return match; }
      return "&lt;" + slash + name + rest + "&gt;";
    });
  }

  function renderMarkdown(text) {
    var source = String(text || "");

    // Without the libraries (for example, offline) show plain text.
    if (!window.marked || !window.DOMPurify) {
      return "<p>" + escapeHtml(source).replace(/\n/g, "<br>") + "</p>";
    }

    try {
      // 1. Protect code, so maths symbols inside code are left alone.
      var codeStore = [];
      var work = source.replace(/```[\s\S]*?```|`[^`\n]+`/g, function (match) {
        codeStore.push(match);
        return "XCODEX" + (codeStore.length - 1) + "X";
      });

      // 2. Pull the maths out, so markdown cannot mangle backslashes and underscores.
      var mathStore = [];
      MATH_PATTERNS.forEach(function (pattern) {
        work = work.replace(pattern.re, function (match, tex) {
          mathStore.push({ tex: tex.trim(), display: pattern.display });
          return "XMATHX" + (mathStore.length - 1) + "X";
        });
      });

      // 3. Show unknown HTML tags as plain text, put the code back,
      //    and convert markdown to HTML.
      work = showUnknownTagsAsText(work);
      work = work.replace(/XCODEX(\d+)X/g, function (match, index) {
        return codeStore[Number(index)];
      });
      var html = window.marked.parse(work, { gfm: true, breaks: true });

      // 4. Remove anything unsafe (scripts, event handlers, images, frames, forms, styles ...).
      html = window.DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true },
        FORBID_TAGS: FORBIDDEN_TAGS,
        FORBID_ATTR: FORBIDDEN_ATTRS
      });

      // 5. Render the maths with KaTeX. KaTeX makes its own safe HTML,
      //    so this happens after sanitising.
      return html.replace(/XMATHX(\d+)X/g, function (match, index) {
        return renderMath(mathStore[Number(index)]);
      });
    } catch (error) {
      return "<p>" + escapeHtml(source).replace(/\n/g, "<br>") + "</p>";
    }
  }

  // Links in answers open in a new tab.
  if (window.DOMPurify) {
    window.DOMPurify.addHook("afterSanitizeAttributes", function (node) {
      if (node.tagName === "A") {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
      }
    });
  }

  /* ------------------------------------------------------------
     Chat messages
     ------------------------------------------------------------ */
  function addUserMessage(text) {
    if (welcomeNode.parentNode) { welcomeNode.remove(); }
    var row = el("div", "msg msg-user");
    var bubble = el("div", "bubble", text); // textContent: never interpreted as HTML
    row.appendChild(bubble);
    els.messages.appendChild(row);
    scrollToBottom();
  }

  function addPendingMessage() {
    var row = el("div", "msg msg-assistant");
    var bubble = el("div", "bubble");
    bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
    row.appendChild(bubble);
    els.messages.appendChild(row);
    scrollToBottom();
    return row;
  }

  function chip(label, value) {
    var node = el("span", "chip");
    node.appendChild(el("b", "", label));
    node.appendChild(document.createTextNode(value));
    return node;
  }

  function fillAssistantMessage(row, data) {
    row.className = "msg msg-assistant";
    row.textContent = "";

    var bubble = el("div", "bubble");

    // Top line: badge, model name and a few small labels
    var head = el("div", "head-row");
    head.appendChild(el("span", "badge badge-" + data.tier, TIER_LABELS[data.tier] || data.tier));
    head.appendChild(el("span", "model-name", data.model_name));
    if (data.tier === "cache" && typeof data.similarity === "number") {
      head.appendChild(chip("similarity", data.similarity.toFixed(3)));
    }
    if (data.mode_requested === "auto" && data.tier !== "cache" && typeof data.similarity === "number") {
      head.appendChild(chip("closest cached question", data.similarity.toFixed(3)));
    }
    if (data.mode_requested !== "auto") {
      head.appendChild(chip("mode", "forced " + data.mode_requested));
    }
    bubble.appendChild(head);

    // The answer itself
    var answer = el("div", "answer");
    answer.innerHTML = renderMarkdown(data.answer);
    bubble.appendChild(answer);

    // Numbers for this request
    var meta = el("div", "meta");
    meta.appendChild(chip("Latency", fmtMs(data.latency_ms)));
    meta.appendChild(chip("Tokens", data.input_tokens + " in / " + data.output_tokens + " out"));
    meta.appendChild(chip("Cost", fmtUsd(data.cost_usd)));
    meta.appendChild(chip("Saved", fmtUsd(data.saved_usd)));
    bubble.appendChild(meta);

    // Why this route
    var why = el("details", "why");
    why.appendChild(el("summary", "", "Why this route?"));
    why.appendChild(el("p", "", data.route_reason));
    bubble.appendChild(why);

    row.appendChild(bubble);
    scrollToBottom();
  }

  function fillErrorMessage(row, message) {
    row.className = "msg msg-assistant msg-error";
    row.textContent = "";
    var bubble = el("div", "bubble", message);
    row.appendChild(bubble);
    scrollToBottom();
  }

  function clearChat() {
    els.messages.textContent = "";
    els.messages.appendChild(welcomeNode);
  }

  /* ------------------------------------------------------------
     Sending a question
     ------------------------------------------------------------ */
  function getMode() {
    var checked = document.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : "auto";
  }

  function setBusy(isBusy) {
    busy = isBusy;
    els.sendBtn.disabled = isBusy;
    els.sendBtn.textContent = isBusy ? "Sending..." : "Send";
  }

  function updateCounter() {
    els.counter.textContent = els.query.value.length + " / " + MAX_QUERY_CHARS;
  }

  function autoGrow() {
    els.query.style.height = "auto";
    els.query.style.height = Math.min(els.query.scrollHeight, 160) + "px";
  }

  async function sendQuery() {
    if (busy) { return; }
    var query = els.query.value.trim();
    if (!query) { return; }
    if (query.length > MAX_QUERY_CHARS) {
      alert("That question is too long. The limit is " + MAX_QUERY_CHARS + " characters.");
      return;
    }

    var mode = getMode();
    setBusy(true);
    addUserMessage(query);
    els.query.value = "";
    updateCounter();
    autoGrow();
    var pending = addPendingMessage();

    try {
      var data = await requestJson("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, mode: mode })
      });
      fillAssistantMessage(pending, data);
    } catch (error) {
      fillErrorMessage(pending, error.message);
    } finally {
      setBusy(false);
      els.query.focus();
      refreshDashboard();
      refreshHealth();
    }
  }

  /* ------------------------------------------------------------
     Status pill and dashboard
     ------------------------------------------------------------ */
  function setStatus(ok, text) {
    els.status.className = "status " + (ok ? "ok" : "bad");
    els.statusText.textContent = text;
  }

  async function refreshHealth() {
    try {
      var health = await requestJson("/health");
      var routerText = health.router_loaded
        ? "router ready"
        : "router not loaded (auto uses the large model)";
      var entries = health.cache ? health.cache.entries : 0;
      var text = "Online \u00b7 " + routerText + " \u00b7 cache: " + entries + " entries";
      if (!health.groq_key_loaded) {
        setStatus(false, "Online, but no Groq key is loaded");
      } else {
        setStatus(true, text);
      }
      els.thresholds.textContent =
        "Router threshold " + Number(health.router_threshold).toFixed(2) +
        " \u00b7 Cache threshold " + Number(health.cache_threshold).toFixed(2);
    } catch (error) {
      setStatus(false, "Server unreachable");
    }
  }

  function updateCards(stats) {
    els.statRequests.textContent = String(stats.total_requests);
    els.statRequestsSub.textContent =
      stats.total_input_tokens + " in / " + stats.total_output_tokens + " out tokens";

    els.statHitRate.textContent = (stats.cache_hit_rate * 100).toFixed(0) + "%";
    els.statHitRateSub.textContent = stats.cache_hits + " of " + stats.total_requests + " requests";

    els.statSavings.textContent = stats.savings_percent.toFixed(1) + "%";
    els.statSavingsSub.textContent = fmtUsd(stats.total_saved_usd) + " saved";

    els.statCost.textContent = fmtUsd(stats.total_cost_usd);
    els.statCostSub.textContent = "baseline " + fmtUsd(stats.total_baseline_cost_usd);

    els.statLatency.textContent = fmtMs(stats.avg_latency_ms);
    els.statMix.textContent = stats.small_calls + " / " + stats.large_calls + " / " + stats.cache_hits;
  }

  function initCharts() {
    if (!window.Chart) {
      els.costNote.textContent = "(charts need internet access to load Chart.js)";
      return;
    }

    var rootStyle = getComputedStyle(document.documentElement);
    var muted = rootStyle.getPropertyValue("--muted").trim() || "#667085";
    var border = rootStyle.getPropertyValue("--border").trim() || "#e2e5eb";
    var small = rootStyle.getPropertyValue("--small").trim() || "#1f9d6b";
    var large = rootStyle.getPropertyValue("--large").trim() || "#7a4fd6";
    var cache = rootStyle.getPropertyValue("--cache").trim() || "#d98200";
    var accent = rootStyle.getPropertyValue("--accent").trim() || "#3b6ef5";

    window.Chart.defaults.color = muted;
    window.Chart.defaults.font.size = 11;

    costChart = new window.Chart(byId("costChart"), {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Actual cost",
            data: [],
            borderColor: accent,
            backgroundColor: "rgba(59, 110, 245, 0.12)",
            fill: true,
            tension: 0.2,
            pointRadius: 0,
            borderWidth: 2
          },
          {
            label: "Always-large baseline",
            data: [],
            borderColor: "#98a2b3",
            borderDash: [6, 4],
            fill: false,
            pointRadius: 0,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "bottom", labels: { boxWidth: 12 } },
          tooltip: {
            callbacks: {
              label: function (context) {
                return context.dataset.label + ": " + fmtUsd(context.parsed.y);
              }
            }
          }
        },
        scales: {
          x: { title: { display: true, text: "Request" }, grid: { display: false } },
          y: {
            beginAtZero: true,
            grid: { color: border },
            ticks: { callback: function (value) { return fmtUsd(value); } }
          }
        }
      }
    });

    tierChart = new window.Chart(byId("tierChart"), {
      type: "doughnut",
      data: {
        labels: ["Small model", "Large model", "Cache"],
        datasets: [{ data: [0, 0, 0], backgroundColor: [small, large, cache], borderWidth: 0 }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        cutout: "60%",
        plugins: { legend: { position: "right", labels: { boxWidth: 12 } } }
      }
    });
  }

  function updateCharts(stats, history) {
    if (costChart) {
      // /history returns newest first; the chart wants oldest first.
      var rows = history.slice().reverse();
      var labels = [];
      var actual = [];
      var baseline = [];
      var actualTotal = 0;
      var baselineTotal = 0;
      rows.forEach(function (row) {
        actualTotal += row.cost_usd;
        baselineTotal += row.baseline_cost_usd;
        labels.push("#" + row.id);
        actual.push(actualTotal);
        baseline.push(baselineTotal);
      });
      costChart.data.labels = labels;
      costChart.data.datasets[0].data = actual;
      costChart.data.datasets[1].data = baseline;
      costChart.update();
      els.costNote.textContent =
        rows.length >= HISTORY_LIMIT ? "(last " + HISTORY_LIMIT + " requests)" : "";
    }
    if (tierChart) {
      tierChart.data.datasets[0].data = [stats.small_calls, stats.large_calls, stats.cache_hits];
      tierChart.update();
    }
  }

  async function refreshDashboard() {
    try {
      var results = await Promise.all([
        requestJson("/stats"),
        requestJson("/history?limit=" + HISTORY_LIMIT)
      ]);
      updateCards(results[0]);
      updateCharts(results[0], results[1]);
    } catch (error) {
      /* the status pill already shows when the server cannot be reached */
    }
  }

  /* ------------------------------------------------------------
     Buttons and keyboard
     ------------------------------------------------------------ */
  async function clearLog() {
    if (!confirm("Delete all logged requests? This cannot be undone.")) { return; }
    try {
      await requestJson("/logs", { method: "DELETE" });
    } catch (error) {
      alert(error.message);
      return;
    }
    refreshDashboard();
  }

  async function clearCacheEntries() {
    if (!confirm("Delete every cached answer? This cannot be undone.")) { return; }
    try {
      await requestJson("/cache", { method: "DELETE" });
    } catch (error) {
      alert(error.message);
      return;
    }
    refreshHealth();
  }

  function updateModeHint() {
    els.modeHint.textContent = MODE_HINTS[getMode()] || "";
  }

  els.sendBtn.addEventListener("click", sendQuery);
  els.query.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendQuery();
    }
  });
  els.query.addEventListener("input", function () {
    updateCounter();
    autoGrow();
  });
  els.clearChatBtn.addEventListener("click", clearChat);
  els.refreshBtn.addEventListener("click", function () {
    refreshDashboard();
    refreshHealth();
  });
  els.clearLogBtn.addEventListener("click", clearLog);
  els.clearCacheBtn.addEventListener("click", clearCacheEntries);

  document.querySelectorAll('input[name="mode"]').forEach(function (radio) {
    radio.addEventListener("change", updateModeHint);
  });

  document.querySelectorAll(".example").forEach(function (button) {
    button.addEventListener("click", function () {
      els.query.value = button.getAttribute("data-query") || "";
      updateCounter();
      autoGrow();
      els.query.focus();
    });
  });

  /* ------------------------------------------------------------
     Start
     ------------------------------------------------------------ */
  initCharts();
  updateModeHint();
  updateCounter();
  refreshHealth();
  refreshDashboard();
  els.query.focus();
})();