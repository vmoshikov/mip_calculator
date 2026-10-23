"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const CTX_STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const PRECISIONS = ["bf16", "fp16", "fp8", "int8", "fp4", "int4"];
const REQ_PRECISIONS = ["native", ...PRECISIONS];
const KV_DTYPES = ["bf16", "fp16", "fp8"];
const PRECISION_LABELS = { native: "нативный (чекпоинт)" };

let cfg = null;   // сохранённый конфиг с сервера
let draft = null; // редактируемая копия настроек

const store = {
  get(k) { try { return JSON.parse(localStorage.getItem("mip." + k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem("mip." + k, JSON.stringify(v)); } catch {} },
};

async function api(method, path, body) {
  const r = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => t.classList.remove("show"), 2600);
}

const nf = (v, d = 0) => Number(v).toLocaleString("ru-RU", { maximumFractionDigits: d, minimumFractionDigits: 0 });
const fmt = {
  gb: (v) => (v >= 1000 ? nf(v / 1000, 2) + " ТБ" : nf(v, v < 10 ? 1 : 0) + " ГБ"),
  tok: (n) => (n >= 1048576 ? nf(n / 1048576, 1) + " млн" : n >= 1024 ? nf(n / 1024) + " тыс." : String(n)),
  big: (n) => (n >= 1e9 ? nf(n / 1e9, 2) + " млрд" : n >= 1e6 ? nf(n / 1e6, 1) + " млн" : n >= 1e3 ? nf(n / 1e3, 1) + " тыс." : nf(n)),
  rub: (v) => (v ? nf(v, v < 10 ? 2 : 0) + " ₽" : "—"),
  rubTiny: (v) => {
    if (!v) return "—";
    if (v >= 0.01) return nf(v, 4) + " ₽";
    const digits = Math.max(2, -Math.floor(Math.log10(v)) + 2);
    return v.toFixed(Math.min(digits, 10)).replace(".", ",") + " ₽";
  },
  int: (v) => nf(v),
  params: (b) => (b >= 1000 ? nf(b / 1000, 2) + " трлн" : nf(b, b < 10 ? 1 : 0) + " млрд"),
  bytes: (v) => (v >= 1e6 ? nf(v / 1e6, 1) + " МБ" : v >= 1024 ? nf(v / 1024, 1) + " КБ" : nf(v) + " Б"),
  ms: (v) => (v >= 1000 ? nf(v / 1000, 2) + " с" : nf(v, v < 10 ? 1 : 0) + " мс"),
  pct: (v) => nf(100 * v, v < 0.1 ? 1 : 0) + "%",
};

/* ---------- вкладки ---------- */
$$(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  $$(".tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  store.set("tab", name);
  document.body.dataset.tab = name;
  if (name === "limits") initLimits();
  if (name === "pd") renderPD();
  if (name === "matrix") renderMatrix();
  if (name === "models") renderModels();
  if (name === "settings") renderSettings();
  if (name === "mcp") renderMCP();
}

/* ---------- форма калькулятора ---------- */
function formRequest() {
  return {
    model: $("#f-model").value,
    gpu: $("#f-gpu").value,
    precision: $("#f-precision").value,
    kv_dtype: $("#f-kv").value,
    context_len: CTX_STEPS[+$("#f-ctx").value],
    concurrency: +$("#f-conc").value || 1,
    replicas: +$("#f-rep").value || 1,
    mem_util: +$("#f-util").value || undefined,
    input_tokens: +$("#f-in").value || undefined,
    output_tokens: +$("#f-out").value || undefined,
    interconnect: $("#f-net").value,
    ttft_slo_ms: +$("#f-slo").value || undefined,
  };
}

function fillSelects() {
  const keep = store.get("form") || {};
  const d = cfg.platform.defaults;
  $("#platformName").textContent = cfg.platform.name;
  $("#f-model").innerHTML = cfg.models
    .map((m) => `<option value="${esc(m.id)}">${esc(m.name)}${m.verified ? "" : " · оценка"}</option>`)
    .join("");
  $("#f-gpu").innerHTML = cfg.gpus.map((g) => `<option value="${esc(g.id)}">${esc(g.name)}</option>`).join("");
  $("#f-net").innerHTML = (cfg.interconnects || [])
    .map((n) => `<option value="${esc(n.id)}">${esc(n.name)}</option>`).join("");
  $("#f-precision").innerHTML = REQ_PRECISIONS
    .map((p) => `<option value="${p}">${PRECISION_LABELS[p] || p}</option>`).join("");
  const pick = (sel, v, fb) => {
    const el = $(sel);
    el.value = v;
    if (el.value !== v) el.value = fb;
  };
  pick("#f-model", keep.model, cfg.models[0]?.id);
  pick("#f-gpu", keep.gpu, cfg.gpus.find((g) => g.id.startsWith("h200"))?.id || cfg.gpus[0]?.id);
  pick("#f-precision", keep.precision, d.precision);
  pick("#f-kv", keep.kv_dtype, d.kv_dtype);
  pick("#f-net", keep.interconnect, d.interconnect);
  $("#f-ctx").value = nearestCtx(keep.context_len || d.context_len);
  $("#f-conc").value = keep.concurrency || d.concurrency;
  $("#f-rep").value = keep.replicas || d.replicas;
  $("#f-util").value = keep.mem_util || cfg.platform.gpu_memory_utilization;
  $("#f-in").value = keep.input_tokens || d.input_tokens;
  $("#f-out").value = keep.output_tokens || d.output_tokens;
  $("#f-slo").value = keep.ttft_slo_ms || d.ttft_slo_ms || "";
}

function nearestCtx(v) {
  let best = 0;
  CTX_STEPS.forEach((s, i) => { if (Math.abs(s - v) < Math.abs(CTX_STEPS[best] - v)) best = i; });
  return best;
}

let calcTimer;
function scheduleCalc() {
  clearTimeout(calcTimer);
  calcTimer = setTimeout(() => {
    calculate();
    if ($("#view-pd").classList.contains("active")) renderPD();
  }, 150);
}
$$("#form select, #form input").forEach((el) => el.addEventListener("input", scheduleCalc));

async function calculate() {
  const req = formRequest();
  store.set("form", req);
  $("#o-ctx").textContent = fmt.tok(req.context_len);
  renderModelMeta(req.model);
  try {
    const [res, cmp] = await Promise.all([
      api("POST", "/api/calculate", req),
      api("POST", "/api/compare", req),
    ]);
    renderKPIs(res);
    renderMemory(res);
    renderCluster(res);
    renderCompare(cmp, req.gpu);
  } catch (e) {
    toast("Ошибка: " + e.message);
  }
}

function archLabel(m) {
  if (m.attention === "mla") return `MLA ${m.kv_lora_rank}+${m.qk_rope_dim}`;
  if (m.attention === "custom") return "сжатый KV";
  return `GQA ${m.kv_heads}×${m.head_dim}`;
}

const verifiedBadge = (v) =>
  v ? `<span class="badge ok">проверено</span>` : `<span class="badge est">оценка</span>`;

function renderModelMeta(id) {
  const m = cfg.models.find((x) => x.id === id);
  if (!m) return;
  $("#modelMeta").innerHTML =
    `<b>${fmt.params(m.total_params_b)}</b> всего · <b>${fmt.params(m.active_params_b)}</b> активных<br>` +
    `${m.layers} слоёв${m.kv_layers ? ` (KV в ${m.kv_layers})` : ""} · ${archLabel(m)}` +
    (m.experts ? ` · ${m.experts_per_tok} из ${m.experts} экспертов` : "") + `<br>` +
    `контекст до ${fmt.tok(m.max_context)} · формат ${m.native_precision}` +
    (m.checkpoint_gb ? ` · чекпоинт ${fmt.gb(m.checkpoint_gb)}` : "") +
    verifiedBadge(m.verified) +
    (m.note ? `<br><span>${esc(m.note)}</span>` : "");
}

function kpi(label, value, sub, cls = "") {
  return `<div class="kpi ${cls}"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
}

function renderKPIs(r) {
  $("#warnings").innerHTML = r.warnings.map((w) => `<div class="warn">${esc(w)}</div>`).join("");
  if (!r.fits) {
    $("#kpis").innerHTML = `
      <div class="kpi bad" style="grid-column:1/-1">
        <div class="label">${esc(r.model.name)} на ${esc(r.gpu.name)}</div>
        <div class="value">${esc(r.reason)}</div>
        <div class="sub">Только по памяти нужно не меньше ${r.min_gpus_by_memory} GPU · веса ${fmt.gb(r.weights_gb)} · KV ${fmt.gb(r.kv_cache_gb)}</div>
      </div>`;
    $("#costs").innerHTML = "";
    return;
  }
  $("#kpis").innerHTML =
    kpi("Всего GPU", r.total_gpus, esc(r.gpu.name), "hero") +
    kpi("На реплику", r.gpus_per_replica, `TP ${r.tp} × PP ${r.pp} · реплик: ${r.replicas}`) +
    kpi("Нод", r.total_nodes, `по ${r.gpu.gpus_per_node} GPU`) +
    kpi("Макс. параллельных", fmt.int(r.max_concurrency), `запросов по ${fmt.tok(r.request.context_len)} на реплику`) +
    kpi("Скорость 1 потока", "~" + nf(r.decode_tok_s), "токенов/с") +
    kpi("Время до 1-го токена", fmt.ms(r.ttft_ms), `prefill ${fmt.tok(r.request.input_tokens)} токенов`);

  $("#costs").innerHTML = `
    <div class="cost-main">
      ${kpi("Генерация в час", fmt.big(r.tokens_per_hour), `токенов · ${fmt.big(r.requests_per_hour)} запросов`, "hero")}
      ${kpi("Стоимость в час", fmt.rub(r.cost_per_hour), `GPU ${fmt.rub(r.gpu_cost_per_hour)}${r.network_cost_per_hour ? " + сеть " + fmt.rub(r.network_cost_per_hour) : ""} · ${fmt.rub(r.cost_per_hour * 730)} в месяц`)}
      ${kpi("За 1 млн токенов", fmt.rub(r.cost_per_1m_tokens), "выходных, prefill включён", "accent")}
      ${kpi("За 1 токен", fmt.rubTiny(r.cost_per_token), "сгенерированный")}
    </div>
    <div class="cost-split">
      <div><span>Вход (prefill)</span><b>${fmt.rub(r.cost_per_1m_input)}</b><small>за 1 млн</small></div>
      <div><span>Выход (decode)</span><b>${fmt.rub(r.cost_per_1m_output)}</b><small>за 1 млн</small></div>
      <div><span>Один запрос</span><b>${fmt.rubTiny(r.cost_per_request)}</b><small>${fmt.tok(r.request.input_tokens)} → ${fmt.tok(r.request.output_tokens)}</small></div>
      <div><span>Доля prefill</span><b>${fmt.pct(r.prefill_share)}</b><small>времени GPU</small></div>
      <div><span>Decode реплики</span><b>${nf(r.throughput_tok_s)}</b><small>токенов/с · упор в ${r.bottleneck}</small></div>
    </div>`;
}

function renderMemory(r) {
  const bar = $("#membar");
  const nums = $("#memnums");
  if (!r.fits) {
    const total = r.weights_gb + r.kv_cache_gb;
    bar.innerHTML = `<div class="seg-w" style="flex-basis:${(100 * r.weights_gb) / total}%">веса ${fmt.gb(r.weights_gb)}</div>
      <div class="seg-kv" style="flex-basis:${(100 * r.kv_cache_gb) / total}%">KV ${fmt.gb(r.kv_cache_gb)}</div>`;
    nums.innerHTML = `<span>Суммарно на модель (не помещается)</span>`;
    return;
  }
  const p = r.per_gpu;
  const segs = [
    ["seg-w", p.weights_gb, "веса"],
    ["seg-kv", p.kv_cache_gb, "KV"],
    ["seg-ov", p.overhead_gb, "накладные"],
    ["seg-fr", Math.max(p.free_gb, 0), "свободно"],
    ["seg-rs", p.reserved_gb, "резерв"],
  ];
  bar.innerHTML = segs.map(([c, v, l]) => {
    const pct = (100 * v) / p.vram_gb;
    return `<div class="${c}" style="flex-basis:${pct}%" title="${l}: ${nf(v, 1)} ГБ">${pct > 11 ? `${l} ${fmt.gb(v)}` : ""}</div>`;
  }).join("");
  nums.innerHTML = `
    <span>VRAM <b>${fmt.gb(p.vram_gb)}</b></span>
    <span>Веса модели всего <b>${fmt.gb(r.weights_gb)}</b></span>
    <span>KV-кэш всего <b>${fmt.gb(r.kv_cache_gb)}</b> (${nf(r.kv_per_token_kb, 1)} КБ/токен${r.state_gb ? `, state ${fmt.gb(r.state_gb)}` : ""})</span>
    <span>Загрузка GPU <b>${Math.round((100 * (p.weights_gb + p.kv_cache_gb + p.overhead_gb)) / p.vram_gb)}%</b></span>`;
}

function renderCluster(r) {
  const el = $("#cluster");
  if (!r.fits) {
    el.innerHTML = `<span class="muted">Нет подходящей раскладки — увеличьте «Макс. GPU на реплику», уменьшите контекст или параллельность, либо выберите другую точность.</span>`;
    $("#clusterHint").textContent = "";
    return;
  }
  const gpn = r.gpu.gpus_per_node;
  const p = r.per_gpu;
  const h = (v) => ((100 * v) / p.vram_gb).toFixed(2) + "%";
  const gpuCell = (label, title) =>
    `<div class="gpu" title="${title}">
       <i class="gw" style="height:${h(p.weights_gb)}"></i>
       <i class="gk" style="height:${h(p.kv_cache_gb)}"></i>
       <i class="go" style="height:${h(p.overhead_gb)}"></i>
       <i style="height:${h(Math.max(p.free_gb, 0))}"></i>
       <i class="gr" style="height:${h(p.reserved_gb)}"></i>
       <span>${label}</span>
     </div>`;
  const idle = `<div class="gpu idle" title="свободный слот"><span>—</span></div>`;

  // Реплики по нодам: при TP меньше ноды упаковываем плотно, иначе — целыми нодами.
  const nodes = [];
  if (r.gpus_per_replica < gpn) {
    let slots = [];
    for (let rep = 0; rep < r.replicas; rep++) {
      for (let i = 0; i < r.gpus_per_replica; i++) {
        slots.push(gpuCell("Р" + (rep + 1), `Реплика ${rep + 1} · TP-ранг ${i}`));
        if (slots.length === gpn) { nodes.push({ title: "", slots }); slots = []; }
      }
    }
    if (slots.length) {
      while (slots.length < gpn) slots.push(idle);
      nodes.push({ title: "", slots });
    }
  } else {
    for (let rep = 0; rep < r.replicas; rep++) {
      for (let s = 0; s < r.pp; s++) {
        const slots = [];
        for (let i = 0; i < gpn; i++) slots.push(gpuCell("Р" + (rep + 1), `Реплика ${rep + 1} · стадия ${s + 1} · TP-ранг ${i}`));
        nodes.push({ title: r.pp > 1 ? `Р${rep + 1} · стадия ${s + 1}` : `Р${rep + 1}`, slots });
      }
    }
  }
  const MAX = 12;
  el.innerHTML =
    nodes.slice(0, MAX).map((n, i) => `
      <div class="node">
        <div class="node-title"><span>Нода ${i + 1}</span><span>${n.title}</span></div>
        <div class="gpus" style="grid-template-columns:repeat(${Math.min(gpn, 4)},44px)">${n.slots.join("")}</div>
      </div>`).join("") +
    (nodes.length > MAX ? `<div class="more">и ещё ${nodes.length - MAX} нод</div>` : "");
  $("#clusterHint").textContent =
    `${r.total_nodes} нод · ${r.total_gpus} GPU · занято ${fmt.gb(p.weights_gb + p.kv_cache_gb + p.overhead_gb)} из ${fmt.gb(p.vram_gb)} на карту`;
}

function renderCompare(list, selected) {
  const fits = list.filter((r) => r.fits && r.cost_per_1m_tokens);
  const maxC = Math.max(1, ...fits.map((r) => r.cost_per_1m_tokens));
  const bestId = list[0]?.fits ? list[0].gpu.id : null;
  $("#compare").innerHTML = list.map((r) => {
    const fill = r.fits
      ? `<div class="cmp-fill" style="width:${(100 * r.cost_per_1m_tokens) / maxC}%">${fmt.rub(r.cost_per_1m_tokens)}</div>`
      : `<div class="cmp-fill no">не помещается (по памяти нужно не меньше ${r.min_gpus_by_memory})</div>`;
    const meta = r.fits
      ? `${r.native ? "" : "⚠ эмуляция · "}<b>${r.total_gpus} GPU</b> · TP${r.tp}×PP${r.pp} · ${fmt.big(r.tokens_per_hour)} ток/ч · ${fmt.rub(r.cost_per_hour)}/ч`
      : "";
    return `<div class="cmp-row ${r.gpu.id === selected ? "sel" : ""}" data-gpu="${esc(r.gpu.id)}">
      <div class="cmp-name">${esc(r.gpu.name)}${r.gpu.id === bestId ? '<span class="best">лучший</span>' : ""}
        <small>${nf(r.gpu.vram_gb)} ГБ · ${nf(r.gpu.bandwidth_gbs / 1000, 1)} ТБ/с</small></div>
      <div class="cmp-track">${fill}</div>
      <div class="cmp-meta">${meta}</div>
    </div>`;
  }).join("");
  $$("#compare .cmp-row").forEach((row) =>
    row.addEventListener("click", () => {
      $("#f-gpu").value = row.dataset.gpu;
      calculate();
    })
  );
}

/* ---------- prefill / decode ---------- */
async function renderPD() {
  const req = formRequest();
  let r;
  try {
    r = await api("POST", "/api/pd", req);
  } catch (e) {
    return toast("Ошибка: " + e.message);
  }
  const q = r.request;
  $("#pdHint").textContent =
    `${fmt.tok(q.input_tokens)} → ${fmt.tok(q.output_tokens)} токенов · decode: ${q.replicas} реплик × ${q.concurrency} параллельных · цель TTFT ${q.ttft_slo_ms ? fmt.ms(q.ttft_slo_ms) : "нет"}`;

  const card = (title, o, cls = "") => {
    if (!o.feasible) {
      return `<div class="pd-card bad"><div class="pd-title">${title}</div><div class="muted">${esc(o.reason)}</div></div>`;
    }
    const disagg = o.mode !== "совмещённый";
    return `<div class="pd-card ${cls}">
      <div class="pd-title">${title}${o.meets_slo ? "" : '<span class="badge est">TTFT выше цели</span>'}</div>
      <div class="pd-sub">${esc(o.label)}</div>
      <div class="pd-big">${fmt.rub(o.cost_per_1m_output)} <small>за 1 млн выходных токенов</small></div>
      <div class="mrows">
        <span>GPU всего</span><b>${o.total_gpus}${disagg ? ` (prefill ${o.prefill_gpus} + decode ${o.decode_gpus})` : ""}</b>
        <span>Стоимость в час</span><b>${fmt.rub(o.cost_per_hour)}${o.network_cost_per_hour ? ` · сеть ${fmt.rub(o.network_cost_per_hour)}` : ""}</b>
        <span>Генерация в час</span><b>${fmt.big(o.output_tokens_per_hour)} токенов</b>
        <span>Время до 1-го токена</span><b>${fmt.ms(o.ttft_ms)}</b>
        ${disagg ? `<span>KV на запрос</span><b>${nf(o.kv_per_request_mb)} МБ · ${fmt.ms(o.transfer_ms)}</b>
        <span>Загрузка сети</span><b>${fmt.pct(o.net_util)}</b>` : ""}
        <span>Один запрос</span><b>${fmt.rubTiny(o.cost_per_request)}</b>
      </div>
      ${o.notes?.length ? `<ul class="pd-notes">${o.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
    </div>`;
  };
  const sav = r.savings_pct;
  $("#pdSummary").innerHTML =
    card("Совмещённый режим", r.colocated) +
    card("Раздельный P/D", r.disagg) +
    card("Лучший вариант", r.best, "best-card") +
    `<div class="pd-verdict ${sav > 0.5 ? "good" : ""}">${
      r.best.mode === "совмещённый"
        ? "Раздельные пулы не окупаются: prefill занимает малую долю GPU-времени, а лишние GPU и сеть стоят дороже. P/D имеет смысл ради стабильной задержки между токенами."
        : `Раздельный P/D дешевле совмещённого на <b>${nf(sav, 1)}%</b> за 1 млн токенов.`
    }</div>`;

  // поток: prefill → сеть → decode
  const o = r.disagg.feasible ? r.disagg : r.alternatives.find((a) => a.mode !== "совмещённый" && a.feasible);
  $("#pdFlow").innerHTML = o ? `
    <div class="flow-box"><b>Prefill</b><span>${o.prefill_instances} × ${o.prefill_gpus_per_instance} ${esc(o.prefill_gpu_name)}</span><small>${o.prefill_gpus} GPU</small></div>
    <div class="flow-link"><span>${esc(o.interconnect_name)}</span><i></i><small>KV ${nf(o.kv_per_request_mb)} МБ за ${fmt.ms(o.transfer_ms)} · загрузка ${fmt.pct(o.net_util)}</small></div>
    <div class="flow-box"><b>Decode</b><span>${o.decode_replicas} × ${o.decode_gpus_per_replica} ${esc(o.decode_gpu_name)}</span><small>${o.decode_gpus} GPU</small></div>` : "";

  const feas = r.alternatives.filter((a) => a.feasible);
  const maxC = Math.max(1, ...feas.map((a) => a.cost_per_1m_output));
  $("#pdTable").innerHTML = `<thead><tr>
      <th>Вариант</th><th>Prefill GPU</th><th>Сеть</th><th>GPU (P + D)</th><th>KV, передача</th><th>Сеть, загрузка</th>
      <th>TTFT</th><th>₽/час</th><th style="min-width:180px">₽ за 1 млн токенов</th></tr></thead><tbody>` +
    r.alternatives.map((a) => {
      if (!a.feasible) {
        return `<tr class="off"><td>${esc(a.mode)}</td><td>${esc(a.prefill_gpu_name || "—")}</td><td>${esc(a.interconnect_name || "—")}</td>
          <td colspan="6" class="muted small">${esc(a.reason)}</td></tr>`;
      }
      const dis = a.mode !== "совмещённый";
      return `<tr class="${a.best ? "best-row" : ""} ${a.meets_slo ? "" : "slo"}" title="${esc((a.notes || []).join("\n"))}">
        <td>${esc(a.mode)}${a.best ? '<span class="best">лучший</span>' : ""}</td>
        <td>${dis ? esc(a.prefill_gpu_name) : "—"}</td>
        <td>${dis ? esc(a.interconnect_name) + (a.rdma ? "" : ' <span class="badge est">без RDMA</span>') : "—"}</td>
        <td>${dis ? `${a.prefill_gpus} + ${a.decode_gpus}` : a.total_gpus}</td>
        <td>${dis ? `${nf(a.kv_per_request_mb)} МБ · ${fmt.ms(a.transfer_ms)}` : "—"}</td>
        <td>${dis ? fmt.pct(a.net_util) : "—"}</td>
        <td>${fmt.ms(a.ttft_ms)}${a.meets_slo ? "" : " ⚠"}</td>
        <td>${fmt.rub(a.cost_per_hour)}</td>
        <td><div class="cell-bar"><i style="width:${(100 * a.cost_per_1m_output) / maxC}%"></i><span>${fmt.rub(a.cost_per_1m_output)}</span></div></td>
      </tr>`;
    }).join("") + `</tbody>`;
}

/* ---------- модели ---------- */
$("#m-sort").addEventListener("input", renderModels);

async function renderModels() {
  let list;
  try {
    list = await api("GET", "/api/models/metrics");
  } catch (e) {
    return toast("Ошибка: " + e.message);
  }
  const key = $("#m-sort").value;
  const val = (x) => (key === "native" ? x.weights_gb.native : key === "kv" ? x.kv_gb_per_1m_tokens : x[key]);
  list.sort((a, b) => val(b) - val(a));
  const byId = Object.fromEntries(cfg.models.map((m) => [m.id, m]));
  $("#mcards").innerHTML = list.map((x) => {
    const m = byId[x.id];
    const w = x.weights_gb;
    return `<div class="mcard" data-id="${esc(x.id)}">
      <div class="mcard-top">
        <div>
          <div class="mcard-name">${esc(x.name)}${verifiedBadge(x.verified)}</div>
          <div class="mcard-vendor">${esc(m.vendor)} · ${archLabel(m)} · ${m.layers} слоёв${m.kv_layers ? ` (${m.kv_layers} с KV)` : ""}</div>
        </div>
        ${x.source ? `<a class="mcard-src" href="${esc(x.source)}" target="_blank" rel="noopener">Hugging Face ↗</a>` : ""}
      </div>
      <div class="hero3">
        <div><div class="v">${fmt.params(x.total_params_b)}</div><div class="l">параметров</div></div>
        <div><div class="v">${fmt.params(x.active_params_b)} <small>${nf(x.active_pct, x.active_pct < 10 ? 1 : 0)}%</small></div><div class="l">активных</div></div>
        <div><div class="v">${fmt.gb(w.native)}</div><div class="l">чекпоинт ${esc(x.native_precision)}</div></div>
      </div>
      <div class="sparsity" title="доля активных параметров"><i style="width:${Math.max(x.active_pct, 1.5)}%"></i></div>
      <div class="mrows">
        <span>Веса FP8 / FP4</span><b>${fmt.gb(w.fp8)} / ${fmt.gb(w.fp4)}</b>
        <span>Эксперты</span><b>${x.experts ? `${x.experts_per_tok} из ${x.experts}` : "плотная модель"}</b>
        <span>KV на токен (FP8)</span><b>${fmt.bytes(x.kv_bytes_per_token.fp8)}</b>
        <span>KV на 1 млн токенов</span><b>${fmt.gb(x.kv_gb_per_1m_tokens)}</b>
        <span>State на запрос</span><b>${x.state_mb_per_seq ? nf(x.state_mb_per_seq) + " МБ" : "—"}</b>
        <span>Макс. контекст</span><b>${fmt.tok(x.max_context)}</b>
        <span>1 запрос на макс. контексте</span><b>${fmt.gb(x.kv_gb_per_seq_at_max_context)}</b>
      </div>
      ${m.note ? `<div class="mnote">${esc(m.note)}</div>` : ""}
    </div>`;
  }).join("");
  $$("#mcards .mcard").forEach((c) =>
    c.addEventListener("click", (e) => {
      if (e.target.closest("a")) return;
      $("#f-model").value = c.dataset.id;
      showTab("calc");
      calculate();
    })
  );
}

/* ---------- матрица ---------- */
async function renderMatrix() {
  const req = formRequest();
  delete req.model;
  delete req.gpu;
  $("#matrixHint").textContent =
    `${PRECISION_LABELS[req.precision] || req.precision} · KV ${req.kv_dtype} · контекст ${fmt.tok(req.context_len)} · ` +
    `${req.concurrency} параллельных · реплик: ${req.replicas}`;
  let mx;
  try {
    mx = await api("POST", "/api/matrix", req);
  } catch (e) {
    return toast("Ошибка: " + e.message);
  }
  const gpus = mx.gpus.map((id) => cfg.gpus.find((g) => g.id === id));
  const models = mx.models.map((id) => cfg.models.find((m) => m.id === id));
  const all = Object.values(mx.cells).flatMap((row) => Object.values(row)).filter((c) => c.fits);
  const maxLog = Math.log2(Math.max(2, ...all.map((c) => c.gpus_per_replica)));
  const color = (n) => {
    const t = Math.log2(n) / maxLog;
    return `background:hsl(206 72% ${93 - t * 40}%);color:${t > 0.55 ? "#fff" : "inherit"}`;
  };
  $("#matrix").innerHTML =
    `<thead><tr><th style="text-align:left">Модель</th>${gpus.map((g) => `<th>${esc(g.name)}</th>`).join("")}</tr></thead><tbody>` +
    models.map((m) => `<tr><td class="mname">${esc(m.name)}${m.verified ? "" : '<span class="badge est">оценка</span>'}</td>` +
      gpus.map((g) => {
        const c = mx.cells[m.id][g.id];
        if (!c.fits) return `<td class="cell no" data-m="${esc(m.id)}" data-g="${esc(g.id)}">✕</td>`;
        return `<td class="cell" style="${color(c.gpus_per_replica)}" data-m="${esc(m.id)}" data-g="${esc(g.id)}">
          ${c.gpus_per_replica}<small>TP${c.tp}×PP${c.pp}${c.cost_per_1m_tokens ? " · " + fmt.rub(c.cost_per_1m_tokens) + "/млн" : ""}</small></td>`;
      }).join("") + `</tr>`).join("") + `</tbody>`;
  $$("#matrix td.cell").forEach((td) =>
    td.addEventListener("click", () => {
      $("#f-model").value = td.dataset.m;
      $("#f-gpu").value = td.dataset.g;
      showTab("calc");
      calculate();
    })
  );
}

/* ---------- настройки ---------- */
const LABELS = {
  id: "ID", name: "Название", arch: "Архитектура", vram_gb: "VRAM, ГБ", bandwidth_gbs: "HBM, ГБ/с",
  tflops_bf16: "TFLOPS BF16", precisions: "Форматы", gpus_per_node: "GPU в ноде", interconnect: "Связь в ноде",
  price_per_hour: "₽/час за GPU", available: "Доступно в MIP",
  vendor: "Разработчик", total_params_b: "Всего, млрд", active_params_b: "Активных, млрд", layers: "Слоёв",
  kv_layers: "Слоёв с KV", attention: "Внимание", kv_heads: "KV-голов", head_dim: "Размер головы",
  kv_lora_rank: "MLA: ранг", qk_rope_dim: "MLA: RoPE", kv_bytes_per_token: "KV, Б/токен (вручную)",
  kv_fixed: "KV в своём формате", kv_extra_bytes_per_token: "Доп. KV, Б/токен", state_mb_per_seq: "State, МБ/запрос",
  hidden_size: "Hidden", attn_topk: "Top-k внимания", experts: "Экспертов", experts_per_tok: "Активных экспертов",
  max_context: "Макс. контекст", native_precision: "Нативный формат", checkpoint_gb: "Чекпоинт, ГБ",
  verified: "Проверено", source: "Источник", note: "Заметка",
  gbps_per_gpu: "Гбит/с на GPU", efficiency: "Эффективность", rdma: "RDMA", latency_us: "Задержка, мкс",
  price_per_node_hour: "₽/час за ноду", same_node_only: "Только внутри ноды",
};
const GPU_COLS = [
  ["id", "text", "w-m"], ["name", "text", "w-l"], ["arch", "text", "w-m"], ["vram_gb", "num"],
  ["bandwidth_gbs", "num"], ["tflops_bf16", "num"], ["precisions", "list", "w-m"], ["gpus_per_node", "num"],
  ["interconnect", "text", "w-l"], ["price_per_hour", "num"], ["available", "num"],
];
const NET_COLS = [
  ["id", "text", "w-m"], ["name", "text", "w-l"], ["gbps_per_gpu", "num"], ["efficiency", "num"], ["rdma", "bool"],
  ["latency_us", "num"], ["price_per_node_hour", "num"], ["same_node_only", "bool"],
];
const MODEL_COLS = [
  ["id", "text", "w-m"], ["name", "text", "w-l"], ["vendor", "text", "w-m"], ["total_params_b", "num"],
  ["active_params_b", "num"], ["layers", "num"], ["kv_layers", "num"], ["attention", ["gqa", "mla", "custom"]],
  ["kv_heads", "num"], ["head_dim", "num"], ["kv_lora_rank", "num"], ["qk_rope_dim", "num"],
  ["kv_bytes_per_token", "num"], ["kv_fixed", "bool"], ["kv_extra_bytes_per_token", "num"], ["state_mb_per_seq", "num"],
  ["hidden_size", "num"], ["attn_topk", "num"], ["experts", "num"], ["experts_per_tok", "num"],
  ["max_context", "num"], ["native_precision", PRECISIONS], ["checkpoint_gb", "num"],
  ["verified", "bool"], ["source", "text", "w-l"], ["note", "text", "w-l"],
];
const PLATFORM_FIELDS = [
  ["name", "text", "Название платформы"], ["currency", "text", "Валюта"],
  ["gpu_memory_utilization", "num", "Доля VRAM для движка"], ["overhead_per_gpu_gb", "num", "Накладные на GPU, ГБ"],
  ["weights_overhead_pct", "num", "Надбавка к весам, %"], ["tp_sizes", "intlist", "Размеры TP"],
  ["max_gpus_per_replica", "num", "Макс. GPU на реплику"], ["mla_dp_attention", "bool", "MLA: делить KV между TP (DP-attention)"],
  ["decode_efficiency", "num", "Эффективность HBM на decode"], ["compute_efficiency", "num", "Эффективность вычислений на decode"],
  ["prefill_efficiency", "num", "Эффективность вычислений на prefill (MFU)"], ["avg_context_fill", "num", "Средняя заполненность контекста"],
  ["layer_overhead_us", "num", "Задержка на слой, мкс"], ["prefill_batch", "num", "Prompt-ов в prefill-инстансе"],
  ["defaults.precision", REQ_PRECISIONS, "По умолчанию: точность весов"], ["defaults.kv_dtype", KV_DTYPES, "По умолчанию: формат KV"],
  ["defaults.context_len", "num", "По умолчанию: контекст"], ["defaults.concurrency", "num", "По умолчанию: параллельных запросов"],
  ["defaults.replicas", "num", "По умолчанию: реплик"], ["defaults.input_tokens", "num", "По умолчанию: токенов на входе"],
  ["defaults.output_tokens", "num", "По умолчанию: токенов на выходе"], ["defaults.interconnect", "text", "По умолчанию: интерконнект (id)"],
  ["defaults.ttft_slo_ms", "num", "По умолчанию: цель TTFT, мс"],
];

const getPath = (o, p) => p.split(".").reduce((a, k) => a?.[k], o);
function setPath(o, p, v) {
  const ks = p.split(".");
  const last = ks.pop();
  ks.reduce((a, k) => (a[k] ??= {}), o)[last] = v;
}

function inputFor(type, value, cls = "") {
  if (Array.isArray(type))
    return `<select>${type.map((o) => `<option value="${o}" ${o === value ? "selected" : ""}>${PRECISION_LABELS[o] || o}</option>`).join("")}</select>`;
  if (type === "bool") return `<input type="checkbox" ${value ? "checked" : ""}>`;
  if (type === "num") return `<input type="number" step="any" class="${cls}" value="${value ?? ""}">`;
  if (type === "list" || type === "intlist")
    return `<input type="text" class="${cls}" value="${esc((value || []).join(", "))}">`;
  return `<input type="text" class="${cls}" value="${esc(value ?? "")}">`;
}

function readInput(el, type) {
  if (type === "bool") return el.checked;
  if (type === "num") return el.value === "" ? 0 : +el.value;
  if (type === "list") return el.value.split(",").map((s) => s.trim()).filter(Boolean);
  if (type === "intlist") return el.value.split(",").map((s) => parseInt(s, 10)).filter((n) => n > 0);
  return el.value;
}

function markDirty() {
  const s = $("#saveStatus");
  s.className = "status dirty";
  s.textContent = "Есть несохранённые изменения";
  $("#rawJson").value = JSON.stringify(draft, null, 2);
}

function renderSettings() {
  if (!draft) draft = structuredClone(cfg);
  draft.interconnects ??= [];
  $("#platformForm").innerHTML = PLATFORM_FIELDS.map(([k, t, label]) =>
    t === "bool"
      ? `<label class="chk" data-k="${k}" title="${k}">${inputFor(t, getPath(draft.platform, k))} ${label}</label>`
      : `<label data-k="${k}" title="${k}">${label}${inputFor(t, getPath(draft.platform, k))}</label>`
  ).join("");
  $$("#platformForm label").forEach((lab) => {
    const k = lab.dataset.k;
    const t = PLATFORM_FIELDS.find((f) => f[0] === k)[1];
    const el = $("input,select", lab);
    el.addEventListener("input", () => { setPath(draft.platform, k, readInput(el, t)); markDirty(); });
  });
  renderEditTable("#gpusTable", "gpus", GPU_COLS);
  renderEditTable("#netTable", "interconnects", NET_COLS);
  renderEditTable("#modelsTable", "models", MODEL_COLS);
  $("#rawJson").value = JSON.stringify(draft, null, 2);
  loadDocs();
}

async function loadDocs() {
  try {
    const [p, w] = await Promise.all([api("GET", "/api/profiles"), api("GET", "/api/workloads")]);
    $("#rawProfiles").value = JSON.stringify(p, null, 2);
    $("#rawWorkloads").value = JSON.stringify(w, null, 2);
  } catch (e) {
    toast("Ошибка: " + e.message);
  }
}

$$("[data-save-doc]").forEach((b) =>
  b.addEventListener("click", async () => {
    const kind = b.dataset.saveDoc;
    const area = kind === "profiles" ? "#rawProfiles" : "#rawWorkloads";
    const status = kind === "profiles" ? "#profStatus" : "#wlStatus";
    try {
      await api("PUT", "/api/" + kind, JSON.parse($(area).value));
      $(status).className = "status";
      $(status).textContent = "Сохранено ✓";
      limitsState.loaded = false;
      toast(kind === "profiles" ? "Профили сохранены" : "Нагрузки сохранены");
    } catch (e) {
      $(status).className = "status err";
      $(status).textContent = e.message;
    }
  })
);

function renderEditTable(sel, key, cols) {
  const t = $(sel);
  t.innerHTML =
    `<thead><tr>${cols.map(([c]) => `<th title="${c}">${LABELS[c] || c}</th>`).join("")}<th></th></tr></thead><tbody>` +
    draft[key].map((row, i) =>
      `<tr data-i="${i}">${cols.map(([c, ty, cls]) => `<td data-c="${c}">${inputFor(ty, row[c], cls)}</td>`).join("")}
       <td><button class="btn del" title="Удалить">✕</button></td></tr>`).join("") + `</tbody>`;
  $$("tbody tr", t).forEach((tr) => {
    const i = +tr.dataset.i;
    $$("td[data-c]", tr).forEach((td) => {
      const [c, ty] = cols.find((x) => x[0] === td.dataset.c);
      const el = $("input,select", td);
      el.addEventListener("input", () => { draft[key][i][c] = readInput(el, ty); markDirty(); });
    });
    $(".del", tr).addEventListener("click", () => {
      draft[key].splice(i, 1);
      markDirty();
      renderSettings();
    });
  });
}

const NEW_ROWS = {
  gpus: (n) => ({ id: "gpu-" + n, name: "Новая GPU", arch: "", vram_gb: 80, bandwidth_gbs: 3000, tflops_bf16: 0, precisions: ["bf16", "fp8"], gpus_per_node: 8, interconnect: "", price_per_hour: 0, available: 0 }),
  interconnects: (n) => ({ id: "net-" + n, name: "Новая сеть", gbps_per_gpu: 200, efficiency: 0.7, rdma: true, latency_us: 10, price_per_node_hour: 0, same_node_only: false }),
  models: (n) => ({ id: "model-" + n, name: "Новая модель", vendor: "", total_params_b: 70, active_params_b: 70, layers: 80, attention: "gqa", kv_heads: 8, head_dim: 128, max_context: 131072, native_precision: "bf16", verified: false, note: "" }),
};
$$("[data-add]").forEach((b) =>
  b.addEventListener("click", () => {
    const key = b.dataset.add;
    draft[key].push(NEW_ROWS[key](draft[key].length + 1));
    markDirty();
    renderSettings();
  })
);

$("#btnSave").addEventListener("click", async () => {
  const s = $("#saveStatus");
  try {
    cfg = await api("PUT", "/api/config", draft);
    draft = structuredClone(cfg);
    s.className = "status";
    s.textContent = "Сохранено ✓";
    fillSelects();
    calculate();
    toast("Настройки сохранены");
  } catch (e) {
    s.className = "status err";
    s.textContent = e.message;
  }
});
$("#btnReload").addEventListener("click", async () => {
  cfg = await api("GET", "/api/config");
  draft = structuredClone(cfg);
  $("#saveStatus").textContent = "";
  renderSettings();
});
$("#btnApplyRaw").addEventListener("click", () => {
  try {
    draft = JSON.parse($("#rawJson").value);
    renderSettings();
    markDirty();
  } catch (e) {
    toast("Ошибка в JSON: " + e.message);
  }
});

/* ---------- MCP ---------- */
async function renderMCP() {
  const url = location.origin + "/mcp";
  $("#mcpUrl").textContent = url;
  $("#mcpClaude").textContent = `claude mcp add --transport http mip-gpu-calc ${url}`;
  $("#mcpStdio").textContent = JSON.stringify({
    mcpServers: { "mip-gpu-calc": { command: "/путь/к/gpucalc", args: ["-stdio", "-config", "/путь/к/config.json"] } },
  }, null, 2);
  try {
    const r = await api("POST", "/mcp", { jsonrpc: "2.0", id: 1, method: "tools/list" });
    $("#mcpTools").innerHTML = r.result.tools
      .map((t) => `<div class="tool"><b>${esc(t.name)}</b><div>${esc(t.description)}</div></div>`).join("");
  } catch (e) {
    $("#mcpTools").textContent = e.message;
  }
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- лимиты SGLang ---------- */
const limitsState = { loaded: false, profiles: [], workloads: [], edited: false, last: null };
const GPU_SHORT = (id) => (cfg.gpus.find((g) => g.id === id)?.name || id);

async function initLimits() {
  if (!limitsState.loaded) {
    try {
      const [p, w] = await Promise.all([api("GET", "/api/profiles"), api("GET", "/api/workloads")]);
      limitsState.profiles = p.profiles || [];
      limitsState.workloads = w.workloads || [];
      limitsState.loaded = true;
    } catch (e) {
      return toast("Ошибка: " + e.message);
    }
    const keep = store.get("limits") || {};
    const count = {};
    limitsState.profiles.forEach((p) => (count[p.model] = (count[p.model] || 0) + 1));
    const models = [...cfg.models].sort((a, b) => (count[b.id] ? 1 : 0) - (count[a.id] ? 1 : 0));
    $("#l-model").innerHTML = models
      .map((m) => `<option value="${esc(m.id)}">${esc(m.name)}${count[m.id] ? ` · профилей: ${count[m.id]}` : " · без замеров"}</option>`).join("");
    $("#l-gpu").innerHTML = cfg.gpus.map((g) => `<option value="${esc(g.id)}">${esc(g.name)}</option>`).join("");
    $("#l-workload").innerHTML = limitsState.workloads
      .map((w) => `<option value="${esc(w.id)}">${esc(w.name)}</option>`).join("");
    $("#l-model").value = keep.model && cfg.models.some((m) => m.id === keep.model) ? keep.model : "deepseek-v4.1-flash";
    fillProfiles(keep.profile);
    if (keep.gpu) $("#l-gpu").value = keep.gpu;
    $("#l-workload").value = keep.workload && limitsState.workloads.some((w) => w.id === keep.workload) ? keep.workload : "agent";
    for (const [id, k] of [["#l-servers", "servers"], ["#l-gpn", "gpn"], ["#l-ttft", "ttft"], ["#l-tpot", "tpot"], ["#l-headroom", "headroom"]]) {
      if (keep[k] !== undefined && keep[k] !== null) $(id).value = keep[k];
    }
    fillWorkloadFields();
  }
  scheduleLimits();
}

function fillProfiles(keepId) {
  const model = $("#l-model").value;
  const list = limitsState.profiles.filter((p) => p.model === model);
  $("#l-profile").innerHTML =
    list.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}${p.measurements?.length ? "" : " · без замеров"}</option>`).join("") +
    `<option value="">Без профиля — оценка по roofline</option>`;
  $("#l-profile").value = keepId && list.some((p) => p.id === keepId) ? keepId : (list[0]?.id || "");
  onProfileChange();
}

function onProfileChange() {
  const p = limitsState.profiles.find((x) => x.id === $("#l-profile").value);
  $("#l-gpu-wrap").hidden = !!p;
  if (p) {
    const g = cfg.gpus.find((x) => x.id === p.gpu);
    $("#l-gpn").value = g?.gpus_per_node || 8;
    const e = p.engine;
    $("#l-profile-meta").innerHTML =
      `<b>${e.gpus}×${esc(GPU_SHORT(p.gpu))}</b> на инстанс · TP ${e.tp}${e.ep ? " · EP " + e.ep : ""}${e.attn_dp ? " · DP-attn " + e.attn_dp : ""}<br>` +
      `SGLang ${esc(e.version || "—")} · замеров: ${p.measurements?.length || 0}` +
      (p.verified ? `<span class="badge ok">проверено</span>` : `<span class="badge est">не проверено</span>`) +
      (p.note ? `<br><span>${esc(p.note)}</span>` : "");
  } else {
    $("#l-profile-meta").innerHTML = "Скорость будет оценена по характеристикам GPU (roofline), раскладка — по калькулятору памяти. Это грубая оценка.";
  }
}

function fillWorkloadFields() {
  const w = limitsState.workloads.find((x) => x.id === $("#l-workload").value);
  if (!w) return;
  const set = (d, a, b) => {
    const fixed = d.distribution === "fixed";
    const buckets = d.distribution === "buckets";
    $(a).value = fixed ? d.value : buckets ? "" : d.p50;
    $(b).value = fixed ? d.value : buckets ? "" : d.p99;
    $(a).disabled = $(b).disabled = buckets;
  };
  set(w.input, "#l-in50", "#l-in99");
  set(w.output, "#l-out50", "#l-out99");
  $("#l-hit").value = Math.round(100 * (w.prefix_cache_hit || 0));
  $("#l-wl-note").textContent = (w.stub ? "Заглушка. " : "") + (w.note || "") +
    (w.input.distribution === "buckets" ? " Гистограмма — правьте во вкладке «Настройки»." : "");
  limitsState.edited = false;
}

function limitsRequest() {
  const w = limitsState.workloads.find((x) => x.id === $("#l-workload").value);
  const req = {
    profile: $("#l-profile").value || undefined,
    model: $("#l-profile").value ? undefined : $("#l-model").value,
    gpu: $("#l-profile").value ? undefined : $("#l-gpu").value,
    ttft_p99_ms: (+$("#l-ttft").value || 40) * 1000,
    tpot_p99_ms: +$("#l-tpot").value || undefined,
    servers: +$("#l-servers").value || 1,
    gpus_per_server: +$("#l-gpn").value || undefined,
    headroom: (+$("#l-headroom").value || 80) / 100,
  };
  if (limitsState.edited && w && w.input.distribution !== "buckets") {
    const dist = (a, b) => {
      const p50 = +$(a).value, p99 = +$(b).value;
      return p50 === p99 ? { distribution: "fixed", value: p50 } : { distribution: "lognormal", p50, p99: Math.max(p99, p50) };
    };
    req.workload_inline = {
      id: "inline", name: "своя нагрузка", stub: false,
      input: dist("#l-in50", "#l-in99"), output: dist("#l-out50", "#l-out99"),
      prefix_cache_hit: Math.min(+$("#l-hit").value || 0, 99) / 100,
    };
  } else {
    req.workload = $("#l-workload").value;
  }
  return req;
}

let limitsTimer;
function scheduleLimits() {
  clearTimeout(limitsTimer);
  limitsTimer = setTimeout(renderLimits, 250);
}
$("#l-model").addEventListener("input", () => { fillProfiles(); scheduleLimits(); });
$("#l-profile").addEventListener("input", () => { onProfileChange(); scheduleLimits(); });
$("#l-workload").addEventListener("input", () => { fillWorkloadFields(); scheduleLimits(); });
["#l-in50", "#l-in99", "#l-out50", "#l-out99", "#l-hit"].forEach((id) =>
  $(id).addEventListener("input", () => { limitsState.edited = true; scheduleLimits(); }));
["#l-gpu", "#l-servers", "#l-gpn", "#l-ttft", "#l-tpot", "#l-headroom"].forEach((id) => $(id).addEventListener("input", scheduleLimits));

async function renderLimits() {
  const req = limitsRequest();
  store.set("limits", {
    model: $("#l-model").value, profile: $("#l-profile").value, gpu: $("#l-gpu").value, workload: $("#l-workload").value,
    servers: $("#l-servers").value, gpn: $("#l-gpn").value, ttft: $("#l-ttft").value, tpot: $("#l-tpot").value, headroom: $("#l-headroom").value,
  });
  $("#l-kpis").innerHTML = `<div class="kpi"><div class="label">Расчёт</div><div class="value">…</div><div class="sub">симуляция очереди SGLang</div></div>`;
  let r;
  try {
    r = await api("POST", "/api/plan", req);
  } catch (e) {
    $("#l-kpis").innerHTML = `<div class="kpi bad" style="grid-column:1/-1"><div class="label">Ошибка</div><div class="value">${esc(e.message)}</div></div>`;
    return;
  }
  limitsState.last = r;
  $("#l-stubs").innerHTML = r.stubs.map((x) => `<div class="stub">${esc(x)}</div>`).join("");
  renderLimitsKPIs(r);
  renderLimitsChart(r);
  renderLimitsTable(r);
  renderLimitsValidation(r);
  renderLimitsLaunch(r);
  renderLimitsCompare(r, req);
}

const sec = (ms) => (ms >= 1000 ? nf(ms / 1000, ms < 10000 ? 1 : 0) + " с" : nf(ms) + " мс");

function renderLimitsKPIs(r) {
  const b = r.best;
  if (!b) {
    $("#l-kpis").innerHTML = `<div class="kpi bad" style="grid-column:1/-1"><div class="label">Цели недостижимы</div>
      <div class="value">${esc(r.notes.find((n) => n.startsWith("цели")) || "ни один лимит не подходит")}</div></div>`;
    return;
  }
  const wl = r.workload;
  $("#l-kpis").innerHTML =
    kpi("--max-running-requests", b.max_running_requests, `одновременно на инстанс · ограничивает: ${esc(b.limited_by)}`, "hero") +
    kpi("Лимит RPS на инстанс", nf(r.rps_limit_per_instance, 2), `${nf(r.request.headroom * 100)}% от предела ${nf(b.max_rps, 2)} запр/с`, "accent") +
    kpi("Инстансов", r.instances, `по ${r.profile.engine.gpus} GPU (${esc(GPU_SHORT(r.profile.gpu))})`) +
    kpi("Лимит RPS на кластер", nf(r.rps_limit_cluster, 1), `≈ ${nf(r.rps_limit_cluster * 3600)} запросов в час`, "accent") +
    kpi("TTFT на пределе", `${sec(b.at_max.ttft_p50_ms)} / ${sec(b.at_max.ttft_p99_ms)}`, `P50 / P99, цель P99 ≤ ${sec(r.request.ttft_p99_ms)}`) +
    kpi("TPOT на пределе", `${nf(b.at_max.tpot_p50_ms, 1)} / ${nf(b.at_max.tpot_p99_ms, 1)} мс`, `P50 / P99${r.request.tpot_p99_ms ? `, цель ≤ ${nf(r.request.tpot_p99_ms)} мс` : " — цель не задана"}`) +
    kpi("Генерация (кластер)", fmt.big(r.output_tokens_per_hour_cluster), `токенов в час при лимите · вход ~${fmt.tok(Math.round(r.workload_mean_input))}, выход ~${fmt.tok(Math.round(r.workload_mean_output))}`);
}

function renderLimitsChart(r) {
  const pts = r.curve || [];
  if (!pts.length) { $("#l-chart").innerHTML = ""; return; }
  const W = 760, H = 280, L = 56, R = 16, T = 14, B = 40;
  const xs = pts.map((p) => p.rps), ys = pts.map((p) => Math.max(p.ttft_p99_ms, 1));
  const slo = r.request.ttft_p99_ms;
  const xmax = Math.max(...xs) * 1.02;
  const ymin = Math.max(Math.min(...ys, slo) / 2, 10), ymax = Math.max(...ys, slo) * 1.6;
  const X = (v) => L + (v / xmax) * (W - L - R);
  const Y = (v) => T + (1 - (Math.log10(v) - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin))) * (H - T - B);
  const ticksY = [];
  for (let e = Math.floor(Math.log10(ymin)); e <= Math.ceil(Math.log10(ymax)); e++) {
    for (const m of [1, 3]) { const v = m * 10 ** e; if (v >= ymin && v <= ymax) ticksY.push(v); }
  }
  const ticksX = Array.from({ length: 6 }, (_, i) => (xmax * i) / 5);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${X(p.rps).toFixed(1)},${Y(Math.max(p.ttft_p99_ms, 1)).toFixed(1)}`).join("");
  const path50 = pts.map((p, i) => `${i ? "L" : "M"}${X(p.rps).toFixed(1)},${Y(Math.max(p.ttft_p50_ms, ymin)).toFixed(1)}`).join("");
  const lim = r.rps_limit_per_instance, mx = r.best?.max_rps;
  $("#l-chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="TTFT P99 от RPS">
    ${ticksY.map((v) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${sec(v)}</text>`).join("")}
    ${ticksX.map((v) => `<text x="${X(v)}" y="${H - B + 16}" text-anchor="middle">${nf(v, xmax < 5 ? 2 : 1)}</text>`).join("")}
    <line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>
    <text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">RPS на инстанс</text>
    <line class="slo" x1="${L}" x2="${W - R}" y1="${Y(slo)}" y2="${Y(slo)}"/>
    <text x="${W - R - 4}" y="${Y(slo) - 6}" text-anchor="end" style="fill:var(--bad)">цель TTFT P99 ${sec(slo)}</text>
    ${mx ? `<line class="lim" x1="${X(lim)}" x2="${X(lim)}" y1="${T}" y2="${H - B}"/><text x="${X(lim) + 4}" y="${T + 12}" style="fill:var(--ok)">лимит ${nf(lim, 2)}</text>` : ""}
    <path class="line2" d="${path50}"/>
    <path class="line" d="${path}"/>
    ${pts.map((p) => `<circle class="dot ${p.stable && p.ttft_p99_ms <= slo ? "" : "bad"}" cx="${X(p.rps)}" cy="${Y(Math.max(p.ttft_p99_ms, 1))}" r="4"><title>RPS ${nf(p.rps, 2)}: TTFT P99 ${sec(p.ttft_p99_ms)}, P50 ${sec(p.ttft_p50_ms)}, TPOT P99 ${nf(p.tpot_p99_ms, 1)} мс, одновременно ~${nf(p.mean_running)}${p.stable ? "" : " · очередь растёт"}</title></circle>`).join("")}
  </svg>`;
  $("#l-chart-hint").textContent = `--max-running-requests ${r.best?.max_running_requests} · сплошная — P99, пунктир — P50 · красные точки — цель нарушена`;
}

function renderLimitsTable(r) {
  const bestMRR = r.best?.max_running_requests;
  $("#l-table").innerHTML = `<thead><tr><th>--max-running-requests</th><th>Предел RPS на инстанс</th><th>На кластер</th>
    <th>TTFT P50</th><th>TTFT P99</th><th>TPOT P50</th><th>TPOT P99</th><th>Одновременно в среднем</th><th>Генерация, ток/с</th><th>Что ограничивает</th></tr></thead><tbody>` +
    r.rows.map((x) => x.feasible
      ? `<tr class="${x.max_running_requests === bestMRR ? "best-row" : ""}">
        <td><b>${x.max_running_requests}</b>${x.max_running_requests === bestMRR ? '<span class="best">рекомендуется</span>' : ""}</td>
        <td>${nf(x.max_rps, 2)}</td><td>${nf(x.max_rps * r.instances, 1)}</td>
        <td>${sec(x.at_max.ttft_p50_ms)}</td><td>${sec(x.at_max.ttft_p99_ms)}</td>
        <td>${nf(x.at_max.tpot_p50_ms, 1)} мс</td><td>${nf(x.at_max.tpot_p99_ms, 1)} мс</td>
        <td>${nf(x.at_max.mean_running)}</td><td>${nf(x.at_max.output_tok_s)}</td><td class="muted">${esc(x.limited_by)}</td></tr>`
      : `<tr class="off"><td><b>${x.max_running_requests}</b></td><td colspan="9" class="muted">цели недостижимы: ${esc(x.limited_by)}</td></tr>`).join("") + `</tbody>`;
}

function errCls(v) {
  const a = Math.abs(v);
  return a <= 20 ? "err-ok" : a <= 50 ? "err-mid" : "err-bad";
}

function renderLimitsValidation(r) {
  const v = r.validation || [];
  $("#l-fit-hint").textContent = v.length
    ? `средняя ошибка: TTFT ${nf(r.validation_ttft_mape_pct)}%, TPOT ${nf(r.validation_tpot_mape_pct)}% · источник скорости: ${r.fit.source}`
    : `замеров нет · источник скорости: ${r.fit.source}`;
  $("#l-valid").innerHTML = v.length
    ? `<thead><tr><th>Параллельность</th><th>Вход → выход</th><th>TTFT замер</th><th>TTFT модель</th><th>Ошибка</th><th>TPOT замер</th><th>TPOT модель</th><th>Ошибка</th></tr></thead><tbody>` +
      v.map((x) => `<tr><td>${x.concurrency}</td><td>${fmt.tok(x.isl)} → ${fmt.tok(x.osl)}</td>
        <td>${sec(x.measured_ttft_ms)}</td><td>${sec(x.pred_ttft_ms)}</td><td class="${errCls(x.err_ttft_pct)}">${x.err_ttft_pct > 0 ? "+" : ""}${nf(x.err_ttft_pct)}%</td>
        <td>${nf(x.measured_tpot_ms, 2)} мс</td><td>${nf(x.pred_tpot_ms, 2)} мс</td><td class="${errCls(x.err_tpot_pct)}">${x.err_tpot_pct > 0 ? "+" : ""}${nf(x.err_tpot_pct)}%</td></tr>`).join("") + `</tbody>`
    : "";
  const p = r.perf;
  $("#l-fit").innerHTML =
    `Сверка: симуляция того же бенчмарка (bench_serving, фиксированная параллельность). Если точек всего две, совпадение TPOT не независимая проверка, а подгонка.<br>` +
    `Подгонка: ${r.fit.details.map(esc).join("; ")}.<br>` +
    `Модель: шаг decode = ${nf(p.decode_base_ms, 2)} мс + n × (${nf(p.decode_per_req_ms, 3)} + ${nf(p.decode_per_req_per_kctx_ms, 4)} × контекст/1000) мс; ` +
    `prefill ≈ ${fmt.big(p.prefill_tok_s)} токенов/с, ${nf(p.prefill_overhead_ms)} мс на батч + ${nf(p.prefill_per_req_ms, 1)} мс на запрос.`;
}

function renderLimitsLaunch(r) {
  $("#l-launch").textContent = r.launch;
  const e = r.profile.engine, c = r.capacity;
  $("#l-engine").innerHTML = `
    <span>SGLang <b>${esc(e.version || "—")}</b></span>
    ${e.image ? `<span>образ <b>${esc(e.image)}</b></span>` : ""}
    <span>GPU на инстанс <b>${e.gpus}</b></span>
    <span>KV-пул <b>${fmt.big(c.kv_pool_tokens)}</b> токенов (${esc(c.source)})</span>
    ${c.state_slots ? `<span>слотов state <b>${c.state_slots}</b></span>` : ""}
    <span>веса на GPU <b>${fmt.gb(c.weights_gb_per_gpu)}</b></span>`;
  const src = r.profile.source;
  $("#l-source").hidden = !src;
  if (src) $("#l-source").href = src;
  $("#l-notes").innerHTML = (r.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
}

async function renderLimitsCompare(r, req) {
  const id = r.profile.id;
  const sib = id.endsWith("-engram-host") ? id.replace(/-engram-host$/, "") : id + "-engram-host";
  const other = limitsState.profiles.find((p) => p.id === sib);
  if (!other) { $("#l-compare-panel").hidden = true; return; }
  $("#l-compare-panel").hidden = false;
  $("#l-compare-title").textContent = "Engram на GPU или в RAM хоста";
  let o;
  try {
    o = await api("POST", "/api/plan", { ...req, profile: sib });
  } catch (e) {
    return;
  }
  const row = (x, label) => `<tr><td>${label}</td><td>${fmt.big(x.capacity.kv_pool_tokens)}</td><td>${fmt.gb(x.capacity.weights_gb_per_gpu)}</td>
    <td>${x.best ? x.best.max_running_requests : "—"}</td><td>${x.best ? nf(x.rps_limit_per_instance, 2) : "—"}</td><td>${x.best ? nf(x.rps_limit_cluster, 1) : "—"}</td>
    <td>${x.best ? sec(x.best.at_max.ttft_p99_ms) : "—"}</td><td>${x.best ? nf(x.best.at_max.kv_peak_pct, 1) + "%" : "—"}</td></tr>`;
  const [gpu, host] = id.endsWith("-engram-host") ? [o, r] : [r, o];
  $("#l-compare").innerHTML = `<thead><tr><th>Вариант</th><th>KV-пул, токенов</th><th>Веса на GPU</th><th>--max-running-requests</th>
    <th>Лимит RPS на инстанс</th><th>На кластер</th><th>TTFT P99</th><th>Пик занятости KV</th></tr></thead><tbody>` +
    row(gpu, "Engram на GPU (по умолчанию)") + row(host, "Engram в RAM хоста (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1)") + `</tbody>` +
    `<caption class="muted small" style="caption-side:bottom;text-align:left;padding-top:8px">Вынос Engram освобождает HBM под KV-пул. Если пик занятости KV далёк от 100%, лимиты не меняются. Влияние выноса на скорость не измерено — в профиле стоит множитель 1,0 (заглушка).</caption>`;
}

/* ---------- запуск ---------- */
(async function init() {
  cfg = await api("GET", "/api/config");
  fillSelects();
  await calculate();
  const tab = store.get("tab");
  document.body.dataset.tab = "calc";
  if (tab && tab !== "calc") showTab(tab);
})();
