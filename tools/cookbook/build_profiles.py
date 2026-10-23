"""Build profiles.default.json (SGLang engine profiles) from SGLang cookbook snippets + manual sources."""
import json
import re
import sys

src = json.load(open(sys.argv[1]))
out_path = sys.argv[2]

HW = {"h100": "h100-80g", "h200": "h200-141g", "b200": "b200-180g", "b300": "b300-288g", "gb300": "gb300-288g"}
DEFAULT_GPUS = {"h100": 8, "h200": 8, "b200": 8, "b300": 8, "gb300": 4}
COOKBOOK = "https://docs.sglang.io/cookbook/autoregressive/"
PAGES = {
    "DeepSeek-V4": "DeepSeek/DeepSeek-V4", "GLM-5.2": "GLM/GLM-5.2", "GLM-5.3": "GLM/GLM-5.3",
    "GLM-5.3-Flash": "GLM/GLM-5.3-Flash", "Qwen3.8": "Qwen/Qwen3.8", "Qwen3.8-Flash-Next": "Qwen/Qwen3.8-Flash-Next",
    "Qwen3.8-27B": "Qwen/Qwen3.8-27B", "Kimi-K3": "Moonshotai/Kimi-K3", "MiniMax-M3": "MiniMax/MiniMax-M3",
    "GigaChat 3.5": "GigaChat/GigaChat3.5",
}
STRATEGY = {"low-latency": "низкая задержка", "balanced": "баланс", "high-throughput": "макс. пропускная способность"}


def model_id(name, variant):
    if name == "DeepSeek-V4":
        return "deepseek-v4-pro" if variant and variant.startswith("pro") else "deepseek-v4-flash"
    return {
        "GLM-5.2": "glm-5.2", "GLM-5.3": "glm-5.3", "GLM-5.3-Flash": "glm-5.3-flash", "Qwen3.8": "qwen3.8-max",
        "Qwen3.8-Flash-Next": "qwen3.8-flash-next", "Qwen3.8-27B": "qwen3.8-27b", "Kimi-K3": "kimi-k3",
        "MiniMax-M3": "minimax-m3", "GigaChat 3.5": "gigachat3.5",
    }.get(name)


def flag_int(flags, *names):
    for f in flags:
        for n in names:
            m = re.match(re.escape(n) + r"[ =](\d+)$", f.strip())
            if m:
                return int(m.group(1))
    return None


def flag_float(flags, name):
    for f in flags:
        m = re.match(re.escape(name) + r"[ =]([\d.]+)$", f.strip())
        if m:
            return float(m.group(1))
    return None


def flag_str(flags, name):
    for f in flags:
        m = re.match(re.escape(name) + r"[ =](\S+)$", f.strip())
        if m:
            return m.group(1)
    return None


def compatible(bm, cm):
    """Benchmark and cell describe the same deployment if no shared key disagrees."""
    shared = set(bm) & set(cm)
    if "hw" not in shared:
        return False
    return all(bm[k] == cm[k] for k in shared)


def engine_from_flags(flags, nnodes):
    clean = [f for f in flags if "{{" not in f and not f.startswith("--trust-remote-code")]
    tp = flag_int(flags, "--tp", "--tp-size") or None
    return {
        "tp": tp,
        "ep": flag_int(flags, "--ep", "--ep-size"),
        "attn_dp": flag_int(flags, "--attn-dp-size", "--dp", "--dp-size"),
        "max_running_requests": flag_int(flags, "--max-running-requests"),
        "mem_fraction_static": flag_float(flags, "--mem-fraction-static"),
        "chunked_prefill_size": flag_int(flags, "--chunked-prefill-size"),
        "cuda_graph_max_bs": flag_int(flags, "--cuda-graph-max-bs", "--cuda-graph-max-bs-decode"),
        "prefill_decode_interval": flag_int(flags, "--prefill-decode-interval"),
        "speculative": flag_str(flags, "--speculative-algorithm"),
        "kv_cache_dtype": flag_str(flags, "--kv-cache-dtype"),
        "nnodes": nnodes or 1,
        "flags": clean,
    }


profiles = []
seen = set()
for key, v in src.items():
    name = v["modelName"]
    page = COOKBOOK + PAGES.get(name, "")
    for b in v["benchmarks"]:
        bm = b["match"]
        if not b.get("speed") or bm.get("hw") not in HW or bm.get("pdMode") in ("prefill", "decode"):
            continue
        if bm.get("nodes", "single") not in ("single", None):
            continue
        variant = bm.get("variant")
        mid = model_id(name, variant)
        if mid == "gigachat3.5":
            continue  # ниже — ручные профили с ёмкостью пулов из того же cookbook
        cell = next((c for c in v["cells"] if c["match"] and compatible(bm, c["match"])), None)
        flags = cell["flags"] if cell else []
        eng = engine_from_flags(flags, cell.get("nnodes") if cell else 1)
        # Qwen3.8-27B замерен на одной GB300 (комментарий в qwen3.8-27b-benchmarks.jsx)
        default_gpus = 1 if mid == "qwen3.8-27b" else DEFAULT_GPUS[bm["hw"]]
        eng["gpus"] = eng["tp"] or default_gpus
        if not cell:
            eng["flags_note"] = "ячейка рецепта не найдена — флаги неизвестны"
        if eng["tp"] is None:
            eng["tp"] = eng["gpus"]
        quant = bm.get("quant") or (cell["match"].get("quant") if cell else None) or "native"
        spec = bm.get("spec")
        parts = [mid, bm["hw"], quant, bm.get("strategy", "default")]
        if variant and variant not in ("default", "flash", "pro"):
            parts.append(variant)
        if spec:
            parts.append(spec)
        if bm.get("kvDsaPair"):
            parts.append(bm["kvDsaPair"])
        pid = "-".join(str(p) for p in parts).lower()
        if pid in seen:
            continue
        seen.add(pid)
        meas = []
        for p in b["speed"]:
            w = p["workload"]
            if p.get("tpot_ms") is None and p.get("ttft_ms") is None:
                continue
            meas.append({
                "isl": w.get("isl"), "osl": w.get("osl"), "concurrency": w.get("max_concurrency"),
                "ttft_ms": p.get("ttft_ms"), "tpot_ms": p.get("tpot_ms"),
                "tok_s_per_gpu": p.get("tokens_per_sec_per_gpu"),
            })
        if not meas:
            continue
        label_variant = f" · {variant}" if variant and variant != "default" else ""
        profiles.append({
            "id": pid,
            "name": f"{name}{label_variant} · {eng['gpus']}×{bm['hw'].upper()} · {quant} · {STRATEGY.get(bm.get('strategy'), bm.get('strategy'))}"
                    + (f" · spec {spec}" if spec else ""),
            "model": mid,
            "gpu": HW[bm["hw"]],
            "quant": quant,
            "strategy": bm.get("strategy"),
            "engine": {"name": "sglang", "version": b.get("sglang_version") or "", "image": v["dockerImages"].get(bm["hw"], ""), **eng},
            "capacity": {},
            "measurements": meas,
            "measurement_kind": "bench_serving, random, фиксированная параллельность (закрытый цикл)",
            "verified": bool(cell and cell.get("verified")),
            "source": page,
            "note": "",
        })

# ---- ручные профили из других источников ----
ISSUE = "https://github.com/sgl-project/sglang/issues/41946"
V41_HT = [
    (1, 1024, 256, 190.98, 4.44, 241.69), (16, 1024, 256, 604.14, 7.6, 2007.51), (64, 1024, 256, 1557.56, 13.43, 4094.59),
    (1, 8192, 1024, 253.88, 4.46, 478.7), (16, 8192, 1024, 2346.31, 8.54, 3322.26), (64, 8192, 1024, 7635.88, 17.6, 5744.12),
]
V41_LL = [
    (1, 1024, 256, 195.67, 1.37, 587.72), (16, 1024, 256, 410.55, 9.95, 1681.01), (32, 1024, 256, 558.63, 15.8, 2150.76),
    (1, 8192, 1024, 258.13, 1.23, 1510.03), (16, 8192, 1024, 889.22, 5.51, 5498.79), (32, 8192, 1024, 1384.33, 9.38, 6585.65),
    (1, 32768, 1024, 939.02, 1.26, 3787.9), (16, 32768, 1024, 3059.92, 13.53, 7898.76),
    (1, 65536, 1024, 1917.45, 1.27, 5167.33), (16, 65536, 1024, 6485.94, 24.83, 8295.27),
]


def rows(data):
    return [{"isl": i, "osl": o, "concurrency": c, "ttft_ms": t, "tpot_ms": p, "tok_s_per_gpu": g} for c, i, o, t, p, g in data]


V41_COMMON = {"name": "sglang", "version": "da64c5cb (dev-dsv41)", "image": "lmsysorg/sglang:dev-dsv41",
              "tp": 4, "ep": 4, "gpus": 4, "nnodes": 1}
for strat, data, extra in (
    ("high-throughput", V41_HT, {"max_running_requests": 256, "flags": ["--tp 4", "--ep-size 4", "--max-running-requests 256"]}),
    ("low-latency", V41_LL, {"mem_fraction_static": 0.8, "cuda_graph_max_bs": 64, "speculative": "DSPARK",
                              "flags": ["--tp 4", "--ep-size 4", "--mem-fraction-static 0.8", "--speculative-algorithm DSPARK",
                                        "--speculative-dspark-block-size 5", "--cuda-graph-max-bs-decode 64"]}),
):
    for host in (False, True):
        eng = {**V41_COMMON, **extra}
        if host:
            eng = {**eng, "env": ["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1", "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank"]}
        profiles.append({
            "id": f"deepseek-v4.1-flash-b300-fp4-{strat}" + ("-engram-host" if host else ""),
            "name": f"DeepSeek-V4.1-Flash · 4×B300 TP4/EP4 · fp4 · {STRATEGY[strat]}" + (" · Engram в RAM хоста" if host else " · Engram на GPU"),
            "model": "deepseek-v4.1-flash",
            "gpu": "b300-288g",
            "quant": "fp4",
            "strategy": strat,
            "engine": eng,
            # Engram в FP8 ≈ 23,6 ГиБ на ранг при TP4 (карточка nvidia/DeepSeek-V4.1-Flash-NVFP4).
            # Веса на GPU и плотность KV-пула — по 24-часовому замеру на 4×B200 (TP4, Engram в RAM):
            # 74,5 ГиБ + 2,1 ГиБ DSpark; Engram в FP8 добавляет ~23,6 ГиБ на ранг; пул ~20 млн токенов на 66,6 ГиБ.
            "capacity": {"weights_gib_per_gpu": 76.6 if host else 100.2, "kv_tokens_per_gib": 300300},
            "perf_scale": {"decode": 1.0, "prefill": 1.0},
            "measurements": rows(data),
            "measurement_kind": "bench_serving, random, фиксированная параллельность (закрытый цикл)",
            "verified": True,
            "source": ISSUE,
            "note": (("Замеры из issue сделаны с Engram на GPU (по умолчанию). Для варианта с Engram в RAM хоста "
                      "скорость не измерена — perf_scale = 1,0 (заглушка); cookbook: вывод побитово тот же, освобождённая HBM уходит в KV-пул.")
                     if host else "Замеры из issue #41946 (B300, TP4/EP4, медианы трёх прогонов).")
                    + (" Ёмкость: веса 74,5 ГиБ + 2,1 ГиБ DSpark на GPU при Engram в RAM (+23,6 ГиБ при Engram на GPU), "
                       "плотность KV-пула ~300 тыс. токенов на ГиБ — по 24-часовому замеру на 4×B200 "
                       "(github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200)."),
        })

profiles.append({
    "id": "gigachat3.5-h100-fp8-high-throughput-capacity",
    "name": "GigaChat 3.5 · 8×H100 · fp8 · ёмкость из cookbook",
    "model": "gigachat3.5", "gpu": "h100-80g", "quant": "fp8", "strategy": "high-throughput",
    "engine": {"name": "sglang", "version": "9d58189c12 (main, 2026-09-22)", "image": "lmsysorg/sglang:latest", "tp": 8, "ep": 8,
               "gpus": 8, "nnodes": 1, "mem_fraction_static": 0.8,
               "flags": ["--tp 8", "--ep 8", "--mem-fraction-static 0.8"]},
    "capacity": {"kv_pool_tokens": 542000, "state_slots": 347},
    "measurements": [
        {"isl": 8192, "osl": 1024, "concurrency": 1, "ttft_ms": 540, "tpot_ms": 14.24, "tok_s_per_gpu": 76},
        {"isl": 8192, "osl": 1024, "concurrency": 16, "ttft_ms": 4930, "tpot_ms": 27.85, "tok_s_per_gpu": 551},
        {"isl": 8192, "osl": 1024, "concurrency": 64, "ttft_ms": 18836, "tpot_ms": 48.0, "tok_s_per_gpu": 913},
    ],
    "measurement_kind": "bench_serving, random, фиксированная параллельность (закрытый цикл)",
    "verified": True, "source": COOKBOOK + "GigaChat/GigaChat3.5",
    "note": "Ёмкость из cookbook: ~51 ГБ весов на GPU, 347 слотов state и 542 тыс. KV-токенов при mem-fraction 0,8.",
})
profiles.append({
    "id": "gigachat3.5-h100-fp8-low-latency-capacity",
    "name": "GigaChat 3.5 · 8×H100 · fp8 · низкая задержка (MTP) · ёмкость из cookbook",
    "model": "gigachat3.5", "gpu": "h100-80g", "quant": "fp8", "strategy": "low-latency",
    "engine": {"name": "sglang", "version": "9d58189c12 (main, 2026-09-22)", "image": "lmsysorg/sglang:latest", "tp": 8, "ep": 8,
               "gpus": 8, "nnodes": 1, "mem_fraction_static": 0.8, "max_running_requests": 80, "speculative": "EAGLE",
               "flags": ["--tp 8", "--ep 8", "--mem-fraction-static 0.8", "--speculative-algorithm EAGLE", "--speculative-num-steps 2",
                         "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 3", "--max-running-requests 80",
                         "--max-mamba-cache-size 240"]},
    "capacity": {"kv_pool_tokens": 185000, "state_slots": 240},
    "measurements": [
        {"isl": 8192, "osl": 1024, "concurrency": 1, "ttft_ms": 583, "tpot_ms": 6.43, "tok_s_per_gpu": 161},
        {"isl": 8192, "osl": 1024, "concurrency": 16, "ttft_ms": 1416, "tpot_ms": 17.12, "tok_s_per_gpu": 850},
        {"isl": 8192, "osl": 1024, "concurrency": 64, "ttft_ms": 52262, "tpot_ms": 22.78, "tok_s_per_gpu": 900},
    ],
    "measurement_kind": "bench_serving, random, фиксированная параллельность (закрытый цикл)",
    "verified": True, "source": COOKBOOK + "GigaChat/GigaChat3.5",
    "note": "С MTP пул state закреплён на 240 слотов, KV-пул падает до 185 тыс. токенов.",
})
profiles.append({
    "id": "mistral-medium-3.5-h200-fp8-default",
    "name": "Mistral Medium 3.5 · 4×H200 TP4 · fp8",
    "model": "mistral-medium-3.5", "gpu": "h200-141g", "quant": "fp8", "strategy": "default",
    "engine": {"name": "sglang", "version": "", "image": "lmsysorg/sglang:latest", "tp": 4, "gpus": 4, "nnodes": 1,
               "flags": ["--tp 4"]},
    "capacity": {},
    "measurements": [
        {"isl": 610, "osl": 268, "concurrency": 1, "ttft_ms": 95.9, "tpot_ms": 14.19, "tok_s_per_gpu": None},
        {"isl": 513, "osl": 262, "concurrency": 100, "ttft_ms": 152.95, "tpot_ms": 42.53, "tok_s_per_gpu": 1652},
    ],
    "measurement_kind": "bench_serving, random (длины случайные — указаны средние), медианы",
    "verified": True, "source": COOKBOOK + "Mistral/Mistral-Medium-3.5", "note": "",
})
profiles.append({
    "id": "kimi-k2.6-h200-int4-default",
    "name": "Kimi-K2.6 · 8×H200 TP8 · int4 · замер K2.5",
    "model": "kimi-k2.6", "gpu": "h200-141g", "quant": "int4", "strategy": "default",
    "engine": {"name": "sglang", "version": "0.5.9", "image": "lmsysorg/sglang:latest", "tp": 8, "gpus": 8, "nnodes": 1,
               "flags": ["--tp 8"]},
    "capacity": {},
    "measurements": [
        {"isl": 610, "osl": 422, "concurrency": 1, "ttft_ms": 176.89, "tpot_ms": 9.22, "tok_s_per_gpu": 32.4},
        {"isl": 496, "osl": 510, "concurrency": 16, "ttft_ms": 374.3, "tpot_ms": 53.25, "tok_s_per_gpu": 63.6},
    ],
    "measurement_kind": "bench_serving, random (длины случайные — указаны средние)",
    "verified": False, "source": COOKBOOK + "Moonshotai/Kimi-K2.6",
    "note": "Cookbook: замер сделан на K2.5 (та же архитектура), SGLang 0.5.9, без DP-attention — вероятно, не оптимальная конфигурация.",
})
profiles.append({
    "id": "minimax-m3.1-flash-b300-nvfp4-low-latency",
    "name": "MiniMax-M3.1-Flash · 4×B300 TP4 · nvfp4 · без замеров скорости",
    "model": "minimax-m3.1-flash", "gpu": "b300-288g", "quant": "nvfp4", "strategy": "low-latency",
    "engine": {"name": "sglang", "version": "", "image": "lmsysorg/sglang:latest", "tp": 4, "gpus": 4, "nnodes": 1,
               "mem_fraction_static": 0.8, "speculative": "DSPARK",
               "flags": ["--tp 4", "--mem-fraction-static 0.8", "--speculative-algorithm DSPARK"]},
    "capacity": {"kv_pool_tokens": 10400000},
    "measurements": [],
    "measurement_kind": "",
    "verified": True, "source": COOKBOOK + "MiniMax/MiniMax-M3.1",
    "note": "Cookbook: KV-пул ~10,4 млн токенов при mem-fraction 0,8 (6,5 млн с MonoKernel). Скорость — оценка по roofline.",
})

profiles.sort(key=lambda p: (p["model"], p["gpu"], p["id"]))
doc = {
    "_comment": ("Профили движка SGLang: конфигурация запуска + замеры. Источник — SGLang cookbook и указанные issue. "
                 "Замените или дополните своими замерами (bench_serving или метрики Prometheus): "
                 "калькулятор сам подгонит модель скорости под точки из measurements."),
    "profiles": profiles,
}
json.dump(doc, open(out_path, "w"), ensure_ascii=False, indent=2)
print(len(profiles), "profiles")
for p in profiles:
    print(f"  {p['id']:60s} m={len(p['measurements'])} tp={p['engine'].get('tp')} gpus={p['engine'].get('gpus')} mrr={p['engine'].get('max_running_requests')}")
