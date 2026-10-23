"""Добавляет результаты `python -m sglang.bench_serving ... --output-file out.jsonl` в замеры профиля.

Пример:
  python -m sglang.bench_serving --backend sglang --dataset-name random --random-input-len 8192 \
      --random-output-len 1024 --random-range-ratio 1 --max-concurrency 16 --num-prompts 80 --output-file bench.jsonl
  python3 tools/bench_to_profile.py bench.jsonl profiles.json deepseek-v4.1-flash-b300-fp4-high-throughput

Каждая строка JSONL — один прогон bench_serving. Берутся max_concurrency, длины, mean_ttft_ms и mean_tpot_ms.
Повторный прогон с теми же параллельностью и длинами заменяет старую точку. Калькулятор подхватит файл без перезапуска.
"""
import json
import sys


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    bench, profiles_path, profile_id = sys.argv[1:]
    doc = json.load(open(profiles_path, encoding="utf-8"))
    prof = next((p for p in doc["profiles"] if p["id"] == profile_id), None)
    if prof is None:
        sys.exit(f"профиль {profile_id!r} не найден в {profiles_path}")
    points = {(m["concurrency"], m["isl"], m["osl"]): m for m in prof.get("measurements", [])}
    added = 0
    for line in open(bench, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        conc = r.get("max_concurrency") or r.get("concurrency")
        isl = r.get("random_input_len") or round(r.get("total_input_tokens", 0) / max(r.get("completed", 1), 1))
        osl = r.get("random_output_len") or round(r.get("total_output_tokens", 0) / max(r.get("completed", 1), 1))
        if not conc or not isl or not osl:
            print("пропускаю строку без max_concurrency или длин", file=sys.stderr)
            continue
        gpus = prof["engine"].get("gpus") or 1
        points[(int(conc), int(isl), int(osl))] = {
            "isl": int(isl), "osl": int(osl), "concurrency": int(conc),
            "ttft_ms": r.get("mean_ttft_ms"), "tpot_ms": r.get("mean_tpot_ms"),
            "tok_s_per_gpu": (r.get("total_throughput") or 0) / gpus or None,
        }
        added += 1
    prof["measurements"] = sorted(points.values(), key=lambda m: (m["isl"], m["osl"], m["concurrency"]))
    prof["measurement_kind"] = "bench_serving (свои замеры)"
    prof.pop("perf_scale", None)
    json.dump(doc, open(profiles_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"добавлено точек: {added}; всего в профиле: {len(prof['measurements'])}")


if __name__ == "__main__":
    main()
