package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// Модель скорости DeepSeek-V4.1-Flash (4×B300, TP4/EP4) должна воспроизводить замеры issue #41946.
func TestPlanFitMatchesMeasurements(t *testing.T) {
	s := testStore(t)
	r, err := PlanLimits(s, PlanRequest{Profile: "deepseek-v4.1-flash-b300-fp4-high-throughput", Workload: "bench-8k-1k"})
	if err != nil {
		t.Fatal(err)
	}
	if r.Fit.Source != "замеры" {
		t.Fatalf("источник скорости %q, ожидались замеры", r.Fit.Source)
	}
	if r.ValidationTTFT > 25 || r.ValidationTPOT > 25 {
		t.Fatalf("средняя ошибка сверки TTFT %.0f%%, TPOT %.0f%% — больше 25%%", r.ValidationTTFT, r.ValidationTPOT)
	}
}

func TestPlanUserCase(t *testing.T) {
	s := testStore(t)
	req := PlanRequest{Profile: "deepseek-v4.1-flash-b300-fp4-high-throughput", Workload: "agent",
		TTFTP99Ms: 40000, Servers: 3, GPUsPerServer: 8}
	r, err := PlanLimits(s, req)
	if err != nil {
		t.Fatal(err)
	}
	if r.Instances != 6 {
		t.Fatalf("3 сервера × 8 GPU при TP4: ожидалось 6 инстансов, получили %d", r.Instances)
	}
	if r.Best == nil || r.Best.At.TTFTp99 > 40000 || r.RPSLimit <= 0 || r.RPSLimit >= r.Best.MaxRPS {
		t.Fatalf("нет корректной рекомендации: %+v лимит %.2f", r.Best, r.RPSLimit)
	}
	if !strings.Contains(r.Launch, "--max-running-requests") {
		t.Fatalf("в команде запуска нет лимита: %s", r.Launch)
	}
	if len(r.Stubs) == 0 {
		t.Fatal("нагрузка-заглушка должна быть помечена")
	}
	// Engram в RAM хоста освобождает HBM под KV-пул
	req.Profile += "-engram-host"
	h, err := PlanLimits(s, req)
	if err != nil {
		t.Fatal(err)
	}
	if h.Capacity.KVPoolTokens <= r.Capacity.KVPoolTokens {
		t.Fatalf("KV-пул с Engram в RAM %d не больше, чем с Engram на GPU %d", h.Capacity.KVPoolTokens, r.Capacity.KVPoolTokens)
	}
}

// Ужесточение цели TTFT не может увеличить допустимый поток.
func TestPlanStricterSLOLowersRPS(t *testing.T) {
	s := testStore(t)
	base := PlanRequest{Profile: "glm-5.3-b300-fp8-low-latency", Workload: "chat", MRR: []int{64}}
	loose, tight := base, base
	loose.TTFTP99Ms, tight.TTFTP99Ms = 30000, 2000
	a, _ := PlanLimits(s, loose)
	b, _ := PlanLimits(s, tight)
	if a.Best == nil || b.Best == nil || b.Best.MaxRPS > a.Best.MaxRPS {
		t.Fatalf("строгая цель дала больше RPS: %+v vs %+v", b.Best, a.Best)
	}
}

func TestPlanInlineWorkloadAndRoofline(t *testing.T) {
	s := testStore(t)
	r, err := PlanLimits(s, PlanRequest{Model: "mistral-small-4", GPU: "h200-141g", TTFTP99Ms: 10000,
		WorkloadInline: &Workload{Input: LengthDist{Distribution: "fixed", Value: 4000},
			Output: LengthDist{Distribution: "lognormal", P50: 300, P99: 1500}}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Fit.Source != "roofline" || r.Best == nil {
		t.Fatalf("roofline-профиль без рекомендации: %s %+v", r.Fit.Source, r.Best)
	}
}

func TestSimLittleLaw(t *testing.T) {
	// при малой нагрузке очередь пуста: одновременных ≈ RPS × время запроса
	sc := simConfig{perf: PerfParams{DecodeBaseMs: 10, PrefillTokS: 1e5, PrefillOverheadMs: 10}, mrr: 64, chunk: 8192}
	wl := Workload{Input: LengthDist{Distribution: "fixed", Value: 1000}, Output: LengthDist{Distribution: "fixed", Value: 100}}
	st := runSim(sc, genOpen(wl, 0, 3000, 2, 1), 300, 0)
	// время запроса ≈ 20 мс prefill + 99 шагов по 10 мс ≈ 1,01 с → ~2 одновременных
	if st.MeanRunning < 1.5 || st.MeanRunning > 2.6 {
		t.Fatalf("в среднем одновременно %.2f, ожидалось ≈ 2", st.MeanRunning)
	}
}

func TestMCPPlanLimits(t *testing.T) {
	srv := &MCPServer{store: testStore(t)}
	resp := srv.Handle([]byte(`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plan_limits",
		"arguments":{"profile":"deepseek-v4.1-flash-b300-fp4-high-throughput","workload":"chat","ttft_p99_ms":40000,"servers":3}}}`))
	if resp.Error != nil {
		t.Fatal(resp.Error.Message)
	}
	out, _ := json.Marshal(resp.Result)
	if !strings.Contains(string(out), "rps_limit_cluster") || strings.Contains(string(out), `"isError":true`) {
		t.Fatalf("неожиданный ответ: %.300s", out)
	}
}
