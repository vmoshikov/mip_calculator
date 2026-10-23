package main

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func testConfig(t *testing.T) Config {
	t.Helper()
	var c Config
	if err := json.Unmarshal(defaultConfigJSON, &c); err != nil {
		t.Fatal(err)
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestKVPerTokenLlama405B(t *testing.T) {
	c := testConfig(t)
	m, _ := c.FindModel("llama-3.1-405b")
	// 2 × 126 слоёв × 8 голов × 128 × 2 байта = 516096 Б
	if got := kvBytesPerToken(m, 2); got != 516096 {
		t.Fatalf("KV байт/токен = %v", got)
	}
}

func TestDeepSeekV32FP8(t *testing.T) {
	c := testConfig(t)
	res, err := Calculate(c, Request{Model: "deepseek-v3.2", GPU: "h100-80g", Precision: "fp8", ContextLen: 32768, Concurrency: 16})
	if err != nil {
		t.Fatal(err)
	}
	// ~690 ГБ весов не помещаются в 8×80 ГБ — нужны две ноды H100.
	if !res.Fits || res.GPUsPerReplica != 16 || res.PP != 2 {
		t.Fatalf("получили fits=%v gpus=%d tp=%d pp=%d", res.Fits, res.GPUsPerReplica, res.TP, res.PP)
	}
	res, _ = Calculate(c, Request{Model: "deepseek-v3.2", GPU: "h200-141g", Precision: "fp8", ContextLen: 32768, Concurrency: 16})
	if res.GPUsPerReplica != 8 {
		t.Fatalf("h200: ожидалось 8 GPU, получили %d", res.GPUsPerReplica)
	}
}

func TestSmallModelPacksReplicas(t *testing.T) {
	c := testConfig(t)
	res, _ := Calculate(c, Request{Model: "gpt-oss-120b", GPU: "b200-180g", Precision: "fp4", Replicas: 3})
	if res.GPUsPerReplica != 1 || res.TotalGPUs != 3 || res.TotalNodes != 1 {
		t.Fatalf("получили %d на реплику, всего %d, нод %d", res.GPUsPerReplica, res.TotalGPUs, res.TotalNodes)
	}
}

func TestNoFitAndWarnings(t *testing.T) {
	c := testConfig(t)
	c.Platform.MaxGPUsPerReplica = 8
	res, _ := Calculate(c, Request{Model: "kimi-k2", GPU: "a100-80g", Precision: "bf16"})
	if res.Fits || res.Reason == "" {
		t.Fatal("kimi-k2 в bf16 не должна помещаться в 8×A100")
	}
	if !strings.Contains(strings.Join(res.Warnings, ";"), "fp8") && len(res.Warnings) == 0 {
		t.Fatal("ожидались предупреждения")
	}
}

func TestMCPToolsCall(t *testing.T) {
	c := testConfig(t)
	s := &Store{cfg: c, path: t.TempDir() + "/c.json"}
	_ = s.Replace(c)
	srv := &MCPServer{store: s}
	resp := srv.Handle([]byte(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"recommend","arguments":{"model":"glm-5.2"}}}`))
	if resp.Error != nil {
		t.Fatal(resp.Error.Message)
	}
	out, _ := json.Marshal(resp.Result)
	if !strings.Contains(string(out), `"fits":true`) {
		t.Fatalf("неожиданный ответ: %s", out[:200])
	}
	if srv.Handle([]byte(`{"jsonrpc":"2.0","method":"notifications/initialized"}`)) != nil {
		t.Fatal("на уведомление не должно быть ответа")
	}
}

func TestNativeCheckpointAndVendorSizing(t *testing.T) {
	c := testConfig(t)
	cases := []struct {
		req  Request
		gpus int
	}{
		// рецепт vLLM: GLM-5.2 FP8 на 8×H200; полный 1M контекста на 8×B200
		{Request{Model: "glm-5.2", GPU: "h200-141g", Precision: "fp8", ContextLen: 131072, Concurrency: 4}, 8},
		{Request{Model: "glm-5.2", GPU: "b200-180g", Precision: "fp8", ContextLen: 1048576, Concurrency: 1}, 8},
		// карточка NVIDIA: DeepSeek-V4.1-Flash на 4× GB300, контекст 1M
		{Request{Model: "deepseek-v4.1-flash", GPU: "b300-288g", Precision: "native", ContextLen: 1048576, Concurrency: 16}, 4},
	}
	for _, tc := range cases {
		res, err := Calculate(c, tc.req)
		if err != nil {
			t.Fatal(err)
		}
		if !res.Fits || res.GPUsPerReplica != tc.gpus {
			t.Errorf("%s на %s: получили fits=%v gpus=%d, ожидалось %d", tc.req.Model, tc.req.GPU, res.Fits, res.GPUsPerReplica, tc.gpus)
		}
	}
}

func TestHybridStateCounted(t *testing.T) {
	c := testConfig(t)
	res, _ := Calculate(c, Request{Model: "qwen3.8-max", GPU: "b300-288g", Precision: "fp8", Concurrency: 100})
	if res.StateGB < 58 || res.StateGB > 59 {
		t.Fatalf("state для 100 запросов = %.1f ГБ, ожидалось ~58,7", res.StateGB)
	}
}

func TestCostInRubles(t *testing.T) {
	c := testConfig(t)
	res, _ := Calculate(c, Request{Model: "deepseek-v4.1-flash", GPU: "b300-288g"})
	if res.TokensPerHour <= 0 || res.CostPerToken <= 0 {
		t.Fatalf("токенов/ч=%v ₽/токен=%v", res.TokensPerHour, res.CostPerToken)
	}
	if got := res.CostPerHour / res.TokensPerHour; math.Abs(got-res.CostPerToken) > 1e-12 {
		t.Fatalf("₽/токен %v ≠ ₽/час / токенов/час %v", res.CostPerToken, got)
	}
	// вход + выход складываются в стоимость запроса
	in, out := float64(res.Request.InputTokens), float64(res.Request.OutputTokens)
	sum := (res.CostPer1MInput*in + res.CostPer1MOutput*out) / 1e6
	if math.Abs(sum-res.CostPerRequest) > 1e-9 {
		t.Fatalf("сумма долей %v ≠ стоимости запроса %v", sum, res.CostPerRequest)
	}
}

func TestDisaggregation(t *testing.T) {
	c := testConfig(t)
	// MoE с коротким prompt: prefill дешёвый, раздельные пулы не окупаются
	moe, err := Disaggregation(c, Request{Model: "glm-5.2", GPU: "h200-141g", Precision: "fp8", InputTokens: 2048, OutputTokens: 512})
	if err != nil {
		t.Fatal(err)
	}
	if moe.Best.Mode != ModeColocated {
		t.Errorf("GLM-5.2 короткий prompt: лучший %q, ожидался совмещённый", moe.Best.Label)
	}
	// плотная 405B с длинным prompt: prefill тяжёлый, P/D дешевле
	dense, _ := Disaggregation(c, Request{Model: "llama-3.1-405b", GPU: "h200-141g", Precision: "fp8",
		ContextLen: 65536, InputTokens: 32768, OutputTokens: 512, Concurrency: 16, TTFTSLOms: 5000})
	if dense.Best.Mode != ModeDisagg || dense.SavingsPct <= 0 {
		t.Errorf("Llama 405B длинный prompt: лучший %q, экономия %.1f%%", dense.Best.Label, dense.SavingsPct)
	}
	if !dense.Best.RDMA {
		t.Errorf("при цели TTFT 5 с лучший вариант должен быть с RDMA, получили %q", dense.Best.Label)
	}
	// NVLink между разными типами GPU невозможен
	for _, a := range dense.Alternatives {
		if a.Interconnect == "nvlink" && a.PrefillGPU != a.DecodeGPU && a.Feasible {
			t.Errorf("NVLink между %s и %s не может быть допустим", a.PrefillGPU, a.DecodeGPU)
		}
	}
}
