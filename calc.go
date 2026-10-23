package main

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

// Байт на один вес, с учётом типичных блочных скейлов у 4-битных форматов.
var precisionBytes = map[string]float64{
	"bf16": 2,
	"fp16": 2,
	"fp8":  1,
	"int8": 1,
	"fp4":  0.5625, // NVFP4: 4 бита + fp8-скейл на 16 значений
	"int4": 0.53,   // AWQ/GPTQ, группа 128
}

// Во сколько раз формат ускоряет тензорные ядра относительно BF16 (при аппаратной поддержке).
var precisionSpeedup = map[string]float64{"bf16": 1, "fp16": 1, "fp8": 2, "int8": 2, "fp4": 4, "int4": 4}

// PrecisionNative — размер весов берётся из официального чекпоинта (смешанные форматы, напр. FP8 + FP4-эксперты).
const PrecisionNative = "native"

var kvBytes = map[string]float64{"bf16": 2, "fp16": 2, "fp8": 1}

type Request struct {
	Model       string  `json:"model"`
	GPU         string  `json:"gpu"`
	Precision   string  `json:"precision,omitempty"`
	KVDtype     string  `json:"kv_dtype,omitempty"`
	ContextLen  int     `json:"context_len,omitempty"`
	Concurrency int     `json:"concurrency,omitempty"`
	Replicas    int     `json:"replicas,omitempty"`
	MemUtil     float64 `json:"mem_util,omitempty"`
	// нагрузка: средний запрос и сеть
	InputTokens  int    `json:"input_tokens,omitempty"`
	OutputTokens int    `json:"output_tokens,omitempty"`
	Interconnect string `json:"interconnect,omitempty"`
	TTFTSLOms    int    `json:"ttft_slo_ms,omitempty"`
}

type PerGPU struct {
	VRAMGB     float64 `json:"vram_gb"`
	UsableGB   float64 `json:"usable_gb"`
	WeightsGB  float64 `json:"weights_gb"`
	KVCacheGB  float64 `json:"kv_cache_gb"`
	OverheadGB float64 `json:"overhead_gb"`
	FreeGB     float64 `json:"free_gb"`     // доступно движку минус веса/KV/накладные
	ReservedGB float64 `json:"reserved_gb"` // память вне gpu_memory_utilization
}

type Result struct {
	Model    Model    `json:"model"`
	GPU      GPU      `json:"gpu"`
	Request  Request  `json:"request"` // с применёнными значениями по умолчанию
	Fits     bool     `json:"fits"`
	Native   bool     `json:"native"` // GPU аппаратно поддерживает формат весов
	Reason   string   `json:"reason,omitempty"`
	Warnings []string `json:"warnings"`

	WeightsGB       float64 `json:"weights_gb"`
	KVCacheGB       float64 `json:"kv_cache_gb"` // логический KV + state, до репликации по TP
	KVPerTokenKB    float64 `json:"kv_per_token_kb"`
	StateGB         float64 `json:"state_gb"`           // рекуррентный state запросов (входит в kv_cache_gb)
	MinGPUsByMemory int     `json:"min_gpus_by_memory"` // ceil(всё / доступно), без правил раскладки

	TP              int    `json:"tp"`
	PP              int    `json:"pp"`
	GPUsPerReplica  int    `json:"gpus_per_replica"`
	NodesPerReplica int    `json:"nodes_per_replica"`
	Replicas        int    `json:"replicas"`
	TotalGPUs       int    `json:"total_gpus"`
	TotalNodes      int    `json:"total_nodes"`
	PerGPU          PerGPU `json:"per_gpu"`
	MaxConcurrency  int    `json:"max_concurrency"` // сколько запросов длины context_len влезает в реплику

	DecodeTokS     float64 `json:"decode_tok_s"`     // скорость одного потока, токенов/с
	ThroughputTokS float64 `json:"throughput_tok_s"` // генерация одной реплики при concurrency, токенов/с
	TokensPerHour  float64 `json:"tokens_per_hour"`  // генерация всех реплик за час
	Bottleneck     string  `json:"bottleneck"`       // память | вычисления

	// совмещённый режим: те же GPU делают и prefill, и decode
	PrefillTokS     float64 `json:"prefill_tok_s"`     // prefill одной реплики, токенов/с
	PrefillShare    float64 `json:"prefill_share"`     // доля GPU-времени на prefill
	TTFTms          float64 `json:"ttft_ms"`           // время до первого токена (без очереди)
	RequestsPerHour float64 `json:"requests_per_hour"` // запросов в час, все реплики

	GPUCostPerHour     float64 `json:"gpu_cost_per_hour"`     // ₽/час за GPU
	NetworkCostPerHour float64 `json:"network_cost_per_hour"` // ₽/час за сеть (если реплика на нескольких нодах)
	CostPerHour        float64 `json:"cost_per_hour"`         // ₽/час всего
	CostPerToken       float64 `json:"cost_per_token"`        // ₽ за 1 сгенерированный токен (с учётом prefill)
	CostPer1MTokens    float64 `json:"cost_per_1m_tokens"`    // ₽ за 1 млн сгенерированных токенов (с учётом prefill)
	CostPer1MInput     float64 `json:"cost_per_1m_input"`     // ₽ за 1 млн входных токенов (доля prefill)
	CostPer1MOutput    float64 `json:"cost_per_1m_output"`    // ₽ за 1 млн выходных токенов (доля decode)
	CostPerRequest     float64 `json:"cost_per_request"`      // ₽ за запрос
}

func applyDefaults(cfg Config, r Request) Request {
	d := cfg.Platform.Defaults
	if r.Precision == "" {
		r.Precision = d.Precision
	}
	if r.KVDtype == "" {
		r.KVDtype = d.KVDtype
	}
	if r.ContextLen <= 0 {
		r.ContextLen = d.ContextLen
	}
	if r.Concurrency <= 0 {
		r.Concurrency = d.Concurrency
	}
	if r.Replicas <= 0 {
		r.Replicas = max(d.Replicas, 1)
	}
	if r.MemUtil <= 0 || r.MemUtil > 1 {
		r.MemUtil = cfg.Platform.GPUMemoryUtilization
	}
	if r.InputTokens <= 0 {
		r.InputTokens = max(d.InputTokens, 1)
	}
	if r.OutputTokens <= 0 {
		r.OutputTokens = max(d.OutputTokens, 1)
	}
	if r.Interconnect == "" {
		r.Interconnect = d.Interconnect
	}
	if r.TTFTSLOms <= 0 {
		r.TTFTSLOms = d.TTFTSLOms
	}
	return r
}

// kvBytesPerToken — байт KV-кэша на один токен по всем слоям.
func kvBytesPerToken(m Model, bytes float64) float64 {
	extra := m.KVExtraPerToken
	switch {
	case m.KVBytesPerToken > 0 && m.KVFixed:
		return m.KVBytesPerToken + extra
	case m.KVBytesPerToken > 0:
		return m.KVBytesPerToken*bytes/2 + extra
	}
	layers := m.Layers
	if m.KVLayers > 0 {
		layers = m.KVLayers
	}
	if m.Attention == "mla" {
		return float64(layers*(m.KVLoraRank+m.QKRopeDim))*bytes + extra
	}
	return float64(2*layers*m.KVHeads*m.HeadDim)*bytes + extra
}

// weightsGB — память под веса и эффективные байты на параметр для формата.
func weightsGB(p Platform, m Model, precision string) (gb, bytesPerParam float64, err error) {
	if precision == PrecisionNative {
		if m.CheckpointGB > 0 {
			return m.CheckpointGB, m.CheckpointGB / m.TotalParamsB, nil
		}
		precision = m.NativePrecision
	}
	b, ok := precisionBytes[precision]
	if !ok {
		return 0, 0, fmt.Errorf("неизвестная точность %q", precision)
	}
	return m.TotalParamsB * b * (1 + p.WeightsOverheadPct/100), b, nil
}

// paramsReadPerStep — сколько млрд параметров читается за один шаг декодинга при batch запросах.
// У MoE с ростом батча задействуется всё больше экспертов: доля = 1 − (1 − k/E)^batch.
func paramsReadPerStep(m Model, batch int) float64 {
	E, k := float64(m.Experts), float64(m.ExpertsPerTok)
	if E <= k || k <= 0 {
		return min(m.TotalParamsB, m.ActiveParamsB*float64(batch))
	}
	expert := (m.TotalParamsB - m.ActiveParamsB) / (E - k)
	shared := max(m.ActiveParamsB-k*expert, 0)
	coverage := 1 - math.Pow(1-k/E, float64(batch))
	return shared + E*expert*coverage
}

type layout struct{ tp, pp int }

// candidateLayouts — раскладки TP×PP по возрастанию числа GPU:
// сначала TP внутри одной ноды, затем целые ноды с pipeline-параллелизмом.
func candidateLayouts(p Platform, g GPU) []layout {
	var out []layout
	for _, tp := range p.TPSizes {
		if tp >= 1 && tp <= g.GPUsPerNode && tp <= p.MaxGPUsPerReplica {
			out = append(out, layout{tp, 1})
		}
	}
	for pp := 2; pp*g.GPUsPerNode <= p.MaxGPUsPerReplica; pp++ {
		out = append(out, layout{g.GPUsPerNode, pp})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].tp*out[i].pp < out[j].tp*out[j].pp })
	return out
}

// kvShards — на сколько частей делится KV одной pipeline-стадии между её TP-рангами.
func kvShards(p Platform, m Model, tp int) int {
	if m.Attention == "mla" {
		if p.MLADataParallelAttn {
			return tp
		}
		return 1 // латентный кэш дублируется на каждом TP-ранге
	}
	if m.KVHeads > 0 {
		return min(tp, m.KVHeads) // головы сверх kv_heads дублируются
	}
	return tp
}

type perf struct {
	tokS       float64 // токенов/с на реплику
	stepSec    float64 // время шага декодинга на одной стадии
	bottleneck string
}

// prefillFLOPsPerToken — 2 × активные параметры + внимание: 4 × слои × hidden × средняя позиция
// (у причинного внимания в среднем I/2, у разреженного DSA — не больше top-k).
func prefillFLOPsPerToken(m Model, inputTokens int) float64 {
	flops := 2 * m.ActiveParamsB * 1e9
	if m.HiddenSize > 0 {
		layers := m.Layers
		if m.KVLayers > 0 {
			layers = m.KVLayers
		}
		pos := float64(inputTokens) / 2
		if m.AttnTopK > 0 {
			pos = min(pos, float64(m.AttnTopK))
		}
		flops += 4 * float64(layers) * float64(m.HiddenSize) * pos
	}
	return flops
}

// prefillTokS — prefill упирается в вычисления.
func prefillTokS(p Platform, m Model, g GPU, gpus int, hwPrecision string, native bool, inputTokens int) float64 {
	eff := p.PrefillEfficiency
	if eff <= 0 {
		eff = 0.5
	}
	speed := 1.0
	if native {
		speed = precisionSpeedup[hwPrecision]
	}
	return float64(gpus) * g.TFLOPSBF16 * 1e12 * eff * speed / prefillFLOPsPerToken(m, inputTokens)
}

// decodePerf — roofline-оценка шага декодинга: max(чтение HBM, вычисления).
func decodePerf(p Platform, m Model, g GPU, l layout, wb, kvPerTok float64, hwPrecision string, native bool, ctx, batch int) perf {
	n := float64(l.tp * l.pp)
	fill := p.AvgContextFill
	if fill <= 0 {
		fill = 0.5
	}
	memEff := p.DecodeEfficiency
	if memEff <= 0 {
		memEff = 0.55
	}
	compEff := p.ComputeEfficiency
	if compEff <= 0 {
		compEff = 0.4
	}
	b := float64(batch)

	weightsRead := paramsReadPerStep(m, batch) * wb / n // ГБ на GPU
	kvRead := b * kvPerTok * float64(ctx) * fill / 1e9 / float64(kvShards(p, m, l.tp)) / float64(l.pp)
	stateRead := b * m.StateMBPerSeq / 1000 / n
	tMem := (weightsRead + kvRead + stateRead) / (g.BandwidthGBs * memEff)

	speed := 1.0
	if native {
		speed = precisionSpeedup[hwPrecision]
	}
	tComp := 0.0
	if g.TFLOPSBF16 > 0 {
		flops := 2 * m.ActiveParamsB * 1e9 * b / n
		tComp = flops / (g.TFLOPSBF16 * 1e12 * compEff * speed)
	}
	overhead := p.LayerOverheadUS
	if overhead <= 0 {
		overhead = 50
	}
	// каждая стадия PP проходит свою часть слоёв
	step := max(tMem, tComp) + float64(m.Layers)/float64(l.pp)*overhead/1e6
	if step <= 0 {
		return perf{}
	}
	bn := "память"
	if tComp > tMem {
		bn = "вычисления"
	}
	// стадии PP работают над разными micro-batch, поэтому пропускная способность = batch / шаг стадии
	return perf{tokS: b / step, stepSec: step, bottleneck: bn}
}

func Calculate(cfg Config, req Request) (*Result, error) {
	m, ok := cfg.FindModel(req.Model)
	if !ok {
		return nil, fmt.Errorf("неизвестная модель %q", req.Model)
	}
	g, ok := cfg.FindGPU(req.GPU)
	if !ok {
		return nil, fmt.Errorf("неизвестный GPU %q", req.GPU)
	}
	req = applyDefaults(cfg, req)
	p := cfg.Platform
	weights, wb, err := weightsGB(p, m, req.Precision)
	if err != nil {
		return nil, err
	}
	hwPrecision := req.Precision
	if hwPrecision == PrecisionNative {
		hwPrecision = m.NativePrecision
	}
	kb, ok := kvBytes[req.KVDtype]
	if !ok {
		return nil, fmt.Errorf("неизвестный формат KV %q (bf16|fp8)", req.KVDtype)
	}

	res := &Result{Model: m, GPU: g, Request: req, Replicas: req.Replicas, Warnings: []string{}}
	res.WeightsGB = weights
	perTok := kvBytesPerToken(m, kb)
	res.KVPerTokenKB = perTok / 1024
	tokens := float64(req.ContextLen) * float64(req.Concurrency)
	res.StateGB = m.StateMBPerSeq * float64(req.Concurrency) / 1000
	res.KVCacheGB = perTok*tokens/1e9 + res.StateGB
	usable := g.VRAMGB*req.MemUtil - p.OverheadPerGPUGB
	if usable > 0 {
		res.MinGPUsByMemory = int(math.Ceil((res.WeightsGB + res.KVCacheGB) / usable))
	}

	res.Native = slices.Contains(g.Precisions, hwPrecision)
	if !res.Native {
		res.Warnings = append(res.Warnings, fmt.Sprintf(
			"%s не поддерживает %s аппаратно — только веса/эмуляция, скорость ниже", g.Name, hwPrecision))
	}
	if m.KVFixed && m.KVBytesPerToken > 0 {
		res.Warnings = append(res.Warnings, "KV-кэш в собственном формате модели — выбор формата KV не влияет")
	}
	if req.KVDtype == "fp8" && !slices.Contains(g.Precisions, "fp8") {
		res.Warnings = append(res.Warnings, fmt.Sprintf("FP8 KV-кэш на %s не аппаратный", g.Name))
	}
	if m.MaxContext > 0 && req.ContextLen > m.MaxContext {
		res.Warnings = append(res.Warnings, fmt.Sprintf(
			"контекст %d больше максимума модели (%d)", req.ContextLen, m.MaxContext))
	}
	if wbNative, ok := precisionBytes[m.NativePrecision]; ok && req.Precision != PrecisionNative && wb < wbNative {
		res.Warnings = append(res.Warnings, fmt.Sprintf(
			"квантование ниже нативного %s — возможна потеря качества", m.NativePrecision))
	}
	if !m.Verified {
		res.Warnings = append(res.Warnings, "параметры модели — оценка, не подтверждены")
	}

	var chosen layout
	for _, l := range candidateLayouts(p, g) {
		n := l.tp * l.pp
		w := res.WeightsGB / float64(n)
		kv := (res.KVCacheGB-res.StateGB)/float64(kvShards(p, m, l.tp))/float64(l.pp) +
			res.StateGB/float64(n) // state линейного внимания шардируется по головам, как веса
		if w+kv+p.OverheadPerGPUGB > g.VRAMGB*req.MemUtil {
			continue
		}
		chosen = l
		res.Fits = true
		res.TP, res.PP, res.GPUsPerReplica = l.tp, l.pp, n
		res.NodesPerReplica = int(math.Ceil(float64(n) / float64(g.GPUsPerNode)))
		res.PerGPU = PerGPU{
			VRAMGB:     g.VRAMGB,
			UsableGB:   g.VRAMGB * req.MemUtil,
			WeightsGB:  w,
			KVCacheGB:  kv,
			OverheadGB: p.OverheadPerGPUGB,
			FreeGB:     g.VRAMGB*req.MemUtil - w - kv - p.OverheadPerGPUGB,
			ReservedGB: g.VRAMGB * (1 - req.MemUtil),
		}
		seqKV := perTok*float64(req.ContextLen)/1e9/float64(kvShards(p, m, l.tp))/float64(l.pp) +
			m.StateMBPerSeq/1000/float64(n)
		if seqKV > 0 {
			res.MaxConcurrency = int((g.VRAMGB*req.MemUtil - w - p.OverheadPerGPUGB) / seqKV)
		}
		break
	}

	if !res.Fits {
		res.Reason = fmt.Sprintf("не помещается в %d× %s (max_gpus_per_replica) при контексте %d и %d параллельных запросах",
			p.MaxGPUsPerReplica, g.Name, req.ContextLen, req.Concurrency)
		return res, nil
	}

	res.TotalGPUs = res.GPUsPerReplica * res.Replicas
	if res.GPUsPerReplica < g.GPUsPerNode {
		// маленькие реплики упаковываются по несколько на ноду
		res.TotalNodes = int(math.Ceil(float64(res.TotalGPUs) / float64(g.GPUsPerNode)))
	} else {
		res.TotalNodes = res.NodesPerReplica * res.Replicas
	}

	single := decodePerf(p, m, g, chosen, wb, perTok, hwPrecision, res.Native, req.ContextLen, 1)
	if single.stepSec > 0 {
		res.DecodeTokS = 1 / (single.stepSec * float64(chosen.pp)) // один токен проходит все стадии
	}
	batch := decodePerf(p, m, g, chosen, wb, perTok, hwPrecision, res.Native, req.ContextLen, req.Concurrency)
	res.ThroughputTokS = batch.tokS
	res.Bottleneck = batch.bottleneck
	res.PrefillTokS = prefillTokS(p, m, g, res.GPUsPerReplica, hwPrecision, res.Native, req.InputTokens)

	// Реплика делит время между prefill и decode: на запрос уходит I/P + O/D секунд «реплики».
	in, out := float64(req.InputTokens), float64(req.OutputTokens)
	if res.PrefillTokS > 0 && res.ThroughputTokS > 0 {
		tPrefill, tDecode := in/res.PrefillTokS, out/res.ThroughputTokS
		res.PrefillShare = tPrefill / (tPrefill + tDecode)
		res.RequestsPerHour = 3600 / (tPrefill + tDecode) * float64(res.Replicas)
		res.TokensPerHour = res.RequestsPerHour * out
		res.TTFTms = 1000 * in / res.PrefillTokS
	}

	res.GPUCostPerHour = float64(res.TotalGPUs) * g.PricePerHour
	if res.PP > 1 {
		ic, ok := cfg.FindInterconnect(req.Interconnect)
		switch {
		case !ok:
			res.Warnings = append(res.Warnings, "реплика на нескольких нодах, а интерконнект не выбран")
		case ic.SameNodeOnly:
			res.Warnings = append(res.Warnings, fmt.Sprintf(
				"%s работает только внутри ноды, а реплике нужно %d нод — выберите IB/RoCE", ic.Name, res.NodesPerReplica))
		default:
			res.NetworkCostPerHour = float64(res.TotalNodes) * ic.PricePerNodeHour
		}
		res.Warnings = append(res.Warnings, fmt.Sprintf(
			"несколько нод: TP=%d × PP=%d, нужен быстрый межузловой интерконнект (IB/RoCE)", res.TP, res.PP))
	}
	res.CostPerHour = res.GPUCostPerHour + res.NetworkCostPerHour
	if res.TokensPerHour > 0 && res.CostPerHour > 0 {
		res.CostPerToken = res.CostPerHour / res.TokensPerHour
		res.CostPer1MTokens = res.CostPerToken * 1e6
		res.CostPerRequest = res.CostPerHour / res.RequestsPerHour
		res.CostPer1MInput = res.CostPerRequest * res.PrefillShare / in * 1e6
		res.CostPer1MOutput = res.CostPerRequest * (1 - res.PrefillShare) / out * 1e6
	}
	if g.Available > 0 && res.TotalGPUs > g.Available {
		res.Warnings = append(res.Warnings, fmt.Sprintf(
			"нужно %d× %s, в MIP доступно %d", res.TotalGPUs, g.Name, g.Available))
	}
	return res, nil
}

// Compare считает один запрос на всех GPU; лучшие (дешевле за токен, затем меньше карт) — первыми.
func Compare(cfg Config, req Request) ([]*Result, error) {
	var out []*Result
	for _, g := range cfg.GPUs {
		r := req
		r.GPU = g.ID
		res, err := Calculate(cfg, r)
		if err != nil {
			return nil, err
		}
		out = append(out, res)
	}
	sort.SliceStable(out, func(i, j int) bool { return better(out[i], out[j]) })
	return out, nil
}

func better(a, b *Result) bool {
	if a.Fits != b.Fits {
		return a.Fits
	}
	if a.Native != b.Native {
		return a.Native
	}
	if a.CostPer1MTokens > 0 && b.CostPer1MTokens > 0 && a.CostPer1MTokens != b.CostPer1MTokens {
		return a.CostPer1MTokens < b.CostPer1MTokens
	}
	return a.TotalGPUs < b.TotalGPUs
}

type MatrixCell struct {
	Fits            bool    `json:"fits"`
	GPUsPerReplica  int     `json:"gpus_per_replica"`
	TotalGPUs       int     `json:"total_gpus"`
	TP              int     `json:"tp"`
	PP              int     `json:"pp"`
	CostPerHour     float64 `json:"cost_per_hour"`
	CostPer1MTokens float64 `json:"cost_per_1m_tokens"`
	TokensPerHour   float64 `json:"tokens_per_hour"`
	Warnings        int     `json:"warnings"`
}

type Matrix struct {
	Request Request                          `json:"request"`
	Models  []string                         `json:"models"`
	GPUs    []string                         `json:"gpus"`
	Cells   map[string]map[string]MatrixCell `json:"cells"` // модель -> GPU -> ячейка
}

func BuildMatrix(cfg Config, req Request) (*Matrix, error) {
	mx := &Matrix{Request: applyDefaults(cfg, req), Cells: map[string]map[string]MatrixCell{}}
	for _, g := range cfg.GPUs {
		mx.GPUs = append(mx.GPUs, g.ID)
	}
	for _, m := range cfg.Models {
		mx.Models = append(mx.Models, m.ID)
		row := map[string]MatrixCell{}
		for _, g := range cfg.GPUs {
			r := req
			r.Model, r.GPU = m.ID, g.ID
			res, err := Calculate(cfg, r)
			if err != nil {
				return nil, err
			}
			row[g.ID] = MatrixCell{
				Fits: res.Fits, GPUsPerReplica: res.GPUsPerReplica, TotalGPUs: res.TotalGPUs, TP: res.TP, PP: res.PP,
				CostPerHour: res.CostPerHour, CostPer1MTokens: res.CostPer1MTokens, TokensPerHour: res.TokensPerHour,
				Warnings: len(res.Warnings),
			}
		}
		mx.Cells[m.ID] = row
	}
	return mx, nil
}

// ModelMetrics — ключевые метрики модели, не зависящие от нагрузки.
type ModelMetrics struct {
	ID              string             `json:"id"`
	Name            string             `json:"name"`
	TotalParamsB    float64            `json:"total_params_b"`
	ActiveParamsB   float64            `json:"active_params_b"`
	ActivePct       float64            `json:"active_pct"`
	Experts         int                `json:"experts,omitempty"`
	ExpertsPerTok   int                `json:"experts_per_tok,omitempty"`
	WeightsGB       map[string]float64 `json:"weights_gb"` // native, bf16, fp8, fp4
	KVBytesPerToken map[string]float64 `json:"kv_bytes_per_token"`
	KVGBPer1MTokens float64            `json:"kv_gb_per_1m_tokens"` // KV в FP8
	KVGBPerSeqMax   float64            `json:"kv_gb_per_seq_at_max_context"`
	StateMBPerSeq   float64            `json:"state_mb_per_seq"`
	MaxContext      int                `json:"max_context"`
	Attention       string             `json:"attention"`
	NativePrecision string             `json:"native_precision"`
	Verified        bool               `json:"verified"`
	Source          string             `json:"source,omitempty"`
}

func Metrics(cfg Config, m Model) ModelMetrics {
	mm := ModelMetrics{
		ID: m.ID, Name: m.Name, TotalParamsB: m.TotalParamsB, ActiveParamsB: m.ActiveParamsB,
		ActivePct: 100 * m.ActiveParamsB / m.TotalParamsB, Experts: m.Experts, ExpertsPerTok: m.ExpertsPerTok,
		StateMBPerSeq: m.StateMBPerSeq, MaxContext: m.MaxContext, Attention: m.Attention,
		NativePrecision: m.NativePrecision, Verified: m.Verified, Source: m.Source,
		WeightsGB:       map[string]float64{},
		KVBytesPerToken: map[string]float64{},
	}
	for _, p := range []string{PrecisionNative, "bf16", "fp8", "fp4"} {
		if gb, _, err := weightsGB(cfg.Platform, m, p); err == nil {
			mm.WeightsGB[p] = gb
		}
	}
	for _, d := range []string{"bf16", "fp8"} {
		mm.KVBytesPerToken[d] = kvBytesPerToken(m, kvBytes[d])
	}
	mm.KVGBPer1MTokens = mm.KVBytesPerToken["fp8"] * 1e6 / 1e9
	mm.KVGBPerSeqMax = mm.KVBytesPerToken["fp8"]*float64(m.MaxContext)/1e9 + m.StateMBPerSeq/1000
	return mm
}

func AllMetrics(cfg Config) []ModelMetrics {
	out := make([]ModelMetrics, 0, len(cfg.Models))
	for _, m := range cfg.Models {
		out = append(out, Metrics(cfg, m))
	}
	return out
}
