package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
)

//go:embed profiles.default.json
var defaultProfilesJSON []byte

//go:embed workloads.default.json
var defaultWorkloadsJSON []byte

// ---------- профили движка ----------

// Measurement — одна точка бенчмарка: bench_serving с фиксированной параллельностью (закрытый цикл).
type Measurement struct {
	ISL         int     `json:"isl"`
	OSL         int     `json:"osl"`
	Concurrency int     `json:"concurrency"`
	TTFTms      float64 `json:"ttft_ms"`
	TPOTms      float64 `json:"tpot_ms"`
	TokSPerGPU  float64 `json:"tok_s_per_gpu,omitempty"`
}

type EngineCfg struct {
	Name                  string   `json:"name"`
	Version               string   `json:"version"`
	Image                 string   `json:"image,omitempty"`
	TP                    int      `json:"tp"`
	EP                    int      `json:"ep,omitempty"`
	AttnDP                int      `json:"attn_dp,omitempty"`
	GPUs                  int      `json:"gpus"`
	NNodes                int      `json:"nnodes,omitempty"`
	MaxRunningRequests    int      `json:"max_running_requests,omitempty"`
	MemFractionStatic     float64  `json:"mem_fraction_static,omitempty"`
	ChunkedPrefillSize    int      `json:"chunked_prefill_size,omitempty"`
	CudaGraphMaxBS        int      `json:"cuda_graph_max_bs,omitempty"`
	PrefillDecodeInterval int      `json:"prefill_decode_interval,omitempty"`
	Speculative           string   `json:"speculative,omitempty"`
	KVCacheDtype          string   `json:"kv_cache_dtype,omitempty"`
	Flags                 []string `json:"flags"`
	Env                   []string `json:"env,omitempty"`
	FlagsNote             string   `json:"flags_note,omitempty"`
}

// Capacity — ёмкость инстанса. Пустые поля вычисляются из памяти GPU и архитектуры модели.
type Capacity struct {
	KVPoolTokens     int     `json:"kv_pool_tokens,omitempty"`      // из лога SGLang: max_total_num_tokens
	StateSlots       int     `json:"state_slots,omitempty"`         // слоты state у гибридных моделей (mamba pool)
	OffloadGiBPerGPU float64 `json:"offload_gib_per_gpu,omitempty"` // веса, вынесенные в RAM хоста (Engram и т. п.)
	WeightsGiBPerGPU float64 `json:"weights_gib_per_gpu,omitempty"` // измеренный объём весов на GPU (лог загрузки)
	KVTokensPerGiB   float64 `json:"kv_tokens_per_gib,omitempty"`   // измеренная плотность KV-пула: токенов на ГиБ памяти пула
}

// PerfParams — модель скорости инстанса.
//
//	шаг decode, мс = decode_base_ms + n × (decode_per_req_ms + decode_per_req_per_kctx_ms × контекст/1000)
//	prefill-батч, мс = prefill_overhead_ms + prefill_per_req_ms × запросов + токены / prefill_tok_s × 1000
//	                   + prefill_attn_ms × работа внимания / 10⁶
type PerfParams struct {
	DecodeBaseMs          float64 `json:"decode_base_ms"`
	DecodePerReqMs        float64 `json:"decode_per_req_ms"`
	DecodePerReqPerKCtxMs float64 `json:"decode_per_req_per_kctx_ms"`
	PrefillTokS           float64 `json:"prefill_tok_s"`
	PrefillOverheadMs     float64 `json:"prefill_overhead_ms"`
	PrefillAttnMs         float64 `json:"prefill_attn_ms"`    // мс на 10⁶ «токен×позиция» внимания
	PrefillPerReqMs       float64 `json:"prefill_per_req_ms"` // накладные на каждый запрос в prefill-батче
}

type PerfScale struct {
	Decode  float64 `json:"decode"`
	Prefill float64 `json:"prefill"`
}

type Profile struct {
	ID              string        `json:"id"`
	Name            string        `json:"name"`
	Model           string        `json:"model"`
	GPU             string        `json:"gpu"`
	Quant           string        `json:"quant"`
	Strategy        string        `json:"strategy"`
	Engine          EngineCfg     `json:"engine"`
	Capacity        Capacity      `json:"capacity"`
	Perf            *PerfParams   `json:"perf,omitempty"`       // заданы вручную — важнее подгонки
	PerfScale       *PerfScale    `json:"perf_scale,omitempty"` // множители к подогнанной модели (заглушка для неизмеренных вариантов)
	Measurements    []Measurement `json:"measurements"`
	MeasurementKind string        `json:"measurement_kind,omitempty"`
	Verified        bool          `json:"verified"`
	Source          string        `json:"source,omitempty"`
	Note            string        `json:"note,omitempty"`
}

type ProfilesDoc struct {
	Comment  string    `json:"_comment,omitempty"`
	Profiles []Profile `json:"profiles"`
}

func (d *ProfilesDoc) Validate() error {
	var errs []error
	seen := map[string]bool{}
	for i, p := range d.Profiles {
		if p.ID == "" {
			errs = append(errs, fmt.Errorf("profiles[%d]: нужен id", i))
		}
		if seen[p.ID] {
			errs = append(errs, fmt.Errorf("profiles[%d]: id %q повторяется", i, p.ID))
		}
		seen[p.ID] = true
		if p.Model == "" || p.GPU == "" || p.Engine.GPUs <= 0 {
			errs = append(errs, fmt.Errorf("профиль %q: нужны model, gpu и engine.gpus > 0", p.ID))
		}
	}
	return errors.Join(errs...)
}

func (d *ProfilesDoc) Find(id string) (Profile, bool) {
	for _, p := range d.Profiles {
		if p.ID == id {
			return p, true
		}
	}
	return Profile{}, false
}

// ---------- профили нагрузки ----------

type LengthDist struct {
	Distribution string       `json:"distribution"` // lognormal | fixed | buckets
	P50          float64      `json:"p50,omitempty"`
	P99          float64      `json:"p99,omitempty"`
	Value        float64      `json:"value,omitempty"`
	Buckets      [][2]float64 `json:"buckets,omitempty"` // [токенов, доля]
}

type Workload struct {
	ID             string     `json:"id"`
	Name           string     `json:"name"`
	Stub           bool       `json:"stub"`
	Input          LengthDist `json:"input"`
	Output         LengthDist `json:"output"`
	PrefixCacheHit float64    `json:"prefix_cache_hit"`
	Note           string     `json:"note,omitempty"`
}

type WorkloadsDoc struct {
	Comment   string     `json:"_comment,omitempty"`
	Workloads []Workload `json:"workloads"`
}

func (d *WorkloadsDoc) Validate() error {
	var errs []error
	seen := map[string]bool{}
	for i, w := range d.Workloads {
		if w.ID == "" || seen[w.ID] {
			errs = append(errs, fmt.Errorf("workloads[%d]: id пустой или повторяется", i))
		}
		seen[w.ID] = true
		for _, l := range []LengthDist{w.Input, w.Output} {
			if err := l.validate(); err != nil {
				errs = append(errs, fmt.Errorf("нагрузка %q: %w", w.ID, err))
			}
		}
		if w.PrefixCacheHit < 0 || w.PrefixCacheHit >= 1 {
			errs = append(errs, fmt.Errorf("нагрузка %q: prefix_cache_hit должен быть в [0, 1)", w.ID))
		}
	}
	return errors.Join(errs...)
}

func (d *WorkloadsDoc) Find(id string) (Workload, bool) {
	for _, w := range d.Workloads {
		if w.ID == id {
			return w, true
		}
	}
	return Workload{}, false
}

func (l LengthDist) validate() error {
	switch l.Distribution {
	case "fixed":
		if l.Value < 1 {
			return errors.New("fixed: нужен value ≥ 1")
		}
	case "lognormal":
		if l.P50 < 1 || l.P99 < l.P50 {
			return errors.New("lognormal: нужны p50 ≥ 1 и p99 ≥ p50")
		}
	case "buckets":
		if len(l.Buckets) == 0 {
			return errors.New("buckets: список пуст")
		}
	default:
		return fmt.Errorf("неизвестное распределение %q (lognormal | fixed | buckets)", l.Distribution)
	}
	return nil
}

// sampler выдаёт длины из распределения; z99 — квантиль 0,99 стандартного нормального.
type sampler func(r *rand.Rand) int

const z99 = 2.326

func (l LengthDist) sampler(maxLen int) sampler {
	clamp := func(v float64) int {
		n := int(math.Round(v))
		if maxLen > 0 && n > maxLen {
			n = maxLen
		}
		return max(n, 1)
	}
	switch l.Distribution {
	case "fixed":
		v := clamp(l.Value)
		return func(*rand.Rand) int { return v }
	case "buckets":
		total := 0.0
		for _, b := range l.Buckets {
			total += b[1]
		}
		return func(r *rand.Rand) int {
			x := r.Float64() * total
			for _, b := range l.Buckets {
				if x -= b[1]; x <= 0 {
					return clamp(b[0])
				}
			}
			return clamp(l.Buckets[len(l.Buckets)-1][0])
		}
	default:
		mu := math.Log(l.P50)
		sigma := (math.Log(l.P99) - mu) / z99
		return func(r *rand.Rand) int { return clamp(math.Exp(mu + sigma*r.NormFloat64())) }
	}
}

// mean — математическое ожидание длины (для подписей и оценок).
func (l LengthDist) mean() float64 {
	switch l.Distribution {
	case "fixed":
		return l.Value
	case "buckets":
		s, w := 0.0, 0.0
		for _, b := range l.Buckets {
			s += b[0] * b[1]
			w += b[1]
		}
		if w == 0 {
			return 0
		}
		return s / w
	default:
		sigma := (math.Log(l.P99) - math.Log(l.P50)) / z99
		return l.P50 * math.Exp(sigma*sigma/2)
	}
}

// ---------- подгонка модели скорости ----------

// maxFitConcurrency — точки с большей параллельностью (насыщение, очередь burst-а) в подгонку не берём:
// планировщик перебирает лимиты до 512.
const maxFitConcurrency = 512

func usableMeasurements(ms []Measurement) []Measurement {
	var out []Measurement
	for _, x := range ms {
		if x.Concurrency > 0 && x.Concurrency <= maxFitConcurrency {
			out = append(out, x)
		}
	}
	return out
}

// quantPrecision переводит обозначение квантизации из рецепта в формат калькулятора.
func quantPrecision(q string, m Model) string {
	q = strings.ToLower(q)
	switch {
	case q == "" || q == "native" || q == "default":
		return PrecisionNative
	case strings.Contains(q, "fp4"):
		if m.NativePrecision == "fp4" {
			return PrecisionNative
		}
		return "fp4"
	case strings.Contains(q, "fp8"):
		if m.NativePrecision == "fp8" {
			return PrecisionNative
		}
		return "fp8"
	case strings.Contains(q, "int4"):
		if m.NativePrecision == "int4" {
			return PrecisionNative
		}
		return "int4"
	case strings.Contains(q, "bf16"):
		return "bf16"
	}
	return PrecisionNative
}

// attnWork — «токен × позиция» внимания при prefill токенов [s, s+n): Σ min(p, topk).
func attnWork(s, n, topk int) float64 {
	sum := func(a, b float64) float64 { return (a + b) * (b - a + 1) / 2 } // Σ p для p ∈ [a, b]
	lo, hi := float64(s), float64(s+n-1)
	if n <= 0 {
		return 0
	}
	if topk <= 0 || hi <= float64(topk) {
		return sum(lo, hi)
	}
	k := float64(topk)
	if lo >= k {
		return k * float64(n)
	}
	return sum(lo, k) + k*(hi-k)
}

// rooflinePerf — модель скорости из характеристик GPU, когда замеров нет или их мало.
func rooflinePerf(cfg Config, m Model, g GPU, prof Profile) PerfParams {
	p := cfg.Platform
	precision := quantPrecision(prof.Quant, m)
	_, wb, err := weightsGB(p, m, precision)
	if err != nil {
		_, wb, _ = weightsGB(p, m, PrecisionNative)
	}
	hw := precision
	if hw == PrecisionNative {
		hw = m.NativePrecision
	}
	l := layout{tp: max(prof.Engine.GPUs, 1), pp: 1}
	kb := kvBytes["bf16"]
	if strings.HasPrefix(prof.Engine.KVCacheDtype, "fp8") {
		kb = kvBytes["fp8"]
	}
	kvTok := kvBytesPerToken(m, kb)
	native := false
	for _, x := range g.Precisions {
		native = native || x == hw
	}
	pf := p
	pf.AvgContextFill = 1
	step := func(ctx, batch int) float64 {
		return decodePerf(pf, m, g, l, wb, kvTok, hw, native, ctx, batch).stepSec * 1000
	}
	s1 := step(0, 1)
	pr := PerfParams{
		DecodeBaseMs:          s1,
		DecodePerReqMs:        max(step(0, 2)-s1, 0.01),
		DecodePerReqPerKCtxMs: max(step(1000, 1)-s1, 0),
		PrefillTokS:           prefillTokS(p, m, g, l.tp, hw, native, 0),
		PrefillOverheadMs:     50,
	}
	if m.HiddenSize > 0 && g.TFLOPSBF16 > 0 {
		layers := m.Layers
		if m.KVLayers > 0 {
			layers = m.KVLayers
		}
		eff := p.PrefillEfficiency
		if eff <= 0 {
			eff = 0.5
		}
		// 4 × слоёв × hidden FLOP на единицу «токен×позиция», 10⁶ единиц
		pr.PrefillAttnMs = 4 * float64(layers) * float64(m.HiddenSize) * 1e6 / (float64(l.tp) * g.TFLOPSBF16 * 1e12 * eff) * 1000
	}
	return pr
}

// lstsq решает min ‖Xβ − y‖ по нормальным уравнениям (до 3 признаков).
func lstsq(X [][]float64, y []float64) ([]float64, bool) {
	k := len(X[0])
	A := make([][]float64, k)
	b := make([]float64, k)
	for i := range A {
		A[i] = make([]float64, k)
	}
	for r, row := range X {
		for i := 0; i < k; i++ {
			b[i] += row[i] * y[r]
			for j := 0; j < k; j++ {
				A[i][j] += row[i] * row[j]
			}
		}
	}
	// Гаусс с выбором главного элемента
	for c := 0; c < k; c++ {
		piv := c
		for r := c + 1; r < k; r++ {
			if math.Abs(A[r][c]) > math.Abs(A[piv][c]) {
				piv = r
			}
		}
		if math.Abs(A[piv][c]) < 1e-12 {
			return nil, false
		}
		A[c], A[piv] = A[piv], A[c]
		b[c], b[piv] = b[piv], b[c]
		for r := 0; r < k; r++ {
			if r == c {
				continue
			}
			f := A[r][c] / A[c][c]
			for j := c; j < k; j++ {
				A[r][j] -= f * A[c][j]
			}
			b[r] -= f * b[c]
		}
	}
	out := make([]float64, k)
	for i := range out {
		out[i] = b[i] / A[i][i]
	}
	return out, true
}

func allNonNeg(v []float64) bool {
	for _, x := range v {
		if x < 0 {
			return false
		}
	}
	return true
}

type FitInfo struct {
	Source  string   `json:"source"` // замеры | замеры + roofline | roofline | вручную
	Points  int      `json:"points"`
	Details []string `json:"details"`
}

// fitPerf подгоняет модель скорости под замеры профиля; недостающие коэффициенты берёт из roofline.
func fitPerf(cfg Config, m Model, g GPU, prof Profile, chunk int) (PerfParams, FitInfo) {
	roof := rooflinePerf(cfg, m, g, prof)
	if prof.Perf != nil {
		return scalePerf(*prof.Perf, prof.PerfScale), FitInfo{Source: "вручную", Details: []string{"параметры заданы в профиле (perf)"}}
	}
	info := FitInfo{Source: "roofline", Details: []string{}}
	pr := roof
	mrr := prof.Engine.MaxRunningRequests
	if skipped := len(prof.Measurements) - len(usableMeasurements(prof.Measurements)); skipped > 0 {
		info.Details = append(info.Details, fmt.Sprintf("%d точек с параллельностью > %d не используются", skipped, maxFitConcurrency))
	}
	prof.Measurements = usableMeasurements(prof.Measurements)

	// decode: tpot = a + n·(b0 + b1·ctx/1000)
	type pt struct{ n, ctx, y float64 }
	var dec []pt
	ctxSet, nSet := map[float64]bool{}, map[float64]bool{}
	for _, x := range prof.Measurements {
		if x.TPOTms <= 0 || x.Concurrency <= 0 {
			continue
		}
		n := float64(x.Concurrency)
		if mrr > 0 && n > float64(mrr) {
			n = float64(mrr)
		}
		ctx := float64(x.ISL) + float64(x.OSL)/2
		dec = append(dec, pt{n, ctx, x.TPOTms})
		ctxSet[ctx], nSet[n] = true, true
	}
	decFitted := false
	if len(dec) >= 3 && len(ctxSet) >= 2 && len(nSet) >= 2 {
		X, y := [][]float64{}, []float64{}
		for _, p := range dec {
			X = append(X, []float64{1, p.n, p.n * p.ctx / 1000})
			y = append(y, p.y)
		}
		if beta, ok := lstsq(X, y); ok && allNonNeg(beta) {
			pr.DecodeBaseMs, pr.DecodePerReqMs, pr.DecodePerReqPerKCtxMs = beta[0], beta[1], beta[2]
			decFitted = true
			info.Details = append(info.Details, fmt.Sprintf("decode: 3 коэффициента по %d точкам", len(dec)))
		}
	}
	if !decFitted && len(nSet) >= 2 {
		X, y := [][]float64{}, []float64{}
		for _, p := range dec {
			X = append(X, []float64{1, p.n})
			y = append(y, p.y-p.n*roof.DecodePerReqPerKCtxMs*p.ctx/1000)
		}
		if beta, ok := lstsq(X, y); ok && allNonNeg(beta) {
			pr.DecodeBaseMs, pr.DecodePerReqMs = beta[0], beta[1]
			decFitted = true
			info.Details = append(info.Details, fmt.Sprintf("decode: база и наклон по %d точкам, вклад контекста — roofline", len(dec)))
		}
	}
	if !decFitted && len(dec) > 0 {
		// одна точка (или вырожденный набор): сохраняем форму roofline, масштабируем под замер
		var sr, sm float64
		for _, p := range dec {
			sm += p.y
			sr += roof.DecodeBaseMs + p.n*(roof.DecodePerReqMs+roof.DecodePerReqPerKCtxMs*p.ctx/1000)
		}
		k := sm / sr
		pr.DecodeBaseMs, pr.DecodePerReqMs, pr.DecodePerReqPerKCtxMs = roof.DecodeBaseMs*k, roof.DecodePerReqMs*k, roof.DecodePerReqPerKCtxMs*k
		decFitted = true
		info.Details = append(info.Details, fmt.Sprintf("decode: форма roofline, масштаб ×%.2f по замерам", k))
	}

	// prefill: по точкам с параллельностью 1 (нет очереди):
	// ttft = t0 × число chunk'ов + isl/P + attn × работа внимания
	type ppt struct{ isl, y float64 }
	var pre []ppt
	islSet := map[float64]bool{}
	for _, x := range prof.Measurements {
		if x.Concurrency == 1 && x.TTFTms > 0 && x.ISL > 0 {
			pre = append(pre, ppt{float64(x.ISL), x.TTFTms})
			islSet[float64(x.ISL)] = true
		}
	}
	chunks := func(isl float64) float64 { return math.Ceil(isl / float64(chunk)) }
	work := func(isl float64) float64 { return attnWork(0, int(isl), 0) / 1e6 }
	preFitted := false
	if len(islSet) >= 3 {
		X, y := [][]float64{}, []float64{}
		for _, p := range pre {
			X = append(X, []float64{chunks(p.isl), p.isl / 1000, work(p.isl)})
			y = append(y, p.y)
		}
		if beta, ok := lstsq(X, y); ok && allNonNeg(beta) && beta[1] > 0 {
			pr.PrefillOverheadMs, pr.PrefillTokS, pr.PrefillAttnMs = beta[0], 1e6/beta[1], beta[2]
			preFitted = true
			info.Details = append(info.Details, fmt.Sprintf("prefill: накладные на chunk, скорость и внимание по %d длинам входа", len(islSet)))
		}
	}
	if !preFitted && len(islSet) >= 2 {
		X, y := [][]float64{}, []float64{}
		for _, p := range pre {
			X = append(X, []float64{chunks(p.isl), p.isl / 1000})
			y = append(y, p.y-roof.PrefillAttnMs*work(p.isl))
		}
		if beta, ok := lstsq(X, y); ok && allNonNeg(beta) && beta[1] > 0 {
			pr.PrefillOverheadMs, pr.PrefillTokS = beta[0], 1e6/beta[1]
			preFitted = true
			info.Details = append(info.Details, fmt.Sprintf("prefill: накладные на chunk и скорость по %d длинам, внимание — roofline", len(islSet)))
		}
	}
	if !preFitted && len(pre) > 0 {
		p := pre[0]
		t0 := min(roof.PrefillOverheadMs, p.y*0.5)
		rest := p.y - t0*chunks(p.isl) - roof.PrefillAttnMs*work(p.isl)
		if rest > 0 {
			pr.PrefillOverheadMs, pr.PrefillTokS = t0, p.isl/rest*1000
			preFitted = true
			info.Details = append(info.Details, "prefill: скорость по одной точке, накладные и внимание — roofline")
		}
	}
	info.Points = len(dec)
	switch {
	case decFitted && preFitted:
		info.Source = "замеры"
	case decFitted || preFitted:
		info.Source = "замеры + roofline"
	}
	if !decFitted {
		info.Details = append(info.Details, "decode: roofline (замеров нет)")
	}
	if !preFitted {
		info.Details = append(info.Details, "prefill: roofline (нет замера при параллельности 1)")
	}
	return scalePerf(pr, prof.PerfScale), info
}

func scalePerf(p PerfParams, s *PerfScale) PerfParams {
	if s == nil {
		return p
	}
	if s.Decode > 0 {
		p.DecodeBaseMs *= s.Decode
		p.DecodePerReqMs *= s.Decode
		p.DecodePerReqPerKCtxMs *= s.Decode
	}
	if s.Prefill > 0 {
		p.PrefillTokS /= s.Prefill
		p.PrefillOverheadMs *= s.Prefill
		p.PrefillAttnMs *= s.Prefill
	}
	return p
}

// ---------- ёмкость ----------

type CapacityInfo struct {
	KVPoolTokens    int      `json:"kv_pool_tokens"`
	StateSlots      int      `json:"state_slots"`
	MaxRunningCap   int      `json:"max_running_cap"` // предел max-running-requests по памяти state (0 — нет)
	WeightsGBPerGPU float64  `json:"weights_gb_per_gpu"`
	PoolGBPerGPU    float64  `json:"pool_gb_per_gpu"`
	KVBytesPerToken float64  `json:"kv_bytes_per_token"` // на инстанс с учётом дублирования по TP
	Source          string   `json:"source"`
	Notes           []string `json:"notes"`
}

const gib = 1.073741824 // ГБ в одном ГиБ

// deriveCapacity — ёмкость KV-пула и слотов state: из профиля, если задана, иначе из памяти.
func deriveCapacity(cfg Config, m Model, g GPU, prof Profile) CapacityInfo {
	ci := CapacityInfo{Notes: []string{}}
	gpus := max(prof.Engine.GPUs, 1)
	mfs := prof.Engine.MemFractionStatic
	if mfs <= 0 {
		mfs = 0.85
		ci.Notes = append(ci.Notes, "mem-fraction-static не задан — взято 0,85")
	}
	w, _, err := weightsGB(cfg.Platform, m, quantPrecision(prof.Quant, m))
	if err != nil {
		w, _, _ = weightsGB(cfg.Platform, m, PrecisionNative)
	}
	ci.WeightsGBPerGPU = w/float64(gpus) - prof.Capacity.OffloadGiBPerGPU*gib
	if prof.Capacity.WeightsGiBPerGPU > 0 {
		ci.WeightsGBPerGPU = prof.Capacity.WeightsGiBPerGPU * gib
	}
	ci.PoolGBPerGPU = max(g.VRAMGB*mfs-ci.WeightsGBPerGPU, 0)

	kb := kvBytes["bf16"]
	if strings.HasPrefix(prof.Engine.KVCacheDtype, "fp8") {
		kb = kvBytes["fp8"]
	}
	shards := kvShards(cfg.Platform, m, gpus)
	if prof.Engine.AttnDP > 1 {
		shards = gpus // DP-attention: каждый ранг держит свои запросы
	}
	ci.KVBytesPerToken = kvBytesPerToken(m, kb)
	perGPUPerTok := ci.KVBytesPerToken / float64(shards)

	poolBytes := ci.PoolGBPerGPU * 1e9
	if m.StateMBPerSeq > 0 && m.KVLayers > 0 {
		// гибрид с линейным вниманием: SGLang делит память после весов между пулом state и KV: --mamba-full-memory-ratio 0,9
		statePerSlot := m.StateMBPerSeq * 1e6 / float64(gpus)
		stateBytes := poolBytes * 0.9 / 1.9
		ci.StateSlots = int(stateBytes / statePerSlot)
		poolBytes -= stateBytes
	}
	ci.KVPoolTokens = int(poolBytes / perGPUPerTok)
	ci.Source = "оценка по памяти"
	if d := prof.Capacity.KVTokensPerGiB; d > 0 {
		ci.KVPoolTokens = int(poolBytes / 1e9 / gib * d)
		ci.Source = "оценка по измеренной плотности пула"
	}
	if prof.Capacity.KVPoolTokens > 0 {
		ci.KVPoolTokens = prof.Capacity.KVPoolTokens
		ci.Source = "профиль (лог SGLang / cookbook)"
	}
	if prof.Capacity.StateSlots > 0 {
		ci.StateSlots = prof.Capacity.StateSlots
	}
	ci.MaxRunningCap = ci.StateSlots
	if ci.PoolGBPerGPU <= 0 {
		ci.Notes = append(ci.Notes, "по оценке веса не помещаются в mem-fraction-static — KV-пул не ограничен в симуляции; уточните capacity по логу SGLang")
	}
	if prof.Capacity.OffloadGiBPerGPU > 0 {
		ci.Notes = append(ci.Notes, fmt.Sprintf("%.1f ГиБ весов на GPU вынесены в RAM хоста — эта память ушла в KV-пул", prof.Capacity.OffloadGiBPerGPU))
	}
	return ci
}

// sortedMRR — кандидаты max-running-requests, не больше предела ёмкости.
func sortedMRR(cands []int, capMax int) []int {
	if len(cands) == 0 {
		cands = []int{8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512}
	}
	var out []int
	for _, c := range cands {
		if c > 0 && (capMax <= 0 || c <= capMax) {
			out = append(out, c)
		}
	}
	if capMax > 0 && capMax <= 1024 && (len(out) == 0 || out[len(out)-1] < capMax) {
		out = append(out, capMax)
	}
	sort.Ints(out)
	return out
}

// chunkSize — --chunked-prefill-size профиля или значение SGLang по умолчанию по объёму памяти GPU.
func chunkSize(prof Profile, g GPU) int {
	if prof.Engine.ChunkedPrefillSize > 0 {
		return prof.Engine.ChunkedPrefillSize
	}
	if g.VRAMGB >= 160 {
		return 16384
	}
	return 8192
}

// calibrateDecode подстраивает модель decode так, чтобы симуляция закрытого цикла (как bench_serving)
// воспроизводила измеренный TPOT: в замерах уже есть помехи от prefill, и без поправки симулятор учёл бы их дважды.
func calibrateDecode(pr PerfParams, prof Profile, sc simConfig) (PerfParams, bool) {
	prof.Measurements = usableMeasurements(prof.Measurements)
	type pt struct {
		x      Measurement
		n, ctx float64
		target float64
	}
	var pts []pt
	ctxSet, nSet := map[float64]bool{}, map[float64]bool{}
	mrr := prof.Engine.MaxRunningRequests
	for _, x := range prof.Measurements {
		if x.TPOTms <= 0 || x.Concurrency <= 0 || x.ISL <= 0 || x.OSL <= 1 {
			continue
		}
		n := float64(x.Concurrency)
		if mrr > 0 {
			n = math.Min(n, float64(mrr))
		}
		ctx := float64(x.ISL) + float64(x.OSL)/2
		pts = append(pts, pt{x, n, ctx, x.TPOTms})
		ctxSet[ctx], nSet[n] = true, true
	}
	if len(pts) < 2 || len(nSet) < 2 {
		return pr, false
	}
	three := len(pts) >= 3 && len(ctxSet) >= 2
	for iter := 0; iter < 6; iter++ {
		X, y := [][]float64{}, []float64{}
		for _, p := range pts {
			if three {
				X = append(X, []float64{1, p.n, p.n * p.ctx / 1000})
				y = append(y, p.target)
			} else {
				X = append(X, []float64{1, p.n})
				y = append(y, p.target-p.n*pr.DecodePerReqPerKCtxMs*p.ctx/1000)
			}
		}
		beta, ok := lstsq(X, y)
		if !ok || !allNonNeg(beta) {
			return pr, iter > 0
		}
		pr.DecodeBaseMs, pr.DecodePerReqMs = beta[0], beta[1]
		if three {
			pr.DecodePerReqPerKCtxMs = beta[2]
		}
		c := sc
		c.perf = pr
		for i := range pts {
			x := pts[i].x
			c.mrr = x.Concurrency
			if mrr > 0 {
				c.mrr = min(c.mrr, mrr)
			}
			s := runSim(c, genFixed(x.ISL, x.OSL, min(max(5*x.Concurrency, 16), 1024)), 0, x.Concurrency)
			if s.TPOTmean > 0 {
				pts[i].target *= math.Max(0.3, math.Min(3, x.TPOTms/s.TPOTmean))
			}
		}
	}
	return pr, true
}

// calibratePrefill подбирает накладные на запрос в prefill-батче и масштаб скорости prefill так, чтобы
// симуляция закрытого цикла воспроизводила измеренный TTFT (при параллельности > 1 запросы делят prefill).
func calibratePrefill(pr PerfParams, prof Profile, sc simConfig) (PerfParams, bool) {
	pts := usableMeasurements(prof.Measurements)
	multi := false
	for _, x := range pts {
		multi = multi || (x.Concurrency > 1 && x.TTFTms > 0)
	}
	if !multi {
		return pr, false
	}
	mrr := prof.Engine.MaxRunningRequests
	loss := func(perReq, speed float64) float64 {
		c := sc
		c.perf = pr
		c.perf.PrefillPerReqMs = perReq
		c.perf.PrefillTokS = pr.PrefillTokS * speed
		l := 0.0
		for _, x := range pts {
			if x.TTFTms <= 0 || x.ISL <= 0 || x.OSL <= 0 {
				continue
			}
			c.mrr = x.Concurrency
			if mrr > 0 {
				c.mrr = min(c.mrr, mrr)
			}
			s := runSim(c, genFixed(x.ISL, x.OSL, min(max(5*x.Concurrency, 16), 1024)), 0, x.Concurrency)
			if s.TTFTmean > 0 {
				l += math.Abs(math.Log(s.TTFTmean / x.TTFTms))
			}
		}
		return l
	}
	golden := func(f func(float64) float64, a, b float64) float64 {
		g := (math.Sqrt(5) - 1) / 2
		c1, c2 := b-g*(b-a), a+g*(b-a)
		f1, f2 := f(c1), f(c2)
		for i := 0; i < 16; i++ {
			if f1 < f2 {
				b, c2, f2 = c2, c1, f1
				c1 = b - g*(b-a)
				f1 = f(c1)
			} else {
				a, c1, f1 = c1, c2, f2
				c2 = a + g*(b-a)
				f2 = f(c2)
			}
		}
		return (a + b) / 2
	}
	perReq, logSpeed := 0.0, 0.0
	base := loss(0, 1)
	for round := 0; round < 3; round++ {
		perReq = golden(func(v float64) float64 { return loss(v, math.Exp(logSpeed)) }, 0, 500)
		logSpeed = golden(func(v float64) float64 { return loss(perReq, math.Exp(v)) }, math.Log(0.25), math.Log(8))
	}
	if loss(perReq, math.Exp(logSpeed)) >= base {
		return pr, false
	}
	pr.PrefillPerReqMs = perReq
	pr.PrefillTokS *= math.Exp(logSpeed)
	return pr, true
}
