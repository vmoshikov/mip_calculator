package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
)

// Планировщик лимитов: для профиля движка и профиля нагрузки подбирает --max-running-requests
// и предельный поток запросов (RPS), при которых TTFT P99 (и, если задано, TPOT P99) укладываются в цель.

type PlanRequest struct {
	Profile        string    `json:"profile,omitempty"`
	Model          string    `json:"model,omitempty"` // без профиля: модель + GPU, скорость по roofline
	GPU            string    `json:"gpu,omitempty"`
	Workload       string    `json:"workload,omitempty"`
	WorkloadInline *Workload `json:"workload_inline,omitempty"`
	TTFTP99Ms      float64   `json:"ttft_p99_ms,omitempty"`
	TPOTP99Ms      float64   `json:"tpot_p99_ms,omitempty"`
	Servers        int       `json:"servers,omitempty"`
	GPUsPerServer  int       `json:"gpus_per_server,omitempty"`
	Instances      int       `json:"instances,omitempty"`
	Headroom       float64   `json:"headroom,omitempty"`
	MRR            []int     `json:"mrr_candidates,omitempty"`
	Requests       int       `json:"requests,omitempty"`
	Seed           int64     `json:"seed,omitempty"`
}

type PlanRow struct {
	MRR       int      `json:"max_running_requests"`
	MaxRPS    float64  `json:"max_rps"` // на инстанс, при котором цели ещё выполняются
	At        SimStats `json:"at_max"`
	LimitedBy string   `json:"limited_by"`
	Feasible  bool     `json:"feasible"`
}

type ValidationRow struct {
	ISL          int     `json:"isl"`
	OSL          int     `json:"osl"`
	Concurrency  int     `json:"concurrency"`
	MeasuredTTFT float64 `json:"measured_ttft_ms"`
	PredTTFT     float64 `json:"pred_ttft_ms"`
	MeasuredTPOT float64 `json:"measured_tpot_ms"`
	PredTPOT     float64 `json:"pred_tpot_ms"`
	ErrTTFTPct   float64 `json:"err_ttft_pct"`
	ErrTPOTPct   float64 `json:"err_tpot_pct"`
}

type CurvePoint struct {
	RPS     float64 `json:"rps"`
	TTFTp99 float64 `json:"ttft_p99_ms"`
	TTFTp50 float64 `json:"ttft_p50_ms"`
	TPOTp99 float64 `json:"tpot_p99_ms"`
	OutTokS float64 `json:"output_tok_s"`
	Running float64 `json:"mean_running"`
	Stable  bool    `json:"stable"`
}

type PlanResult struct {
	Request         PlanRequest     `json:"request"`
	Profile         Profile         `json:"profile"`
	Model           Model           `json:"model"`
	GPU             GPU             `json:"gpu"`
	Workload        Workload        `json:"workload"`
	WorkloadMeanIn  float64         `json:"workload_mean_input"`
	WorkloadMeanOut float64         `json:"workload_mean_output"`
	Perf            PerfParams      `json:"perf"`
	Fit             FitInfo         `json:"fit"`
	Capacity        CapacityInfo    `json:"capacity"`
	Validation      []ValidationRow `json:"validation"`
	ValidationTTFT  float64         `json:"validation_ttft_mape_pct"`
	ValidationTPOT  float64         `json:"validation_tpot_mape_pct"`
	Rows            []PlanRow       `json:"rows"`
	Best            *PlanRow        `json:"best,omitempty"`
	Instances       int             `json:"instances"`
	RPSLimit        float64         `json:"rps_limit_per_instance"` // с запасом headroom
	RPSCluster      float64         `json:"rps_limit_cluster"`
	TokPerHour      float64         `json:"output_tokens_per_hour_cluster"`
	Curve           []CurvePoint    `json:"curve"`
	Launch          string          `json:"launch"`
	Stubs           []string        `json:"stubs"`
	Notes           []string        `json:"notes"`
}

// syntheticProfile — профиль без замеров: раскладка из калькулятора памяти, скорость по roofline.
func syntheticProfile(cfg Config, model, gpu string) (Profile, error) {
	res, err := Calculate(cfg, Request{Model: model, GPU: gpu, Precision: PrecisionNative, Concurrency: 1, ContextLen: 8192})
	if err != nil {
		return Profile{}, err
	}
	if !res.Fits {
		return Profile{}, fmt.Errorf("модель не помещается: %s", res.Reason)
	}
	return Profile{
		ID: "roofline-" + model + "-" + gpu, Name: fmt.Sprintf("%s · %d×%s · без замеров (roofline)", res.Model.Name, res.GPUsPerReplica, res.GPU.Name),
		Model: model, GPU: gpu, Quant: "native", Strategy: "roofline",
		Engine: EngineCfg{Name: "sglang", TP: res.TP, GPUs: res.GPUsPerReplica, NNodes: res.NodesPerReplica,
			Flags: []string{fmt.Sprintf("--tp %d", res.GPUsPerReplica)}},
		Note: "Синтетический профиль: скорость оценена по характеристикам GPU, замеров нет.",
	}, nil
}

func PlanLimits(st *Store, req PlanRequest) (*PlanResult, error) {
	cfg := st.Get()
	var prof Profile
	if req.Profile != "" {
		p, ok := func() (Profile, bool) { d := st.Profiles.Get(); return d.Find(req.Profile) }()
		if !ok {
			return nil, fmt.Errorf("неизвестный профиль %q", req.Profile)
		}
		prof = p
	} else {
		p, err := syntheticProfile(cfg, req.Model, req.GPU)
		if err != nil {
			return nil, err
		}
		prof = p
	}
	m, ok := cfg.FindModel(prof.Model)
	if !ok {
		return nil, fmt.Errorf("модели %q нет в %s — файл, вероятно, создан старой версией: удалите его (создастся заново) или добавьте модель", prof.Model, st.Path())
	}
	g, ok := cfg.FindGPU(prof.GPU)
	if !ok {
		return nil, fmt.Errorf("профиль %q ссылается на неизвестный GPU %q", prof.ID, prof.GPU)
	}
	var wl Workload
	switch {
	case req.WorkloadInline != nil:
		wl = *req.WorkloadInline
		if wl.Name == "" {
			wl.Name = "своя нагрузка"
		}
		d := WorkloadsDoc{Workloads: []Workload{wl}}
		if wl.ID == "" {
			d.Workloads[0].ID = "inline"
		}
		if err := d.Validate(); err != nil {
			return nil, err
		}
	default:
		id := req.Workload
		if id == "" {
			id = "chat"
		}
		w, ok := func() (Workload, bool) { d := st.Workloads.Get(); return d.Find(id) }()
		if !ok {
			return nil, fmt.Errorf("неизвестный профиль нагрузки %q", id)
		}
		wl = w
	}
	if req.TTFTP99Ms <= 0 {
		req.TTFTP99Ms = 40000
	}
	if req.Headroom <= 0 || req.Headroom > 1 {
		req.Headroom = 0.8
	}
	if req.Requests <= 0 {
		req.Requests = 4000
	}
	req.Requests = min(req.Requests, 20000)
	if req.Seed == 0 {
		req.Seed = 42
	}

	res := &PlanResult{Request: req, Profile: prof, Model: m, GPU: g, Workload: wl, Stubs: []string{}, Notes: []string{}}
	res.WorkloadMeanIn, res.WorkloadMeanOut = wl.Input.mean(), wl.Output.mean()
	chunk := chunkSize(prof, g)
	res.Perf, res.Fit = fitPerf(cfg, m, g, prof, chunk)
	res.Capacity = deriveCapacity(cfg, m, g, prof)

	if wl.Stub {
		res.Stubs = append(res.Stubs, "профиль нагрузки — заглушка: замените длины и prefix cache своими данными (workloads.json)")
	}
	if res.Fit.Source == "roofline" {
		res.Stubs = append(res.Stubs, "скорость — оценка по roofline, замеров для профиля нет")
	}
	if prof.PerfScale != nil && (prof.PerfScale.Decode != 1 || prof.PerfScale.Prefill != 1 || strings.Contains(prof.Note, "заглушка")) {
		res.Stubs = append(res.Stubs, "perf_scale профиля — заглушка (влияние варианта на скорость не измерено)")
	}
	if strings.Contains(prof.Note, "заглушка") && prof.PerfScale != nil {
		res.Notes = append(res.Notes, prof.Note)
	}
	res.Notes = append(res.Notes, res.Capacity.Notes...)
	if res.Capacity.Source == "оценка по памяти" {
		res.Notes = append(res.Notes, "KV-пул оценён по памяти — точное значение берите из лога SGLang (max_total_num_tokens) и пропишите в capacity.kv_pool_tokens")
	}

	sc := simConfig{
		perf: res.Perf, kvPool: res.Capacity.KVPoolTokens, stateSlots: res.Capacity.StateSlots,
		chunk: chunk, interval: prof.Engine.PrefillDecodeInterval, maxCtx: m.MaxContext,
	}
	if prof.Engine.ChunkedPrefillSize <= 0 {
		res.Notes = append(res.Notes, fmt.Sprintf("chunked-prefill-size не задан — взято значение SGLang по умолчанию для этой GPU (%d)", chunk))
	}
	if prof.Perf == nil && (res.Fit.Source == "замеры" || res.Fit.Source == "замеры + roofline") {
		// prefill и decode влияют друг на друга в симуляции — калибруем попеременно
		for round := 0; round < 2; round++ {
			if p, ok := calibratePrefill(res.Perf, prof, sc); ok {
				res.Perf, sc.perf = p, p
			}
			if p, ok := calibrateDecode(res.Perf, prof, sc); ok {
				res.Perf, sc.perf = p, p
			}
		}
		res.Fit.Details = append(res.Fit.Details,
			"prefill и decode откалиброваны симуляцией закрытого цикла под измеренные TTFT и TPOT при параллельности > 1")
	}

	// сверка модели с замерами (закрытый цикл, как bench_serving --max-concurrency)
	var et, ep []float64
	for _, x := range usableMeasurements(prof.Measurements) {
		if x.Concurrency <= 0 || x.ISL <= 0 || x.OSL <= 0 {
			continue
		}
		vc := sc
		vc.mrr = x.Concurrency
		if prof.Engine.MaxRunningRequests > 0 {
			vc.mrr = min(vc.mrr, prof.Engine.MaxRunningRequests)
		}
		n := min(max(5*x.Concurrency, 16), 2048)
		s := runSim(vc, genFixed(x.ISL, x.OSL, n), 0, x.Concurrency)
		row := ValidationRow{ISL: x.ISL, OSL: x.OSL, Concurrency: x.Concurrency,
			MeasuredTTFT: x.TTFTms, PredTTFT: s.TTFTmean, MeasuredTPOT: x.TPOTms, PredTPOT: s.TPOTmean}
		if x.TTFTms > 0 {
			row.ErrTTFTPct = 100 * (s.TTFTmean - x.TTFTms) / x.TTFTms
			et = append(et, math.Abs(row.ErrTTFTPct))
		}
		if x.TPOTms > 0 {
			row.ErrTPOTPct = 100 * (s.TPOTmean - x.TPOTms) / x.TPOTms
			ep = append(ep, math.Abs(row.ErrTPOTPct))
		}
		res.Validation = append(res.Validation, row)
	}
	res.ValidationTTFT, res.ValidationTPOT = mean(et), mean(ep)

	// перебор max-running-requests
	capMax := res.Capacity.MaxRunningCap
	cands := sortedMRR(req.MRR, capMax)
	rows := make([]PlanRow, len(cands))
	var wg sync.WaitGroup
	for i, mrr := range cands {
		wg.Add(1)
		go func(i, mrr int) {
			defer wg.Done()
			c := sc
			c.mrr = mrr
			rows[i] = searchMaxRPS(c, wl, req)
			rows[i].MRR = mrr
		}(i, mrr)
	}
	wg.Wait()
	res.Rows = rows

	// лучший — наименьший лимит, дающий не меньше 95% максимального RPS: дальше рост лимита
	// почти не добавляет пропускной способности, но увеличивает TPOT и риск нехватки памяти
	top := 0.0
	for _, r := range rows {
		if r.Feasible {
			top = math.Max(top, r.MaxRPS)
		}
	}
	best := -1
	for i, r := range rows {
		if r.Feasible && r.MaxRPS >= 0.95*top {
			best = i
			break
		}
	}
	if best < 0 {
		res.Notes = append(res.Notes, "цели недостижимы даже при минимальной нагрузке: один запрос уже дольше цели TTFT/TPOT — нужен другой профиль, GPU или более короткий контекст")
	} else {
		b := rows[best]
		res.Best = &b
		res.RPSLimit = b.MaxRPS * req.Headroom
		res.Curve = curve(sc, b.MRR, b.MaxRPS, wl, req)
	}

	gpusPerServer := req.GPUsPerServer
	if gpusPerServer <= 0 {
		gpusPerServer = g.GPUsPerNode
	}
	switch {
	case req.Instances > 0:
		res.Instances = req.Instances
	case req.Servers > 0:
		res.Instances = req.Servers * gpusPerServer / max(prof.Engine.GPUs, 1)
		if res.Instances == 0 {
			res.Notes = append(res.Notes, fmt.Sprintf("одному инстансу нужно %d GPU, а на %d серверах по %d GPU их меньше",
				prof.Engine.GPUs, req.Servers, gpusPerServer))
		}
	default:
		res.Instances = 1
	}
	res.RPSCluster = res.RPSLimit * float64(res.Instances)
	if res.Best != nil {
		res.TokPerHour = res.Best.At.OutTokS * req.Headroom * float64(res.Instances) * 3600
	}
	if res.Instances > 1 {
		res.Notes = append(res.Notes, "кластерный лимит = лимит инстанса × число инстансов: роутер должен распределять запросы равномерно (round-robin или cache-aware)")
	}
	res.Launch = launchCommand(prof, m, res.Best)
	return res, nil
}

// searchMaxRPS — максимальный RPS на инстанс, при котором выполняются цели (бинарный поиск по RPS).
func searchMaxRPS(sc simConfig, wl Workload, req PlanRequest) PlanRow {
	ok := func(s SimStats) bool {
		return s.Stable && s.TTFTp99 <= req.TTFTP99Ms && (req.TPOTP99Ms <= 0 || s.TPOTp99 <= req.TPOTP99Ms)
	}
	run := func(rps float64) SimStats {
		return runSim(sc, genOpen(wl, sc.maxCtx, req.Requests, rps, req.Seed), req.Requests/10, 0)
	}
	lo, hi := 0.0, 0.0
	var loStats SimStats
	r := 0.05
	s := run(r)
	if ok(s) {
		lo, loStats = r, s
		for i := 0; i < 24; i++ {
			r *= 2
			s = run(r)
			if !ok(s) {
				hi = r
				break
			}
			lo, loStats = r, s
		}
		if hi == 0 {
			return PlanRow{MaxRPS: lo, At: loStats, Feasible: true, LimitedBy: "предел поиска"}
		}
	} else {
		hi = r
		for i := 0; i < 12; i++ {
			r /= 2
			s = run(r)
			if ok(s) {
				lo, loStats = r, s
				break
			}
			hi = r
		}
		if lo == 0 {
			return PlanRow{Feasible: false, At: s, LimitedBy: limitReason(sc, s, req)}
		}
	}
	for i := 0; i < 10; i++ {
		mid := (lo + hi) / 2
		s := run(mid)
		if ok(s) {
			lo, loStats = mid, s
		} else {
			hi = mid
		}
	}
	return PlanRow{MaxRPS: lo, At: loStats, Feasible: true, LimitedBy: limitReason(sc, run(hi), req)}
}

// limitReason — что первым ломается сразу за найденным пределом.
func limitReason(sc simConfig, s SimStats, req PlanRequest) string {
	switch {
	case req.TPOTP99Ms > 0 && s.TPOTp99 > req.TPOTP99Ms:
		return "TPOT P99"
	case sc.kvPool > 0 && s.KVPeakPct >= 95:
		return "KV-пул"
	case sc.stateSlots > 0 && s.MaxRunning >= sc.stateSlots:
		return "слоты state"
	case s.MeanRunning >= 0.9*float64(sc.mrr):
		return "max-running-requests: очередь на допуск"
	default:
		return "prefill: очередь на обработку входа"
	}
}

func curve(sc simConfig, mrr int, maxRPS float64, wl Workload, req PlanRequest) []CurvePoint {
	sc.mrr = mrr
	fr := []float64{0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.35}
	out := make([]CurvePoint, len(fr))
	var wg sync.WaitGroup
	for i, f := range fr {
		wg.Add(1)
		go func(i int, rps float64) {
			defer wg.Done()
			s := runSim(sc, genOpen(wl, sc.maxCtx, req.Requests, rps, req.Seed), req.Requests/10, 0)
			out[i] = CurvePoint{RPS: rps, TTFTp99: s.TTFTp99, TTFTp50: s.TTFTp50, TPOTp99: s.TPOTp99,
				OutTokS: s.OutTokS, Running: s.MeanRunning, Stable: s.Stable}
		}(i, maxRPS*f)
	}
	wg.Wait()
	sort.Slice(out, func(i, j int) bool { return out[i].RPS < out[j].RPS })
	return out
}

func launchCommand(p Profile, m Model, best *PlanRow) string {
	model := m.ID
	if strings.HasPrefix(m.Source, "https://huggingface.co/") {
		model = strings.TrimPrefix(m.Source, "https://huggingface.co/")
	}
	var b strings.Builder
	for _, e := range p.Engine.Env {
		b.WriteString(e + " \\\n")
	}
	b.WriteString("python3 -m sglang.launch_server \\\n  --model-path " + model + " \\\n  --trust-remote-code")
	hasMRR := false
	for _, f := range p.Engine.Flags {
		if strings.HasPrefix(f, "--max-running-requests") && best != nil {
			f = fmt.Sprintf("--max-running-requests %d", best.MRR)
			hasMRR = true
		}
		if strings.HasPrefix(f, "--model-path") {
			continue
		}
		b.WriteString(" \\\n  " + f)
	}
	if best != nil && !hasMRR {
		b.WriteString(fmt.Sprintf(" \\\n  --max-running-requests %d", best.MRR))
	}
	return b.String()
}
