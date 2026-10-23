package main

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
)

// Дискретно-событийная модель одного инстанса SGLang:
//   - очередь ожидания FIFO; запрос допускается, если свободен слот max-running-requests,
//     хватает KV-пула (вход + выход) и слотов state у гибридных моделей;
//   - prefill приоритетнее decode (как планировщик SGLang по умолчанию): пока есть допущенные
//     запросы без prefill, идут prefill-батчи по chunked-prefill-size токенов; с
//     --prefill-decode-interval N между prefill-батчами идут минимум N шагов decode;
//   - шаг decode генерирует по токену всем запросам в декодинге, время шага зависит от их числа и контекста.

type simConfig struct {
	perf       PerfParams
	mrr        int
	kvPool     int // токенов, 0 — без ограничения
	stateSlots int // 0 — без ограничения
	chunk      int // chunked-prefill-size
	interval   int // prefill-decode-interval
	maxCtx     int
}

type simReq struct {
	arrival   float64
	in, out   int
	uncached  int // токенов входа, которые надо посчитать (за вычетом prefix cache)
	prefilled int
	firstTok  float64
	done      float64
	decStart  int // номер глобального шага decode, с которого запрос в декодинге
}

type finishItem struct {
	step int
	idx  int
}
type finishHeap []finishItem

func (h finishHeap) Len() int           { return len(h) }
func (h finishHeap) Less(i, j int) bool { return h[i].step < h[j].step }
func (h finishHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *finishHeap) Push(x any)        { *h = append(*h, x.(finishItem)) }
func (h *finishHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

type SimStats struct {
	Requests    int     `json:"requests"`
	TTFTp50     float64 `json:"ttft_p50_ms"`
	TTFTp90     float64 `json:"ttft_p90_ms"`
	TTFTp99     float64 `json:"ttft_p99_ms"`
	TTFTmean    float64 `json:"ttft_mean_ms"`
	TPOTp50     float64 `json:"tpot_p50_ms"`
	TPOTp99     float64 `json:"tpot_p99_ms"`
	TPOTmean    float64 `json:"tpot_mean_ms"`
	OutTokS     float64 `json:"output_tok_s"`
	MeanRunning float64 `json:"mean_running"`
	MaxRunning  int     `json:"max_running"`
	KVPeakPct   float64 `json:"kv_peak_pct"`
	Stable      bool    `json:"stable"`
}

func percentile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(math.Ceil(q*float64(len(sorted)))) - 1
	return sorted[min(max(i, 0), len(sorted)-1)]
}

// runSim прогоняет заранее заданные запросы (arrival уже проставлен, кроме закрытого цикла).
// closedConc > 0 — закрытый цикл как в bench_serving --max-concurrency: следующий запрос
// приходит в момент завершения предыдущего.
func runSim(sc simConfig, reqs []simReq, warm int, closedConc int) SimStats {
	n := len(reqs)
	if closedConc > 0 {
		for i := range reqs {
			reqs[i].arrival = math.Inf(1)
		}
		for i := 0; i < min(closedConc, n); i++ {
			reqs[i].arrival = 0
		}
	}
	next := 0       // следующий ещё не пришедший запрос (по порядку)
	nextClosed := 0 // в закрытом цикле: сколько запросов уже выпущено
	if closedConc > 0 {
		nextClosed = min(closedConc, n)
	}
	var waiting []int
	wHead := 0
	var prefilling []int
	var decoding int
	fin := &finishHeap{}
	var sumIn, sumDecStart float64

	t := 0.0
	step := 0 // глобальный счётчик шагов decode
	sinceP := 1 << 30
	kvUsed := 0
	completed := 0
	var runArea, lastT float64
	maxRun, kvPeak := 0, 0

	running := func() int { return len(prefilling) + decoding }
	finish := func(i int) {
		reqs[i].done = t
		kvUsed -= reqs[i].in + reqs[i].out
		completed++
		if closedConc > 0 && nextClosed < n {
			reqs[nextClosed].arrival = t
			nextClosed++
		}
	}

	limit := 0
	for completed < n {
		limit++
		if limit > 50_000_000 {
			break
		}
		// поступления
		for next < n && reqs[next].arrival <= t {
			waiting = append(waiting, next)
			next++
		}
		// допуск
		for wHead < len(waiting) {
			i := waiting[wHead]
			need := reqs[i].in + reqs[i].out
			if running() >= sc.mrr || (sc.stateSlots > 0 && running() >= sc.stateSlots) {
				break
			}
			if sc.kvPool > 0 && kvUsed+need > sc.kvPool && running() > 0 {
				break
			}
			wHead++
			kvUsed += need
			prefilling = append(prefilling, i)
		}
		if wHead > 4096 && wHead*2 > len(waiting) {
			waiting = append([]int(nil), waiting[wHead:]...)
			wHead = 0
		}
		if r := running(); r > maxRun {
			maxRun = r
		}
		kvPeak = max(kvPeak, kvUsed)

		var dt float64
		switch {
		case len(prefilling) > 0 && (decoding == 0 || sinceP >= sc.interval):
			// prefill-батч до chunk токенов
			budget := sc.chunk
			tokens, touched := 0, 0
			work := 0.0
			var doneIdx []int
			for _, i := range prefilling {
				if budget <= 0 {
					break
				}
				r := &reqs[i]
				left := r.uncached - r.prefilled
				take := min(left, budget)
				// позиции считаемых токенов: после закэшированного префикса
				startPos := r.in - r.uncached + r.prefilled
				work += attnWork(startPos, take, 0)
				touched++
				r.prefilled += take
				tokens += take
				budget -= take
				if r.prefilled >= r.uncached {
					doneIdx = append(doneIdx, i)
				}
			}
			dt = sc.perf.PrefillOverheadMs + sc.perf.PrefillPerReqMs*float64(touched) +
				float64(tokens)/sc.perf.PrefillTokS*1000 + sc.perf.PrefillAttnMs*work/1e6
			runArea += float64(running()) * dt
			t += dt / 1000
			// запросы с законченным prefill получают первый токен и уходят в decode
			k := 0
			for _, i := range prefilling {
				if reqs[i].prefilled >= reqs[i].uncached {
					reqs[i].firstTok = t
					if reqs[i].out <= 1 {
						finish(i)
						continue
					}
					reqs[i].decStart = step
					decoding++
					sumIn += float64(reqs[i].in)
					sumDecStart += float64(step)
					heap.Push(fin, finishItem{step: step + reqs[i].out - 1, idx: i})
					continue
				}
				prefilling[k] = i
				k++
			}
			prefilling = prefilling[:k]
			sinceP = 0
		case decoding > 0:
			nd := float64(decoding)
			// средний контекст: вход + уже сгенерированные токены
			ctx := (sumIn + nd*float64(step) - sumDecStart + nd) / nd
			dt = sc.perf.DecodeBaseMs + nd*(sc.perf.DecodePerReqMs+sc.perf.DecodePerReqPerKCtxMs*ctx/1000)
			runArea += float64(running()) * dt
			t += dt / 1000
			step++
			sinceP++
			for fin.Len() > 0 && (*fin)[0].step <= step {
				it := heap.Pop(fin).(finishItem)
				decoding--
				sumIn -= float64(reqs[it.idx].in)
				sumDecStart -= float64(reqs[it.idx].decStart)
				finish(it.idx)
			}
		default:
			if next < n && !math.IsInf(reqs[next].arrival, 1) {
				t = max(t, reqs[next].arrival)
			} else if closedConc > 0 && next < n {
				// в закрытом цикле следующий запрос появится при завершении — сюда не попадаем
				t += 0.001
			} else if wHead >= len(waiting) {
				completed = n // ничего не осталось
			}
		}
		lastT = t
	}

	st := SimStats{Requests: n - warm, MaxRunning: maxRun}
	if lastT > 0 {
		st.MeanRunning = runArea / 1000 / lastT
	}
	if sc.kvPool > 0 {
		st.KVPeakPct = 100 * float64(kvPeak) / float64(sc.kvPool)
	}
	var ttft, tpot []float64
	outTok := 0
	first, last := math.Inf(1), 0.0
	for i := warm; i < n; i++ {
		r := reqs[i]
		if r.done == 0 && r.firstTok == 0 {
			continue
		}
		ttft = append(ttft, (r.firstTok-r.arrival)*1000)
		if r.out > 1 {
			tpot = append(tpot, (r.done-r.firstTok)*1000/float64(r.out-1))
		}
		outTok += r.out
		first = math.Min(first, r.arrival)
		last = math.Max(last, r.done)
	}
	if last > first {
		st.OutTokS = float64(outTok) / (last - first)
	}
	sort.Float64s(ttft)
	sort.Float64s(tpot)
	st.TTFTp50, st.TTFTp90, st.TTFTp99 = percentile(ttft, .5), percentile(ttft, .9), percentile(ttft, .99)
	st.TPOTp50, st.TPOTp99 = percentile(tpot, .5), percentile(tpot, .99)
	st.TTFTmean, st.TPOTmean = mean(ttft), mean(tpot)

	// устойчивость: очередь не растёт — TTFT последней пятой части не намного больше средней трети
	st.Stable = true
	if closedConc == 0 && n-warm >= 50 {
		var mid, tail []float64
		m := n - warm
		for i := warm; i < n; i++ {
			v := (reqs[i].firstTok - reqs[i].arrival) * 1000
			switch {
			case i-warm >= m*2/5 && i-warm < m*3/5:
				mid = append(mid, v)
			case i-warm >= m*4/5:
				tail = append(tail, v)
			}
		}
		if mt, tt := mean(mid), mean(tail); tt > 2*mt && tt-mt > 2000 {
			st.Stable = false
		}
	}
	return st
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// genOpen — запросы с пуассоновскими поступлениями интенсивности rps.
func genOpen(wl Workload, maxCtx, n int, rps float64, seed int64) []simReq {
	r := rand.New(rand.NewSource(seed))
	in := wl.Input.sampler(maxCtx)
	out := wl.Output.sampler(0)
	reqs := make([]simReq, n)
	t := 0.0
	for i := range reqs {
		t += r.ExpFloat64() / rps
		a, b := in(r), out(r)
		if maxCtx > 0 && a+b > maxCtx {
			a = max(maxCtx-b, 1)
		}
		unc := max(int(math.Round(float64(a)*(1-wl.PrefixCacheHit))), 1)
		reqs[i] = simReq{arrival: t, in: a, out: b, uncached: unc}
	}
	return reqs
}

// genFixed — n одинаковых запросов (для сверки с bench_serving).
func genFixed(isl, osl, n int) []simReq {
	reqs := make([]simReq, n)
	for i := range reqs {
		reqs[i] = simReq{in: isl, out: osl, uncached: isl}
	}
	return reqs
}
