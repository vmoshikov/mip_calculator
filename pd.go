package main

import (
	"fmt"
	"math"
	"sort"
)

// Раздельный prefill/decode (P/D disaggregation): prompt обрабатывает отдельный пул GPU,
// готовый KV-кэш передаётся по сети в пул decode. Плюс — decode не прерывается prefill-ами
// и пулы можно собрать на разных GPU; минус — лишние GPU на prefill и сеть под передачу KV.

const (
	ModeColocated = "совмещённый"
	ModeDisagg    = "раздельный P/D"
)

type PDOption struct {
	Mode             string `json:"mode"`
	Label            string `json:"label"`
	PrefillGPU       string `json:"prefill_gpu,omitempty"`
	PrefillGPUName   string `json:"prefill_gpu_name,omitempty"`
	DecodeGPU        string `json:"decode_gpu"`
	DecodeGPUName    string `json:"decode_gpu_name"`
	Interconnect     string `json:"interconnect,omitempty"`
	InterconnectName string `json:"interconnect_name,omitempty"`
	RDMA             bool   `json:"rdma"`

	PrefillInstances       int `json:"prefill_instances"`
	PrefillGPUsPerInstance int `json:"prefill_gpus_per_instance"`
	PrefillGPUs            int `json:"prefill_gpus"`
	DecodeReplicas         int `json:"decode_replicas"`
	DecodeGPUsPerReplica   int `json:"decode_gpus_per_replica"`
	DecodeGPUs             int `json:"decode_gpus"`
	TotalGPUs              int `json:"total_gpus"`
	TotalNodes             int `json:"total_nodes"`

	KVPerRequestMB float64 `json:"kv_per_request_mb"` // KV-кэш одного prompt, передаваемый в decode
	TransferMs     float64 `json:"transfer_ms"`       // время передачи KV
	NetUtil        float64 `json:"net_util"`          // загрузка сети decode-пула (1 = 100%)
	TTFTms         float64 `json:"ttft_ms"`           // prefill + передача KV, без очереди

	RequestsPerHour     float64 `json:"requests_per_hour"`
	OutputTokensPerHour float64 `json:"output_tokens_per_hour"`
	GPUCostPerHour      float64 `json:"gpu_cost_per_hour"`
	NetworkCostPerHour  float64 `json:"network_cost_per_hour"`
	CostPerHour         float64 `json:"cost_per_hour"`
	CostPer1MOutput     float64 `json:"cost_per_1m_output"` // ₽ за 1 млн выходных токенов, всё включено
	CostPerRequest      float64 `json:"cost_per_request"`

	Feasible bool     `json:"feasible"`
	MeetsSLO bool     `json:"meets_slo"` // TTFT укладывается в ttft_slo_ms
	Reason   string   `json:"reason"`
	Notes    []string `json:"notes"`
	Best     bool     `json:"best"`
}

type PDResult struct {
	Request      Request    `json:"request"`
	Colocated    PDOption   `json:"colocated"`
	Disagg       PDOption   `json:"disagg"`       // тот же GPU под prefill, выбранный интерконнект
	Alternatives []PDOption `json:"alternatives"` // все варианты, лучшие первыми
	SavingsPct   float64    `json:"savings_pct"`  // экономия лучшего варианта относительно совмещённого
	Best         PDOption   `json:"best"`
}

func colocatedOption(cfg Config, req Request) (PDOption, error) {
	res, err := Calculate(cfg, req)
	if err != nil {
		return PDOption{}, err
	}
	o := PDOption{
		Mode: ModeColocated, Label: "Совмещённый · " + res.GPU.Name,
		DecodeGPU: res.GPU.ID, DecodeGPUName: res.GPU.Name,
		DecodeReplicas: res.Replicas, DecodeGPUsPerReplica: res.GPUsPerReplica, DecodeGPUs: res.TotalGPUs,
		TotalGPUs: res.TotalGPUs, TotalNodes: res.TotalNodes, TTFTms: res.TTFTms,
		RequestsPerHour: res.RequestsPerHour, OutputTokensPerHour: res.TokensPerHour,
		GPUCostPerHour: res.GPUCostPerHour, NetworkCostPerHour: res.NetworkCostPerHour, CostPerHour: res.CostPerHour,
		CostPer1MOutput: res.CostPer1MTokens, CostPerRequest: res.CostPerRequest,
		Feasible: res.Fits, Reason: res.Reason, Notes: []string{},
	}
	if res.Fits {
		o.Notes = append(o.Notes, fmt.Sprintf("prefill занимает %.0f%% времени GPU и прерывает decode — скачки задержки между токенами",
			100*res.PrefillShare))
	}
	if res.PP > 1 {
		if ic, ok := cfg.FindInterconnect(req.Interconnect); ok {
			o.Interconnect, o.InterconnectName, o.RDMA = ic.ID, ic.Name, ic.RDMA
		}
	}
	return o, nil
}

// disaggOption — пул decode из запроса + пул prefill на prefillGPU, KV передаётся по ic.
func disaggOption(cfg Config, req Request, dec *Result, prefillGPU GPU, ic Interconnect) PDOption {
	p := cfg.Platform
	m := dec.Model
	o := PDOption{
		Mode:       ModeDisagg,
		Label:      fmt.Sprintf("P/D · prefill %s → decode %s · %s", prefillGPU.Name, dec.GPU.Name, ic.Name),
		PrefillGPU: prefillGPU.ID, PrefillGPUName: prefillGPU.Name,
		DecodeGPU: dec.GPU.ID, DecodeGPUName: dec.GPU.Name,
		Interconnect: ic.ID, InterconnectName: ic.Name, RDMA: ic.RDMA,
		Notes: []string{},
	}
	if !dec.Fits {
		o.Reason = "пул decode: " + dec.Reason
		return o
	}
	o.DecodeReplicas, o.DecodeGPUsPerReplica, o.DecodeGPUs = dec.Replicas, dec.GPUsPerReplica, dec.TotalGPUs

	batch := p.PrefillBatch
	if batch <= 0 {
		batch = 4
	}
	pre, err := Calculate(cfg, Request{
		Model: req.Model, GPU: prefillGPU.ID, Precision: req.Precision, KVDtype: req.KVDtype,
		ContextLen: req.InputTokens, Concurrency: batch, Replicas: 1, MemUtil: req.MemUtil,
	})
	if err != nil || !pre.Fits {
		o.Reason = "модель не помещается на prefill-инстанс " + prefillGPU.Name
		return o
	}
	o.PrefillGPUsPerInstance = pre.GPUsPerReplica
	perInstance := pre.PrefillTokS

	if ic.SameNodeOnly {
		if prefillGPU.ID != dec.GPU.ID || pre.GPUsPerReplica+dec.GPUsPerReplica > dec.GPU.GPUsPerNode {
			o.Reason = fmt.Sprintf("%s работает только внутри ноды: prefill (%d) + decode (%d) не помещаются в %d GPU одной ноды",
				ic.Name, pre.GPUsPerReplica, dec.GPUsPerReplica, dec.GPU.GPUsPerNode)
			return o
		}
	}

	in, out := float64(req.InputTokens), float64(req.OutputTokens)
	// Пропускная способность задаётся пулом decode, prefill-инстансов ставим столько, чтобы его загрузить.
	rate := dec.ThroughputTokS * float64(dec.Replicas) / out // запросов/с
	o.PrefillInstances = max(1, int(math.Ceil(rate*in/perInstance)))
	o.PrefillGPUs = o.PrefillInstances * pre.GPUsPerReplica

	kb := kvBytes[req.KVDtype]
	kvBytesReq := in*kvBytesPerToken(m, kb) + m.StateMBPerSeq*1e6
	o.KVPerRequestMB = kvBytesReq / 1e6
	linkGBs := ic.GbpsPerGPU / 8 * ic.Efficiency // ГБ/с на одну GPU
	// шарды KV идут параллельно со всех TP-рангов prefill на ранги decode
	lanes := float64(min(pre.GPUsPerReplica, dec.GPUsPerReplica))
	o.TransferMs = 1000*kvBytesReq/(lanes*linkGBs*1e9) + ic.LatencyUS/1000
	if !ic.RDMA {
		o.TransferMs *= 1.5 // копирование через память хоста
		o.Notes = append(o.Notes, "без RDMA: KV копируется через CPU/RAM хоста, выше задержка и нагрузка на CPU")
	}

	// загрузка сети: весь поток KV должен пройти через NIC decode-пула и prefill-пула
	need := rate * kvBytesReq / 1e9 // ГБ/с
	capacity := linkGBs * float64(min(o.DecodeGPUs, o.PrefillGPUs))
	o.NetUtil = need / capacity
	if o.NetUtil > 1 {
		rate /= o.NetUtil
		o.Notes = append(o.Notes, fmt.Sprintf("сеть — узкое место: нужно %.1f ГБ/с, доступно %.1f ГБ/с, пропускная способность урезана",
			need, capacity))
	} else if o.NetUtil > 0.7 {
		o.Notes = append(o.Notes, fmt.Sprintf("сеть загружена на %.0f%% — мало запаса под пики", 100*o.NetUtil))
	}

	o.TTFTms = 1000*in/perInstance + o.TransferMs
	o.RequestsPerHour = rate * 3600
	o.OutputTokensPerHour = o.RequestsPerHour * out
	o.TotalGPUs = o.PrefillGPUs + o.DecodeGPUs
	prefillNodes := int(math.Ceil(float64(o.PrefillGPUs) / float64(prefillGPU.GPUsPerNode)))
	o.TotalNodes = dec.TotalNodes + prefillNodes
	if ic.SameNodeOnly {
		o.TotalNodes = int(math.Ceil(float64(o.TotalGPUs) / float64(dec.GPU.GPUsPerNode)))
	}
	o.GPUCostPerHour = float64(o.PrefillGPUs)*prefillGPU.PricePerHour + float64(o.DecodeGPUs)*dec.GPU.PricePerHour
	o.NetworkCostPerHour = float64(o.TotalNodes) * ic.PricePerNodeHour
	o.CostPerHour = o.GPUCostPerHour + o.NetworkCostPerHour
	if o.OutputTokensPerHour > 0 {
		o.CostPer1MOutput = o.CostPerHour / o.OutputTokensPerHour * 1e6
		o.CostPerRequest = o.CostPerHour / o.RequestsPerHour
	}
	if dec.PP > 1 && ic.SameNodeOnly {
		o.Reason = "пул decode занимает несколько нод — NVLink не подходит"
		return o
	}
	o.Feasible = true
	return o
}

// Disaggregation считает совмещённый режим, раздельный P/D и альтернативы
// (каждый GPU под prefill × каждый интерконнект) для одной модели и пула decode.
func Disaggregation(cfg Config, req Request) (*PDResult, error) {
	req = applyDefaults(cfg, req)
	ic, ok := cfg.FindInterconnect(req.Interconnect)
	if !ok {
		return nil, fmt.Errorf("неизвестный интерконнект %q", req.Interconnect)
	}
	dec, err := Calculate(cfg, req)
	if err != nil {
		return nil, err
	}
	out := &PDResult{Request: req}
	if out.Colocated, err = colocatedOption(cfg, req); err != nil {
		return nil, err
	}
	out.Disagg = disaggOption(cfg, req, dec, dec.GPU, ic)
	out.Disagg.MeetsSLO = out.Disagg.Feasible && (req.TTFTSLOms <= 0 || out.Disagg.TTFTms <= float64(req.TTFTSLOms))

	alts := []PDOption{out.Colocated}
	for _, g := range cfg.GPUs {
		for _, n := range cfg.Interconnects {
			alts = append(alts, disaggOption(cfg, req, dec, g, n))
		}
	}
	for i := range alts {
		o := &alts[i]
		o.MeetsSLO = o.Feasible && (req.TTFTSLOms <= 0 || o.TTFTms <= float64(req.TTFTSLOms))
		if o.Feasible && !o.MeetsSLO {
			o.Notes = append(o.Notes, fmt.Sprintf("TTFT %.0f мс выше цели %d мс", o.TTFTms, req.TTFTSLOms))
		}
	}
	out.Colocated = alts[0]
	sort.SliceStable(alts, func(i, j int) bool {
		a, b := alts[i], alts[j]
		if a.Feasible != b.Feasible {
			return a.Feasible
		}
		if a.MeetsSLO != b.MeetsSLO {
			return a.MeetsSLO
		}
		if a.CostPer1MOutput != b.CostPer1MOutput {
			return a.CostPer1MOutput < b.CostPer1MOutput
		}
		return a.TTFTms < b.TTFTms
	})
	if len(alts) > 0 && alts[0].Feasible {
		alts[0].Best = true
		out.Best = alts[0]
		if c := out.Colocated.CostPer1MOutput; c > 0 && out.Best.CostPer1MOutput > 0 {
			out.SavingsPct = 100 * (c - out.Best.CostPer1MOutput) / c
		}
	}
	out.Alternatives = alts
	return out, nil
}
