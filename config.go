package main

import (
	_ "embed"
	"errors"
	"fmt"
	"path/filepath"
)

//go:embed config.default.json
var defaultConfigJSON []byte

// Model — описание LLM. Поля архитектуры определяют размер KV-кэша.
type Model struct {
	ID              string  `json:"id"`
	Name            string  `json:"name"`
	Vendor          string  `json:"vendor"`
	TotalParamsB    float64 `json:"total_params_b"`                     // все веса, млрд
	ActiveParamsB   float64 `json:"active_params_b"`                    // активных на токен (MoE), млрд
	Layers          int     `json:"layers"`                             // слоёв трансформера
	KVLayers        int     `json:"kv_layers,omitempty"`                // слоёв с полным KV (гибридное внимание); 0 = все
	Attention       string  `json:"attention"`                          // gqa | mla | custom
	KVHeads         int     `json:"kv_heads,omitempty"`                 // gqa
	HeadDim         int     `json:"head_dim,omitempty"`                 // gqa
	KVLoraRank      int     `json:"kv_lora_rank,omitempty"`             // mla
	QKRopeDim       int     `json:"qk_rope_dim,omitempty"`              // mla
	KVBytesPerToken float64 `json:"kv_bytes_per_token,omitempty"`       // ручное значение (в bf16), важнее архитектуры
	KVFixed         bool    `json:"kv_fixed,omitempty"`                 // kv_bytes_per_token в собственном формате модели, kv_dtype не влияет
	KVExtraPerToken float64 `json:"kv_extra_bytes_per_token,omitempty"` // индексер / вспомогательные кэши, абсолютно
	StateMBPerSeq   float64 `json:"state_mb_per_seq,omitempty"`         // state линейного внимания / окна, не зависит от контекста
	HiddenSize      int     `json:"hidden_size,omitempty"`              // размерность скрытого слоя — для FLOPs внимания
	AttnTopK        int     `json:"attn_topk,omitempty"`                // разреженное внимание (DSA): токенов на запрос, 0 = полное
	Experts         int     `json:"experts,omitempty"`                  // маршрутизируемых экспертов (MoE), 0 = dense
	ExpertsPerTok   int     `json:"experts_per_tok,omitempty"`          // активных экспертов на токен
	MaxContext      int     `json:"max_context"`
	NativePrecision string  `json:"native_precision"`        // основной формат весов официального чекпоинта
	CheckpointGB    float64 `json:"checkpoint_gb,omitempty"` // размер офиц. чекпоинта, используется при precision "native"
	Verified        bool    `json:"verified"`
	Source          string  `json:"source,omitempty"`
	Note            string  `json:"note,omitempty"`
}

// GPU — тип ускорителя, доступный на платформе.
type GPU struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	Arch         string   `json:"arch"`
	VRAMGB       float64  `json:"vram_gb"`
	BandwidthGBs float64  `json:"bandwidth_gbs"`
	TFLOPSBF16   float64  `json:"tflops_bf16"`
	Precisions   []string `json:"precisions"` // аппаратно поддерживаемые форматы
	GPUsPerNode  int      `json:"gpus_per_node"`
	Interconnect string   `json:"interconnect"`
	PricePerHour float64  `json:"price_per_hour"` // ₽/час за одну GPU, 0 = неизвестно
	Available    int      `json:"available"`      // свободно в MIP, 0 = без ограничений
}

// Interconnect — сеть для передачи KV-кэша между пулами prefill и decode и для PP между нодами.
type Interconnect struct {
	ID               string  `json:"id"`
	Name             string  `json:"name"`
	GbpsPerGPU       float64 `json:"gbps_per_gpu"`        // пропускная способность на одну GPU, Гбит/с
	Efficiency       float64 `json:"efficiency"`          // реальная / пиковая
	RDMA             bool    `json:"rdma"`                // GPUDirect RDMA, без копий через CPU
	LatencyUS        float64 `json:"latency_us"`          // задержка на передачу
	PricePerNodeHour float64 `json:"price_per_node_hour"` // ₽/час за ноду: NIC, порты коммутатора
	SameNodeOnly     bool    `json:"same_node_only"`      // только внутри ноды (NVLink)
}

type Defaults struct {
	Precision    string `json:"precision"`
	KVDtype      string `json:"kv_dtype"`
	ContextLen   int    `json:"context_len"`
	Concurrency  int    `json:"concurrency"`
	Replicas     int    `json:"replicas"`
	InputTokens  int    `json:"input_tokens"`  // токенов в запросе (prompt)
	OutputTokens int    `json:"output_tokens"` // токенов в ответе
	Interconnect string `json:"interconnect"`
	TTFTSLOms    int    `json:"ttft_slo_ms"` // целевое время до первого токена, 0 = без ограничения
}

// Platform — общие параметры расчёта для MIP.
type Platform struct {
	Name                 string   `json:"name"`
	Currency             string   `json:"currency"`
	GPUMemoryUtilization float64  `json:"gpu_memory_utilization"` // доля VRAM, доступная движку
	OverheadPerGPUGB     float64  `json:"overhead_per_gpu_gb"`    // активации, CUDA graphs, буферы NCCL
	WeightsOverheadPct   float64  `json:"weights_overhead_pct"`   // неквантованные эмбеддинги/нормы, скейлы
	TPSizes              []int    `json:"tp_sizes"`               // допустимые размеры TP внутри ноды
	MaxGPUsPerReplica    int      `json:"max_gpus_per_replica"`
	MLADataParallelAttn  bool     `json:"mla_dp_attention"`   // шардировать MLA-кэш по TP-рангам
	DecodeEfficiency     float64  `json:"decode_efficiency"`  // реальная / пиковая пропускная способность HBM
	ComputeEfficiency    float64  `json:"compute_efficiency"` // реальная / пиковая производительность тензорных ядер
	AvgContextFill       float64  `json:"avg_context_fill"`   // средняя заполненность контекста запроса (0..1)
	PrefillEfficiency    float64  `json:"prefill_efficiency"` // MFU на prefill
	LayerOverheadUS      float64  `json:"layer_overhead_us"`  // фиксированная задержка на слой за шаг decode: all-reduce TP, запуск ядер
	PrefillBatch         int      `json:"prefill_batch"`      // одновременных prompt в prefill-инстансе
	Defaults             Defaults `json:"defaults"`
}

type Config struct {
	Platform      Platform       `json:"platform"`
	GPUs          []GPU          `json:"gpus"`
	Interconnects []Interconnect `json:"interconnects"`
	Models        []Model        `json:"models"`
}

func (c *Config) FindInterconnect(id string) (Interconnect, bool) {
	for _, n := range c.Interconnects {
		if n.ID == id {
			return n, true
		}
	}
	return Interconnect{}, false
}

func (c *Config) FindModel(id string) (Model, bool) {
	for _, m := range c.Models {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

func (c *Config) FindGPU(id string) (GPU, bool) {
	for _, g := range c.GPUs {
		if g.ID == id {
			return g, true
		}
	}
	return GPU{}, false
}

func (c *Config) Validate() error {
	var errs []error
	p := c.Platform
	if p.GPUMemoryUtilization <= 0 || p.GPUMemoryUtilization > 1 {
		errs = append(errs, errors.New("platform.gpu_memory_utilization: должно быть в (0,1]"))
	}
	if len(p.TPSizes) == 0 {
		errs = append(errs, errors.New("platform.tp_sizes: список не может быть пустым"))
	}
	if p.MaxGPUsPerReplica <= 0 {
		errs = append(errs, errors.New("platform.max_gpus_per_replica: должно быть > 0"))
	}
	if _, ok := precisionBytes[p.Defaults.Precision]; !ok && p.Defaults.Precision != PrecisionNative {
		errs = append(errs, fmt.Errorf("platform.defaults.precision: неизвестная точность %q", p.Defaults.Precision))
	}
	seen := map[string]bool{}
	for i, g := range c.GPUs {
		if g.ID == "" {
			errs = append(errs, fmt.Errorf("gpus[%d]: нужен id", i))
		}
		if seen["g:"+g.ID] {
			errs = append(errs, fmt.Errorf("gpus[%d]: id %q повторяется", i, g.ID))
		}
		seen["g:"+g.ID] = true
		if g.VRAMGB <= 0 || g.BandwidthGBs <= 0 || g.GPUsPerNode <= 0 {
			errs = append(errs, fmt.Errorf("GPU %q: vram_gb, bandwidth_gbs, gpus_per_node должны быть > 0", g.ID))
		}
	}
	for i, n := range c.Interconnects {
		if n.ID == "" {
			errs = append(errs, fmt.Errorf("interconnects[%d]: нужен id", i))
		}
		if seen["n:"+n.ID] {
			errs = append(errs, fmt.Errorf("interconnects[%d]: id %q повторяется", i, n.ID))
		}
		seen["n:"+n.ID] = true
		if n.GbpsPerGPU <= 0 || n.Efficiency <= 0 || n.Efficiency > 1 {
			errs = append(errs, fmt.Errorf("интерконнект %q: gbps_per_gpu > 0, efficiency в (0,1]", n.ID))
		}
	}
	if d := p.Defaults.Interconnect; d != "" && !seen["n:"+d] {
		errs = append(errs, fmt.Errorf("platform.defaults.interconnect: неизвестный интерконнект %q", d))
	}
	for i, m := range c.Models {
		if m.ID == "" {
			errs = append(errs, fmt.Errorf("models[%d]: нужен id", i))
		}
		if seen["m:"+m.ID] {
			errs = append(errs, fmt.Errorf("models[%d]: id %q повторяется", i, m.ID))
		}
		seen["m:"+m.ID] = true
		if _, ok := precisionBytes[m.NativePrecision]; !ok {
			errs = append(errs, fmt.Errorf("модель %q: неизвестная native_precision %q", m.ID, m.NativePrecision))
		}
		if m.TotalParamsB <= 0 || m.ActiveParamsB <= 0 || m.Layers <= 0 {
			errs = append(errs, fmt.Errorf("модель %q: total_params_b, active_params_b, layers должны быть > 0", m.ID))
		}
		if m.KVBytesPerToken == 0 {
			switch m.Attention {
			case "gqa":
				if m.KVHeads <= 0 || m.HeadDim <= 0 {
					errs = append(errs, fmt.Errorf("модель %q: для gqa нужны kv_heads и head_dim", m.ID))
				}
			case "mla":
				if m.KVLoraRank <= 0 {
					errs = append(errs, fmt.Errorf("модель %q: для mla нужен kv_lora_rank", m.ID))
				}
			default:
				errs = append(errs, fmt.Errorf("модель %q: attention — gqa или mla (для custom нужен kv_bytes_per_token)", m.ID))
			}
		}
	}
	return errors.Join(errs...)
}

// Store — конфиг, профили движка и профили нагрузки. Каждый хранится в своём JSON-файле
// рядом с config.json; правки файлов подхватываются при следующем чтении.
type Store struct {
	*JSONStore[Config]
	Profiles  *JSONStore[ProfilesDoc]
	Workloads *JSONStore[WorkloadsDoc]
}

func NewStore(cfgPath string) (*Store, error) {
	cfg, err := NewJSONStore(cfgPath, defaultConfigJSON, (*Config).Validate)
	if err != nil {
		return nil, err
	}
	dir := filepath.Dir(cfgPath)
	prof, err := NewJSONStore(filepath.Join(dir, "profiles.json"), defaultProfilesJSON, (*ProfilesDoc).Validate)
	if err != nil {
		return nil, err
	}
	wl, err := NewJSONStore(filepath.Join(dir, "workloads.json"), defaultWorkloadsJSON, (*WorkloadsDoc).Validate)
	if err != nil {
		return nil, err
	}
	return &Store{JSONStore: cfg, Profiles: prof, Workloads: wl}, nil
}

func (s *Store) UpsertModel(m Model) error {
	c := s.Get()
	models := append([]Model(nil), c.Models...)
	replaced := false
	for i := range models {
		if models[i].ID == m.ID {
			models[i], replaced = m, true
		}
	}
	if !replaced {
		models = append(models, m)
	}
	c.Models = models
	return s.Replace(c)
}

func (s *Store) UpsertGPU(g GPU) error {
	c := s.Get()
	gpus := append([]GPU(nil), c.GPUs...)
	replaced := false
	for i := range gpus {
		if gpus[i].ID == g.ID {
			gpus[i], replaced = g, true
		}
	}
	if !replaced {
		gpus = append(gpus, g)
	}
	c.GPUs = gpus
	return s.Replace(c)
}

func (s *Store) DeleteModel(id string) error {
	c := s.Get()
	var out []Model
	for _, m := range c.Models {
		if m.ID != id {
			out = append(out, m)
		}
	}
	if len(out) == len(c.Models) {
		return fmt.Errorf("модель %q не найдена", id)
	}
	c.Models = out
	return s.Replace(c)
}

func (s *Store) DeleteGPU(id string) error {
	c := s.Get()
	var out []GPU
	for _, g := range c.GPUs {
		if g.ID != id {
			out = append(out, g)
		}
	}
	if len(out) == len(c.GPUs) {
		return fmt.Errorf("GPU %q не найден", id)
	}
	c.GPUs = out
	return s.Replace(c)
}
