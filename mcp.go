package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
)

// Минимальный MCP-сервер (JSON-RPC 2.0): только инструменты, транспорты stdio и Streamable HTTP.

const mcpLatestProtocol = "2025-06-18"

var mcpProtocols = []string{"2025-06-18", "2025-03-26", "2024-11-05"}

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type mcpTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
	handler     func(s *Store, args json.RawMessage) (any, error)
}

func obj(props map[string]any, required ...string) map[string]any {
	o := map[string]any{"type": "object", "properties": props}
	if len(required) > 0 {
		o["required"] = required
	}
	return o
}

var calcProps = map[string]any{
	"model":         map[string]any{"type": "string", "description": "id модели (см. list_models)"},
	"gpu":           map[string]any{"type": "string", "description": "id GPU (см. list_gpus)"},
	"precision":     map[string]any{"type": "string", "enum": []string{"native", "bf16", "fp16", "fp8", "int8", "fp4", "int4"}, "description": "точность весов; native = размер официального чекпоинта; по умолчанию из платформы"},
	"kv_dtype":      map[string]any{"type": "string", "enum": []string{"bf16", "fp16", "fp8"}, "description": "формат KV-кэша"},
	"context_len":   map[string]any{"type": "integer", "description": "макс. токенов на запрос (вход + выход), под него резервируется KV"},
	"concurrency":   map[string]any{"type": "integer", "description": "параллельных запросов на реплику"},
	"replicas":      map[string]any{"type": "integer", "description": "число реплик модели"},
	"mem_util":      map[string]any{"type": "number", "description": "доля VRAM, доступная движку, 0..1"},
	"input_tokens":  map[string]any{"type": "integer", "description": "средняя длина входа (prompt), токенов"},
	"output_tokens": map[string]any{"type": "integer", "description": "средняя длина ответа, токенов"},
	"interconnect":  map[string]any{"type": "string", "description": "id интерконнекта (см. list_interconnects)"},
	"ttft_slo_ms":   map[string]any{"type": "integer", "description": "целевое время до первого токена, мс"},
}

func without(m map[string]any, key string) map[string]any {
	out := map[string]any{}
	for k, v := range m {
		if k != key {
			out[k] = v
		}
	}
	return out
}

func decodeArgs[T any](raw json.RawMessage) (T, error) {
	var v T
	if len(raw) == 0 {
		return v, nil
	}
	err := json.Unmarshal(raw, &v)
	return v, err
}

var mcpTools = []mcpTool{
	{
		Name:        "list_models",
		Description: "Список LLM в MIP: число параметров и архитектура внимания.",
		InputSchema: obj(map[string]any{}),
		handler:     func(s *Store, _ json.RawMessage) (any, error) { return map[string]any{"models": s.Get().Models}, nil },
	},
	{
		Name: "model_metrics",
		Description: "Ключевые метрики моделей без привязки к нагрузке: всего/активных параметров, размер весов в native/bf16/fp8/fp4, " +
			"KV на токен, ГБ KV на 1 млн токенов, рекуррентный state на запрос, макс. контекст. Можно указать id.",
		InputSchema: obj(map[string]any{"id": map[string]any{"type": "string", "description": "id модели; пусто = все"}}),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			a, err := decodeArgs[struct{ ID string }](raw)
			if err != nil {
				return nil, err
			}
			cfg := s.Get()
			if a.ID == "" {
				return map[string]any{"models": AllMetrics(cfg)}, nil
			}
			m, ok := cfg.FindModel(a.ID)
			if !ok {
				return nil, fmt.Errorf("неизвестная модель %q", a.ID)
			}
			return Metrics(cfg, m), nil
		},
	},
	{
		Name:        "list_gpus",
		Description: "Типы GPU NVIDIA в MIP: VRAM, пропускная способность памяти, форматы, цена ₽/час, свободное количество.",
		InputSchema: obj(map[string]any{}),
		handler:     func(s *Store, _ json.RawMessage) (any, error) { return map[string]any{"gpus": s.Get().GPUs}, nil },
	},
	{
		Name:        "list_interconnects",
		Description: "Сети для передачи KV-кэша между prefill и decode и для PP между нодами: NVLink, InfiniBand, RoCE, TCP — скорость, RDMA, ₽/час за ноду.",
		InputSchema: obj(map[string]any{}),
		handler: func(s *Store, _ json.RawMessage) (any, error) {
			return map[string]any{"interconnects": s.Get().Interconnects}, nil
		},
	},
	{
		Name:        "get_platform",
		Description: "Параметры расчёта платформы MIP и значения по умолчанию (утилизация памяти, накладные, размеры TP, эффективность).",
		InputSchema: obj(map[string]any{}),
		handler:     func(s *Store, _ json.RawMessage) (any, error) { return s.Get().Platform, nil },
	},
	{
		Name: "calculate",
		Description: "Сколько GPU заданного типа нужно для модели в MIP (совмещённый prefill+decode): раскладка TP×PP, ноды, " +
			"память на GPU (веса / KV / накладные), макс. параллельность, токенов в час, стоимость в ₽ за час, за токен и за 1 млн токенов.",
		InputSchema: obj(calcProps, "model", "gpu"),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			req, err := decodeArgs[Request](raw)
			if err != nil {
				return nil, err
			}
			return Calculate(s.Get(), req)
		},
	},
	{
		Name:        "compare_gpus",
		Description: "Расчёт одной модели на всех типах GPU, лучшие первыми (помещается, нативный формат, дешевле за 1 млн токенов, меньше карт).",
		InputSchema: obj(without(calcProps, "gpu"), "model"),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			req, err := decodeArgs[Request](raw)
			if err != nil {
				return nil, err
			}
			res, err := Compare(s.Get(), req)
			return map[string]any{"results": res}, err
		},
	},
	{
		Name:        "recommend",
		Description: "Рекомендация типа и количества GPU для модели в MIP (самый дешёвый за токен подходящий вариант).",
		InputSchema: obj(without(calcProps, "gpu"), "model"),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			req, err := decodeArgs[Request](raw)
			if err != nil {
				return nil, err
			}
			res, err := Compare(s.Get(), req)
			if err != nil {
				return nil, err
			}
			if len(res) == 0 || !res[0].Fits {
				return map[string]any{"fits": false, "message": "ни один тип GPU не подходит при этих параметрах"}, nil
			}
			return res[0], nil
		},
	},
	{
		Name:        "matrix",
		Description: "GPU на реплику и ₽ за 1 млн токенов для всех моделей × всех GPU при одинаковой нагрузке.",
		InputSchema: obj(without(without(calcProps, "gpu"), "model")),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			req, err := decodeArgs[Request](raw)
			if err != nil {
				return nil, err
			}
			return BuildMatrix(s.Get(), req)
		},
	},
	{
		Name: "pd_disaggregation",
		Description: "Стоимость связки prefill/decode: совмещённый режим против раздельных пулов P/D с передачей KV-кэша " +
			"по NVLink / InfiniBand / RoCE (RDMA) / TCP. Возвращает размер пулов, объём и время передачи KV, загрузку сети, TTFT, " +
			"₽/час и ₽ за 1 млн выходных токенов, а также альтернативы (каждый GPU под prefill × каждый интерконнект), лучшие первыми.",
		InputSchema: obj(calcProps, "model", "gpu"),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			req, err := decodeArgs[Request](raw)
			if err != nil {
				return nil, err
			}
			return Disaggregation(s.Get(), req)
		},
	},
	{
		Name: "upsert_model",
		Description: "Добавить или обновить модель (по id). Поля: id, name, vendor, total_params_b, active_params_b, layers, kv_layers, " +
			"attention (gqa|mla|custom), kv_heads, head_dim, kv_lora_rank, qk_rope_dim, kv_bytes_per_token, kv_fixed, kv_extra_bytes_per_token, " +
			"state_mb_per_seq, experts, experts_per_tok, max_context, native_precision, checkpoint_gb, verified, source, note.",
		InputSchema: obj(map[string]any{"model": map[string]any{"type": "object"}}, "model"),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			a, err := decodeArgs[struct{ Model Model }](raw)
			if err != nil {
				return nil, err
			}
			if err := s.UpsertModel(a.Model); err != nil {
				return nil, err
			}
			return map[string]any{"ok": true, "model": a.Model}, nil
		},
	},
	{
		Name: "upsert_gpu",
		Description: "Добавить или обновить тип GPU (по id). Поля: id, name, arch, vram_gb, bandwidth_gbs, tflops_bf16, " +
			"precisions[], gpus_per_node, interconnect, price_per_hour (₽/час), available.",
		InputSchema: obj(map[string]any{"gpu": map[string]any{"type": "object"}}, "gpu"),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			a, err := decodeArgs[struct{ GPU GPU }](raw)
			if err != nil {
				return nil, err
			}
			if err := s.UpsertGPU(a.GPU); err != nil {
				return nil, err
			}
			return map[string]any{"ok": true, "gpu": a.GPU}, nil
		},
	},
	{
		Name:        "delete_model",
		Description: "Удалить модель по id.",
		InputSchema: obj(map[string]any{"id": map[string]any{"type": "string"}}, "id"),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			a, err := decodeArgs[struct{ ID string }](raw)
			if err != nil {
				return nil, err
			}
			return map[string]any{"ok": true}, s.DeleteModel(a.ID)
		},
	},
	{
		Name:        "delete_gpu",
		Description: "Удалить тип GPU по id.",
		InputSchema: obj(map[string]any{"id": map[string]any{"type": "string"}}, "id"),
		handler: func(s *Store, raw json.RawMessage) (any, error) {
			a, err := decodeArgs[struct{ ID string }](raw)
			if err != nil {
				return nil, err
			}
			return map[string]any{"ok": true}, s.DeleteGPU(a.ID)
		},
	},
}

type MCPServer struct{ store *Store }

// Handle обрабатывает одно JSON-RPC сообщение; для уведомлений возвращает nil.
func (m *MCPServer) Handle(raw []byte) *rpcResponse {
	var req rpcRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return &rpcResponse{JSONRPC: "2.0", ID: json.RawMessage("null"), Error: &rpcError{-32700, "ошибка разбора JSON"}}
	}
	if len(req.ID) == 0 {
		return nil // уведомление (notifications/initialized, cancelled, ...)
	}
	resp := &rpcResponse{JSONRPC: "2.0", ID: req.ID}
	switch req.Method {
	case "initialize":
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		_ = json.Unmarshal(req.Params, &p)
		version := mcpLatestProtocol
		if slices.Contains(mcpProtocols, p.ProtocolVersion) {
			version = p.ProtocolVersion
		}
		resp.Result = map[string]any{
			"protocolVersion": version,
			"capabilities":    map[string]any{"tools": map[string]any{"listChanged": false}},
			"serverInfo":      map[string]any{"name": "mip-gpu-calc", "title": "Калькулятор GPU для MIP", "version": "1.0.0"},
			"instructions": "Калькулятор размера кластера для LLM в MIP (платформа управляемого инференса). " +
				"Сначала list_models / list_gpus / list_interconnects, затем calculate, compare_gpus, recommend или pd_disaggregation. " +
				"Память: веса + KV-кэш (context_len × concurrency) + накладные на GPU ≤ VRAM × mem_util. " +
				"Стоимость — в рублях: ₽/час, ₽ за токен и за 1 млн сгенерированных токенов с учётом prefill.",
		}
	case "ping":
		resp.Result = map[string]any{}
	case "tools/list":
		resp.Result = map[string]any{"tools": mcpTools}
	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			resp.Error = &rpcError{-32602, "некорректные параметры"}
			break
		}
		idx := slices.IndexFunc(mcpTools, func(t mcpTool) bool { return t.Name == p.Name })
		if idx < 0 {
			resp.Error = &rpcError{-32602, "неизвестный инструмент: " + p.Name}
			break
		}
		out, err := mcpTools[idx].handler(m.store, p.Arguments)
		if err != nil {
			resp.Result = map[string]any{
				"content": []map[string]any{{"type": "text", "text": "ошибка: " + err.Error()}},
				"isError": true,
			}
			break
		}
		text, _ := json.MarshalIndent(out, "", "  ")
		result := map[string]any{"content": []map[string]any{{"type": "text", "text": string(text)}}}
		if strings.HasPrefix(strings.TrimSpace(string(text)), "{") {
			result["structuredContent"] = out
		}
		resp.Result = result
	default:
		resp.Error = &rpcError{-32601, "метод не найден: " + req.Method}
	}
	return resp
}

// ServeStdio — JSON-RPC построчно через stdin/stdout.
func (m *MCPServer) ServeStdio(in io.Reader, out io.Writer) error {
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	enc := json.NewEncoder(out)
	for sc.Scan() {
		line := sc.Bytes()
		if len(strings.TrimSpace(string(line))) == 0 {
			continue
		}
		if resp := m.Handle(line); resp != nil {
			if err := enc.Encode(resp); err != nil {
				return err
			}
		}
	}
	return sc.Err()
}

// ServeHTTP — транспорт Streamable HTTP с обычными JSON-ответами (без SSE).
func (m *MCPServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if origin := r.Header.Get("Origin"); origin != "" && !allowedOrigin(origin, r.Host) {
		http.Error(w, "запрещённый Origin", http.StatusForbidden)
		return
	}
	switch r.Method {
	case http.MethodPost:
	case http.MethodDelete:
		w.WriteHeader(http.StatusNoContent) // без сессий: завершать нечего
		return
	default:
		w.Header().Set("Allow", "POST, DELETE")
		http.Error(w, "метод не поддерживается", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 4<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	resp := m.Handle(body)
	if resp == nil {
		w.WriteHeader(http.StatusAccepted)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// allowedOrigin блокирует DNS-rebinding с чужих сайтов, пропуская встроенный UI и локальные инструменты.
func allowedOrigin(origin, host string) bool {
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	if u.Host == host {
		return true
	}
	h := u.Hostname()
	return h == "localhost" || h == "127.0.0.1" || h == "::1"
}

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
}
