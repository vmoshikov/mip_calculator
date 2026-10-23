package main

import (
	"embed"
	"encoding/json"
	"flag"
	"io/fs"
	"net/http"
	"os"
	"time"
)

//go:embed web
var webFS embed.FS

func main() {
	addr := flag.String("addr", envOr("ADDR", ":8080"), "адрес HTTP-сервера")
	cfgPath := flag.String("config", envOr("CONFIG", "config.json"), "путь к конфигу; рядом лежат profiles.json и workloads.json (создаются из значений по умолчанию, если их нет)")
	stdio := flag.Bool("stdio", false, "работать как MCP-сервер через stdio вместо HTTP")
	flag.Parse()

	store, err := NewStore(*cfgPath)
	if err != nil {
		logf("конфиг: %v", err)
		os.Exit(1)
	}
	mcp := &MCPServer{store: store}

	if *stdio {
		if err := mcp.ServeStdio(os.Stdin, os.Stdout); err != nil {
			logf("stdio: %v", err)
			os.Exit(1)
		}
		return
	}

	mux := http.NewServeMux()
	registerAPI(mux, store)
	mux.Handle("/mcp", mcp)
	static, _ := fs.Sub(webFS, "web")
	mux.Handle("/", http.FileServerFS(static))

	srv := &http.Server{Addr: *addr, Handler: logRequests(mux), ReadHeaderTimeout: 10 * time.Second}
	logf("Калькулятор GPU для MIP: http://localhost%s  ·  MCP: http://localhost%s/mcp  ·  конфиг: %s, %s, %s",
		*addr, *addr, *cfgPath, store.Profiles.Path(), store.Workloads.Path())
	if err := srv.ListenAndServe(); err != nil {
		logf("http: %v", err)
		os.Exit(1)
	}
}

func registerAPI(mux *http.ServeMux, store *Store) {
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })

	mux.HandleFunc("GET /api/config", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, store.Get(), nil) })
	mux.HandleFunc("PUT /api/config", func(w http.ResponseWriter, r *http.Request) {
		var c Config
		if !readJSON(w, r, &c) {
			return
		}
		writeJSON(w, c, store.Replace(c))
	})

	mux.HandleFunc("POST /api/calculate", func(w http.ResponseWriter, r *http.Request) {
		var req Request
		if readJSON(w, r, &req) {
			res, err := Calculate(store.Get(), req)
			writeJSON(w, res, err)
		}
	})
	mux.HandleFunc("POST /api/compare", func(w http.ResponseWriter, r *http.Request) {
		var req Request
		if readJSON(w, r, &req) {
			res, err := Compare(store.Get(), req)
			writeJSON(w, res, err)
		}
	})
	mux.HandleFunc("POST /api/matrix", func(w http.ResponseWriter, r *http.Request) {
		var req Request
		if readJSON(w, r, &req) {
			res, err := BuildMatrix(store.Get(), req)
			writeJSON(w, res, err)
		}
	})

	mux.HandleFunc("POST /api/pd", func(w http.ResponseWriter, r *http.Request) {
		var req Request
		if readJSON(w, r, &req) {
			res, err := Disaggregation(store.Get(), req)
			writeJSON(w, res, err)
		}
	})
	mux.HandleFunc("GET /api/profiles", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, store.Profiles.Get(), nil) })
	mux.HandleFunc("PUT /api/profiles", func(w http.ResponseWriter, r *http.Request) {
		var d ProfilesDoc
		if readJSON(w, r, &d) {
			writeJSON(w, d, store.Profiles.Replace(d))
		}
	})
	mux.HandleFunc("GET /api/workloads", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, store.Workloads.Get(), nil) })
	mux.HandleFunc("PUT /api/workloads", func(w http.ResponseWriter, r *http.Request) {
		var d WorkloadsDoc
		if readJSON(w, r, &d) {
			writeJSON(w, d, store.Workloads.Replace(d))
		}
	})
	mux.HandleFunc("POST /api/plan", func(w http.ResponseWriter, r *http.Request) {
		var req PlanRequest
		if readJSON(w, r, &req) {
			res, err := PlanLimits(store, req)
			writeJSON(w, res, err)
		}
	})
	mux.HandleFunc("GET /api/models/metrics", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, AllMetrics(store.Get()), nil)
	})
	mux.HandleFunc("PUT /api/models", func(w http.ResponseWriter, r *http.Request) {
		var m Model
		if readJSON(w, r, &m) {
			writeJSON(w, m, store.UpsertModel(m))
		}
	})
	mux.HandleFunc("DELETE /api/models/{id}", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]bool{"ok": true}, store.DeleteModel(r.PathValue("id")))
	})
	mux.HandleFunc("PUT /api/gpus", func(w http.ResponseWriter, r *http.Request) {
		var g GPU
		if readJSON(w, r, &g) {
			writeJSON(w, g, store.UpsertGPU(g))
		}
	})
	mux.HandleFunc("DELETE /api/gpus/{id}", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]bool{"ok": true}, store.DeleteGPU(r.PathValue("id")))
	})
}

func readJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20)).Decode(v); err != nil {
		writeJSON(w, nil, err)
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, v any, err error) {
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		v = map[string]string{"error": err.Error()}
	}
	_ = json.NewEncoder(w).Encode(v)
}

func logRequests(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		h.ServeHTTP(w, r)
		if r.URL.Path != "/healthz" {
			logf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Microsecond))
		}
	})
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
