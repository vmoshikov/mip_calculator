package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"
)

// JSONStore хранит значение в памяти и в JSON-файле. Если файла нет, он создаётся из встроенного
// значения по умолчанию; если файл изменили на диске, он перечитывается при следующем Get.
type JSONStore[T any] struct {
	mu       sync.RWMutex
	path     string
	val      T
	mtime    time.Time
	validate func(*T) error
}

func NewJSONStore[T any](path string, def []byte, validate func(*T) error) (*JSONStore[T], error) {
	s := &JSONStore[T]{path: path, validate: validate}
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		if err := os.WriteFile(path, def, 0o644); err != nil {
			return nil, fmt.Errorf("запись %s по умолчанию: %w", path, err)
		}
	}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *JSONStore[T]) Path() string { return s.path }

func (s *JSONStore[T]) load() error {
	st, err := os.Stat(s.path)
	if err != nil {
		return err
	}
	raw, err := os.ReadFile(s.path)
	if err != nil {
		return err
	}
	var v T
	if err := json.Unmarshal(raw, &v); err != nil {
		return fmt.Errorf("разбор %s: %w", s.path, err)
	}
	if s.validate != nil {
		if err := s.validate(&v); err != nil {
			return fmt.Errorf("некорректный %s: %w", s.path, err)
		}
	}
	s.val, s.mtime = v, st.ModTime()
	return nil
}

// Get возвращает текущее значение и перечитывает файл, если он изменился.
func (s *JSONStore[T]) Get() T {
	s.mu.RLock()
	st, err := os.Stat(s.path)
	stale := err == nil && !st.ModTime().Equal(s.mtime)
	v := s.val
	s.mu.RUnlock()
	if !stale {
		return v
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.load(); err != nil {
		// оставляем последнее рабочее значение: недописанный файл не должен ломать сервис
		logf("не удалось перечитать %s, оставляю прежнее: %v", s.path, err)
	}
	return s.val
}

func (s *JSONStore[T]) Replace(v T) error {
	if s.validate != nil {
		if err := s.validate(&v); err != nil {
			return err
		}
	}
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, s.path); err != nil {
		return err
	}
	st, err := os.Stat(s.path)
	if err != nil {
		return err
	}
	s.val, s.mtime = v, st.ModTime()
	return nil
}
