# Калькулятор GPU для MIP

Сервис, который отвечает на три вопроса о запуске большой языковой модели на **MIP (платформа управляемого инференса)**:

1. **Сколько GPU NVIDIA нужно**: раскладка TP×PP, число нод, распределение памяти на каждой карте.
2. **Сколько токенов в час это даст**: скорость обработки входа (prefill), генерации (decode) и одного потока.
3. **Сколько это стоит в рублях**: за час, за 1 млн токенов, за один токен и за запрос. Отдельно считается, выгоднее ли разнести prefill и decode на разные пулы GPU с передачей KV-кэша по RDMA/RoCE.
4. **Какие лимиты поставить инсталляции SGLang**: `--max-running-requests` и предел RPS на инстанс и кластер, при которых выполняется цель TTFT P99 (и TPOT P99). Модель скорости подгоняется под замеры из SGLang cookbook или под ваши собственные.

Написан на Go без внешних зависимостей, это один бинарник. Внутри веб-интерфейс, REST API и MCP-сервер для агентов.

---

## Содержание

- [Быстрый старт](#быстрый-старт)
- [Интерфейс](#интерфейс)
- [Лимиты SGLang: max-running-requests и RPS](#лимиты-sglang-max-running-requests-и-rps)
- [Методика расчёта](#методика-расчёта)
- [Настройка](#настройка)
- [REST API](#rest-api)
- [MCP для агентов](#mcp-для-агентов)
- [Модели и GPU в комплекте](#модели-и-gpu-в-комплекте)
- [Структура проекта](#структура-проекта)
- [Ограничения](#ограничения)
- [Источники](#источники)

---

## Быстрый старт

Нужен Go 1.22 или новее.

```bash
cd gpucalc
go build -o gpucalc .
./gpucalc
```

Откройте <http://localhost:8080>. При первом запуске рядом с конфигом появятся три файла:

| Файл | Что внутри |
|---|---|
| `config.json` | платформа, GPU, сети, модели (архитектура) |
| `profiles.json` | профили движка SGLang: флаги запуска, версия, ёмкость, замеры скорости (112 профилей из cookbook) |
| `workloads.json` | профили нагрузки: распределения длин входа и выхода, prefix cache — **заглушки** |

| Флаг | Переменная окружения | По умолчанию | Назначение |
|---|---|---|---|
| `-addr` | `ADDR` | `:8080` | адрес HTTP-сервера |
| `-config` | `CONFIG` | `config.json` | путь к конфигу; `profiles.json` и `workloads.json` лежат в том же каталоге. Отсутствующие файлы создаются из встроенных значений |
| `-stdio` | — | выключен | работать как MCP-сервер через stdin/stdout вместо HTTP |

**Docker:**

```bash
docker build -t mip-gpucalc .
docker run -p 8080:8080 -v $PWD/data:/data mip-gpucalc   # конфиг: ./data/config.json
```

**Тесты:** `go test ./...`

> **Обновление со старой версии.** Встроенный список моделей изменился. Если `config.json` создан старой версией, удалите его (или перенесите свои правки): иначе в нём не будет новых моделей, на которые ссылаются профили SGLang.

---

## Интерфейс

На вкладках «Калькулятор», «Prefill / Decode» и «Матрица» слева общая панель параметров: модель, GPU, точность весов, формат KV-кэша, контекст, число одновременных запросов на реплику, реплик и доля VRAM. Ниже — нагрузка: токенов на входе и на выходе, сеть между нодами и цель по времени до первого токена (TTFT). У вкладки «Лимиты SGLang» своя форма.

| Вкладка | Что показывает |
|---|---|
| **Калькулятор** | число GPU и нод, раскладку TP×PP, максимум параллельных запросов, скорость одного потока, TTFT, токенов в час, ₽/час, ₽ за 1 млн токенов и за 1 токен, раздельно вход и выход, ₽ за запрос. Ниже: полоса памяти одной GPU, схема кластера по нодам, сравнение всех GPU по ₽ за 1 млн токенов |
| **Лимиты SGLang** | рекомендуемый `--max-running-requests`, лимит RPS на инстанс и кластер, TTFT и TPOT на пределе, график TTFT P99 от RPS, таблица по всем лимитам, сравнение вариантов (Engram на GPU или в RAM), сверка модели с замерами, команда запуска |
| **Prefill / Decode** | совмещённый режим против раздельных пулов: схема «prefill → сеть → decode», карточки вариантов, вывод об экономии, таблица альтернатив (каждый GPU под prefill × каждая сеть) |
| **Модели** | карточки моделей с ключевыми метриками: параметры, доля активных, размер чекпоинта, веса в FP8/FP4, эксперты, KV на токен и на 1 млн токенов, state на запрос, контекст, ссылка на Hugging Face |
| **Матрица** | GPU на реплику и ₽ за 1 млн токенов для всех моделей × всех GPU |
| **Настройки** | редактирование платформы, GPU, сетей и моделей в таблицах; редакторы `profiles.json` и `workloads.json`; правка конфига JSON целиком |
| **MCP** | адрес сервера, команды подключения и список инструментов |

Клик по строке сравнения, ячейке матрицы или карточке модели открывает соответствующий расчёт.

---

## Лимиты SGLang: max-running-requests и RPS

### Что отвечает

Типичный вопрос эксплуатации: какой поставить `--max-running-requests` и какой лимит RPS на роутере для инсталляции модели, чтобы выполнялась цель, например TTFT P99 < 40 с. Например: DeepSeek-V4.1-Flash, 3 сервера B300, TP4, веса fp4/fp8, Engram на GPU или в RAM хоста.

### Термины

| Термин | Что значит |
|---|---|
| `--max-running-requests` | сколько запросов инстанс SGLang обрабатывает **одновременно** (в батче). Остальные ждут допуска в очереди — это ожидание входит в TTFT |
| RPS | входящий поток: сколько новых запросов **в секунду** приходит на инстанс. По закону Литтла: одновременных ≈ RPS × среднее время запроса |
| TTFT | время до первого токена: ожидание допуска + обработка входа (prefill) |
| TPOT | время на каждый следующий токен; растёт с числом одновременных запросов |
| инстанс | один процесс SGLang на группе GPU. 3 сервера по 8 B300 при TP4 = 6 инстансов |

В калькуляторе памяти поле «Одновременных запросов на реплику» — то же самое, что `--max-running-requests`, а не RPS.

### Как считается

1. **Профиль движка** (`profiles.json`) — конфигурация из SGLang cookbook: GPU, TP/EP, флаги, версия, образ и замеры `bench_serving`: TTFT и TPOT при параллельности 1…512 и разных длинах входа.
2. **Модель скорости** подгоняется под замеры:
   - шаг decode = `база + n × (на запрос + на запрос × контекст/1000)`;
   - prefill-батч = `накладные на батч + на запрос × число запросов + токены / скорость + внимание`.
   - Чего не хватает в замерах, берётся из roofline по характеристикам GPU и архитектуре модели.
3. **Калибровка.** Замеры cookbook сделаны в закрытом цикле: bench_serving держит N запросов в работе. Симулятор прогоняет тот же бенчмарк и подстраивает prefill и decode, пока не совпадут измеренные TTFT и TPOT. Иначе помехи от prefill, которые уже есть в TPOT, учитывались бы дважды. Результат сверки показан в таблице «Насколько модель совпадает с замерами».
4. **Ёмкость:** KV-пул в токенах и слоты state у гибридных моделей. Берётся из профиля (`capacity.kv_pool_tokens` из лога SGLang), из измеренной плотности пула или оценивается по памяти: `VRAM × mem-fraction-static − веса`.
5. **Симуляция очереди одного инстанса** (`sim.go`):
   - запросы приходят пуассоновским потоком с длинами из профиля нагрузки;
   - допуск, если свободен слот `max-running-requests` и хватает KV-пула и слотов state;
   - prefill приоритетнее decode, как в планировщике SGLang по умолчанию, батчами по `chunked-prefill-size`, с учётом `--prefill-decode-interval`;
   - prefix cache уменьшает число считаемых токенов входа.
6. **Поиск предела.** Для каждого `max-running-requests` (8…512) бинарным поиском ищется максимальный RPS, при котором TTFT P99 ≤ цели, TPOT P99 ≤ цели (если задана) и очередь не растёт. Рекомендуется наименьший лимит, дающий не меньше 95% максимального RPS: дальше рост лимита почти не добавляет пропускной способности, но ухудшает TPOT.
7. **Лимит для роутера** = предел × запас (по умолчанию 80%). Кластер = лимит × число инстансов.

### Пример: DeepSeek-V4.1-Flash на 3×8 B300, TP4/EP4

Профиль `deepseek-v4.1-flash-b300-fp4-high-throughput` (замеры [issue #41946](https://github.com/sgl-project/sglang/issues/41946)). Средняя ошибка модели на замерах: TTFT 11%, TPOT 12%. Нагрузка — заглушка «агенты»: вход p50 30 тыс. / p99 120 тыс., выход p50 800 / p99 6 тыс., prefix cache 90%. Цель TTFT P99 < 40 с, запас 80%.

| Цель | `--max-running-requests` | Предел RPS на инстанс | Лимит на кластер (6 инстансов) | TTFT P50 / P99 на пределе | TPOT P99 |
|---|---|---|---|---|---|
| только TTFT P99 < 40 с | 384 | 2,25 | 10,8 | 0,37 с / 7,4 с | 165 мс |
| + TPOT P99 ≤ 60 мс | 96 | 1,86 | 8,9 | 0,38 с / 7,7 с | 48 мс |

Что видно из расчёта:

- **Цель TTFT P99 < 40 с почти не ограничивает.** При `max-running-requests` от 64 и выше предел задаёт пропускная способность. Ниже предела TTFT P99 держится в единицах секунд, выше — очередь растёт без ограничения. Реальный рычаг — лимит RPS на роутере, и выбор между пропускной способностью и TPOT.
- **Engram в RAM хоста** на B300 лимиты не меняет. KV-пул вырастает с 38,4 до 45,5 млн токенов, но пиковая занятость пула при этой нагрузке около 40%. Вынос имеет смысл, только если упираетесь в KV-пул: очень длинные контексты при большом числе одновременных запросов. Влияние выноса на скорость не измерено — в профиле стоит множитель 1,0 (заглушка).
- **Low-latency профиль** (DSpark) даёт меньший предел: 1,27 против 2,25 RPS. Модель хуже описывает его замеры (ошибка до 70%, у speculative decoding нелинейная зависимость).

Цифры зависят от нагрузки. Пока профиль нагрузки — заглушка, это иллюстрация метода, а не рекомендация для прода.

### Как заменить заглушки своими данными

**Нагрузка** (`workloads.json` или поля формы на вкладке). Нужны распределения длин входа и выхода и доля prefix cache:

| Что | Откуда взять |
|---|---|
| длины входа и выхода, p50 / p99 | логи запросов (роутер, шлюз, приложение: `usage.prompt_tokens`, `usage.completion_tokens`); средние — из счётчиков SGLang `sglang:prompt_tokens_total` и `sglang:generation_tokens_total`, делённых на число запросов |
| доля prefix cache | `sglang:cache_hit_rate` |
| проверка результата | `sglang:num_running_reqs`, `sglang:num_queue_reqs`, гистограммы `sglang:time_to_first_token_seconds` и `sglang:inter_token_latency_seconds` при известном RPS |

Формат распределения: `lognormal` (p50 + p99), `fixed` (value) или `buckets` (`[[токенов, доля], ...]`).

**Замеры скорости** — свой прогон `bench_serving` на вашей версии SGLang и вашем железе:

```bash
python -m sglang.bench_serving --backend sglang --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1 \
  --max-concurrency 16 --num-prompts 80 --output-file bench.jsonl
# повторить для --max-concurrency 1, 16, 64, 128 и нескольких длин входа
python3 tools/bench_to_profile.py bench.jsonl profiles.json <id профиля>
```

Калькулятор подхватит файл без перезапуска и переподгонит модель. Чем больше точек (разная параллельность и длины), тем точнее. Ёмкость — из лога запуска SGLang: `max_total_num_tokens` → `capacity.kv_pool_tokens`.

**Новая версия движка** — скопируйте профиль с новым `id`, укажите `engine.version` и добавьте замеры этой версии. Профили разных версий можно сравнить на вкладке.

**Обновление из SGLang cookbook:** `tools/cookbook/update.sh` скачивает сниппеты конфигураций и бенчмарков cookbook и пересобирает `profiles.default.json`.

---

## Методика расчёта

Все формулы находятся в `calc.go` (память, скорость, стоимость) и `pd.go` (prefill/decode).

### 1. Память

На каждой GPU должно выполняться:

```
веса / (TP × PP)  +  KV / шарды / PP  +  state / (TP × PP)  +  накладные  ≤  VRAM × доля VRAM
```

| Составляющая | Формула |
|---|---|
| Веса, точность `native` | размер официального чекпоинта `checkpoint_gb`. Нужен для смешанных форматов, например FP8 и эксперты в FP4 |
| Веса, остальные форматы | `параметры × байт формата × (1 + weights_overhead_pct)`. Байт: bf16 = 2, fp8/int8 = 1, fp4 ≈ 0,56, int4 ≈ 0,53 |
| KV на токен, GQA | `2 × слоёв с KV × KV-голов × размер головы × байт KV` |
| KV на токен, MLA | `слоёв × (kv_lora_rank + qk_rope_dim) × байт KV` |
| Дополнительный KV | `kv_extra_bytes_per_token`, например индексер DSA у GLM-5.2 и DeepSeek-V3.2 |
| Свой формат KV | `kv_bytes_per_token` с `kv_fixed` (сжатый FP4-KV у DeepSeek-V4.1-Flash) |
| KV всего | `KV на токен × контекст × параллельных запросов` |
| Шарды KV | GQA: `min(TP, KV-голов)`. MLA: 1, кэш дублируется на каждом TP-ранге, если не включён `mla_dp_attention` |
| State | `state_mb_per_seq × параллельных запросов`: линейное внимание у гибридов Qwen3.8, от контекста не зависит |

Раскладки перебираются по возрастанию числа GPU: сначала TP из `tp_sizes` внутри одной ноды, затем целые ноды с конвейерным параллелизмом (PP) до `max_gpus_per_replica`. Берётся первая, в которую всё помещается.

### 2. Скорость

**Decode** считается как roofline на один шаг:

```
шаг = max(чтение HBM, вычисления) + layer_overhead_us × слоёв / PP
```

- **Чтение HBM** — прочитанные за шаг веса плюс KV всех запросов (`× avg_context_fill`), делённые на `пропускную способность HBM × decode_efficiency`.
- У MoE доля задействованных экспертов при batch запросов равна `1 − (1 − k/E)^batch`. Общие слои читаются всегда.
- **Вычисления:** `2 × активные параметры × batch / (TFLOPS × compute_efficiency × ускорение формата)`. Ускорение fp8 ×2, fp4 ×4, и только если GPU поддерживает формат аппаратно.
- **Задержка на слой** покрывает all-reduce между TP-рангами и запуск ядер.
- **Пропускная способность реплики:** `batch / шаг`. **Скорость одного потока:** `1 / (шаг при batch = 1 × PP)`.

**Prefill** упирается в вычисления:

```
FLOP на токен = 2 × активные параметры + 4 × слоёв × hidden × min(вход / 2, attn_topk)
скорость prefill = GPU × TFLOPS × prefill_efficiency × ускорение формата / FLOP на токен
```

### 3. Стоимость в рублях (совмещённый режим)

На один запрос реплика тратит `вход / скорость prefill + выход / скорость decode` секунд.

```
запросов в час   = 3600 / (вход / prefill + выход / decode) × реплик
токенов в час    = запросов в час × токенов на выходе
₽ в час          = GPU × price_per_hour  +  ноды × price_per_node_hour (только если PP > 1)
₽ за токен       = ₽ в час / токенов в час
₽ за 1 млн       = ₽ за токен × 1 000 000
₽ за запрос      = ₽ в час / запросов в час
вход / выход     = ₽ за запрос делится пропорционально времени prefill и decode
```

Пример: GLM-5.2, FP8, 16× H200 (TP 8 × PP 2), 64 параллельных запроса, вход 4096, выход 512 токенов.

| Показатель | Значение |
|---|---|
| Скорость одного потока | ~164 токенов/с |
| Пропускная способность реплики | ~2 373 токенов/с, упор в память |
| Генерация в час | ~7,76 млн токенов |
| Стоимость | 11 100 ₽/час → ~1 430 ₽ за 1 млн токенов → 0,00143 ₽ за токен |
| Один запрос | ~0,73 ₽ |

### 4. Prefill / Decode: раздельные пулы

В совмещённом режиме одни и те же GPU обрабатывают вход и генерируют ответ, prefill прерывает генерацию. В раздельном режиме (P/D disaggregation) вход обрабатывает отдельный пул GPU, а готовый KV-кэш передаётся по сети в пул decode.

1. **Пул decode** берётся из параметров формы: GPU, реплики, параллельные запросы. Его пропускная способность задаёт поток запросов: `запросов/с = decode-токенов/с / токенов на выходе`.
2. **Пул prefill:** инстанс подбирается под модель с `prefill_batch` одновременных входов. Число инстансов = `⌈запросов/с × вход / скорость инстанса⌉`.
3. **Передача KV:** объём на запрос = `вход × KV на токен + state`. Время = `объём / (линий × Гбит/с / 8 × efficiency) + задержка`, где линий = `min(GPU инстанса prefill, GPU реплики decode)`. Без RDMA время умножается на 1,5 из-за копирования через память хоста.
4. **Загрузка сети:** `поток KV / ёмкость NIC пула`. Если больше 100%, пропускная способность урезается, и вариант помечается.
5. **TTFT** = `вход / скорость инстанса prefill + передача KV`.
6. **Стоимость:** `GPU prefill × цена + GPU decode × цена + ноды × цена сети за ноду`.

Ограничение NVLink: работает только внутри ноды, поэтому оба пула должны быть на одном типе GPU и вместе поместиться в одну ноду.

**Альтернативы** — это совмещённый режим плюс каждый тип GPU под prefill × каждая сеть. Порядок: допустимые → укладываются в цель TTFT → дешевле за 1 млн токенов → меньше TTFT.

Что показывает расчёт:

| Сценарий | Совмещённый | Лучший вариант | Итог |
|---|---|---|---|
| GLM-5.2 (MoE), вход 2 тыс. | — | совмещённый | prefill занимает малую долю GPU-времени, лишний пул не окупается |
| Llama 3.1 405B (dense), вход 32 тыс. → 512, 16 параллельных, цель TTFT 5 с | 14 566 ₽/млн, TTFT 3,9 с | prefill 12× B200 → decode 8× H200 по RoCE 200G: KV 8,5 ГБ за 113 мс, сеть загружена на 5%, TTFT 3,1 с, **9 437 ₽/млн** | дешевле на **35%** |

Даже когда раздельные пулы дороже, они дают стабильную задержку между токенами: генерация не прерывается обработкой входа.

---

## Настройка

Данные лежат в трёх файлах: `config.json`, `profiles.json`, `workloads.json`. Править можно тремя способами:

- **Вручную в файле:** изменения подхватываются без перезапуска при следующем запросе. Если файл битый, остаётся предыдущий рабочий конфиг.
- **Во вкладке «Настройки».**
- **Через REST API или MCP** (`upsert_model`, `upsert_gpu`, `PUT /api/config`, `PUT /api/profiles`, `PUT /api/workloads`).

> ⚠️ **Цены GPU и сети в `config.default.json` — заглушки.** Подставьте тарифы MIP.

### `platform` — платформа

| Поле | По умолчанию | Смысл |
|---|---|---|
| `name`, `currency` | MIP, `RUB` | название и валюта |
| `gpu_memory_utilization` | 0,9 | доля VRAM, которую получает движок (vLLM/SGLang) |
| `overhead_per_gpu_gb` | 3 | активации, CUDA graphs, буферы NCCL на каждой GPU |
| `weights_overhead_pct` | 3 | надбавка к весам: неквантованные эмбеддинги, скейлы |
| `tp_sizes` | 1, 2, 4, 8 | допустимые размеры TP внутри ноды |
| `max_gpus_per_replica` | 64 | предел GPU на одну реплику |
| `mla_dp_attention` | false | делить MLA-кэш между TP-рангами (DP-attention) |
| `decode_efficiency` | 0,55 | реальная доля пиковой пропускной способности HBM |
| `compute_efficiency` | 0,4 | доля пиковых TFLOPS на decode |
| `prefill_efficiency` | 0,5 | доля пиковых TFLOPS на prefill (MFU) |
| `avg_context_fill` | 0,5 | средняя заполненность контекста при чтении KV |
| `layer_overhead_us` | 50 | фиксированная задержка на слой за шаг decode, мкс |
| `prefill_batch` | 4 | одновременных входов в одном prefill-инстансе |
| `defaults.*` | — | значения формы по умолчанию: `precision` (`native`), `kv_dtype` (`fp8`), `context_len` (32 768), `concurrency` (64), `replicas` (1), `input_tokens` (4096), `output_tokens` (512), `interconnect` (`roce-400`), `ttft_slo_ms` (2000) |

### `gpus` — GPU

| Поле | Смысл |
|---|---|
| `id`, `name`, `arch` | идентификатор, название, архитектура |
| `vram_gb` | память, ГБ |
| `bandwidth_gbs` | пропускная способность HBM, ГБ/с |
| `tflops_bf16` | плотные TFLOPS в BF16 |
| `precisions` | аппаратно поддерживаемые форматы (`bf16`, `fp8`, `fp4`, `int8`) |
| `gpus_per_node`, `interconnect` | GPU в ноде, связь внутри ноды |
| `price_per_hour` | **₽/час за одну GPU** |
| `available` | сколько карт свободно в MIP (0 — без ограничения). Если не хватает, выводится предупреждение |

### `interconnects` — сеть

| Поле | Смысл |
|---|---|
| `id`, `name` | идентификатор, название |
| `gbps_per_gpu` | пропускная способность на одну GPU, Гбит/с (rail-optimized: одна NIC на GPU) |
| `efficiency` | реальная доля пиковой скорости |
| `rdma` | GPUDirect RDMA. Без него передача медленнее в 1,5 раза |
| `latency_us` | задержка на передачу, мкс |
| `price_per_node_hour` | **₽/час за ноду**: сетевые карты, порты коммутатора |
| `same_node_only` | работает только внутри ноды (NVLink) |

### `models` — модели

| Поле | Смысл |
|---|---|
| `id`, `name`, `vendor` | идентификатор, название, разработчик |
| `total_params_b`, `active_params_b` | всего и активных на токен параметров, млрд |
| `layers`, `kv_layers` | слоёв всего и слоёв с полным KV (у гибридов; 0 — все) |
| `attention` | `gqa`, `mla` или `custom` (тогда нужен `kv_bytes_per_token`) |
| `kv_heads`, `head_dim` | для GQA |
| `kv_lora_rank`, `qk_rope_dim` | для MLA |
| `kv_bytes_per_token`, `kv_fixed` | KV на токен вручную: в bf16, либо как есть при `kv_fixed` |
| `kv_extra_bytes_per_token` | дополнительные кэши (индексер DSA), байт на токен |
| `state_mb_per_seq` | state линейного внимания на запрос, МБ |
| `hidden_size`, `attn_topk` | для FLOPs внимания на prefill; top-k у разреженного внимания |
| `experts`, `experts_per_tok` | число экспертов MoE и сколько активно на токен |
| `max_context` | максимальный контекст |
| `native_precision`, `checkpoint_gb` | основной формат и размер официального чекпоинта |
| `verified`, `source`, `note` | данные проверены, ссылка на источник, заметка |

Добавить модель можно через вкладку «Настройки» (кнопка «+ Модель») или агентом через `upsert_model`. Значения берутся из `config.json` модели на Hugging Face:

| Поле калькулятора | Поле в `config.json` модели |
|---|---|
| `layers` | `num_hidden_layers` |
| `kv_heads` | `num_key_value_heads` |
| `head_dim`, `hidden_size` | `head_dim`, `hidden_size` |
| `kv_lora_rank`, `qk_rope_dim` | `kv_lora_rank`, `qk_rope_head_dim` |
| `experts`, `experts_per_tok` | `n_routed_experts` / `num_experts`, `num_experts_per_tok` |
| `kv_layers` | число слоёв `full_attention` в `layer_types` |

### `profiles.json` — профили движка SGLang

| Поле | Смысл |
|---|---|
| `id`, `name`, `model`, `gpu`, `quant`, `strategy` | идентификатор, подпись, модель и GPU из `config.json`, квантизация, стратегия рецепта |
| `engine` | `name`, `version`, `image`, `tp`, `ep`, `attn_dp`, `gpus` (GPU на инстанс), `max_running_requests`, `mem_fraction_static`, `chunked_prefill_size`, `cuda_graph_max_bs`, `prefill_decode_interval`, `speculative`, `kv_cache_dtype`, `flags[]`, `env[]` |
| `capacity` | `kv_pool_tokens` (из лога: `max_total_num_tokens`), `state_slots`, `weights_gib_per_gpu`, `kv_tokens_per_gib`, `offload_gib_per_gpu` (веса в RAM хоста). Пустые поля оцениваются по памяти |
| `measurements[]` | точки bench_serving: `concurrency`, `isl`, `osl`, `ttft_ms`, `tpot_ms`, `tok_s_per_gpu` (средние значения) |
| `perf` | параметры модели скорости вручную — тогда подгонка не выполняется |
| `perf_scale` | множители `decode` и `prefill` к подогнанной модели (например, для неизмеренного варианта) |
| `verified`, `source`, `note` | проверен ли рецепт в cookbook, ссылка, заметка |

### `workloads.json` — профили нагрузки

| Поле | Смысл |
|---|---|
| `id`, `name`, `stub` | идентификатор, подпись; `stub: true` — заглушка, расчёт помечается предупреждением |
| `input`, `output` | распределение длины: `{"distribution": "lognormal", "p50", "p99"}`, `{"distribution": "fixed", "value"}` или `{"distribution": "buckets", "buckets": [[токенов, доля], ...]}` |
| `prefix_cache_hit` | доля токенов входа, взятых из prefix cache (0…0,99) |

---

## REST API

| Метод и путь | Описание |
|---|---|
| `GET /api/config` | весь конфиг |
| `PUT /api/config` | заменить конфиг целиком (с проверкой) |
| `POST /api/calculate` | расчёт для одной модели и одного GPU |
| `POST /api/compare` | то же по всем GPU, лучшие первыми |
| `POST /api/pd` | совмещённый режим, раздельный P/D и альтернативы |
| `POST /api/matrix` | модели × GPU |
| `GET /api/models/metrics` | ключевые метрики моделей |
| `POST /api/plan` | лимиты SGLang: `{profile \| model+gpu, workload \| workload_inline, ttft_p99_ms, tpot_p99_ms?, servers, gpus_per_server, instances?, headroom?, mrr_candidates?}` |
| `GET/PUT /api/profiles` | профили движка |
| `GET/PUT /api/workloads` | профили нагрузки |
| `PUT /api/models`, `DELETE /api/models/{id}` | добавить/обновить или удалить модель |
| `PUT /api/gpus`, `DELETE /api/gpus/{id}` | добавить/обновить или удалить GPU |
| `GET /healthz` | проверка живости |

Параметры запроса для `calculate`, `compare`, `pd` и `matrix` необязательны, кроме `model` и `gpu`. Недостающие берутся из `platform.defaults`.

```json
{
  "model": "glm-5.2",
  "gpu": "h200-141g",
  "precision": "fp8",
  "kv_dtype": "fp8",
  "context_len": 32768,
  "concurrency": 64,
  "replicas": 1,
  "mem_util": 0.9,
  "input_tokens": 4096,
  "output_tokens": 512,
  "interconnect": "roce-400",
  "ttft_slo_ms": 2000
}
```

**Пример расчёта:**

```bash
curl -s localhost:8080/api/calculate -d '{"model":"glm-5.2","gpu":"h200-141g","precision":"fp8"}'
```

```json
{
  "fits": true,
  "tp": 8, "pp": 2,
  "gpus_per_replica": 16, "total_gpus": 16, "total_nodes": 2,
  "max_concurrency": 96,
  "decode_tok_s": 164.2,
  "throughput_tok_s": 2373.4,
  "ttft_ms": 21.7,
  "tokens_per_hour": 7762517,
  "cost_per_hour": 11100,
  "cost_per_1m_tokens": 1429.95,
  "cost_per_token": 0.00143,
  "cost_per_1m_input": 16.35,
  "cost_per_1m_output": 1299.12,
  "cost_per_request": 0.732,
  "bottleneck": "память",
  "warnings": ["несколько нод: TP=8 × PP=2, нужен быстрый межузловой интерконнект (IB/RoCE)"]
}
```

Ответ сокращён. Полный ответ содержит также модель, GPU, память на одну GPU (`per_gpu`) и другие поля.

**Пример prefill/decode:**

```bash
curl -s localhost:8080/api/pd -d '{
  "model":"llama-3.1-405b","gpu":"h200-141g","precision":"fp8",
  "context_len":65536,"input_tokens":32768,"output_tokens":512,"concurrency":16,"ttft_slo_ms":5000
}'
```

В ответе `colocated` — совмещённый режим, `disagg` — раздельный на том же GPU с выбранной сетью, `best` — лучший вариант, `alternatives` — все варианты, `savings_pct` — экономия лучшего относительно совмещённого.

---

## MCP для агентов

Сервер поддерживает протокол MCP (версии 2025-06-18, 2025-03-26 и 2024-11-05) с двумя транспортами:

- **Streamable HTTP** — `http://localhost:8080/mcp`.
- **stdio** — флаг `-stdio`.

Запросы из браузера с чужим `Origin` отклоняются (защита от DNS-rebinding).

**Claude Code:**

```bash
claude mcp add --transport http mip-gpu-calc http://localhost:8080/mcp
```

**Claude Desktop, Cursor и другие клиенты через stdio:**

```json
{
  "mcpServers": {
    "mip-gpu-calc": {
      "command": "/путь/к/gpucalc",
      "args": ["-stdio", "-config", "/путь/к/config.json"]
    }
  }
}
```

| Инструмент | Что делает |
|---|---|
| `list_models` | список моделей с архитектурой |
| `model_metrics` | ключевые метрики моделей (все или по `id`) |
| `list_gpus` | GPU: VRAM, HBM, форматы, ₽/час, остаток |
| `list_interconnects` | сети: скорость, RDMA, задержка, ₽/час за ноду |
| `get_platform` | параметры платформы и значения по умолчанию |
| `calculate` | GPU, раскладка, память, токены в час, ₽ за час, токен и 1 млн |
| `compare_gpus` | расчёт на всех GPU, лучшие первыми |
| `recommend` | самый дешёвый за токен подходящий GPU |
| `pd_disaggregation` | совмещённый режим против P/D с передачей KV по NVLink, IB, RoCE или TCP, плюс альтернативы |
| `matrix` | модели × GPU |
| `list_profiles` | профили SGLang: GPU, TP/EP, версия, флаги, число замеров (фильтры `model`, `gpu`) |
| `list_workloads` | профили нагрузки (заглушки помечены) |
| `plan_limits` | `--max-running-requests` и лимит RPS на инстанс и кластер под цель TTFT P99 / TPOT P99 |
| `upsert_model`, `upsert_gpu` | добавить или обновить модель или GPU |
| `delete_model`, `delete_gpu` | удалить |

Примеры вопросов агенту:
- «Какие `max-running-requests` и лимит RPS поставить DeepSeek-V4.1-Flash на 3 серверах B300 TP4, чтобы TTFT P99 был меньше 40 с? Сравни Engram на GPU и в RAM.»
- «Сколько H200 нужно для GLM-5.3 на 200 одновременных запросов с контекстом 64K и сколько будет стоить 1 млн токенов?»

---

## Модели и GPU в комплекте

**Модели.** Семейства GigaChat, GLM, Qwen, DeepSeek, Kimi, MiniMax, Mistral. Архитектура — из `config.json` моделей на Hugging Face и SGLang cookbook (октябрь 2026). Профили — сколько конфигураций SGLang с замерами есть в `profiles.json`.

| Модель | Параметры (всего / активных) | Внимание | Контекст | Профилей SGLang |
|---|---|---|---|---|
| GigaChat 3.5 (432B-A28B) | 434 млрд / 28 млрд | гибрид MLA + GatedDeltaNet | 256 тыс. | 2 (H100) |
| GigaChat 3.1 Ultra (702B-A36B) | 702 млрд / 36 млрд | MLA | 256 тыс. | — |
| GigaChat 3.1 Lightning (10B-A1.8B) | 10 млрд / 1,8 млрд | MLA | 256 тыс. | — |
| GLM-5.2 | 753 млрд / 40 млрд | MLA + DSA | 1 млн | 15 |
| GLM-5.3 | 753 млрд / ~40 млрд | MLA + DSA | 1 млн | 15 |
| GLM-5.3-Flash | 320 млрд / 18 млрд | гибрид DSA + KDA | 1 млн | 7 |
| Qwen3.8-Max (2.4T-A95B) | 2,4 трлн / 95 млрд | гибрид GQA + GDN | 256 тыс. | — |
| Qwen3.8-Flash-Next | 180 млрд / 6 млрд | гибрид GQA + GDN | 256 тыс. | — |
| Qwen3.8-27B | 27 млрд, плотная | гибрид GQA + GDN | 256 тыс. | 6 (GB300) |
| DeepSeek-V4.1-Flash | 763 млрд / 16 млрд | сжатый KV, DSA | 1 млн | 4 (B300) |
| DeepSeek-V4-Flash | 284 млрд / 13 млрд | сжатый KV, DSA | 1 млн | 30 |
| DeepSeek-V4-Pro | 1,6 трлн / 49 млрд | сжатый KV, DSA | 1 млн | 18 |
| DeepSeek-V3.2 | 685 млрд / 37 млрд | MLA + DSA | 160 тыс. | — |
| Kimi K3 | 2,8 трлн / 104 млрд | гибрид MLA + KDA | 1 млн | 8 (B300) |
| Kimi K2.6 | 1,04 трлн / 32 млрд | MLA | 256 тыс. | 1 (H200) |
| MiniMax-M3 | 428 млрд / 23 млрд | GQA + sparse | 1 млн | 4 |
| MiniMax-M3.1-Flash | ~428 млрд / ~23 млрд | GQA + sparse | 1 млн | 1 (без замеров) |
| Mistral Large 3 | 675 млрд / 41 млрд | MLA | 256 тыс. | — |
| Mistral Medium 3.5 | 128 млрд, плотная | GQA | 256 тыс. | 1 (H200) |
| Mistral Small 4 | 119 млрд / 6,5 млрд | MLA | 256 тыс. | — |
| Devstral 2 | 123 млрд, плотная | GQA | 256 тыс. | — |

Для моделей без профиля вкладка «Лимиты» строит профиль по roofline — это грубая оценка, о чём выводится предупреждение. Модели с пометкой «оценка» в интерфейсе имеют неподтверждённые поля (см. `note` в конфиге).

**GPU.** Цены — заглушки.

| GPU | VRAM | HBM | Форматы | ₽/час |
|---|---|---|---|---|
| A100 80GB SXM | 80 ГБ | 2,0 ТБ/с | bf16, int8 | 270 |
| H100 80GB SXM | 80 ГБ | 3,35 ТБ/с | bf16, fp8 | 520 |
| H200 141GB SXM | 141 ГБ | 4,8 ТБ/с | bf16, fp8 | 680 |
| B200 | 180 ГБ | 8 ТБ/с | bf16, fp8, fp4 | 950 |
| B300 | 288 ГБ | 8 ТБ/с | bf16, fp8, fp4 | 1 250 |
| GB300 (NVL72) | 288 ГБ | 8 ТБ/с | bf16, fp8, fp4 | 1 300 |

**Сети.** Цены — заглушки.

| Сеть | Гбит/с на GPU | RDMA | ₽/час за ноду |
|---|---|---|---|
| NVLink (внутри ноды) | 3 600 | да | 0 |
| InfiniBand NDR 400G | 400 | да | 160 |
| RoCE v2 400G | 400 | да | 110 |
| RoCE v2 200G | 200 | да | 70 |
| Ethernet 2×100G TCP | 25 | нет | 25 |

**Сверка с рекомендациями вендоров** (закреплена тестами):

| Конфигурация | Рекомендация | Источник |
|---|---|---|
| GLM-5.2, FP8 | 8× H200; для контекста 1M — 8× B200 | рецепт vLLM |
| DeepSeek-V4.1-Flash | 4× GB300, контекст 1M | карточка NVIDIA |

---

## Структура проекта

```
gpucalc/
├── main.go              # запуск, флаги, HTTP-маршруты и REST API
├── config.go            # типы конфига, проверка, хранилище с автоперечитыванием файла
├── config.default.json  # встроенный конфиг: платформа, GPU, сети, модели
├── calc.go              # память, раскладка TP×PP, скорость, стоимость, сравнение, матрица, метрики
├── pd.go                # prefill/decode: раздельные пулы, передача KV, альтернативы
├── mcp.go               # MCP-сервер (JSON-RPC 2.0): stdio и Streamable HTTP
├── jsonstore.go         # хранение JSON-файлов с автоперечитыванием
├── profiles.go          # профили SGLang и нагрузки, подгонка модели скорости, ёмкость, калибровка
├── profiles.default.json# встроенные профили SGLang (из cookbook и issue)
├── workloads.default.json # встроенные профили нагрузки (заглушки)
├── sim.go               # симулятор очереди и планировщика SGLang
├── plan.go              # подбор max-running-requests и лимита RPS, сверка, кривая
├── calc_test.go         # тесты: память, сверка с вендорами, рубли, P/D, MCP
├── plan_test.go         # тесты планировщика лимитов
├── tools/
│   ├── bench_to_profile.py  # результаты bench_serving → замеры профиля
│   └── cookbook/            # обновление профилей из SGLang cookbook (update.sh)
├── web/                 # интерфейс, встраивается в бинарник (go:embed)
│   ├── index.html
│   ├── app.js
│   └── style.css
├── Dockerfile
└── README.md
```

Статика встроена в бинарник, поэтому после правок в `web/` нужно пересобрать проект.

---

## Ограничения

- Это **оценка для планирования**, а не замена нагрузочному тесту. На вкладках «Калькулятор» и «Prefill / Decode» скорость считается по характеристикам GPU (roofline); подгонка — коэффициентами `decode_efficiency`, `compute_efficiency`, `prefill_efficiency` и `layer_overhead_us`. На вкладке «Лимиты SGLang» модель подгоняется под замеры профиля.
- **Нагрузка в `workloads.json` — заглушки.** Пока они не заменены вашими данными, рекомендации лимитов — иллюстрация метода.
- Замеры cookbook сделаны на синтетике (`random`, фиксированные длины, без prefix cache). На реальной нагрузке скорость может отличаться: другое распределение длин, принятие draft-токенов speculative decoding, попадания в кэш.
- Профили со speculative decoding (DSpark, EAGLE/MTP) модель описывает хуже: у них нелинейная зависимость TPOT от параллельности и контекста. Смотрите ошибку в таблице сверки.
- Симулятор не моделирует вытеснение и повторный prefill при нехватке KV, HiCache, DP-attention с балансировкой между рангами и неравномерность роутера.
- Влияние выноса Engram в RAM хоста на скорость не измерено (множитель 1,0).
- **Цены GPU и сети — заглушки.**
- Совмещённый режим чуть оптимистичен: prefill и decode считаются разделёнными по времени без потерь на переключение.
- В калькуляторе памяти не учитываются спекулятивное декодирование, prefix caching и очереди — для этого вкладка «Лимиты SGLang».
- У DeepSeek-V4.1-Flash сжатый KV передаётся по сети медленнее, чем на самом деле. Таблицы Engram (~49 ГиБ) vLLM по умолчанию держит в RAM хоста, а калькулятор считает их на GPU, то есть с запасом.
- Размер чекпоинта Qwen3.8-Max (~4,9 ТБ) взят из сторонних обзоров. Для Qwen3.8-Flash-Next размер оценён как 180 млрд × 2 байта.

---

## Источники

- GLM-5.2: [карточка на Hugging Face](https://huggingface.co/zai-org/GLM-5.2), [рецепт vLLM](https://recipes.vllm.ai/zai-org/GLM-5.2)
- DeepSeek-V4.1-Flash: [карточка](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json), [NVFP4 от NVIDIA](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)
- Qwen3.8: [2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), [Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [27B](https://huggingface.co/Qwen/Qwen3.8-27B), [блог Qwen](https://qwen.ai/blog?id=qwen3.8)
- [SGLang cookbook](https://docs.sglang.io/cookbook/intro) — рецепты и замеры; исходные данные: [docs/src/snippets/configs](https://github.com/sgl-project/sglang/tree/main/docs/src/snippets/configs)
- DeepSeek-V4.1-Flash на B300: [issue #41946](https://github.com/sgl-project/sglang/issues/41946) (бенчмарк TP4/EP4), [24-часовой замер на 4×B200](https://github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200) (ёмкость пула, Engram в RAM), [рецепт cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)
- GigaChat: [3.5-432B-A28B](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B), [3.1-702B-A36B](https://huggingface.co/ai-sage/GigaChat3.1-702B-A36B), [3.1-10B-A1.8B](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B), [рецепт cookbook](https://docs.sglang.io/cookbook/autoregressive/GigaChat/GigaChat3.5)
- GLM: [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- Kimi: [K3](https://huggingface.co/moonshotai/Kimi-K3), [K2.6](https://huggingface.co/moonshotai/Kimi-K2.6)
- MiniMax: [M3](https://huggingface.co/MiniMaxAI/MiniMax-M3), [M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
- Mistral: [Large 3](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512), [Medium 3.5](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B), [Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603), [Devstral 2](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)
- [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
