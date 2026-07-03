# GigaAM UI v2.0.0

Веб-интерфейс для распознавания русской речи на базе [GigaAM](https://github.com/salute-developers/GigaAM) v3.

## Что нового в v2

- Пакетная загрузка и обработка аудио/видео
- Поддержка WEBM, MP4, MKV и других видеоформатов
- Объединённые SRT/VTT/TXT и ZIP с результатами
- Безопасная очистка рабочего кэша и TTL заданий
- Модульная архитектура (`gigaam_ui/`) и тесты (pytest)

## Возможности

- Загрузка одного или нескольких аудио/видео файлов
- Поддержка `.webm`, `.mp4`, `.mkv`, `.mov`, `.avi` и других форматов через FFmpeg
- Нормализация в WAV, mono, PCM s16le, 16 kHz
- Пакетная очередь с отдельным статусом каждого файла
- Изменение порядка файлов перед обработкой (drag-and-drop)
- Отдельные результаты: TXT, JSON, SRT, VTT
- Объединённые SRT/VTT/TXT с глобальными таймкодами + ZIP со всеми результатами
- Безопасная очистка рабочего кэша (без удаления модели GigaAM)
- Автоочистка завершённых заданий по TTL

## Быстрый старт

```powershell
docker compose build
docker compose up -d
```

UI: http://localhost:8000

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MODEL_NAME` | `v3_e2e_rnnt` | Модель GigaAM |
| `CHUNK_SECONDS` | `20` | Длина чанка для длинных файлов |
| `MAX_FILE_MB` | `1000` | Лимит размера загрузки |
| `JOB_TTL_SECONDS` | `3600` | TTL завершённых заданий в памяти |
| `JOB_TTL_CHECK_INTERVAL` | `60` | Интервал проверки TTL (сек) |

## API

| Endpoint | Описание |
|----------|----------|
| `POST /api/transcribe` | Один файл (обратная совместимость) |
| `POST /api/batch` | Пакетная обработка (`files[]`, опционально `order`) |
| `GET /api/status/<job_id>` | Статус задания |
| `POST /api/clear` | Очистка (`workspace`, `outputs`, `logs`, `cache`) |
| `GET /download/<filename>` | Скачивание результата |

### Очистка

- `workspace` — inputs, outputs, logs, temp, история заданий (модель не удаляется)
- `cache` — кэш Hugging Face (модель потребует повторной загрузки)
- Очистка блокируется во время активной обработки (HTTP 409)

## Тесты

```powershell
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

Обязательный сценарий WEBM: `tests/test_api.py::test_integration_webm_processing`

## Проверка GPU

```powershell
docker exec -it gigaam-ui python3 -c "import torch; print(torch.cuda.is_available())"
```

## Почему GigaAM из GitHub

PyPI `gigaam 0.1.0` поддерживает только модели до v2. Для `v3_e2e_rnnt` пакет устанавливается из GitHub.
