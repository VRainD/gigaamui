# GigaAM UI Fixed

Почему эта версия отличается:
- ставит **актуальный GigaAM из GitHub**, а не старый `gigaam 0.1.0` с PyPI;
- использует Flask вместо Gradio;
- контейнер стартует без загрузки модели;
- есть индикаторы прогресса в UI.

## Важно

Официальный GitHub GigaAM уже содержит модели v3, включая `v3_e2e_rnnt`, а PyPI `gigaam 0.1.0` знает только модели до v2. Поэтому для v3 ставим пакет из GitHub.

## Команды

```powershell
docker compose down --remove-orphans
docker rm -f gigaam-ui 2>$null
docker image prune -af
docker builder prune -af
docker compose build --no-cache
docker compose up -d --force-recreate
docker logs -f gigaam-ui
```

Открыть UI:

http://localhost:8000

## Проверка torch

```powershell
docker exec -it gigaam-ui python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```
# gigaamui
