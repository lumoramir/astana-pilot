# Запуск с нуля: Windows 11 + NVIDIA RTX 4060

Всё ниже проверено на ноутбуке с RTX 4060 Laptop (8 ГБ), Python 3.11. Команды выполняются в PowerShell или в терминале Visual Studio / VS Code.

## 0. Что нужно один раз
* **Python 3.10–3.12** с python.org (галочка *Add python.exe to PATH*).
* **Драйвер NVIDIA** (свежий Game Ready/Studio). CUDA Toolkit ставить **не нужно** — PyTorch приносит его сам.
* **Git** (git-scm.com). Для работы в **Visual Studio 2022**: установите рабочую нагрузку *Python development*, затем
  `File → Open → Folder` → папка проекта; интерпретатор: `Python Environments → Add Environment → Existing → .venv\Scripts\python.exe`.
  (VS Code работает так же: *Python: Select Interpreter* → `.venv`.)

## 1. Установка
```bat
cd путь\к\astana-av
scripts\setup_windows.bat
```
Скрипт создаёт `.venv`, ставит проект (`pip install -e ".[train,view,dev]"`) и **в конце** — PyTorch с CUDA. Порядок важен: обычный `pip install
stable-baselines3` может подменить CUDA-сборку на CPU-версию (так было у нас). Проверка:
```bat
.venv\Scripts\astana check
```
Должно закончиться `ALL CHECKS PASSED` и строкой `CUDA available: True ... RTX 4060`. Если CUDA = False:
```bat
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
```

## 2. Карты (реальные районы Астаны из OpenStreetMap)
```bat
.venv\Scripts\astana map --preset expo       :: EXPO, сфера Нур Алем, кольцевые дороги, MEGA Silk Way
.venv\Scripts\astana map --preset nurzhol    :: правительственный квартал, эстакады над Ишимом, клеверная развязка
.venv\Scripts\astana map --list-presets
```
Ответы OSM лежат в `assets\osm_cache`, поэтому повторный запуск не требует интернета. Если Overpass «занят» (502/504) — скрипт сам
пробует зеркала и ждёт; можно просто запустить через несколько минут. Свой район: `astana map --center 51.1285,71.438 --half-size 1400 --name moy`.
Файл карты ~17–35 МБ, в Git он **не** попадает (см. `.gitignore`) — его всегда можно пересобрать.

## 3. Посмотреть
```bat
.venv\Scripts\astana view --map assets\maps\astana_expo.pkl          :: 3D, эксперт-водитель
.venv\Scripts\astana view --map assets\maps\astana_nurzhol.pkl --camera drone
.venv\Scripts\astana view2d --map assets\maps\astana_expo.pkl        :: лёгкая 2D-версия
```
Клавиши: `C`/`1`–`7` камеры · `K` кинорежиссёр · `T` день/закат/ночь/снег · `L` лидар · `H` HUD · `G` взять управление (WASD) ·
`ПРОБЕЛ` пауза · `+`/`-` скорость симуляции · `F12` скриншот · `V` запись видео · `R` новый маршрут · `Esc` выход.
Слабое железо: `--quality low`. По умолчанию качество выбирается автоматически; если окно рисуется встроенной графикой, выполните один раз
`astana gpu` (ставит предпочтение NVIDIA для python.exe) и перезапустите.

## 4. Обучение
Быстрая проверка конвейера (≈10 минут):
```bat
scripts\run_smoke.bat
```
Ночное обучение (8 часов), на двух реальных районах + синтетических городах:
```bat
.venv\Scripts\astana train --run-name night1 --hours 8 --maps assets\maps\astana_expo.pkl,assets\maps\astana_nurzhol.pkl,synthetic:0,synthetic:1
```
* Что происходит: демонстрации эксперта → клонирование (BC) → прогрев критика → DAgger → PPO с адаптивным куррикулумом.
* **Прервали / выключили свет?** Запустите ту же команду — она продолжит с `runs\night1\checkpoints\latest.zip`.
  `--timesteps` — целевой итог, а не добавка. `--fresh` — начать заново.
* Ноутбук не должен засыпать: питание от сети, режим «Высокая производительность», крышку не закрывать.
* Смотреть метрики: `.venv\Scripts\tensorboard --logdir runs` → http://localhost:6006, группа **custom/**:
  `success`, `collision_rate`, `collision/*`, `viol_any`, `viol/*`, `unneeded_slow_frac`, `aeb_frac`, `difficulty`, `speed_kmh`.
* Хорошие ориентиры «готово» (**цели, не обещания**): success ≥ 90 % на seen и на unseen (разница ≤ 10 п.п.), collision_rate ≤ 2 %,
  viol_any ≤ 2 %, aeb_frac мало, difficulty дошёл до 1.0.

### Plot
```bat
.venv\Scripts\astana plot night1 --watch --open
```
Dashboard of 8 charts in runs\night1\training_curves.png, refreshed every minute. Several runs: `astana plot night1 baseline`. Interactive: `tensorboard --logdir runs`.

### Что реально ожидать после ночи (без обещаний)
* Агент **не** ездит «без дороги»: ему дают маршрут из карты (узлы OSM), а он учится держать полосу, соблюдать светофоры/зебры/знаки и
  не врезаться. Это не GPS-навигатор и не «понимание города»; позволительная формулировка: *«на маршруте, заданном по OSM»*.
* 8 часов на RTX 4060 — это **около 15 млн шагов** (измерено после исправлений мира: ~600 шагов/с; до них было 150–215).
* Берите `runs\<имя>\best_model.zip` (лучшая версия за всё обучение), а не только `final_model.zip`. В консоли строки `[stability]` значат, что
  успех обвалился и обучение вернулось к лучшим весам. Проверка 40 мин (1,3 млн шагов, сложность 1.0): сеть 43–50 % успеха против
  эксперта 67–73 %; на незнакомых районах не хуже, чем на знакомых (не заучивание). Подробности: `docs/TRAINING_NOTES.md`. Ориентир (цель, не гарантия): success растёт кривой с «полкой»; если после 3–4 ч
  `custom/success` на unseen < 30 %, смотрите графики (`astana plot`) и меняйте куррикулум, а не ждите дольше.
* Длинные сложные маршруты — это то, что проверяет режим **Grand tour** (2–3 км, 40–60 событий: светофоры, зебры, тоннели, эстакады).
  Если сеть застревает — это честный результат; запишите его в `docs/RESULTS.md` и покажите график + что именно улучшили.
* Для показа комиссии: `astana view --map ... --model runs\night1\final_model.zip --tour` (или кнопка **Grand tour** в Studio), рядом —
  график `training_curves.png` и таблица «seen / unseen» из кнопки **Test network**.

### Загрузка и тест готовой сети (кнопки в Studio)
* **Load network…** — выбираете `final_model.zip` (например, обученный на другом ПК); файл копируется в `runs\imported\<имя>\`.
* **Test network** — запускает `astana eval` на тех же сидах, что и эксперт; таблица success / collisions / violations на seen и unseen
  районах появляется в Studio. Из консоли: `astana eval --model <zip> --episodes 60`.

### Режим наблюдателя (Fly around)
`W A S D` полёт · `Q`/`E` вниз/вверх · `Shift` быстрее · **правая кнопка мыши** (или `Tab`) — осмотр · колесо — скорость полёта ·
`[` `]` или `1`–`9` — перелёт к хот-споту после аудита · `F` — следовать за ближайшей машиной · `T` свет · `H` справка · `Esc`.
Стрелки тоже работают, но не обязательны.

## 5. Оценка (это ваши доказательства для комиссии)
```bat
.venv\Scripts\astana eval --model runs\night1\final_model.zip --episodes 60 ^
    --seen-maps assets\maps\astana_expo.pkl --unseen-maps assets\maps\astana_nurzhol.pkl,synthetic:100
```
Результат — `runs\night1\eval\report.md` (+ `results.json`, `episodes.csv`): таблицы «модель vs эксперт» на одинаковых сидах, интервалы
Уилсона, кто во что врезался, нарушения по правилам, «лишняя медлительность», проверка «выучил ли карту». **Скопируйте итоговые числа в
`docs\RESULTS.md`** — честно, включая плохие.

Эксперимент «абляция» (сильный аргумент): обучите вариант `--no-bc` и сравните по тем же сидам; опишите разницу.

## 6. Аудит безопасности района
```bat
.venv\Scripts\astana audit --map assets\maps\astana_expo.pkl --episodes 300
.venv\Scripts\astana audit --map assets\maps\astana_expo.pkl --model runs\night1\final_model.zip --variants "baseline;limit0.8;green30"
```
Папка `reports\audit_astana_expo\`: `risk_heatmap.png`, `hotspots.csv`, `hotspots.geojson` (откройте в QGIS или Google Earth), `report.md`.
Чем больше поездок, тем стабильнее рейтинг (96 поездок — лишь иллюстрация, для выводов нужно ≥ 500).

## 7. Материалы для презентации
```bat
.venv\Scripts\astana record --map assets\maps\astana_expo.pkl --model runs\night1\final_model.zip --duration 90 --out assets\videos\demo.mp4
.venv\Scripts\astana view --map assets\maps\astana_nurzhol.pkl --preset night      :: затем F12 — скриншоты
.venv\Scripts\astana still --map assets\maps\astana_expo.pkl --pos=0,820,-1150 --look=0,0,0 --out docs\img --name aerial
```
Запись идёт без окна (`--headless`), камера «кинорежиссёр» меняет планы каждые 5–9 секунд. 90 секунд при 1080p занимают несколько минут.

## 8. GitHub
```bat
git init
git add .
git commit -m "Astana Pilot: rule-compliant driving agent and road-safety digital twin"
git branch -M main
git remote add origin https://github.com/<ваш-логин>/astana-pilot.git
git push -u origin main
```
* В репозиторий **не** кладём: `runs\`, `assets\maps\*.pkl`, `.venv\`, большие видео (залейте на YouTube/Google Drive, ссылку — в README).
  Обученную модель `final_model.zip` (≈ 5 МБ) можно положить в GitHub *Release*.
* В `README.md` замените картинки в `docs\img\` на свои и добавьте реальные числа из `eval`.
* Добавьте *Topics*: `autonomous-driving`, `reinforcement-learning`, `openstreetmap`, `ppo`, `digital-twin`, `road-safety`.

## 9. Если что-то не так
| Симптом | Что делать |
|---|---|
| `CUDA available: False` | переустановить torch (см. п. 1) |
| `Overpass is unavailable` | подождать, повторить; зеркала перебираются автоматически; кэш в `assets\osm_cache` |
| окно тормозит | `--quality medium` или `low`; `astana gpu`; закрыть программы, грузящие GPU |
| `UnicodeEncodeError` в консоли | `chcp 65001` и `set PYTHONUTF8=1` |
| обучение «не растёт» | смотрите `custom/success` и `custom/difficulty`; сначала `scripts\run_smoke.bat`; `--maps` должен существовать |
| `ModuleNotFoundError: astana_pilot` | запускайте из папки проекта с активированным `.venv` или `pip install -e .` |
