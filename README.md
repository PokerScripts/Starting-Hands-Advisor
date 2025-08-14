# Starting Hands Advisor

`advisor.py` — это консольный помощник по префлоп-решениям в **No-Limit Texas Hold'em** для форматов **6-max** и **9-max**.  
Скрипт подсказывает оптимальное действие на основе упрощённых чартов стартовых рук: **open-raise**, защита против открытия, игра против 3-бета и 4-бета.

> ⚠️ Важно: данный инструмент **не является GTO-калькулятором** и предназначен для обучения, тренировки и анализа, а не для использования во время игры в реальном времени.

---

## 🚀 Возможности

- **Рекомендации по руке и позиции**
- **Вывод диапазонов** для позиции или ситуации
- **Экспорт в матрицу 13×13** (ASCII/Unicode)
- **Подробное объяснение решения** (источник, комментарии)
- **Проверка корректности пресета** (`validate`)
- Работа **в одном файле** без зависимостей
- Возможность **строгого режима** (`--strict`) или использования **эвристик**

---

## 📂 Структура

```
Starting-Hands-Advisor/
│
├── advisor.py     # основной исполняемый скрипт
├── README.md      # документация
└── LICENSE        # лицензия MIT
```

---

## 🔧 Установка и запуск

1. Убедитесь, что у вас установлен **Python 3.10+**.
2. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/PokerScripts/Starting-Hands-Advisor.git
   cd Starting-Hands-Advisor
   ```
3. Запустите скрипт:
   ```bash
   python advisor.py --help
   ```

---

## 📜 Примеры использования

### 1. Рекомендация на префлопе (open-raise)
```bash
python advisor.py advise --cards A9s --pos CO --action first_in
```
```
RECOMMENDATION: RAISE (2.5bb) | ALT: FOLD | CONF: HIGH
```

### 2. Защита большого блайнда против BTN 2.5bb
```bash
python advisor.py advise --cards KTo --pos BB --action vs --vs open --from BTN --sizing 2.5
```
```
RECOMMENDATION: CALL | ALT: 3BET | CONF: MED
```

### 3. Экспорт чарта открытия с BTN
```bash
python advisor.py export --pos BTN --scenario open --format ascii
```

### 4. Подробное объяснение решения
```bash
python advisor.py explain --cards A5s --pos SB --action vs --vs open --from CO
```
```
- Preset: Basic6.defend.SB_vs_CO_open_2.5 -> 3bet-bluff
- Action: 3BET (conf=HIGH)
- Alternative: CALL
- Note: блокер на тузов + играбельность в 3bet-потах
```

---

## ⚙️ Основные команды CLI

| Команда       | Назначение |
|---------------|------------|
| `advise`      | Рекомендация по заданной руке, позиции и сценарию |
| `range`       | Вывести набор рук для позиции/сценария |
| `export`      | Экспорт диапазона в виде матрицы 13×13 |
| `explain`     | Подробное объяснение выбранного действия |
| `validate`    | Проверка корректности встроенных чартов |

---

## 📑 Примеры сценариев

**Открытие с CO:**
```bash
python advisor.py advise --cards 77 --pos CO --action first_in
```

**Против 3-бета:**
```bash
python advisor.py advise --cards AQs --pos CO --action vs --vs 3bet --from BTN
```

**Строгий режим:**
```bash
python advisor.py advise --cards K9o --pos UTG --action first_in --strict
```

---

## 🗂 Пресеты

Встроено два пресета:
- **Basic6** — упрощённый диапазон для 6-max
- **Basic9** — консервативный диапазон для 9-max

Выбор пресета:
```bash
python advisor.py advise --cards AJo --pos MP --action first_in --preset Basic9
```

---

## ⚠️ Ограничения

- Диапазоны упрощены и не соответствуют точному GTO.
- Скрипт не подключается к онлайн-румам и не анализирует реальный поток рук.
- Использование во время онлайн-игры может нарушать правила покер-румов.
