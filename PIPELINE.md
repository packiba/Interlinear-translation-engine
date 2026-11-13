# Bilingual Book Pipeline (BLB)

Полная система для создания двуязычных книг в формате BLB для приложения Billing Reader.

## Формат BLB

BLB (Bilingual Book) — это JSON контейнер с версией 1, содержащий:
- Метаинформацию (название, автор, языки)
- Встроенную обложку в формате PNG (base64)
- Все главы с выровненными по предложениям переводами

## Полный конвейер обработки

### Этап 1: Парсинг PDF → JSON

```bash
python bl_book_parser.py pdf/book_ru.pdf -o book_ru.json -v
python bl_book_parser.py pdf/book_bg.pdf -o book_bg.json -v
```

Парсер автоматически:
- Распознаёт структуру книги (иерархия глав)
- Чистит служебный текст (сноски, колонтитулы, номера страниц)
- Разбивает текст на абзацы
- Восстанавливает переносы слов

### Этап 2: Выравнивание по главам

```bash
python align_json_chapters.py book_ru.json book_bg.json \
  -o1 book_ru_aligned.json -o2 book_bg_aligned.json
```

Выравнивает главы 1:1 между двумя версиями.

### Этап 3: GPU Выравнивание предложений

```bash
python bl_sentence_aligner.py book_ru_aligned.json book_bg_aligned.json \
  -o book_aligned.json
```

Использует LaBSE эмбединги и DP движок на GPU для выравнивания предложений.

### Этап 4: Создание BLB файла

```bash
python create_blb.py book_aligned.json book_ru.json pdf/book_ru.pdf \
  -o book_ru_bg.blb
```

Создаёт финальный BLB файл с обложкой и метаинформацией.

## Быстрый старт (одна команда)

Создан скрипт-обёртка `make_blb.py` для одношагового конвейера:

```bash
python make_blb.py pdf/book_ru.pdf pdf/book_bg.pdf -o book.blb
```

## Примеры использования

### Для "Дети капитана Гранта"

```bash
python make_blb.py \
  'D:/coding/Python Projects/IT WAP/DKG/Dieti kapitana Granta - Zhiul' Gabriel' Viern.pdf' \
  'D:/coding/Python Projects/IT WAP/DKG/Jules-Verne_-_Detsata_na_kapitan_Grant_-_1678-b.pdf' \
  -o Deti_Kapitana_Granta_RU_BG.blb
```

## Требования

- Python 3.9+
- PyMuPDF (fitz)
- NumPy
- sentence-transformers (для выравнивания)
- GPU с CUDA поддержкой (рекомендуется для выравнивания предложений)

## Структура BLB файла

```json
{
  "blb_version": 1,
  "book": {
    "title_src": "Название оригинала",
    "title_tgt": "Название перевода",
    "author": "Автор",
    "lang_src": "ru",
    "lang_tgt": "bg"
  },
  "cover": {
    "mime": "image/png",
    "width": 1275,
    "height": 1650,
    "data_base64": "iVBORw0KG..."
  },
  "chapters": [
    {
      "pair_num": 1,
      "title_src": "Глава I. Название",
      "title_tgt": "Глава I. Название",
      "pairs": [
        {"src": "Русское предложение.", "tgt": "Болгарско преложение."},
        ...
      ]
    },
    ...
  ]
}
```

## Переменные окружения

- `BL_EMBED_MODEL` — модель эмбеддингов (по умолчанию: `sentence-transformers/LaBSE`)
- `BL_ENGINES` — движки выравнивания (по умолчанию: `dp`; также можно `bertalign`)
- `BL_VERBOSE` — уровень логирования (0=тихо, 1=базовое, 2=подробное)

## Поддерживаемые форматы PDF

- Calibre (с оглавлением)
- Word экспорт
- Prince (типографическая верстка)
- Самодельные слияния (pypdf)
- Наборы с эпиграфами (без заголовков)

## Качество парсинга

Система поддерживает:
- **Русский, болгарский, английский, испанский, немецкий, французский, итальянский**
- **Различные романские и кириллические варианты**: римские цифры с кириллическими гомоглифами, болгарские составные числа
- **Служебный текст**: аннотации, предисловия, приложения, сноски
- **Форматирование**: диалоги, цитаты, лирические отступления, эпиграфы

## Статистика качества

Для "Дети капитана Гранта":
- Глав в парах: 70/70 (100%)
- Доля неправильно распознанных абзацев: < 1%
- Точность выравнивания предложений: GPU-качество (LaBSE)
