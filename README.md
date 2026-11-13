# Interlinear Translation Engine

Многоязычная система выравнивания предложений для двуязычных книг.

Универсальный парсер PDF + GPU-ускоренное выравнивание предложений (LaBSE) + BLB формат для Bilingual Reader.

## Результаты

**Вариант 1 (Попарное выравнивание)**
- Дети капитана Гранта: RU-BG (70 глав, 2647 пар)
- Золотой жук: RU-BG (2 главы, 379 пар)
- Маленький принц: RU-EN (25 глав, 1447 пар)
- Маленький принц: RU-FR (26 глав, 1479 пар)

**Вариант 2 (Многоязычное выравнивание - эксперимент)**
- Маленький принц: RU-EN-ES-FR (23 глав, 1473 пары, BLB v2)

## Компоненты

1. **bl_book_parser.py** (54 KB) - Универсальный парсер PDF для 26+ раскладок
2. **bl_sentence_aligner.py** (92 KB) - GPU выравнивание предложений (LaBSE)
3. **align_json_chapters.py** - Выравнивание глав
4. **create_blb.py** - BLB v1 генератор
5. **process_book_pair.py** - Полная автоматизация (PDF → BLB)
6. **align_multilingual.py** - Многоязычный парсер глав
7. **multilang_sentence_aligner.py** - Многоязычное выравнивание предложений
8. **create_multilang_blb.py** - BLB v2 генератор

## BLB формат

**v1 (двуязычный)** - 1473 пары RU-EN или RU-FR
**v2 (многоязычный)** - 1473 пары RU-EN-ES-FR в одном файле

## Статус

✅ **Вариант 1** - Production Ready
⚠️  **Вариант 2** - Experimental (работает, но требует синхронизации индексов)

## Документация

- `PIPELINE.md` - Детальное описание конвейера
- `PROCESS_BOOKS_README.md` - Руководство по обработке
- `EXPERIMENT_V2_REPORT.md` - Отчёт многоязычного эксперимента

## Git история

```
ee08600 - docs: Add comprehensive Variant 2 experiment report
1c29313 - Variant 2: Experimental multilingual alignment
b865281 - Initial commit: Bilingual book PDF parser and alignment engine
```

2026-11-13
