# Автоматическая обработка пар книг

Новый скрипт `process_book_pair.py` полностью автоматизирует конвейер:

```
PDF + PDF → (парсинг) → JSON + JSON
         → (выравнивание глав) → выровненные JSON
         → (GPU выравнивание предложений) → выровненные предложения
         → (создание BLB) → финальный BLB файл
```

## 🚀 Использование

### Способ 1: Простой (рекомендуется)

```bash
python process_book_pair.py "D:\coding\Python Projects\IT WAP\ZJ"
```

Скрипт автоматически:
1. Найдёт два PDF файла в папке
2. Выполнит весь конвейер
3. Создаст `zj_final.blb` в той же папке
4. Удалит промежуточные файлы (очистит папку)

### Способ 2: Сохранить промежуточные файлы

```bash
python process_book_pair.py "D:\coding\Python Projects\IT WAP\ZJ" --keep-intermediate
```

Оставит все JSON файлы для отладки/анализа.

## 📂 Структура папок

### Исходная папка (например, `D:\my_books\book1\`)
```
book1/
├── book_ru.pdf          ← Русская версия
├── book_bg.pdf          ← Болгарская версия
└── (все промежуточные файлы создаются здесь)
```

### После обработки (без --keep-intermediate)
```
book1/
├── book_ru.pdf
├── book_bg.pdf
└── book1_final.blb      ← ГОТОВО К ИСПОЛЬЗОВАНИЮ
```

### После обработки (с --keep-intermediate)
```
book1/
├── book_ru.pdf
├── book_bg.pdf
├── book1_final.blb
├── book1_src.json
├── book1_tgt.json
├── book1_src_aligned.json
├── book1_tgt_aligned.json
├── book1_aligned_sentences.json
└── book1_aligned_sentences_unaligned.json
```

## 🔍 Как работает определение языков

Скрипт автоматически определяет:
- **Источник** = большего PDF (обычно более полное издание)
- **Цель** = меньшего PDF (обычно сокращённое издание)

Если это не совпадает с вашими ожиданиями, переименуйте файлы так, чтобы исходный был больше.

## 📝 Примеры

### Пример 1: Дети капитана Гранта
```bash
# Содержимое папки:
# D:\books\granta\
#   ├── granta_ru.pdf (12 МБ)
#   ├── granta_bg.pdf (5.3 МБ)

python process_book_pair.py "D:\books\granta"

# Результат:
# D:\books\granta\granta_final.blb (готов!)
```

### Пример 2: Золотой жук
```bash
python process_book_pair.py "D:\coding\Python Projects\IT WAP\ZJ" --keep-intermediate

# Результат:
# D:\coding\Python Projects\IT WAP\ZJ\zj_final.blb (+ все JSON файлы)
```

## ⚙️ Опции

```bash
python process_book_pair.py <folder> [--keep-intermediate]

Arguments:
  folder                Папка с двумя PDF файлами
  
Options:
  --keep-intermediate   Сохранить JSON файлы (по умолчанию удаляются)
```

## ⏱️ Время обработки

- **Маленькая книга** (100-300 стр.): 5-15 минут
- **Средняя книга** (300-600 стр.): 15-30 минут
- **Большая книга** (600+ стр.): 30-60 минут

Большая часть времени — это GPU выравнивание предложений (это нормально).

## 🆘 Если что-то пошло не так

### Ошибка: "Found only X PDF file(s)"
Убедитесь, что в папке ровно 2 PDF файла.

### Ошибка: "Parsing failed"
Может быть проблема с кодировкой PDF. Попробуйте:
```bash
python process_book_pair.py <folder> --keep-intermediate
```
И посмотрите на выходные JSON файлы.

### Ошибка: "Chapters mismatch"
Это нормально, если версии значительно отличаются. Скрипт автоматически выберет минимальное количество глав.

### Ошибка CUDA / GPU
Если GPU не доступен, выравнивание предложений будет медленнее, но всё равно сработает.

## 📊 Что получается на выходе

BLB файл содержит:
- ✓ Встроенную обложку (PNG из первой страницы)
- ✓ Все главы с выровненными предложениями
- ✓ Метаинформацию (название, автор, языки)
- ✓ Полностью готов для Billing Reader

## 🔗 Связанные файлы

- `process_book_pair.py` — этот скрипт
- `bl_book_parser.py` — парсер PDF → JSON
- `align_json_chapters.py` — выравниватель глав
- `bl_sentence_aligner.py` — GPU выравниватель предложений
- `create_blb.py` — создатель BLB файлов
