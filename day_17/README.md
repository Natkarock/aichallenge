# autopub-pipeline — CLI пайплайн для изменений в коде по ТЗ

## Что делает пайплайн

1. **Git**: клон/обновление репозитория, создание ветки `auto/<slug>-<date>`.
2. **ТЗ**: загрузка Google Docs → очистка → удаление стоп-слов → краткое summary.
3. **Снимок проекта**: индексирует весь репозиторий (текстовые файлы), формирует компактный snapshot: дерево + head каждого файла (ограничение по символам).
4. **Генерация изменений**: LLM получает snapshot и краткое ТЗ и возвращает **строго JSON**: список изменяемых/новых файлов и `change_notes`.
5. **Применение**: перезаписывает файлы согласно JSON, создаёт `change_notes.md`, коммитит и пушит ветку.

> По умолчанию исключаются директории: `.git`, `node_modules`, `build`, `dist`, `out`, `target`, `__pycache__`, `venv`, `.venv` и пр., а также бинарные/крупные файлы.

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="you@example.com"
# (опционально для https push)
export GIT_HTTPS_TOKEN="ghp_xxx"
```

## Запуск

```bash
python -m autopub.cli   --repo "https://github.com/user/project.git"   --gdoc "https://docs.google.com/document/d/<DOC_ID>/edit?usp=sharing"   --workdir "/tmp/workspace"   --branch-prefix "auto"   --include "**/*"   --exclude "**/*.png,**/*.jpg,**/*.pdf,**/*.apk,**/*.aab,**/*.ipa,**/*.so,**/*.jar,**/*.keystore,**/*.lock"   --max-file-chars 6000   --max-files 500
```

### Важные флаги

- `--include` — какие пути разрешено анализировать/обновлять (glob, через запятую). По умолчанию `**/*`.
- `--exclude` — исключения (glob), по умолчанию исключаются бинарники и lock-файлы.
- `--max-file-chars` — сколько символов head читать из каждого файла для контекста (по умолчанию 6000).
- `--max-files` — максимум файлов в snapshot (по умолчанию 500, чтобы не распух контекст).

## Безопасность

- Применяем только те файлы, которые попадают под `--include` и не попадают под `--exclude` + проходят текстовую эвристику (не бинарь).
- Бинарники никогда не отправляются в LLM и не перезаписываются.
