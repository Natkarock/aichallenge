# autopub-pipeline — CLI пайплайн для авто‑подготовки релиза

Пайплайн делает 5 шагов:
1) Клонирует/обновляет репозиторий по ссылке (git).
2) Тянет ТЗ из Google Docs (по публичной ссылке), прогоняет через мини‑пайплайн:
   очистка HTML/мусора → удаление стоп‑слов → сжатие (summary LLM) → поиск по документации репозитория.
3) Создаёт новую ветку.
4) Генерирует изменения через LLM и перезаписывает файлы + собирает `release_notes.md`.
5) Коммитит и пушит ветку в удалённый репозиторий.

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export GIT_USER_NAME="Your Name"
export GIT_USER_EMAIL="you@example.com"
# (опционально для https):
export GIT_HTTPS_TOKEN="ghp_xxx"
```

## Запуск

```bash
python -m autopub.cli   --repo "https://github.com/user/project.git"   --gdoc "https://docs.google.com/document/d/<DOC_ID>/edit?usp=sharing"   --workdir "/tmp/workspace"   --branch-prefix "auto"   --targets "README.md,CHANGELOG.md,docs/*.md,app/build.gradle*"
```

## Пояснения
- Google Docs должен быть доступен «по ссылке». Иначе укажи локальный `.txt/.md` вместо `--gdoc`.
- Файлы, которые реально изменяются, ограничены `--targets` (безопасность!).
- Изменения применяются как полная перезапись файла, не diff‑патчи.

MIT.
