#!/usr/bin/env python3
"""Generate the bilingual interview question bank from TalentMe knowledge cards.

The source of truth is the website's `lib/mindmaps/cards/**/*.ts` bank:
477 `KnowledgeCard` objects, each with 5 interviewer follow-up questions.

Output
------
- `QUESTION-BANK.md`              — root index (cards, question counts, links)
- `Question-Bank/<Module>.md`     — 5 questions per card, grouped by module

English questions are shown inline; Chinese questions are collapsed in a
`<details>` block so each module page stays scannable. Every card links back to
the interactive card page on TalentMe (UTM tagged), where the full answer,
derivation, and code live.

Usage
-----
  python3 scripts/generate_question_bank.py
  python3 scripts/generate_question_bank.py --dry-run
  python3 scripts/generate_question_bank.py --check   # exit 1 if out of date
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(
    os.environ.get(
        "TALENTME_CARD_SOURCE",
        "/home/suiyaoc/TalentMe/website/frontend/lib/mindmaps/cards",
    )
)

SITE = "https://talentme.airsota.com"
UTM = "utm_source=github&utm_medium=question-bank&utm_campaign=vault"

MODULE_ORDER: List[str] = [
    "Math",
    "ML",
    "DL",
    "LLM",
    "RL",
    "Multimodal",
    "AI_Infra",
    "AI_Engineering",
    "System_Design",
    "MLE",
    "AIE",
    "RS",
    "DS",
]

MODULE_META: Dict[str, Tuple[str, str, str, str]] = {
    # module: (icon, nameEn, nameZh, section)
    "Math": ("📐", "Math & Optimization", "数理基础与优化", "Foundations"),
    "ML": ("📊", "Classic Machine Learning", "经典机器学习", "Foundations"),
    "DL": ("🧠", "Deep Learning Foundations", "深度学习基础", "Foundations"),
    "LLM": ("⚡", "Large Language Models", "大语言模型", "Foundations"),
    "RL": ("🎮", "RL & Post-Training Alignment", "强化学习与后训练对齐", "Foundations"),
    "Multimodal": ("👁️", "Multimodal & Diffusion", "多模态与扩散模型", "Foundations"),
    "AI_Infra": ("🖥️", "AI Infrastructure", "AI 基础设施", "Foundations"),
    "AI_Engineering": ("🤖", "AI Engineering & Agents", "AI 工程与智能体", "Foundations"),
    "System_Design": ("🏗️", "Industrial System Design", "工业级系统设计", "Foundations"),
    "MLE": ("💻", "MLE Role Track", "机器学习工程师", "Interviews"),
    "AIE": ("🚀", "AIE Role Track", "AI 系统工程师", "Interviews"),
    "RS": ("🎓", "RS Role Track", "算法科学家", "Interviews"),
    "DS": ("📈", "DS Role Track", "数据科学家", "Interviews"),
}

PAIR_RE = re.compile(
    r'\{\s*zh:\s*"((?:[^"\\]|\\.)*)"\s*,\s*en:\s*"((?:[^"\\]|\\.)*)"\s*,?\s*\}',
    re.DOTALL,
)
STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"', re.DOTALL)


# --------------------------------------------------------------------------- #
# JavaScript / TypeScript extraction helpers
# --------------------------------------------------------------------------- #
def decode_js_string(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except (json.JSONDecodeError, ValueError):
        return (
            raw.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )


def joined_strings(block: str) -> str:
    """Join a concatenated TS string expression into plain text."""
    return "".join(decode_js_string(part) for part in STRING_RE.findall(block))


def field_string(text: str, field: str) -> str:
    match = re.search(rf'\b{field}:\s*"((?:[^"\\]|\\.)*)"', text)
    return decode_js_string(match.group(1)) if match else ""


def block_until(text: str, field: str, next_fields: List[str]) -> str:
    match = re.search(rf"\b{field}:\s*", text)
    if not match:
        return ""
    start = match.end()
    end = len(text)
    for next_field in next_fields:
        found = re.search(rf"\n\s*{next_field}:\s*", text[start:])
        if found:
            end = min(end, start + found.start())
    return text[start:end]


def extract_array(text: str, field: str) -> str:
    match = re.search(rf"\b{field}:\s*\[", text)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[start + 1:index]
    return ""


# --------------------------------------------------------------------------- #
# Card model
# --------------------------------------------------------------------------- #
def parse_card(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    if "KnowledgeCard = {" not in text:
        return None
    card_id = re.search(r'\bid:\s*"([^"]+)"', text)
    module_id = re.search(r'\bmoduleId:\s*"([^"]+)"', text)
    if not card_id or not module_id:
        return None

    definition_zh = joined_strings(
        block_until(text, "definitionZh", ["definitionEn"])
    )
    definition_en = joined_strings(
        block_until(text, "definitionEn", ["examPoints"])
    )
    questions = [
        (decode_js_string(zh), decode_js_string(en))
        for zh, en in PAIR_RE.findall(extract_array(text, "examPoints"))
    ]
    if not questions:
        return None

    return {
        "id": card_id.group(1),
        "module": module_id.group(1),
        "nameEn": field_string(text, "nameEn") or card_id.group(1),
        "nameZh": field_string(text, "nameZh") or card_id.group(1),
        "definitionEn": definition_en,
        "definitionZh": definition_zh,
        "questions": questions,
        "path": path,
    }


def load_cards(source: Path) -> Dict[str, List[dict]]:
    by_module: Dict[str, List[dict]] = {module: [] for module in MODULE_ORDER}
    for path in sorted(source.rglob("*.ts")):
        if path.name.startswith("_"):
            continue
        card = parse_card(path)
        if not card:
            continue
        by_module.setdefault(card["module"], []).append(card)
    for module in by_module:
        by_module[module].sort(key=lambda card: (card["nameEn"].lower(), card["id"]))
    return by_module


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def excerpt(text: str, limit: int = 260) -> str:
    text = re.sub(r"\*\*|`", "", text)
    text = re.sub(r"^(Definition|定义)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    for separator in (". ", "。", "；", "; "):
        cut = text.find(separator, 120)
        if 0 < cut <= limit:
            return text[: cut + len(separator)].strip()
    clipped = text[:limit].rstrip()
    # Never end inside an inline math span.
    if clipped.count("$") % 2:
        clipped = clipped[: clipped.rfind("$")].rstrip()
    return clipped + "…"


def card_url(module: str, card_id: str) -> str:
    return (
        f"{SITE}/resources/tech/roadmap/mindmap/{module}/card/{card_id}"
        f"?{UTM}&utm_content={card_id}"
    )


def flashcards_url() -> str:
    return f"{SITE}/tools/flashcards?{UTM}"


def mindmap_url(module: str) -> str:
    return f"{SITE}/resources/tech/roadmap/mindmap/{module}?{UTM}"


def render_card(index: int, card: dict) -> str:
    module = card["module"]
    lines = [
        f"## {index}. {card['nameEn']} · {card['nameZh']}",
        "",
        f"> {excerpt(card['definitionEn'])}",
        "",
        "**Interview questions**",
        "",
    ]
    for question_index, (_, en) in enumerate(card["questions"], start=1):
        lines.append(f"{question_index}. {en}")
    lines.extend(["", "<details>", "<summary>中文题目</summary>", ""])
    for question_index, (zh, _) in enumerate(card["questions"], start=1):
        lines.append(f"{question_index}. {zh}")
    lines.extend(
        [
            "",
            "</details>",
            "",
            f"[Read the full answer, derivation & code →]({card_url(module, card['id'])})",
            "",
            "---",
            "",
        ]
    )
    return "\n".join(lines)


def render_module(module: str, cards: List[dict]) -> str:
    icon, name_en, name_zh, _ = MODULE_META.get(
        module, ("📄", module, module, "")
    )
    header = [
        f"# {icon} {name_en} · {name_zh} — Interview Question Bank",
        "",
        f"> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · "
        f"**{len(cards)} cards · {len(cards) * 5} questions**",
        f"> [🃏 Practice all 1,898 flashcards]({flashcards_url()}) · "
        f"[🧭 Open the interactive mindmap]({mindmap_url(module)})",
        "",
        "English questions are shown inline; expand **中文题目** for the Chinese "
        "version. Each card links to the full answer, derivation, and code.",
        "",
        "---",
        "",
    ]
    body = [render_card(index, card) for index, card in enumerate(cards, start=1)]
    footer = ["← Back to [the full question bank index](../QUESTION-BANK.md)", ""]
    return "\n".join(header) + "\n" + "\n".join(body) + "\n".join(footer)


def render_root(by_module: Dict[str, List[dict]]) -> str:
    total_cards = sum(len(cards) for cards in by_module.values())
    lines = [
        "# 🎯 TalentMe Interview Question Bank",
        "",
        f"**{total_cards} knowledge cards · {total_cards * 5:,} interviewer questions** "
        "across 13 modules — English + 中文.",
        "",
        f"> [🧭 Interactive mindmaps]({SITE}/resources/tech/roadmap?{UTM}) · "
        f"[🃏 1,898 flashcards]({flashcards_url()}) · "
        f"[🎯 1:1 mock interview]({SITE}/contact?{UTM})",
        "",
        "Each card below has **5 real interviewer follow-ups**. The questions are "
        "here; the full answers, derivations, and Pure Python code are on the "
        "linked card pages.",
        "",
        "| Module | Cards | Questions | Question bank |",
        "|---|:--:|:--:|---|",
    ]
    for module in MODULE_ORDER:
        cards = by_module.get(module, [])
        if not cards:
            continue
        icon, name_en, name_zh, section = MODULE_META[module]
        lines.append(
            f"| {icon} {name_en} · {name_zh} | {len(cards)} | {len(cards) * 5} | "
            f"[Open](Question-Bank/{module}.md) |"
        )
    lines.extend(
        [
            "",
            "---",
            "",
            "Generated by [`scripts/generate_question_bank.py`](scripts/generate_question_bank.py) "
            "from the TalentMe knowledge-card bank. Do not edit by hand.",
            "",
            "← Back to [the vault README](README.md)",
            "",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def write_if_changed(path: Path, content: str, dry_run: bool) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == content:
        return False
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"{'would write' if dry_run else 'wrote'}: {path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_dir():
        print(f"error: card source not found: {source}", file=sys.stderr)
        return 2

    by_module = load_cards(source)
    cards = [card for module_cards in by_module.values() for card in module_cards]
    if not cards:
        print("error: no knowledge cards parsed", file=sys.stderr)
        return 2

    dry_run = args.dry_run or args.check
    changed = 0
    changed += write_if_changed(
        REPO_ROOT / "QUESTION-BANK.md", render_root(by_module), dry_run
    )
    for module in MODULE_ORDER:
        module_cards = by_module.get(module, [])
        if not module_cards:
            continue
        changed += write_if_changed(
            REPO_ROOT / "Question-Bank" / f"{module}.md",
            render_module(module, module_cards),
            dry_run,
        )

    print()
    print(f"source:     {source}")
    print(f"cards:      {len(cards)}")
    print(f"questions:  {len(cards) * 5}")
    print(f"modules:    {sum(1 for m in by_module.values() if m)}")
    print(f"files:      {changed} changed")

    if args.check:
        if changed:
            print("\ncheck failed: question bank is out of date.")
            return 1
        print("\ncheck passed: question bank is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
