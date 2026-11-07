#!/usr/bin/env python3
"""Sync the TalentMe website Tech Vault into this GitHub repository.

What it does
------------
1. Copies `content/tech/{Foundations,Interviews}/**/*.md` from the website repo.
2. Injects an idempotent bilingual CTA block into every guide:
   - top: read the interactive version on TalentMe (UTM tagged)
   - bottom: practice with flashcards / book a 1:1 mock interview (UTM tagged)
3. Regenerates navigation indexes:
   - `INDEX.md` (root)
   - `Foundations/README.md`, `Interviews/README.md`
   - one `README.md` per category (13 modules)

Usage
-----
  python3 scripts/sync_vault.py                 # sync + inject + rebuild indexes
  python3 scripts/sync_vault.py --dry-run       # show what would change
  python3 scripts/sync_vault.py --check         # exit 1 if the repo drifts from the site
  python3 scripts/sync_vault.py --prune         # also delete target files missing from source

Source path can be overridden with `TALENTME_TECH_CONTENT`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(
    os.environ.get(
        "TALENTME_TECH_CONTENT",
        "/home/suiyaoc/TalentMe/website/frontend/content/tech",
    )
)

SITE = "https://talentme.airsota.com"
UTM = "utm_source=github&utm_medium=article&utm_campaign=tech-vault"
SECTIONS = ("Foundations", "Interviews")

CATEGORY_META: Dict[str, Tuple[str, str, str]] = {
    "Math": ("📐", "Math & Optimization", "数理基础与优化"),
    "ML": ("📊", "Classic Machine Learning", "经典机器学习"),
    "DL": ("🧠", "Deep Learning Foundations", "深度学习基础"),
    "LLM": ("⚡", "Large Language Models", "大语言模型"),
    "RL": ("🎮", "RL & Post-Training Alignment", "强化学习与后训练对齐"),
    "Multimodal": ("👁️", "Multimodal & Diffusion", "多模态与扩散模型"),
    "AI_Infra": ("🖥️", "AI Infrastructure & Distributed Training", "AI 基础设施与分布式训练"),
    "AI_Engineering": ("🤖", "AI Engineering & Compound Systems", "AI 工程与复合系统"),
    "System_Design": ("🏗️", "Industrial System Design", "工业级系统设计"),
    "MLE": ("💻", "MLE Role Track", "机器学习工程师"),
    "AIE": ("🚀", "AIE Role Track", "AI 系统工程师"),
    "RS": ("🎓", "RS Role Track", "算法科学家"),
    "DS": ("📈", "DS Role Track", "数据科学家"),
}

FM_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)
TOP_RE = re.compile(
    r"\n?<!-- talentme:top:start -->.*?<!-- talentme:top:end -->\n?", re.DOTALL
)
BOTTOM_RE = re.compile(
    r"\n?<!-- talentme:bottom:start -->.*?<!-- talentme:bottom:end -->\n?", re.DOTALL
)


# --------------------------------------------------------------------------- #
# Frontmatter helpers
# --------------------------------------------------------------------------- #
def parse_frontmatter(text: str) -> Dict[str, str]:
    """Parse the small subset of YAML we need, without a PyYAML dependency."""
    match = FM_RE.match(text)
    if not match:
        return {}
    data: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        item = re.match(r"^([A-Za-z][A-Za-z0-9_]*):\s*(.*)$", line)
        if not item:
            continue
        key, value = item.group(1), item.group(2).strip()
        if value.startswith(('"', "'")) and value.endswith(('"', "'")):
            value = value[1:-1]
        data[key] = value.replace('\\"', '"')
    return data


def guide_title(frontmatter: Dict[str, str], lang: str, fallback: str) -> str:
    if lang == "zh":
        return frontmatter.get("titleZh") or frontmatter.get("title") or fallback
    return (
        frontmatter.get("titleEn")
        or frontmatter.get("title")
        or fallback.replace("-", " ").title()
    )


# --------------------------------------------------------------------------- #
# CTA blocks
# --------------------------------------------------------------------------- #
def article_url(category: str, slug: str) -> str:
    return f"{SITE}/resources/p/{category}/{slug}?{UTM}&utm_content={slug}"


def flashcards_url(slug: str) -> str:
    return f"{SITE}/tools/flashcards?{UTM}&utm_content={slug}-flashcards"


def coaching_url(slug: str) -> str:
    return f"{SITE}/contact?{UTM}&utm_content={slug}-coaching"


def top_cta(lang: str, category: str, slug: str) -> str:
    url = article_url(category, slug)
    if lang == "zh":
        body = (
            f"> 📖 **在 TalentMe 上以交互式导图、原子知识卡和 1,898 张闪卡阅读本指南** "
            f"→ [打开交互版]({url})"
        )
    else:
        body = (
            "> 📖 **Read this guide with interactive mindmaps, atomic cards, and "
            f"1,898 flashcards on TalentMe** → [Open the interactive version]({url})"
        )
    return (
        "<!-- talentme:top:start -->\n"
        f"{body}\n"
        "<!-- talentme:top:end -->"
    )


def bottom_cta(lang: str, category: str, slug: str) -> str:
    practice = flashcards_url(slug)
    coaching = coaching_url(slug)
    if lang == "zh":
        body = (
            "---\n\n"
            f"🎯 **准备好检验自己了吗？** 用 [1,898 张免费闪卡]({practice}) 练习本题，"
            f"或预约一场 [1v1 模拟面试]({coaching})。"
        )
    else:
        body = (
            "---\n\n"
            f"🎯 **Ready to test yourself?** Practice this topic with "
            f"[1,898 free flashcards]({practice}) or book a "
            f"[1:1 mock interview]({coaching})."
        )
    return (
        "<!-- talentme:bottom:start -->\n"
        f"{body}\n"
        "<!-- talentme:bottom:end -->"
    )


def inject_cta(text: str, lang: str, category: str, slug: str) -> str:
    """Remove old CTA blocks and insert fresh ones. Safe to run repeatedly."""
    text = TOP_RE.sub("", text)
    text = BOTTOM_RE.sub("", text)

    match = FM_RE.match(text)
    if not match:
        # Structural docs without frontmatter: leave untouched.
        return text

    frontmatter = match.group(0)
    body = text[match.end():].strip("\n")
    return (
        f"{frontmatter}\n"
        f"{top_cta(lang, category, slug)}\n\n"
        f"{body.rstrip()}\n\n"
        f"{bottom_cta(lang, category, slug)}\n"
    )


# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #
def iter_source_files(source: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (section, category, path) for every bilingual guide."""
    for section in SECTIONS:
        section_dir = source / section
        if not section_dir.is_dir():
            continue
        for category_dir in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            for path in sorted(category_dir.glob("*.md")):
                if path.name.endswith((".en.md", ".zh.md")):
                    yield section, category_dir.name, path


def iter_top_level_files(source: Path) -> Iterable[Path]:
    """Section-level docs such as `Foundations/taxonomy*.md` (copied verbatim)."""
    for section in SECTIONS:
        section_dir = source / section
        if not section_dir.is_dir():
            continue
        for path in sorted(section_dir.glob("*.md")):
            yield path


def desired_content(path: Path, section: str, category: str) -> str:
    raw = path.read_text(encoding="utf-8")
    lang = "zh" if path.name.endswith(".zh.md") else "en"
    slug = re.sub(r"\.(en|zh)\.md$", "", path.name)
    return inject_cta(raw, lang, category, slug)


def sync(source: Path, dry_run: bool, prune: bool) -> Tuple[int, int, int, List[str]]:
    written = 0
    unchanged = 0
    orphans: List[str] = []
    expected: set[Path] = set()

    for section, category, path in iter_source_files(source):
        relative = path.relative_to(source)
        target = REPO_ROOT / relative
        expected.add(relative)
        new_text = desired_content(path, section, category)

        old_text = target.read_text(encoding="utf-8") if target.exists() else None
        if old_text == new_text:
            unchanged += 1
            continue

        written += 1
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_text, encoding="utf-8")
        print(f"{'would write' if dry_run else 'wrote'}: {relative}")

    # Section-level docs (taxonomy & friends): copy verbatim, no CTA injection.
    for path in iter_top_level_files(source):
        relative = path.relative_to(source)
        expected.add(relative)
        target = REPO_ROOT / relative
        new_text = path.read_text(encoding="utf-8")
        old_text = target.read_text(encoding="utf-8") if target.exists() else None
        if old_text == new_text:
            unchanged += 1
            continue
        written += 1
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_text, encoding="utf-8")
        print(f"{'would write' if dry_run else 'wrote'}: {relative}")

    # Report (and optionally prune) guides that no longer exist upstream.
    for section in SECTIONS:
        section_dir = REPO_ROOT / section
        if not section_dir.is_dir():
            continue
        for path in sorted(section_dir.rglob("*.md")):
            if path.name == "README.md":
                continue
            relative = path.relative_to(REPO_ROOT)
            if relative not in expected:
                orphans.append(str(relative))
                if prune and not dry_run:
                    path.unlink()
                    print(f"pruned: {relative}")

    return written, unchanged, len(orphans), orphans


# --------------------------------------------------------------------------- #
# Index generation
# --------------------------------------------------------------------------- #
def collect_guides(source: Path) -> Dict[str, Dict[str, List[dict]]]:
    tree: Dict[str, Dict[str, List[dict]]] = {section: {} for section in SECTIONS}
    for section in SECTIONS:
        section_dir = source / section
        if not section_dir.is_dir():
            continue
        for category_dir in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            category = category_dir.name
            entries: Dict[str, dict] = {}
            for path in sorted(category_dir.glob("*.md")):
                if path.name.endswith(".zh.md"):
                    lang, slug = "zh", path.name[:-6]
                elif path.name.endswith(".en.md"):
                    lang, slug = "en", path.name[:-6]
                else:
                    continue
                fm = parse_frontmatter(path.read_text(encoding="utf-8"))
                entry = entries.setdefault(
                    slug,
                    {"slug": slug, "titleEn": "", "titleZh": "", "date": fm.get("date", "")},
                )
                entry["titleEn" if lang == "en" else "titleZh"] = guide_title(
                    fm, lang, slug
                )
            if entries:
                tree[section][category] = [entries[s] for s in sorted(entries)]
    return tree


def category_readme(section: str, category: str, guides: List[dict]) -> str:
    icon, name_en, name_zh = CATEGORY_META.get(category, ("📄", category, category))
    root = "../.."
    lines = [
        f"# {icon} {name_en} · {name_zh}",
        "",
        f"> Part of the [TalentMe AI/ML/LLM Interview Vault]({root}/README.md) · "
        f"[🌐 Read the interactive version]({SITE}/resources/tech?{UTM}) · "
        f"[🧭 Open the mindmap]({SITE}/resources/tech/roadmap/mindmap/{category}?{UTM})",
        "",
        f"**{len(guides)} bilingual guides** in this module.",
        "",
        "| # | Guide | English | 中文 |",
        "|:--:|---|---|---|",
    ]
    for index, guide in enumerate(guides, start=1):
        slug = guide["slug"]
        lines.append(
            f"| {index} | {guide['titleEn']} | "
            f"[EN]({slug}.en.md) | [中文]({slug}.zh.md) |"
        )
    lines.extend(["", "---", f"← Back to [{section}]({root}/{section}/README.md)", ""])
    return "\n".join(lines)


def section_readme(section: str, categories: Dict[str, List[dict]]) -> str:
    total = sum(len(v) for v in categories.values())
    lines = [
        f"# {'📖' if section == 'Foundations' else '🎯'} {section}",
        "",
        f"> Part of the [TalentMe AI/ML/LLM Interview Vault](../README.md). "
        f"**{total} bilingual guides** across {len(categories)} modules.",
        "",
        "| Module | Guides | Index |",
        "|---|:--:|---|",
    ]
    for category, guides in categories.items():
        icon, name_en, name_zh = CATEGORY_META.get(category, ("📄", category, category))
        lines.append(
            f"| {icon} {name_en} · {name_zh} | {len(guides)} | "
            f"[Open]({category}/README.md) |"
        )
    lines.extend(["", f"← Back to [the vault index](../INDEX.md)", ""])
    return "\n".join(lines)


def root_index(tree: Dict[str, Dict[str, List[dict]]]) -> str:
    lines = [
        "# 📚 TalentMe AI/ML/LLM Interview Vault — Full Index",
        "",
        f"> [🌐 Interactive version]({SITE}/resources/tech?{UTM}) · "
        f"[🃏 1,898 flashcards]({SITE}/tools/flashcards?{UTM}) · "
        f"[🎯 2,385-question bank](QUESTION-BANK.md) · "
        f"[💬 1:1 mock interview]({SITE}/contact?{UTM})",
        "",
        "Every guide is bilingual (English + 中文). Pick a module or jump straight "
        "into a role track.",
        "",
    ]
    for section in SECTIONS:
        categories = tree.get(section, {})
        total = sum(len(v) for v in categories.values())
        lines.extend(
            [
                f"## {'📖 Foundations' if section == 'Foundations' else '🎯 Interviews'} "
                f"({total} guides)",
                "",
                "| Module | Guides | Index |",
                "|---|:--:|---|",
            ]
        )
        for category, guides in categories.items():
            icon, name_en, name_zh = CATEGORY_META.get(category, ("📄", category, category))
            lines.append(
                f"| {icon} {name_en} · {name_zh} | {len(guides)} | "
                f"[Open]({section}/{category}/README.md) |"
            )
        lines.append("")
    lines.extend(
        [
            "---",
            "",
            "Generated by [`scripts/sync_vault.py`](scripts/sync_vault.py). "
            "Do not edit the tables by hand.",
            "",
        ]
    )
    return "\n".join(lines)


def write_indexes(source: Path, dry_run: bool) -> int:
    tree = collect_guides(source)
    changed = 0
    files: Dict[Path, str] = {}

    files[REPO_ROOT / "INDEX.md"] = root_index(tree)
    for section, categories in tree.items():
        total = sum(len(v) for v in categories.values())
        if total:
            files[REPO_ROOT / section / "README.md"] = section_readme(section, categories)
        for category, guides in categories.items():
            files[REPO_ROOT / section / category / "README.md"] = category_readme(
                section, category, guides
            )

    for path, content in files.items():
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old == content:
            continue
        changed += 1
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(f"{'would write' if dry_run else 'wrote'}: {path.relative_to(REPO_ROOT)}")

    return changed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_dir():
        print(f"error: source directory not found: {source}", file=sys.stderr)
        return 2

    written, unchanged, orphan_count, orphans = sync(
        source, dry_run=args.dry_run or args.check, prune=args.prune
    )
    index_changed = write_indexes(source, dry_run=args.dry_run or args.check)

    print()
    print(f"source:     {source}")
    print(f"guides:     {written + unchanged} processed ({written} changed, {unchanged} unchanged)")
    print(f"indexes:    {index_changed} changed")
    print(f"orphans:    {orphan_count}")
    if orphans:
        for orphan in orphans[:20]:
            print(f"  - {orphan}")
        if len(orphans) > 20:
            print(f"  ... and {len(orphans) - 20} more")

    if args.check and (written or index_changed or orphans):
        print("\ncheck failed: repository is out of sync with the website content.")
        return 1

    if args.check:
        print("\ncheck passed: repository is in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
