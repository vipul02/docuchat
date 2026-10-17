"""DocuChat RAG engine: load documents, chunk them, retrieve with BM25, answer with Claude + citations."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

MODEL = "claude-opus-5-5"
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K = 5

SYSTEM_PROMPT = (
    "You are DocuChat, a company knowledge assistant. Answer only from the provided documents. "
    "Be concise and specific. If the documents do not contain the answer, say you don't know and "
    "suggest who the user could ask. Never invent policies, numbers or dates."
)

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was "
    "what when where which who will with do does can my our we you your".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


@dataclass
class Chunk:
    id: int
    source: str
    section: str
    text: str
    tokens: list[str] = field(repr=False, default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.source} - {self.section}" if self.section else self.source


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # optional dependency

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text on headings so chunks keep a meaningful section name."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def chunk_text(body: str) -> list[str]:
    if len(body) <= CHUNK_CHARS:
        return [body]
    chunks, start = [], 0
    while start < len(body):
        end = min(start + CHUNK_CHARS, len(body))
        if end < len(body):  # break on a paragraph or sentence boundary when possible
            cut = max(body.rfind("\n\n", start, end), body.rfind(". ", start, end))
            if cut > start + CHUNK_CHARS // 2:
                end = cut + 1
        chunks.append(body[start:end].strip())
        if end >= len(body):
            break
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c]


class KnowledgeBase:
    """In-memory BM25 index over document chunks."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.df: Counter[str] = Counter()
        self.avg_len = 0.0

    def add_document(self, name: str, text: str) -> int:
        self.chunks = [c for c in self.chunks if c.source != name]  # re-adding replaces
        added = 0
        for section, body in split_sections(text) or [("", text)]:
            for piece in chunk_text(body):
                self.chunks.append(Chunk(len(self.chunks), name, section, piece, tokenize(section + " " + piece)))
                added += 1
        self._reindex()
        return added

    def load_folder(self, folder: Path) -> None:
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".md", ".txt", ".pdf"}:
                self.add_document(path.name, read_document(path))

    def _reindex(self) -> None:
        for i, c in enumerate(self.chunks):
            c.id = i
        self.df = Counter(t for c in self.chunks for t in set(c.tokens))
        self.avg_len = sum(len(c.tokens) for c in self.chunks) / max(len(self.chunks), 1)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def search(self, query: str, k: int = TOP_K, k1: float = 1.5, b: float = 0.75) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        n = len(self.chunks)
        scored = []
        for c in self.chunks:
            tf = Counter(c.tokens)
            score = 0.0
            for t in terms:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(c.tokens) / self.avg_len))
            if score > 0:
                scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def has_claude_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def answer(kb: KnowledgeBase, question: str) -> dict:
    """Retrieve relevant chunks and answer. Returns {answer, citations, sources, mode}."""
    hits = kb.search(question)
    sources = [{"title": c.title, "text": c.text, "score": round(s, 2)} for c, s in hits]
    if not hits:
        return {
            "answer": "I couldn't find anything about that in the knowledge base.",
            "citations": [], "sources": [], "mode": "retrieval",
        }
    if not has_claude_credentials():
        return _extractive_answer(question, hits, sources)
    return _claude_answer(question, hits, sources)


def _extractive_answer(question: str, hits, sources) -> dict:
    """Offline fallback: return the best-matching passages with their sources."""
    best, _ = hits[0]
    return {
        "answer": f"Here's what your documents say ({best.title}):\n\n{best.text}",
        "citations": [{"title": best.title, "cited_text": best.text[:220]}],
        "sources": sources,
        "mode": "retrieval",
    }


def _claude_answer(question: str, hits, sources) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c.text},
            "title": c.title,
            "citations": {"enabled": True},
        }
        for c, _ in hits
    ]
    try:
        # Server-side fallback: if Claude Opus 5.5 declines, the API retries on a fallback model in the same call.
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
        )
    except anthropic.AuthenticationError:
        return {"answer": "Claude API key is invalid.", "citations": [], "sources": sources, "mode": "error"}
    except anthropic.RateLimitError:
        return {"answer": "Rate limited by the Claude API. Try again shortly.", "citations": [], "sources": sources, "mode": "error"}
    except anthropic.APIStatusError as e:
        return {"answer": f"Claude API error ({e.status_code}).", "citations": [], "sources": sources, "mode": "error"}
    except anthropic.APIConnectionError:
        return {"answer": "Couldn't reach the Claude API.", "citations": [], "sources": sources, "mode": "error"}

    if response.stop_reason == "refusal":
        return {"answer": "The model declined to answer this question.", "citations": [], "sources": sources, "mode": "claude"}

    parts, citations, seen = [], [], set()
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for cite in getattr(block, "citations", None) or []:
            key = (cite.document_title, cite.cited_text)
            if key not in seen:
                seen.add(key)
                citations.append({"title": cite.document_title, "cited_text": cite.cited_text})
                parts.append(f" [{len(citations)}]")
    return {"answer": "".join(parts).strip(), "citations": citations, "sources": sources, "mode": "claude"}
