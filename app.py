"""DocuChat web server. Run: python app.py, then open http://localhost:8000"""

from __future__ import annotations

import json
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from rag import KnowledgeBase, answer, has_claude_credentials, read_document

ROOT = Path(__file__).parent
kb = KnowledgeBase()
kb.load_folder(ROOT / "sample_docs")


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = (ROOT / "static" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self._json({"sources": kb.sources, "chunks": len(kb.chunks), "ai": has_claude_credentials()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        if self.path == "/api/ask":
            question = json.loads(data or b"{}").get("question", "").strip()
            if not question:
                return self._json({"error": "question is required"}, 400)
            return self._json(answer(kb, question))
        if self.path == "/api/upload":
            name = Path(unquote(self.headers.get("X-Filename", "upload.txt"))).name
            if Path(name).suffix.lower() not in {".md", ".txt", ".pdf"}:
                return self._json({"error": "Only .md, .txt and .pdf files are supported"}, 400)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                path.write_bytes(data)
                try:
                    text = read_document(path)
                except ImportError:
                    return self._json({"error": "Install pypdf to upload PDFs"}, 400)
            added = kb.add_document(name, text)
            return self._json({"added_chunks": added, "sources": kb.sources})
        self._json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"DocuChat running on http://localhost:{port} ({len(kb.chunks)} chunks, AI answers: {has_claude_credentials()})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
