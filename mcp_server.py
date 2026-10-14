#!/usr/bin/env python3
"""
mcp_server.py — an MCP server that exposes a retrieval corpus as tools.

    printf '%s\\n' '<jsonrpc request>' | python3 mcp_server.py

===============================================================================
WHAT THIS IS
===============================================================================
MCP (Model Context Protocol) is how an agent reaches tooling and data that live
somewhere other than in its prompt. The server speaks **JSON-RPC 2.0 over stdio**: the
client writes one JSON object per line, the server writes one back.

    initialize    -> capabilities and server info
    tools/list    -> what this server can do, with JSON Schemas
    tools/call    -> run one, return content

⚠️ WHY IT IS SMALL AND STILL REAL. The value of MCP is not the transport, it is the
**contract**: a tool declares its schema, the client validates against it, and the
model never sees the corpus until it asks for it. That contract is fully exercised by
three methods, so that is all this implements.

⚠️ AND IT IS DELIBERATELY STATELESS. Every request is answered from the request alone —
no session carried between calls. An MCP server that leaks state between tool calls is
very hard to test and very easy to get subtly wrong.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "retrieval"
SERVER_VERSION = "1.0.0"

# ─── The corpus ──────────────────────────────────────────────────────────────
#
# ⚠️ Small on purpose. The protocol is what is being demonstrated, and a large corpus
# would only make the same three methods slower to read.

CORPUS = {
    "chunking": "Chunk size is what breaks retrieval. A chunk that is too large "
                "dilutes the embedding; one that is too small loses the context the "
                "answer needed. Offset-preserving splits keep the source locatable.",
    "hybrid": "Hybrid retrieval combines lexical and dense search because neither "
              "wins everywhere. An identifier is best found by exact match; a "
              "paraphrase is best found by embedding. Reciprocal rank fusion merges "
              "the two rankings by position, not by score.",
    "abstention": "Abstention is not a failure. A retrieval system that always "
                  "returns passages will answer questions the corpus does not "
                  "cover, confidently and wrongly. Coverage decides when to refuse.",
    "evaluation": "Evaluation needs a control. A judge that agrees with a human "
                  "ninety percent of the time can still be unable to fail the one "
                  "case that must fail. Measure sensitivity, not just agreement.",
    "vectors": "Vector search trades recall for latency. Exact search is a scan and "
               "is always correct; an approximate graph index is faster and returns "
               "most of the true neighbours. Report both numbers or neither.",
}

_WORD = __import__("re").compile(r"[a-z0-9]+")


# ⚠️⚠️ STOPWORDS ARE NOT OPTIONAL, AND THIS IS THE BUG THE TEST FOUND.
#
# The first version counted every word over two characters. Measured: the query
# "what is the capital of Peru" scored coverage 1.00 and RETURNED A PASSAGE, because
# "what" and "the" both appear somewhere in the corpus.
#
# ⚠️ Coverage over raw tokens measures whether the corpus knows the SENTENCE, not
# whether it knows the SUBJECT. Function words are present in every document, so they
# contribute nothing to the decision and everything to the score. Four documents make
# it worse: one matching "the" is already a quarter of the corpus.
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "what", "when", "where", "which",
    "who", "whom", "how", "why", "does", "did", "doing", "done", "can", "could",
    "should", "would", "will", "with", "without", "into", "onto", "from", "that",
    "this", "these", "those", "there", "their", "them", "then", "than", "but",
    "not", "you", "your", "its", "it's", "has", "have", "had", "been", "being",
    "about", "above", "after", "before", "between", "over", "under", "more",
    "most", "some", "any", "all", "each", "other", "such", "only", "own", "same",
}


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    text: str


class Index:
    """BM25 over the corpus. ⚠️ Enough to be real; not the point of the exercise."""

    K1, B = 1.5, 0.75

    def __init__(self, docs: dict[str, str]):
        self.ids = sorted(docs)
        self.toks = {i: tokenize(docs[i]) for i in self.ids}
        self.docs = docs
        self.avg = sum(len(t) for t in self.toks.values()) / max(len(self.toks), 1)
        self.df: Counter[str] = Counter()
        for toks in self.toks.values():
            for term in set(toks):
                self.df[term] += 1
        self.n = len(self.ids)

    def _idf(self, term: str) -> float:
        # ⚠️ Smoothed. Unsmoothed idf goes negative for a term in every document and
        # silently removes the most common words from every ranking.
        return math.log(1 + (self.n - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = tokenize(query)
        if not q:
            return []
        out: list[Hit] = []
        for doc_id in self.ids:
            toks = self.toks[doc_id]
            tf = Counter(toks)
            score = 0.0
            for term in set(q):
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.K1 * (1 - self.B + self.B * len(toks) / self.avg)
                score += self._idf(term) * (f * (self.K1 + 1)) / denom
            if score > 0:
                out.append(Hit(doc_id, score, self.docs[doc_id]))
        # ⚠️ Deterministic tie-break on id, so a repeated run is a repeated answer.
        out.sort(key=lambda h: (-h.score, h.doc_id))
        return out[:k]

    def coverage(self, query: str) -> float:
        """⚠️ The abstention signal. Similarity does not separate answerable from
        unanswerable; the fraction of the query the corpus actually knows does."""
        q = set(tokenize(query))
        if not q:
            return 0.0
        known = {t for t in q if t in self.df}
        if not known:
            return 0.0
        return len({t for t in known if any(t in set(self.toks[i]) for i in self.ids)}) / len(known)


INDEX = Index(CORPUS)
MIN_COVERAGE = 0.5

# ─── The tool contract ───────────────────────────────────────────────────────
#
# ⚠️ THE SCHEMAS ARE THE PRODUCT. A tool without an inputSchema is a function the model
# has to guess at. `additionalProperties: false` matters — without it a model invents
# arguments and the server silently ignores them.

TOOLS = [
    {
        "name": "search_corpus",
        "description": (
            "Search the approved corpus and return the best-matching passages. "
            "Returns an abstention result when the corpus does not cover the query, "
            "rather than returning passages that do not answer it."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language question."},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_document",
        "description": "Fetch one document by id. Use after search_corpus to read a passage in full.",
        "inputSchema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string", "enum": sorted(CORPUS)}},
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_documents",
        "description": "List the document ids available in the corpus.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class ToolError(Exception):
    """⚠️ A tool that fails returns isError, not a crash. A server that dies on bad
    input takes the whole agent session with it."""


def call_tool(name: str, args: dict) -> tuple[str, bool]:
    """Return (text, is_error)."""
    if name == "list_documents":
        return "\n".join(sorted(CORPUS)), False

    if name == "get_document":
        doc_id = args.get("doc_id")
        if doc_id not in CORPUS:
            return f"unknown document {doc_id!r}", True
        return CORPUS[doc_id], False

    if name == "search_corpus":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "query must be a non-empty string", True
        k = args.get("k", 3)
        if not isinstance(k, int) or not 1 <= k <= 10:
            return "k must be an integer between 1 and 10", True

        coverage = INDEX.coverage(query)
        if coverage < MIN_COVERAGE:
            # ⚠️ NOT AN ERROR. An abstention is a correct answer, and returning it as
            # isError would teach the client to retry — which is how a refusal becomes
            # an outage. The model should be told plainly that the corpus is silent.
            return (f"No passage in the corpus covers this question "
                    f"(coverage {coverage:.2f}). Do not answer from general knowledge.",
                    False)

        hits = INDEX.search(query, k=k)
        if not hits:
            return "No matching passage.", False
        body = "\n\n".join(f"[{h.doc_id}] (score {h.score:.2f})\n{h.text}" for h in hits)
        return body, False

    return f"unknown tool {name!r}", True


# ─── JSON-RPC ────────────────────────────────────────────────────────────────

def handle(req: dict) -> dict | None:
    """⚠️ Returns None for a NOTIFICATION (no `id`). Replying to one is a protocol
    violation, and it is the kind of thing a strict client rejects the session over."""
    rid = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    def ok(result):
        return None if rid is None else {"jsonrpc": "2.0", "id": rid, "result": result}

    if method == "initialize":
        client = params.get("protocolVersion", PROTOCOL_VERSION)
        return ok({
            "protocolVersion": client,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    if method in ("notifications/initialized", "notifications/cancelled"):
        return None

    if method == "ping":
        return ok({})

    if method == "tools/list":
        return ok({"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            text, is_error = call_tool(name, args)
        except Exception as e:                                   # pragma: no cover
            text, is_error = f"tool raised: {e}", True
        # ⚠️ Tool failures travel as content with isError=true, NOT as a JSON-RPC
        # error. The difference matters: a JSON-RPC error is a broken server, and a
        # client is entitled to stop talking to it.
        return ok({"content": [{"type": "text", "text": text}], "isError": is_error})

    if rid is None:
        return None
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve(stdin=sys.stdin, stdout=sys.stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}
        else:
            resp = handle(req)
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
