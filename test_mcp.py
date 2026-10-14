"""test_mcp.py — the protocol, exercised end to end over a real stdin/stdout pipe.

    python3 test_mcp.py
"""

import json
import subprocess
import sys
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def rpc(requests: list[dict]) -> list[dict]:
    """⚠️ Drives the server as a CLIENT DOES — a separate process, over a pipe, one
    JSON object per line. Importing the module and calling handle() would prove the
    functions work and say nothing about whether the transport does."""
    payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "mcp_server.py")],
        input=payload, capture_output=True, text=True, timeout=20,
    )
    return [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]


def main() -> int:
    print("test_mcp.py — the MCP server over stdio\n")

    responses = rpc([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "t", "version": "1"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "search_corpus", "arguments": {"query": "how does hybrid retrieval fuse rankings"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "search_corpus", "arguments": {"query": "what is the capital of Peru"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "get_document", "arguments": {"doc_id": "abstention"}}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "get_document", "arguments": {"doc_id": "nope"}}},
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
         "params": {"name": "search_corpus", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
         "params": {"name": "no_such_tool", "arguments": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 9, "method": "nonexistent/method"},
    ])

    # ⚠️ NINE responses for TEN requests: the notification must NOT be answered.
    # Replying to one is a protocol violation a strict client rejects the session over.
    check("a notification gets no reply", len(responses) == 9,
          f"{len(responses)} response(s) for 10 requests")

    by_id = {}
    for r in responses:
        if "id" in r:
            by_id[r["id"]] = r

    init = by_id.get(1, {}).get("result", {})
    check("initialize returns server info and tools capability",
          init.get("serverInfo", {}).get("name") == "retrieval" and "tools" in init.get("capabilities", {}),
          f"protocolVersion echoed as {init.get('protocolVersion')!r}")

    tools = by_id.get(2, {}).get("result", {}).get("tools", [])
    check("tools/list returns all three tools", len(tools) == 3,
          ", ".join(t["name"] for t in tools))
    check("every tool declares an inputSchema",
          all(isinstance(t.get("inputSchema"), dict) for t in tools))
    check("schemas refuse extra properties",
          all(t["inputSchema"].get("additionalProperties") is False for t in tools),
          "without this a model invents arguments and the server silently ignores them")

    hit = by_id.get(3, {}).get("result", {})
    text = hit.get("content", [{}])[0].get("text", "")
    check("search_corpus finds the right passage", "[hybrid]" in text,
          text.splitlines()[0][:60] if text else "empty")
    check("a successful tool call is not an error", hit.get("isError") is False)

    # ⚠️ THE HEADLINE BEHAVIOUR. An uncovered question must abstain, and the
    # abstention must NOT be isError — a client that sees an error retries, and a
    # refusal that looks like a fault is how a refusal becomes an outage.
    miss = by_id.get(4, {}).get("result", {})
    mtext = miss.get("content", [{}])[0].get("text", "")
    check("an uncovered query ABSTAINS", "No passage" in mtext, mtext[:66])
    check("abstention is NOT reported as an error", miss.get("isError") is False,
          "an error invites a retry; a refusal is a correct answer")
    check("abstention tells the model not to answer from memory",
          "general knowledge" in mtext)

    doc = by_id.get(5, {}).get("result", {})
    check("get_document returns the text",
          "Abstention is not a failure" in doc.get("content", [{}])[0].get("text", ""))
    check("an unknown document is an error",
          by_id.get(6, {}).get("result", {}).get("isError") is True)
    check("a missing required argument is an error",
          by_id.get(7, {}).get("result", {}).get("isError") is True)
    check("an unknown tool is an error",
          by_id.get(8, {}).get("result", {}).get("isError") is True)

    err = by_id.get(9, {}).get("error", {})
    check("an unknown METHOD is a JSON-RPC error, not tool content",
          err.get("code") == -32601, f"code {err.get('code')}")
    check("tool failures never masquerade as JSON-RPC errors",
          all("error" not in by_id[i] for i in (6, 7, 8)),
          "a JSON-RPC error says the server is broken, and a client may hang up")

    print()
    if FAILURES:
        print(f"  ❌ {len(FAILURES)} failed: {', '.join(FAILURES)}")
        return len(FAILURES)
    print("  ✅ the protocol contract holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
