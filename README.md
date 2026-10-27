# mcp-retrieval-server

**A retrieval corpus exposed over MCP, so an agent reaches it as tools rather than as prompt text.**

```bash
python3 test_mcp.py        # 16 protocol checks, no dependencies
python3 mcp_server.py      # then speak JSON-RPC on stdin
```

## Why MCP rather than a prompt

 Putting the corpus in the prompt costs tokens on every call, cannot be scoped per
request, and gives the model no way to say *"that document isn't there."* A tool call
does all three: the model asks, the server decides what it is allowed to see, and a
miss is a real answer.

## The protocol

JSON-RPC 2.0 over stdio — one object per line in, one per line out.

| Method | |
|---|---|
| `initialize` | protocol version, capabilities, server info |
| `tools/list` | the tool set, each with a JSON Schema |
| `tools/call` | run one, return content |

```bash
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | python3 mcp_server.py
```

## The three decisions worth explaining

**1. Schemas set `additionalProperties: false`.** Without it a model invents
arguments, the server silently ignores them, and the failure is invisible — the tool
returns a plausible answer to a question nobody asked.

**2. An abstention is content, not an error.** When the corpus does not cover a
question the server returns `isError: false` and a sentence saying so:

```json
{"content": [{"type": "text",
  "text": "No passage in the corpus covers this question (coverage 0.00). Do not answer from general knowledge."}],
 "isError": false}
```

 **A JSON-RPC error means the server is broken**, and a client is entitled to stop
talking to it or retry. A refusal is neither — returning it as an error is how a
correct outcome becomes a retry storm.

**3. Tool failures never masquerade as JSON-RPC errors.** An unknown `doc_id`, a
missing argument and an unknown tool all return `isError: true` with content, so the
model can correct itself. Only an unknown *method* produces a protocol error — that
one really is a broken client.

##  The bug the tests found

The first version scored coverage over every word longer than two characters.
Measured:

```
"what is the capital of Peru"  ->  coverage 1.00  ->  RETURNED A PASSAGE
```

`what` and `the` both occur somewhere in the corpus, so the question looked
answerable.  **Coverage over raw tokens measures whether the corpus knows the
*sentence*, not the *subject*.** Function words appear in every document — they
contribute nothing to the decision and everything to the score, and with four
documents one match is already a quarter of the corpus.

Fixed with a stopword filter. The same question now abstains at coverage `0.00`.

## What this is not

 **The retriever is small on purpose** — four documents and a BM25, because the
protocol is the subject and a large corpus would only make the same three methods
slower to read. The interesting parts are the **contract** (schemas, error taxonomy,
abstention-as-content) and the transport, not the ranking.

## Test coverage

16 checks, driven over a **real subprocess pipe** rather than by importing the module:

- a notification gets **no reply** — replying to one is a protocol violation
- every tool declares an `inputSchema` that refuses extra properties
- an uncovered query **abstains**, and the abstention is **not** an error
- tool failures return `isError: true`; only an unknown method is a JSON-RPC error

 One shared response-parsing bug would otherwise pass every in-process test and
break every real client.
