#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-test}" in
  test) python3 test_mcp.py ;;
  serve) exec python3 mcp_server.py ;;
  demo)
    printf '\\n' \
      '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
      '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
      '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_corpus","arguments":{"query":"how do fusion rankings work"}}}' \
      | python3 mcp_server.py
    ;;
  *) sed -n '2,7p' "$0" ;;
esac
