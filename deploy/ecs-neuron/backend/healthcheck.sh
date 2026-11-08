#!/bin/bash
# ECS container health check. ECS caps startPeriod at 300 s, but startup (S3 pull of ~25 GB + ~70 s device load)
# can take longer, so report healthy while the entrypoint's startup marker is younger than STARTUP_GRACE_S.
# Once /health answers 200 the marker is removed and only /health counts.
PORT=${PORT:-1337}
MARKER=${STARTUP_MARKER:-/tmp/backend-starting}
GRACE=${STARTUP_GRACE_S:-1200}
if curl -fsS -m 4 "http://127.0.0.1:$PORT/health" >/dev/null; then
  rm -f "$MARKER"
  exit 0
fi
if [ -f "$MARKER" ] && [ $(( $(date +%s) - $(cat "$MARKER") )) -lt "$GRACE" ]; then
  exit 0
fi
exit 1
