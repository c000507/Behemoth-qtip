#!/bin/bash
# Host-side check (no NeuronCores) of Qwen-format tool calling through neuron_server_fake (built by host_test.sh) on a
# Qwen3.5 tokenizer directory (tokenizer.json, tokenizer_config.json, chat_template.jinja; no weights): the prompt the
# server renders for a tools conversation (assistant tool_calls with OpenAI string arguments, tool results) against a
# golden made with HF apply_chat_template, and - the fake model samples pseudo-random tokens - that tool_choice
# required / a named function give syntactically valid calls (grammar + parser + finish_reason "tool_calls", streamed
# or not), and that an answer to a prompt that opens <think> starts as reasoning_content.
#   neuron/cpp/qwen_tools_test.sh [OUTDIR with neuron_server_fake, default build-host] [TOKENIZER_DIR, default $BEHEMOTH_QWEN_TOKENIZER]
set -e
OUT=$(cd "${1:-build-host}" && pwd)
TOK=${2:-$BEHEMOTH_QWEN_TOKENIZER}
if [ -z "$TOK" ] || [ ! -f "$TOK/tokenizer.json" ]; then echo "SKIP qwen_tools: no Qwen tokenizer directory (BEHEMOTH_QWEN_TOKENIZER)"; exit 0; fi
exec python3 -I - "$OUT/neuron_server_fake" "$TOK" <<'PY'
import glob, json, os, socket, subprocess, sys, tempfile, time, urllib.request

binary, tok = sys.argv[1], sys.argv[2]
work = tempfile.mkdtemp()
CONVERSATION = json.loads('{"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris and Oslo?"}, {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\", \\"days\\": 2, \\"metric\\": true, \\"note\\": \\"a <b>\\\\nc\\", \\"opts\\": {\\"k\\": [1, null]}}"}}, {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Oslo\\", \\"metric\\": false}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "{\\"temp\\": 21}"}, {"role": "tool", "tool_call_id": "call_2", "content": "rain"}, {"role": "assistant", "content": null, "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "ping", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "call_3", "content": "pong"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather \\u00e9t\\u00e9", "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Paris", "Oslo"]}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}, "required": ["city", "metric"]}}}, {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}]}')
GOLDEN = '<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"type": "function", "function": {"name": "get_weather", "description": "Weather été", "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Paris", "Oslo"]}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}, "required": ["city", "metric"]}}}\n{"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nBe brief.<|im_end|>\n<|im_start|>user\nWeather in Paris and Oslo?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nChecking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n<parameter=note>\na <b>\nc\n</parameter>\n<parameter=opts>\n{"k": [1, null]}\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=metric>\nfalse\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{"temp": 21}\n</tool_response>\n<tool_response>\nrain\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=ping>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\npong\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n'

checks = 0
def check(ok, what):
    global checks
    checks += 1
    if not ok:
        raise SystemExit("FAIL qwen_tools: " + what)

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
vocab = max(a["id"] for a in json.load(open(tok + "/tokenizer.json"))["added_tokens"]) + 1
env = dict(os.environ, BEHEMOTH_NEURON_NEFF_DIR=work, BEHEMOTH_DIAGNOSTIC_DIR=work + "/diag", BEHEMOTH_FAKE_VOCAB=str(vocab),
           BEHEMOTH_REQUEST_LOG="0")
env.pop("BEHEMOTH_TOOLS", None)
log = open(work + "/stderr.log", "w+")
server = subprocess.Popen([binary, "--model", tok, "--port", str(port)], stderr=log, stdout=subprocess.DEVNULL, env=env)
try:
    for _ in range(600):
        try: urllib.request.urlopen("http://127.0.0.1:%d/health" % port); break
        except Exception: time.sleep(0.1)
    def post(body):
        r = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
        return urllib.request.urlopen(r).read().decode()
    def rendered():  # the prompt of the latest request
        return json.load(open(max(glob.glob(work + "/diag/*.json"), key=os.path.getmtime)))["rendered_prompt"]
    log.seek(0)
    check("Tool calling: Qwen format" in log.read(), "the tokenizer directory is not detected as Qwen format")

    # 1. the prompt: tools block, assistant tool_calls (string arguments parsed for the template), tool results
    post(dict(CONVERSATION, max_tokens=2))
    check(rendered() == GOLDEN, "rendered tools conversation differs from the HF golden")
    check(rendered().endswith("<|im_start|>assistant\n<think>\n"), "generation prompt")

    weather = {"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string", "enum": ["Paris", "Oslo"]}, "days": {"type": "integer"},
                                         "metric": {"type": "boolean"}}, "required": ["city", "metric"]}}}
    ping = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}
    ask = [{"role": "user", "content": "Weather in Paris?"}]
    def valid(call):
        check(call["type"] == "function" and call["id"].startswith("call_"), "call shape")
        a = json.loads(call["function"]["arguments"])
        if call["function"]["name"] == "ping":
            check(a == {}, "ping takes no arguments")
        else:
            check(call["function"]["name"] == "get_weather", "an offered function")
            check(a["city"] in ("Paris", "Oslo") and isinstance(a["metric"], bool), "required arguments, typed")
            check(set(a) <= {"city", "days", "metric"} and type(a.get("days", 0)) is int, "declared arguments, typed")
            check(list(a) == [k for k in ("city", "days", "metric") if k in a], "declared order")

    # 2. required (one call), under the grammar from the first token (reasoning_effort none)
    base = {"messages": ask, "tools": [weather, ping], "reasoning_effort": "none", "max_tokens": 400, "temperature": 1.0}
    for seed in range(4):
        c = json.loads(post(dict(base, tool_choice="required", parallel_tool_calls=False, seed=seed)))["choices"][0]
        check(c["finish_reason"] == "tool_calls" and c["message"]["content"] is None, "required: finish_reason tool_calls, content null")
        check(len(c["message"]["tool_calls"]) == 1, "required, parallel_tool_calls false: one call")
        valid(c["message"]["tool_calls"][0])
    check("<tools>\n" + json.dumps(weather) in rendered(), "tools rendered")
    # 3. a named function
    named = {"type": "function", "function": {"name": "get_weather"}}
    c = json.loads(post(dict(base, tool_choice=named, parallel_tool_calls=False, seed=7)))["choices"][0]
    check(c["finish_reason"] == "tool_calls" and c["message"]["tool_calls"][0]["function"]["name"] == "get_weather", "named")
    valid(c["message"]["tool_calls"][0])
    # 4. parallel calls: every call valid (the fake model may go on calling up to the token limit)
    c = json.loads(post(dict(base, tool_choice="required", seed=3, max_tokens=300)))["choices"][0]
    check(c["finish_reason"] in ("tool_calls", "length") and len(c["message"]["tool_calls"]) >= 1, "parallel")
    for call in c["message"]["tool_calls"][:-1] if c["finish_reason"] == "length" else c["message"]["tool_calls"]:
        valid(call)
    # 5. streamed: name first, then argument fragments that add up to the arguments
    calls, content, finish = [], "", None
    for line in post(dict(base, tool_choice="required", parallel_tool_calls=False, seed=1, stream=True)).split("\n"):
        if not line.startswith("data: {"):
            continue
        choice = json.loads(line[6:])["choices"]
        if not choice:
            continue
        finish = choice[0].get("finish_reason") or finish
        delta = choice[0].get("delta", {})
        content += delta.get("content") or ""
        for item in delta.get("tool_calls", []):
            if "id" in item:
                check(item["index"] == len(calls) and item["function"]["arguments"] == "", "stream: a call starts with its name")
                calls.append({"id": item["id"], "type": item["type"], "function": {"name": item["function"]["name"], "arguments": ""}})
            else:
                calls[item["index"]]["function"]["arguments"] += item["function"]["arguments"]
    check(finish == "tool_calls" and content == "" and len(calls) == 1, "stream: one call, no content, finish_reason tool_calls")
    valid(calls[0])
    # 6. the prompt opened <think>: free answers start as reasoning (auto, no tools), and so do gated ones unless
    #    reasoning_effort is none; pseudo-random tokens do not close the block
    for extra in ({"tools": [weather], "tool_choice": "auto"}, {}, {"tools": [weather], "tool_choice": "required"},
                  {"response_format": {"type": "json_object"}}):
        m = json.loads(post(dict({"messages": ask, "max_tokens": 24, "temperature": 1.0, "seed": 5}, **extra)))["choices"][0]
        check(m["finish_reason"] == "length" and m["message"]["reasoning_content"] and not m["message"]["content"] and
              "tool_calls" not in m["message"], "think block opened by the prompt: reasoning_content (%s)" % ",".join(extra))
    # 6b. enable_thinking=false (top level or chat_template_kwargs): the template's closed empty block, so the answer is
    #     content, not reasoning; true / absent keep the opened block
    for kw, closed in (({"enable_thinking": False}, True), ({"chat_template_kwargs": {"enable_thinking": False}}, True),
                       ({"chat_template_kwargs": {"enable_thinking": True}}, False)):
        m = json.loads(post(dict({"messages": ask, "max_tokens": 12, "temperature": 1.0, "seed": 5}, **kw)))["choices"][0]
        check(rendered().endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") == closed, "enable_thinking prompt %s" % kw)
        check(bool(m["message"]["content"]) == closed and bool(m["message"]["reasoning_content"]) != closed, "enable_thinking split %s" % kw)
    # 7. tool_choice none: tools are not rendered and nothing is parsed
    m = json.loads(post({"messages": ask, "tools": [weather], "tool_choice": "none", "max_tokens": 8}))["choices"][0]
    check("<tools>" not in rendered() and "tool_calls" not in m["message"], "tool_choice none")
    # 8. other constrained answers still work (JSON from the first token)
    m = json.loads(post({"messages": ask, "response_format": {"type": "json_object"}, "reasoning_effort": "none",
                         "max_tokens": 400, "temperature": 1.0, "seed": 2}))["choices"][0]
    if m["finish_reason"] == "stop":
        check(isinstance(json.loads(m["message"]["content"]), dict), "json_object")
    print("PASS qwen_tools: %d checks" % checks)
finally:
    server.kill()
PY
