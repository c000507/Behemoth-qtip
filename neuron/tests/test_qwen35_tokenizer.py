"""Server tokenizer path (glm53::Tokenizer via neuron/cpp/tok_check) vs HF AutoTokenizer on a checkpoint directory:
encode of raw texts and chat-template rendering + ids for plain / system / multi-turn / tool conversations.
usage: python neuron/tests/test_qwen35_tokenizer.py MODEL_DIR TOK_CHECK_BINARY"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

TEXTS = ['Hello world', 'The quick brown fox jumps over the lazy dog.', '  leading spaces and\n\nnewlines\ttab', 'def f(x):\n    return x**2  # π ≈ 3.14159',
         '日本語のテキストと한국어 그리고 émojis 🚀🔥', "I'd've said it's 1234567 dollars, isn't it?", '<|im_start|>user\nhi<|im_end|>\n']
CONVS = [
    [{'role': 'user', 'content': 'What is the capital of France?'}],
    [{'role': 'system', 'content': 'You are terse.'}, {'role': 'user', 'content': 'Explain RoPE in one sentence.'}],
    [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': '<think>\nthinking\n</think>\n\nHello!'}, {'role': 'user', 'content': 'bye'}],
]
TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather by city', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
TOOL_CONV = [{'role': 'user', 'content': 'Weather in Paris?'},
             {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'}}]},
             {'role': 'tool', 'content': '{"temp": 21}'}, {'role': 'user', 'content': 'and London?'}]


def run(binary, model, *args):
    return json.loads(subprocess.run([binary, model, *args], check=True, capture_output=True, text=True).stdout)


def main():
    model, binary = sys.argv[1], sys.argv[2]
    hf = AutoTokenizer.from_pretrained(model)
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for t in TEXTS:
            Path(d, 't.txt').write_text(t)
            ours = run(binary, model, '--text', str(Path(d, 't.txt')))['ids']
            ref = hf.encode(t, add_special_tokens=False)
            ok = ours == ref
            bad += not ok
            print(f'text {t[:40]!r:44s} {"OK" if ok else "MISMATCH"} ours={ours[:12]} hf={ref[:12]}')
        for i, conv in enumerate(CONVS + [TOOL_CONV]):
            tools = TOOLS if conv is TOOL_CONV else None
            Path(d, 'm.json').write_text(json.dumps(conv))
            Path(d, 'tools.json').write_text(json.dumps(tools))
            try:
                ours = run(binary, model, str(Path(d, 'm.json')), *([str(Path(d, 'tools.json'))] if tools else []))
            except subprocess.CalledProcessError as e:
                print(f'conv {i}: tok_check FAILED: {e.stderr[-300:]}')
                bad += 1
                continue
            hf_conv = json.loads(json.dumps(conv))
            for m in hf_conv:
                for c in m.get('tool_calls', []):
                    c['function']['arguments'] = json.loads(c['function']['arguments'])
            ref_text = hf.apply_chat_template(hf_conv, tools=tools, tokenize=False, add_generation_prompt=True)
            ref_ids = hf.encode(ref_text, add_special_tokens=False)
            ok = ours['ids'] == ref_ids
            bad += not ok
            print(f'conv {i} ({len(conv)} msgs{", tools" if tools else ""}): {"OK" if ok else "MISMATCH"} ({len(ours["ids"])} vs {len(ref_ids)} ids)')
            if not ok:
                print('  ours:', repr(ours['rendered'][:600]))
                print('  hf  :', repr(ref_text[:600]))
    print('mismatches:', bad)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
