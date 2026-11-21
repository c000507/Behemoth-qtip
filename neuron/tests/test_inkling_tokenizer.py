"""Server tokenizer path (glm53::Tokenizer via neuron/cpp/tok_check, with neuron/inkling_chat_template.jinja: the checkpoint's
template rewritten for minja) vs HF (tokenizers + the ORIGINAL chat_template.jinja through transformers' jinja):
encode of raw texts, chat rendering + ids for plain / system / multi-turn / reasoning conversations.
usage: python neuron/tests/test_inkling_tokenizer.py OURS_DIR HF_DIR TOK_CHECK_BINARY
  OURS_DIR: tokenizer.json, tokenizer_config.json, chat_template.jinja = inkling_chat_template.jinja
  HF_DIR:   the same with the checkpoint's original chat_template.jinja"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

TEXTS = ['Hello world', 'The quick brown fox jumps over the lazy dog.', '  leading spaces and\n\nnewlines\ttab', 'def f(x):\n    return x**2  # π ≈ 3.14159',
         '日本語のテキストと한국어 그리고 émojis 🚀🔥', "I'd've said it's 1234567 dollars, isn't it?", 'CamelCaseWord HTTPServer XMLHttpRequest',
         '<|message_user|><|content_text|>hi<|end_message|>', 'tabs\t\tand   triple   spaces\r\nwindows lines', '1+1=2; 12345678901234; 3.14e-10']
CONVS = [
    [{'role': 'user', 'content': 'What is the capital of France?'}],
    [{'role': 'system', 'content': 'You are terse.'}, {'role': 'user', 'content': 'Explain attention in one sentence.'}],
    [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'Hello!'}, {'role': 'user', 'content': 'bye'}],
    [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'Hello!', 'reasoning_content': 'The user greets me.'},
     {'role': 'user', 'content': 'Write a haiku about the sea.'}],
]


def run(binary, model, *args):
    return json.loads(subprocess.run([binary, model, *args], check=True, capture_output=True, text=True).stdout)


def main():
    ours_dir, hf_dir, binary = sys.argv[1], sys.argv[2], sys.argv[3]
    hf = AutoTokenizer.from_pretrained(hf_dir)
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for t in TEXTS:
            Path(d, 't.txt').write_text(t)
            ours = run(binary, ours_dir, '--text', str(Path(d, 't.txt')))['ids']
            ref = hf.encode(t, add_special_tokens=False)
            ok = ours == ref
            bad += not ok
            print(f'text {t[:40]!r:44s} {"OK" if ok else "MISMATCH"} ours={ours[:10]} hf={ref[:10]}')
        for i, conv in enumerate(CONVS):
            Path(d, 'm.json').write_text(json.dumps(conv))
            ours = run(binary, ours_dir, str(Path(d, 'm.json')))
            ref_text = hf.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            ref_ids = hf.encode(ref_text, add_special_tokens=False)
            ok = ours['rendered'] == ref_text and ours['ids'] == ref_ids
            bad += not ok
            print(f'conv {i}: {"OK" if ok else "MISMATCH"} ({len(ref_ids)} ids)')
            if not ok:
                print('  ours:', repr(ours['rendered'])[:400])
                print('  hf:  ', repr(ref_text)[:400])
    print('mismatches:', bad)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
