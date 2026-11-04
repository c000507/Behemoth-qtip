"""Compare native BPE with Hugging Face tokenizers, including >2048-token input.

python tokenizer_parity.py --binary BUILD/glm53 --model CHECKPOINT_OR_ASSETS
"""
import argparse
import json
import pathlib
import subprocess
import tempfile

from tokenizers import Tokenizer
from jinja2 import Environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    args = parser.parse_args()
    hf = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    hf.no_truncation()
    hf.no_padding()
    cases = [
        "Hello, world!", "I'M here; it's 1234567890 and 0001234567.",
        " \t \n\nCafé naïve 中文 日本語 العربية Ελληνικά 😀 🧑‍💻\r\n",
        "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n<think>test</think>",
        "[gMASK]<sop><|user|>hi<|assistant|><think>",
        "!?!...\n\n", "  trailing whitespace \t  ", "word " * 4096,
    ]
    with tempfile.TemporaryDirectory(prefix="glm53-tokenizer-") as folder:
        root = pathlib.Path(folder)
        for i, text in enumerate(cases):
            input_path, output = root / f"{i}.txt", root / f"{i}.json"
            input_path.write_text(text)
            subprocess.run([str(args.binary), "tokenize", "--model", str(args.model),
                            "--input", str(input_path), "--out", str(output)], check=True)
            native = json.loads(output.read_text())["tokens"]
            expected = hf.encode(text, add_special_tokens=False).ids
            assert native == expected, (i, native[:40], expected[:40])
            print("PASS tokenizer case", i, "tokens", len(native))
        assert len(native) > 2048, "serialized tokenizer truncation was applied"
        config = json.loads((args.model / "tokenizer_config.json").read_text())
        template = config.get("chat_template")
        if template is None:
            template = (args.model / "chat_template.jinja").read_text()
        if isinstance(template, list):
            template = next(t["template"] for t in template if t["name"] == "default")
        env = Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        def token(key):
            value = config.get(key)
            return value.get("content", "") if isinstance(value, dict) else value or ""
        for i, messages in enumerate([
            [dict(role="user", content="Hello")],
            [dict(role="system", content="Be concise."), dict(role="user", content="Test 中文 😀")],
            [dict(role="user", content="Hi"), dict(role="assistant", content=" A "), dict(role="user", content="Again")],
        ]):
            request, output = root / f"chat-{i}.json", root / f"chat-{i}.txt"
            request.write_text(json.dumps(messages))
            subprocess.run([str(args.binary), "render", "--model", str(args.model),
                            "--messages", str(request), "--out", str(output)], check=True)
            expected = env.from_string(template).render(messages=messages, tools=None,
                bos_token=token("bos_token"), eos_token=token("eos_token"), add_generation_prompt=True)
            assert output.read_text() == expected, (i, repr(output.read_text()), repr(expected))
            print("PASS chat template case", i)


if __name__ == "__main__":
    main()
