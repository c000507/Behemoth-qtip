"""Experimental single-request OpenAI-style chat server for LlamaRunner."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
import time
import uuid

import numpy as np
from llama import LlamaRunner


class ChatService:
    def __init__(self, runner, tokenizer, model):
        self.runner, self.tokenizer, self.model = runner, tokenizer, model
        self.lock = Lock()
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        if backend is not None:
            backend.no_truncation()

    def complete(self, request):
        messages = request.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a non-empty array')
        if request.get('stream', False):
            raise ValueError('Streaming is not supported')
        if request.get('temperature', 0) != 0:
            raise ValueError('Only greedy sampling (temperature 0) is supported')
        if request.get('model', self.model) != self.model:
            raise ValueError('Unknown model')
        limit = request.get('max_tokens', 128)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError('max_tokens must be a positive integer')
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    truncation=False)
        if len(tokens) + limit - 1 > self.runner.max_context:
            raise ValueError('Prompt and completion exceed model context')
        generation_config = getattr(self.runner, 'generation_config', {})
        eos = generation_config.get('eos_token_id', self.runner.checkpoint.config.get('eos_token_id', self.tokenizer.eos_token_id))
        eos = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
        with self.lock:
            self.runner.reset()
            logits = self.runner.prefill(tokens)
            generated = []
            finish = 'length'
            for i in range(limit):
                token = int(np.argmax(logits)); generated.append(token)
                if token in eos:
                    finish = 'stop'; break
                if i + 1 < limit:
                    logits = self.runner.forward_token(token)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
                'created': int(time.time()), 'model': self.model,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                             'finish_reason': finish}],
                'usage': {'prompt_tokens': len(tokens), 'completion_tokens': len(generated),
                          'total_tokens': len(tokens) + len(generated)}}


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            payload = json.dumps(value).encode('utf-8')
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_json(404, {'error': {'message': 'Not found'}}); return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length <= 0 or length > 4 * 1024 * 1024:
                    raise ValueError('Invalid request length')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict): raise ValueError('Request must be a JSON object')
                self.send_json(200, service.complete(body))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {'error': {'message': str(exc)}})
            except Exception:
                self.send_json(500, {'error': {'message': 'Inference failed; see server logs'}})
                import traceback
                traceback.print_exc()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--model', default='behemoth-neuron')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', action='store_true')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    runner = LlamaRunner(args.checkpoint, simulate=not args.device)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(ChatService(runner, tokenizer, args.model)))
    print(f'Listening on {args.host}:{args.port}', flush=True)
    server.serve_forever()

if __name__ == '__main__': main()
