"""Exercise the HTTP chat endpoint without loading model weights."""
from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import sys
import threading
import urllib.request
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import ChatService, handler_for


class FakeTokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['truncation'] is False and kwargs['add_generation_prompt']
        return [10, 11]
    def decode(self, tokens, skip_special_tokens): return ','.join(map(str, tokens))

class FakeRunner:
    max_context = 8
    checkpoint = type('Checkpoint', (), {'config': {'eos_token_id': 2}})()
    def reset(self): self.calls = 0
    def prefill(self, tokens): return np.array([0., 1., 0.])
    def forward_token(self, token): return np.array([0., 0., 1.])


def main():
    service = ChatService(FakeRunner(), FakeTokenizer(), 'test')
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % server.server_port,
              data=json.dumps({'model':'test', 'messages':[{'role':'user','content':'hello'}], 'max_tokens':4}).encode(),
              headers={'Content-Type':'application/json'})
        response = json.load(urllib.request.urlopen(req))
        assert response['choices'][0]['message']['content'] == '1,2'
        assert response['choices'][0]['finish_reason'] == 'stop'
        assert response['usage']['total_tokens'] == 4
        try: service.complete({'messages':[{}], 'stream':True})
        except ValueError: pass
        else: raise AssertionError('Streaming should be rejected')
        try: service.complete({'messages':[{}], 'max_tokens':8})
        except ValueError: pass
        else: raise AssertionError('Overlong request should be rejected')
        print('Experimental chat endpoint handles greedy generation, EOS and validation')
    finally:
        server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': main()
