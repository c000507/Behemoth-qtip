"""End-to-end tiny FP8 Llama and cached decoding against NumPy."""
import json
import os
from pathlib import Path
import sys
import tempfile
import ml_dtypes
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_checkpoint import write_checkpoint
from llama import LlamaRunner
from backend import bf16
from server import ChatService
from test_server import FakeTokenizer


def main():
    rng = np.random.default_rng(48)
    h, heads, kvheads, dim, vocab = 64, 2, 1, 32, 32
    def random(shape): return rng.normal(0, .08, shape).astype(np.float32)
    tensors = {}
    def put(name, value, dtype='F32'): tensors[name] = (dtype, value)
    emb, lm = random((vocab,h)), random((vocab,h))
    put('model.embed_tokens.weight', emb); put('lm_head.weight', lm)
    put('model.norm.weight', np.ones(h, np.float32))
    p = 'model.layers.0.'
    put(p+'input_layernorm.weight', np.ones(h, np.float32))
    put(p+'post_attention_layernorm.weight', np.ones(h, np.float32))
    sizes = {'self_attn.q_proj': (h,h), 'self_attn.k_proj': (32,h),
             'self_attn.v_proj': (32,h), 'self_attn.o_proj': (h,h),
             'mlp.gate_proj': (h,h), 'mlp.up_proj': (h,h), 'mlp.down_proj': (h,h)}
    weights = {}
    for name, shape in sizes.items():
        w = random(shape).astype(ml_dtypes.float8_e4m3fn)
        weights[name] = w.astype(np.float32)
        put(p+name+'.weight', w.view(np.uint8), 'F8_E4M3')
        put(p+name+'.weight_scale', np.ones(shape[0],np.float32))
    def norm(x): return bf16(x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5))
    def linear(name, x): return bf16(bf16(x) @ weights[name].T)
    keys, vals = [], []
    def ref(token, pos):
        state = bf16(emb[token:token+1]); x=norm(state)
        q=linear('self_attn.q_proj',x).reshape(heads,dim)
        k=linear('self_attn.k_proj',x).reshape(kvheads,dim)
        v=linear('self_attn.v_proj',x).reshape(kvheads,dim)
        angle = pos * 10000 ** (-np.arange(0,dim,2,dtype=np.float32)/dim)
        def rope(a): return bf16(np.concatenate((a[:,:dim//2]*np.cos(angle)-a[:,dim//2:]*np.sin(angle), a[:,dim//2:]*np.cos(angle)+a[:,:dim//2]*np.sin(angle)),axis=1))
        q,k=rope(q),rope(k); keys.append(k); vals.append(v)
        ks,vs=np.stack(keys),np.stack(vals); att=[]
        for head in range(heads):
            scores=ks[:,0] @ q[head] / np.sqrt(dim); ex=np.exp(scores-scores.max())
            att.append(ex @ vs[:,0] / ex.sum())
        state=bf16(state+linear('self_attn.o_proj',bf16(np.array(att).reshape(1,-1))))
        x=norm(state); gate=linear('mlp.gate_proj',x); up=linear('mlp.up_proj',x)
        act=bf16(gate/(1+np.exp(-gate))*up)
        state=bf16(state+linear('mlp.down_proj',act)); return norm(state) @ lm.T
    with tempfile.TemporaryDirectory() as d:
        path = Path(d); write_checkpoint(path, tensors)
        (path/'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':h,'intermediate_size':h,
            'num_attention_heads':heads,'num_key_value_heads':kvheads,'num_hidden_layers':1,
            'vocab_size':vocab,'max_position_embeddings':32,'rms_norm_eps':1e-5}))
        runner=LlamaRunner(path, simulate=os.environ.get("NEURON_TEST_DEVICE") != "1")
        for pos, token in enumerate((3, 8, 19)):
            got=runner.forward_token(token); expected=ref(token,pos).reshape(-1)
            np.testing.assert_allclose(got, expected, rtol=.015, atol=.002)
        assert runner.position == 3 and len(runner.keys[0]) == 3
        previous = runner.backend.attention
        def fail(*args): raise RuntimeError('injected failure')
        runner.backend.attention = fail
        try: runner.forward_token(5)
        except RuntimeError: pass
        else: raise AssertionError('Expected injected failure')
        assert runner.position == 3 and len(runner.keys[0]) == 3 and len(runner.values[0]) == 3
        runner.backend.attention = previous
        runner.reset(); assert runner.position == 0 and not runner.keys[0]
        reply = ChatService(runner, FakeTokenizer(), 'tiny').complete({'model':'tiny', 'messages':[{'role':'user','content':'hi'}], 'max_tokens':1})
        assert reply['usage']['prompt_tokens'] == 2 and reply['usage']['completion_tokens'] == 1
        print('Tiny FP8 Llama checkpoint, GQA prefill and cached decode match NumPy')

if __name__ == '__main__': main()
