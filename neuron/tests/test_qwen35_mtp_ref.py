"""qwen35_moe_ref.Mtp (the checkpoint's multi-token-prediction head) against transformers modules on a tiny random config, f32.

transformers ignores `mtp.*` (`_keys_to_ignore_on_load_unexpected`), so the oracle is composed from ITS modules the way
vLLM (Qwen3_5MultiTokenPredictor) and llama.cpp (qwen35moe graph_mtp) run the head: Qwen3_5MoeRMSNorm x 2 + Linear(2H, H)
-> a Qwen3_5MoeTextModel with ONE full_attention layer fed through inputs_embeds (its layer = mtp.layers.0, its final norm
= mtp.norm, its DynamicCache = the MTP KV cache) -> the main model's lm_head. The hidden state fed in is the main model's
last_hidden_state (after the final norm). Checked: a prefill over (h[p], x[p+1]) pairs, single steps with the carried
cache, and 4-step draft chains (h <- the head's own output, x <- its greedy draft) against HF with a copied cache.
usage: python neuron/tests/test_qwen35_mtp_ref.py"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen35_moe_ref as R

torch.manual_seed(0)
from transformers import Qwen3_5MoeForCausalLM  # noqa: E402
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig  # noqa: E402

for name in ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated'):
    setattr(M, name, None)
M.is_fast_path_available = False


class HfMtp(torch.nn.Module):
    def __init__(self, cfg, experts):
        super().__init__()
        c = dict(cfg)
        c.update(num_hidden_layers=1, layer_types=['full_attention'], num_experts=experts)
        hc = Qwen3_5MoeTextConfig(**c)
        hc._attn_implementation = 'eager'
        self.body = M.Qwen3_5MoeTextModel(hc)           # layers.0 = mtp.layers.0, norm = mtp.norm (embed_tokens unused)
        self.fc = torch.nn.Linear(2 * hc.hidden_size, hc.hidden_size, bias=False)
        self.pre_fc_norm_embedding = M.Qwen3_5MoeRMSNorm(hc.hidden_size, eps=hc.rms_norm_eps)
        self.pre_fc_norm_hidden = M.Qwen3_5MoeRMSNorm(hc.hidden_size, eps=hc.rms_norm_eps)

    def load(self, w, cfg):
        g = R.Cfg({**cfg, 'num_hidden_layers': 1, 'layer_types': ['full_attention']})
        sd = self.body.state_dict()
        E = w['mtp.layers.0.mlp.gate.weight'].shape[0]
        ex = lambda e, n: w[f'mtp.layers.0.mlp.experts.{e}.{n}_proj.weight']
        new = {}
        for k in sd:
            if k == 'embed_tokens.weight':
                new[k] = sd[k]
            elif k.endswith('experts.gate_up_proj'):
                new[k] = torch.stack([torch.cat((ex(e, 'gate'), ex(e, 'up'))) for e in range(E)]).reshape(sd[k].shape)
            elif k.endswith('experts.down_proj'):
                new[k] = torch.stack([ex(e, 'down') for e in range(E)]).reshape(sd[k].shape)
            else:
                new[k] = w['mtp.' + k].reshape(sd[k].shape)
        self.body.load_state_dict(new)
        self.fc.weight.data.copy_(w['mtp.fc.weight'])
        self.pre_fc_norm_embedding.weight.data.copy_(w['mtp.pre_fc_norm_embedding.weight'])
        self.pre_fc_norm_hidden.weight.data.copy_(w['mtp.pre_fc_norm_hidden.weight'])

    def forward(self, h, emb, pos, cache):
        x = self.fc(torch.cat((self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(h)), dim=-1))
        out = self.body(inputs_embeds=x.unsqueeze(0), position_ids=torch.as_tensor(pos).reshape(1, -1), past_key_values=cache, use_cache=True)
        return out.last_hidden_state[0], out.past_key_values


def main():
    cfg = dict(R.TINY)
    hf_cfg = Qwen3_5MoeTextConfig(**cfg)
    hf_cfg._attn_implementation = 'eager'
    model = Qwen3_5MoeForCausalLM(hf_cfg).float().eval()
    w = R.random_weights(cfg, seed=1, prefix='model.')
    sd = model.state_dict()
    model.load_state_dict({k: v.reshape(sd[k].shape) for k, v in w.items()})
    E_mtp = 12                                            # another expert count than the main layers (pruned checkpoints)
    wm = R.random_mtp_weights(cfg, seed=3, experts=E_mtp)
    hf = HfMtp(cfg, E_mtp).float().eval()
    hf.load(wm, cfg)
    ref, mtp = R.Ref(cfg, w), R.Mtp(cfg, wm)
    embed, head = w['model.embed_tokens.weight'], w['lm_head.weight']

    T, steps, worst = 13, 5, 0.0
    tokens = torch.randint(0, cfg['vocab_size'], (T + steps + 1,))
    with torch.no_grad():
        h_hf = model.model(tokens.reshape(1, -1)).last_hidden_state[0]           # post final norm
    _, h_ref = ref.forward(tokens, ref.init_state(), return_hidden=True)
    assert (h_ref - h_hf).abs().max() / h_hf.abs().max() < 2e-5

    def check(a, b, what):
        nonlocal worst
        rel = ((a - b).abs().max() / b.abs().max()).item()
        worst = max(worst, rel)
        print(f'{what}: rel max |diff| {rel:.2e}')
        assert rel < 5e-5, (what, rel)

    # prefill: entries p = 0 .. T-1 pair h[p] with the embedding of token p + 1
    with torch.no_grad():
        o_hf, cache = hf(h_hf[:T], embed[tokens[1:T + 1]], torch.arange(T), None)
    st = mtp.init_state()
    o_ref = mtp.step(h_ref[:T], embed[tokens[1:T + 1]], torch.arange(T), st)
    check(o_ref, o_hf, f'prefill of {T} entries')
    check(o_ref @ head.T, model.lm_head(o_hf).detach(), 'prefill logits')
    # steps with the carried cache; at each a 4-token greedy draft chain on copies of the caches
    all_o, all_d = [o_ref], []
    for i in range(steps):
        p = T + i
        with torch.no_grad():
            o_hf, cache = hf(h_hf[p:p + 1], embed[tokens[p + 1:p + 2]], [p], cache)
        o_ref = mtp.step(h_ref[p:p + 1], embed[tokens[p + 1:p + 2]], [p], st)
        check(o_ref, o_hf, f'step at {p}')
        all_o.append(o_ref)
        c2, s2 = copy.deepcopy(cache), {k: v.clone() for k, v in st.items()}
        ch, cr, drafts = o_hf, o_ref, []
        for j in range(1, 4):
            d_hf, d_ref = int(model.lm_head(ch).argmax(-1)), int((cr @ head.T).argmax(-1))
            assert d_hf == d_ref
            drafts.append(d_ref)
            with torch.no_grad():
                ch, c2 = hf(ch, embed[d_hf].reshape(1, -1), [p + j], c2)
            cr = mtp.step(cr, embed[d_ref].reshape(1, -1), [p + j], s2)
            check(cr, ch, f'  chain step {j} at {p + j}')
        drafts.append(int((cr @ head.T).argmax(-1)))
        all_d.append(drafts)
        assert st['k'].shape[0] == p + 1                                           # the chain did not touch the carried state
    # the batched chains of the acceptance study == the step-by-step chains
    o_all = torch.cat(all_o)
    got = mtp.chains(o_all[T:], torch.arange(T, T + steps), st['k'], st['v'], embed, head, 3)
    assert got.tolist() == all_d, (got.tolist(), all_d)
    print('batched chains == step-by-step chains:', got.tolist())
    print(f'OK: qwen35_moe_ref.Mtp == HF-composed MTP head (worst rel {worst:.2e})')


if __name__ == '__main__':
    main()
