import json, struct, numpy as np, sys
root="/mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output/"
idx=json.load(open(root+"model.safetensors.index.json"))["weight_map"]
def tensor(name):
    fn=root+idx[name]
    with open(fn,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n)); meta=hdr[name]
        a,b=meta["data_offsets"]; f.seek(8+n+a); buf=f.read(b-a)
    dt={"I16":np.int16,"F16":np.float16,"F32":np.float32,"BF16":np.uint16}[meta["dtype"]]
    return np.frombuffer(buf,dtype=dt).reshape(meta["shape"]), meta
layers=[int(x) for x in sys.argv[1:]] or [0]
for L in layers:
  for proj in ["self_attn.q_proj","self_attn.o_proj","mlp.gate_proj","mlp.down_proj"]:
    name=f"model.layers.{L}.{proj}.codes"
    c,meta=tensor(name)
    cb,mb=tensor(f"model.layers.{L}.{proj}.codebooks")
    u=c.astype(np.uint16).ravel()
    h=np.bincount(u,minlength=65536).astype(np.int64)
    s=np.sort(h)[::-1]; cs=np.cumsum(s)/u.size
    used=(h>0).sum()
    ent=-(h[h>0]/u.size*np.log2(h[h>0]/u.size)).sum()
    print(f"L{L} {proj} codes shape={meta['shape']} dtype={meta['dtype']} book shape={mb['shape']} {mb['dtype']} used={used} entropy={ent:.2f} bits  coverage top1K={cs[1023]:.3f} top2K={cs[2047]:.3f} top4K={cs[4095]:.3f} top8K={cs[8191]:.3f} top16K={cs[16383]:.3f} top32K={cs[32767]:.3f} max_freq={s[0]/u.size:.4f}")
    cbf=cb.astype(np.float32).ravel()
    print(f"   codebook: min={cbf.min():.4f} max={cbf.max():.4f} absmean={np.abs(cbf).mean():.4f} std={cbf.std():.4f}")
