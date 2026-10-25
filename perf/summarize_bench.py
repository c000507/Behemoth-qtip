import json,sys,glob
for fn in sys.argv[1:]:
    d=json.load(open(fn))
    best={}
    for e in d["results"]:
        if e.get("status")!="measured": continue
        best.setdefault(e["projection"],{})[e["mode"]]=(e["median_ms"],e["cross_path_pass"],e["rows"])
    tot=0; r=None
    print(fn)
    for k,v in best.items():
        modes=sorted(v.items(), key=lambda x:x[1][0])
        bm=[m for m in modes if m[0]!="auto"][0]
        tot+=bm[1][0]; r=bm[1][2]
        s=" ".join("%s=%.3f%s"%(m,ms,"" if ok else "!") for m,(ms,ok,_) in modes[:6])
        print("  %-18s best %-16s %.3f ms | %s"%(k,bm[0],bm[1][0],s))
    print("  rows=%d layer total(best) %.3f ms -> x88 = %.1f ms; per-row %.1f ms"%(r,tot,tot*88,tot*88/r))
