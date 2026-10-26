import sys,re,statistics as st,datetime as dt
tag=sys.argv[1]; L=open(f'/home/charles/od_exp/{tag}.log').read().splitlines()
def ts(s): return dt.datetime.strptime(s[:19],'%Y-%m-%d_%H:%M:%S')
t0=ts([l for l in L if 'START' in l][-1].split()[0])
ends=[l for l in L if ' END ' in l or 'CRASH' in l]
t1=ts(ends[0].split()[0]) if ends else dt.datetime.now()
tps=[float(m.group(1)) for l in L for m in [re.search(r'batch \d+ rc=0 tok/s=([\d.]+)',l)] if m]
c1=[l.split('=',2)[-1] for l in L if ' c1' in l]
rows=[]
for line in open('/tmp/gpu_poll2.csv'):
    p=line.strip().split(',')
    if p[0]=='ts' or len(p)<17: continue
    try: t=dt.datetime.strptime(p[0][:19],'%Y-%m-%d_%H:%M:%S')
    except: continue
    if t0<=t<=t1:
        try: rows.append([float(x) for x in p[9:17]])
        except: pass
ld=[r for r in rows if r[0]>60e6 or r[1]>60e6]
def s(i,sc): v=[r[i]/sc for r in ld]; return f'{st.mean(v):.0f}/{max(v):.0f}' if v else 'NA'
print(f'{tag}: dur={(t1-t0).total_seconds()/60:.1f}min crashed={"CRASH" in "".join(L)} batches={len(tps)} c4tok/s median={st.median(tps[1:] or tps or [0]):.2f} c1={c1}')
print(f'  loaded samples={len(ld)} P0 avg/pk={s(0,1e6)}W P1={s(1,1e6)}W f0={s(2,1e6)}MHz f1={s(3,1e6)}MHz T0={s(4,1e3)}C T1={s(5,1e3)}C V0={s(6,1)}mV V1={s(7,1)}mV')
