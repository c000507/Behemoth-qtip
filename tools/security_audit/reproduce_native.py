"""Bounded, local-only reproducers for the 2026-11-09 security audit.

Compile the actual repository parsers; run adversarial inputs in child
processes. Does not open network sockets or access user checkpoint files.
Current vulnerable code produces crashes; this is an audit probe, not a passing
regression suite. --out retains logs and the synthetic test fixtures.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'grammar': r'''#include "behemoth/gbnf.hpp"
#include <iostream>
int main(int argc,char**argv) {
  int n=std::stoi(argv[1]);
  try {auto g=behemoth::grammar::ParseGbnf("root ::= "+std::string(n,'(')+"\"a\""+std::string(n,')')+"\n");
       std::cout<<"accepted "<<g.rules.size()<<" rules\n";}
  catch(const std::exception&e){std::cout<<"rejected: "<<e.what()<<"\n";}
}''',
    'safetensors': r'''#include "safetensors.hpp"
#include <iostream>
int main(int argc,char**argv) {
  try {qtip::SafeTensorsReader r(argv[1]); auto v=r.read("x"); std::cout<<"read "<<v.size()<<" bytes\n";}
  catch(const std::exception&e){std::cout<<"rejected: "<<e.what()<<"\n";}
}''',
    'scheduler': r'''#include "behemoth/scheduler.hpp"
#include <iostream>
int main() {
  behemoth::SchedulerPolicy p; p.max_requests=1;
  behemoth::Scheduler s(64,64,64,p,256);
  std::vector<int> prompt(64,1);
  for(int i=0;i<1000;++i) s.submit(prompt,1,{});
  std::cout<<"batch_cap=1 pending="<<s.pending()<<"\n";
}''',
}


def limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    resource.setrlimit(resource.RLIMIT_CPU, (20,20))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    units={'grammar':['runtime/src/gbnf.cpp'], 'safetensors':['src/safetensors.cpp'],
           'scheduler':['runtime/src/scheduler.cpp','runtime/src/cache.cpp']}
    for name,source in SOURCES.items():
        file=a.out/(name+'.cpp');file.write_text(source)
        flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer'] if name=='safetensors' else []
        subprocess.run(['c++','-std=c++20','-O2','-g',*flags,'-I'+str(ROOT/'runtime/include'),
                        '-I'+str(ROOT/'src'),'-I'+str(ROOT/'third_party'),str(file),
                        *[str(ROOT/u) for u in units[name]],'-o',str(a.out/name)],check=True)

    def tensor(path, offsets, data=b''):
        h=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':offsets}}).encode()
        path.write_bytes(struct.pack('<Q',len(h))+h+data)

    bad=a.out/'bad-offsets';bad.mkdir(exist_ok=True)
    tensor(bad/'model.safetensors',[])
    outside=a.out/'outside.safetensors';tensor(outside,[0,4],b'TEST')
    traversal=a.out/'traversal';traversal.mkdir(exist_ok=True)
    (traversal/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'../outside.safetensors'}}))
    cases={'grammar_control':['grammar','100'],'grammar_deep':['grammar','50000'],
           'bad_offsets':['safetensors',str(bad)],'shard_traversal':['safetensors',str(traversal)],
           'queue_admission':['scheduler']}
    results={}
    for name,(exe,*args) in cases.items():
        r=subprocess.run([str(a.out/exe),*args],capture_output=True,text=True,timeout=25,preexec_fn=limits,
                         env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0:abort_on_error=1'))
        (a.out/(name+'.log')).write_text(r.stdout+r.stderr)
        results[name]={'returncode':r.returncode,'stdout':r.stdout,
                       'stderr_excerpt':'\n'.join(r.stderr.splitlines()[:12])}
    (a.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
