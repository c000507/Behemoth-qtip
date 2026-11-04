"""Exercise GLM dense/MoE dispatch through the shared CUDA CLI and HTTP server."""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=pathlib.Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS='4')
    probe = subprocess.run([str(args.build / 'behemoth_cuda_tests'), '--probe'], env=env)
    if probe.returncode:
        return probe.returncode
    def run(name, *argv):
        subprocess.run([str(args.build / name), *map(str, argv)], check=True, env=env)
    with tempfile.TemporaryDirectory(prefix='behemoth-cuda-architectures-') as folder:
        root = pathlib.Path(folder)
        request = root / 'tokens.json'
        request.write_text(json.dumps(dict(tokens=list(range(1,8)), logit_positions=list(range(7)))))
        for kind, flag in (('moe','--fixture'), ('dense','--dense-fixture')):
            source = root / kind
            run('behemoth_cuda_glm_fixtures', flag, source)
            quant = root / (kind + '-nvfp4')
            run('behemoth_cuda_glm_reference', 'quantize', '--model', source, '--out', quant)
            for model in (source, quant):
                expected, actual = root / (model.name+'-reference.json'), root / (model.name+'-shared.json')
                run('behemoth_cuda_glm_reference', 'trace', '--model', model, '--tokens', request, '--device', '0', '--out', expected)
                run('behemoth_cuda', '--model', model, '--tokens', request, '--context', '64', '--max-tokens', '0', '--trace', actual)
                assert json.loads(expected.read_text())['logits'] == json.loads(actual.read_text())['logits'], model.name
                print('PASS shared CUDA architecture dispatch:', model.name)
        model = root / 'http-fixture'
        run('behemoth_cuda_glm_fixtures', '--server-fixture', model)
        with socket.socket() as s:
            s.bind(('127.0.0.1',0))
            port = s.getsockname()[1]
        with (root / 'server.log').open('w') as log:
            server = subprocess.Popen([str(args.build / 'behemoth_cuda_server'), '--model', str(model),
                '--context', '64', '--port', str(port), '--name', 'glm-test'], stdout=log, stderr=log, env=env)
            try:
                base = f'http://127.0.0.1:{port}'
                for _ in range(200):
                    if server.poll() is not None:
                        raise RuntimeError((root / 'server.log').read_text())
                    try:
                        with urllib.request.urlopen(base+'/health',timeout=1) as response:
                            assert json.load(response)['healthy']
                        break
                    except OSError:
                        time.sleep(.05)
                else:
                    raise TimeoutError('Shared CUDA server startup')
                def chat():
                    request=urllib.request.Request(base+'/v1/chat/completions',
                        data=json.dumps(dict(model='glm-test',messages=[dict(role='user',content='hi')],
                            max_tokens=3,temperature=0,seed=17)).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=30) as response:
                        return json.load(response)
                first,second=chat(),chat()
                assert first['choices']==second['choices'], 'Shared server leaked GLM state'
                assert isinstance(first['choices'][0]['message']['content'],str)
                print('PASS shared CUDA HTTP GLM dispatch and state reset')
            finally:
                server.terminate()
                server.wait(timeout=5)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
