"""Experimental compiled-kernel cache for the pinned NKI 0.7 standalone API.

This uses NKI internals and must be revalidated when the SDK changes. Arrays
are still transferred through host memory on every call. The cache only
avoids recompilation and model reload; it does not make weights device resident.
"""
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
import tempfile
import numpy as np


class KernelCache:
    def __init__(self):
        if version('nki').split('+')[0] != '0.7.0':
            raise RuntimeError('Compiled-kernel cache requires NKI 0.7.0; use cache_kernels=False on other versions')
        self._entries = {}
        self._temp = tempfile.TemporaryDirectory(prefix='behemoth-neuron-cache-')

    def run(self, kernel, *args):
        from nki.framework.compiled import StandaloneKernel, compile_kernel_to_nir
        from nki.compiler.driver import _compile_bir_to_neff, _build_nki_return_value
        instance = kernel._to_subclass(StandaloneKernel)
        bound = dict(instance._bind_args(args, {}))
        signature = tuple((name, ('array', value.shape, value.dtype.str, str(value.dtype)) if isinstance(value,np.ndarray)
                           else ('scalar', type(value).__name__, value)) for name,value in bound.items())
        key = (kernel.func, signature)
        entry = self._entries.get(key)
        if entry is None:
            directory = Path(self._temp.name) / str(len(self._entries))
            directory.mkdir()
            opts = replace(instance._compile_opts(), artifacts_dir=str(directory),
                           output_path=str(directory/'kernel.neff'))
            frontend = instance._frontend_cls(enable_backend_opt=instance._enable_backend_opt)
            bir = compile_kernel_to_nir(instance, inputs=bound, compile_opts=opts,
                                        frontend=frontend, enable_cache=False)
            tensors = {name:np.ascontiguousarray(value) for name,value in bound.items() if isinstance(value,np.ndarray)}
            compiled = _compile_bir_to_neff(bir, opts, tensors)
            entry = (compiled, _build_nki_return_value)
            self._entries[key] = entry
        compiled, build_result = entry
        inputs = {name:np.ascontiguousarray(value) for name,value in bound.items() if isinstance(value,np.ndarray)}
        inputs = compiled.prepare_inputs(inputs)
        outputs = compiled.prepare_outputs()
        compiled.execute(inputs, outputs)
        return build_result(compiled, outputs)
