"""Experimental compiled-kernel cache for the pinned NKI 0.7 standalone API.

This uses NKI internals and must be revalidated when the SDK changes. Compiled
kernels are kept with loaded NEFF models, device-resident input/output tensors
and per-call results: array arguments marked resident are uploaded once (the
same numpy object skips re-upload), other arrays are refreshed in place with
write_from_numpy, and only the outputs return through host memory. The cache
does not fuse kernel launches; dispatch count is unchanged.
"""
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
import tempfile
import numpy as np


class _KernelState:
    __slots__ = ('compiled', 'build_result', 'model', 'inputs', 'outputs')

    def __init__(self, compiled, build_result):
        self.compiled = compiled
        self.build_result = build_result
        self.model = None
        self.inputs = {}
        self.outputs = {}


class KernelCache:
    def __init__(self):
        if version('nki').split('+')[0] != '0.7.0':
            raise RuntimeError('Compiled-kernel cache requires NKI 0.7.0; use cache_kernels=False on other versions')
        self._entries = {}
        # Resident weight tensors shared across kernel signatures, keyed by the
        # identity of the (kept-alive) numpy source array. Without this, layers
        # sharing one signature would overwrite each other's device weights.
        self._resident = {}
        self._temp = tempfile.TemporaryDirectory(prefix='behemoth-neuron-cache-')

    def run(self, kernel, *args, resident=()):
        from nki.framework.compiled import StandaloneKernel, compile_kernel_to_nir
        from nki.compiler.driver import _compile_bir_to_neff, _build_nki_return_value
        from nki.runtime import NrtpyModel, NrtpyTensor
        instance = kernel._to_subclass(StandaloneKernel)
        bound = dict(instance._bind_args(args, {}))
        signature = tuple((name, ('array', value.shape, value.dtype.str, str(value.dtype)) if isinstance(value,np.ndarray)
                           else ('scalar', type(value).__name__, value)) for name,value in bound.items())
        key = (kernel.func, signature)
        state = self._entries.get(key)
        if state is None:
            directory = Path(self._temp.name) / str(len(self._entries))
            directory.mkdir()
            opts = replace(instance._compile_opts(), artifacts_dir=str(directory),
                           output_path=str(directory/'kernel.neff'))
            frontend = instance._frontend_cls(enable_backend_opt=instance._enable_backend_opt)
            bir = compile_kernel_to_nir(instance, inputs=bound, compile_opts=opts,
                                        frontend=frontend, enable_cache=False)
            tensors = {name:np.ascontiguousarray(value) for name,value in bound.items() if isinstance(value,np.ndarray)}
            compiled = _compile_bir_to_neff(bir, opts, tensors)
            state = _KernelState(compiled, _build_nki_return_value)
            self._entries[key] = state
        if state.model is None:
            state.model = NrtpyModel.load_from_neff(neff_path=state.compiled.neff_path,
                                                    core_id=0, cc_enabled=False,
                                                    rank_id=0, world_size=1)
            state.outputs = {name: NrtpyTensor.from_numpy(array, name=name, core_id=0)
                             for name, array in state.compiled.prepare_outputs().items()}
        tensors = {}
        for index, (name, value) in enumerate(bound.items()):
            if not isinstance(value, np.ndarray):
                continue
            if index in resident:
                stored = self._resident.get(id(value))
                if stored is not None and stored[0].shape == value.shape and stored[0].dtype == value.dtype:
                    tensors[name] = stored[1]
                    continue
                tensor = NrtpyTensor.from_numpy(np.ascontiguousarray(value), name=name, core_id=0)
                self._resident[id(value)] = (value, tensor)
                tensors[name] = tensor
                continue
            stored = state.inputs.get(name)
            if stored is not None and stored[1].shape == value.shape and stored[1].dtype == value.dtype:
                tensor = stored[0]
                tensor.write_from_numpy(np.ascontiguousarray(value))
                state.inputs[name] = (tensor, value)
            else:
                tensor = NrtpyTensor.from_numpy(np.ascontiguousarray(value), name=name, core_id=0)
                state.inputs[name] = (tensor, value)
            tensors[name] = tensor
        state.model(state.compiled.prepare_inputs(tensors), outputs=state.outputs, save_trace=False)
        return state.build_result(state.compiled, {name: tensor.numpy()
                                                   for name, tensor in state.outputs.items()})
