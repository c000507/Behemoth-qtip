"""Experimental compiled-kernel cache for the pinned NKI 0.7 standalone API.

This uses NKI internals and must be revalidated when the SDK changes. Compiled
kernels are kept with loaded NEFF models, device-resident input/output tensors
and per-call results: array arguments marked resident are uploaded once (the
same numpy object skips re-upload), other arrays are refreshed in place with
write_from_numpy, and only the outputs return through host memory. With a
cache_dir, NEFF files and manifests persist across processes so later runs skip
neuronx-cc entirely. The cache does not fuse kernel launches.
"""
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import numpy as np


class _Spec:
    __slots__ = ('name', 'shape', 'dtype', 'element_size_bytes')

    def __init__(self, entry):
        self.name = entry['name']
        self.shape = tuple(entry['shape'])
        self.dtype = entry['dtype']
        self.element_size_bytes = entry.get('element_size_bytes')


class _LoadedKernel:
    """CompiledKernel stand-in rebuilt from a persisted manifest + NEFF."""

    def __init__(self, manifest):
        self.neff_path = manifest['neff_path']
        self.input_specs = [_Spec(e) for e in manifest['input_specs']]
        self.output_specs = [_Spec(e) for e in manifest['output_specs']]
        self._aliases = {int(k): v for k, v in manifest['aliases'].items()}
        self._none = tuple(manifest['none_positions'])
        self.is_tuple_return = manifest['is_tuple_return']

    @property
    def none_positions(self):
        return self._none

    @property
    def input_output_aliases(self):
        return dict(self._aliases)

    def prepare_outputs(self):
        from nki.compiler import _dtypes
        result = {}
        for spec in self.output_specs:
            try:
                np_dtype = _dtypes.dtype.from_name(spec.dtype).to_numpy()
            except ValueError:
                if not spec.element_size_bytes:
                    raise ValueError(f"Cannot determine element size for output '{spec.name}'")
                np_dtype = np.dtype(f"V{spec.element_size_bytes}")
            result[spec.name] = np.empty(spec.shape, dtype=np_dtype)
        return result

    def prepare_inputs(self, inputs):
        result = dict(inputs)
        suffix = _must_alias_suffix()
        for input_name in self.input_output_aliases.values():
            if input_name in result:
                result[f"{input_name}{suffix}"] = result.pop(input_name)
        return result


def _must_alias_suffix():
    from nki.compiler.ncc_driver import MUST_ALIAS_SUFFIX
    return MUST_ALIAS_SUFFIX


class _KernelState:
    __slots__ = ('compiled', 'build_result', 'model', 'inputs', 'outputs', 'resident')

    def __init__(self, compiled, build_result):
        self.compiled = compiled
        self.build_result = build_result
        self.model = None
        self.inputs = {}
        self.outputs = {}
        # Resident tensors are per NEFF model: device buffers created for one
        # loaded kernel are never passed to another model's execution.
        self.resident = {}


class KernelCache:
    def __init__(self, cache_dir=None):
        if version('nki').split('+')[0] != '0.7.0':
            raise RuntimeError('Compiled-kernel cache requires NKI 0.7.0; use cache_kernels=False on other versions')
        self._entries = {}
        self._temp = tempfile.TemporaryDirectory(prefix='behemoth-neuron-cache-')
        self._dir = Path(cache_dir).resolve() if cache_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, key):
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:24]
        return self._dir / f'kernel-{digest}'

    def _load_persisted(self, key):
        if not self._dir:
            return None
        directory = self._manifest_path(key)
        manifest_file = directory / 'manifest.json'
        if not manifest_file.exists():
            return None
        try:
            manifest = json.loads(manifest_file.read_text())
            manifest['neff_path'] = str(directory / 'kernel.neff')
            if not Path(manifest['neff_path']).exists():
                return None
            return _KernelState(_LoadedKernel(manifest), None)
        except (OSError, ValueError, KeyError):
            return None

    def _persist(self, key, state):
        if not self._dir:
            return
        compiled = state.compiled
        if isinstance(compiled, _LoadedKernel):
            return
        directory = self._manifest_path(key)
        try:
            directory.mkdir(exist_ok=True)
            shutil.copyfile(compiled.neff_path, directory / 'kernel.neff')
            suffix = _must_alias_suffix()
            aliases = {str(k): v for k, v in compiled.input_output_aliases.items()}

            def spec(spec):
                return {'name': spec.name, 'shape': list(spec.shape), 'dtype': spec.dtype,
                        'element_size_bytes': getattr(spec, 'element_size_bytes', 0) or 0}
            manifest = {'neff_path': 'kernel.neff',
                        'input_specs': [spec(s) for s in compiled.input_specs],
                        'output_specs': [spec(s) for s in compiled.output_specs],
                        'aliases': aliases, 'none_positions': list(compiled.none_positions),
                        'is_tuple_return': compiled.is_tuple_return}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
        except (OSError, AttributeError, ValueError):
            shutil.rmtree(directory, ignore_errors=True)

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
            state = self._load_persisted(key)
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
                self._persist(key, state)
            self._entries[key] = state
        if state.build_result is None:
            state.build_result = _build_nki_return_value
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
                stored = state.resident.get(id(value))
                if stored is not None and stored[0].shape == value.shape and stored[0].dtype == value.dtype:
                    tensors[name] = stored[1]
                    continue
                tensor = NrtpyTensor.from_numpy(np.ascontiguousarray(value), name=name, core_id=0)
                state.resident[id(value)] = (value, tensor)
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
