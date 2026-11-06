"""Rewrite F8E4M3FN -> F8E4M3 (IEEE-style E4M3, supported on Trn1/Trn2) in a serialized HloModuleProto.

neuronx-cc rejects F8E4M3FN on TRN1/TRN2 (NCC_EVRF051) but accepts F8E4M3. PyTorch only has
float8_e4m3fn, so graphs traced with FN parameters are retargeted here. Encodings are identical
for exponents below 15; weights must be scaled so no code reaches 240 (see to_fp8_rows).

usage: tools_patch_fp8_hlo.py in.hlo out.hlo
"""
import sys

from libneuronxla.proto import hlo_pb2



def primitive_types():
    # PrimitiveType lives in xla_data_pb2 next to hlo_pb2
    import importlib
    mod = importlib.import_module(hlo_pb2.__name__.rsplit('.', 1)[0] + '.xla_data_pb2')
    return mod.PrimitiveType


def patch(message, src, dst, counter):
    for field in message.DESCRIPTOR.fields:
        if field.name == 'element_type' and getattr(message, 'element_type') == src:
            message.element_type = dst
            counter[0] += 1
        elif field.type == field.TYPE_MESSAGE:
            value = getattr(message, field.name)
            if field.label == field.LABEL_REPEATED:
                for item in value:
                    if hasattr(item, 'DESCRIPTOR'):
                        patch(item, src, dst, counter)
            elif message.HasField(field.name):
                patch(value, src, dst, counter)


def main():
    pt = primitive_types()
    module = hlo_pb2.HloModuleProto()
    with open(sys.argv[1], 'rb') as f:
        module.ParseFromString(f.read())
    counter = [0]
    patch(module, pt.F8E4M3FN, pt.F8E4M3, counter)
    with open(sys.argv[2], 'wb') as f:
        f.write(module.SerializeToString())
    print(f'retargeted {counter[0]} F8E4M3FN element types to F8E4M3')


if __name__ == '__main__':
    main()
