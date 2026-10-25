#!/usr/bin/env python3
"""Создаёт yolov8n_npu.onnx — копию yolov8n.onnx с дополнительным выходом
логитов классов (/model.22/Split_output_1, 1x80x8400).

Зачем: штатный выход YOLOv8n — Concat(боксы 0..640, классы 0..1). При
INT8-квантовании на весь тензор берётся один масштаб (~2.5), и вероятности
0..1 округляются в 0. Отдельный выход логитов (до Sigmoid) квантуется со
своим масштабом; Sigmoid считается в float в mergeYoloOutputs().

Требуется: pip3 install --user onnx
Запуск:     python3 tools/make_npu_model.py [src/yolov8n.onnx] [src/yolov8n_npu.onnx]
"""
import sys
import onnx
from onnx import TensorProto

src = sys.argv[1] if len(sys.argv) > 1 else 'src/yolov8n.onnx'
dst = sys.argv[2] if len(sys.argv) > 2 else 'src/yolov8n_npu.onnx'

m = onnx.load(src)
name = '/model.22/Split_output_1'  # логиты классов, 1x80x8400 для 640x640/80 классов

if not any(o.name == name for o in m.graph.output):
    vi = onnx.ValueInfoProto()
    vi.name = name
    vi.type.tensor_type.elem_type = TensorProto.FLOAT
    for d in (1, 80, 8400):
        dim = vi.type.tensor_type.shape.dim.add()
        dim.dim_value = d
    m.graph.output.append(vi)

onnx.checker.check_model(m)
onnx.save(m, dst)
print('saved', dst)
