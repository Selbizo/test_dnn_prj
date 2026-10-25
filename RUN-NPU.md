# NPU inference (TIM-VX / OpenCV dnn)

YOLOv8n on the MIPI camera via the Amlogic NPU, using OpenCV 4.12's TIM-VX backend.

## Run

```bash
./run_npu.sh --npu [options]
```

`run_npu.sh` sets `LD_LIBRARY_PATH` with the patched `libGAL.so` first and runs
the binary as root (needed for `/dev/galcore`). Do not run `facedetection --npu`
directly without it — the system `libGAL.so` (v6.4.6) mismatches the loaded
galcore driver (v6.4.8) and segfaults on context creation.

Key options: `--duration <sec>` (stop after N seconds), `--obj-thresh`, `--nms`.

## What it prints (terminal NPU/runtime info)

OpenCV version, whether TIM-VX is in the build, `/dev/galcore` access,
`galcore enableNN=255`, libGAL status, and available backend/targets
(`TIMVX/NPU`).

## Behavior / caveat

- First forward compiles the TIM-VX graph (~10–18 s one-time cost).
- Steady-state forward ≈ 1170 ms/frame — about the same as the CPU path
  (~1131 ms). TIM-VX offloads only ~151 of 431 layers; the rest run on
  CPU/OpenCL, so there is **no net speedup** vs CPU in this configuration.

## Files

- `src/main.cpp` — camera capture + `--npu` path (TIM-VX backend, INT8).
- `run_npu.sh` — launches with the patched libGAL first, as root.
- Build: `cmake --build build -j4`.
