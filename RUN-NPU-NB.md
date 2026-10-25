# Полный NPU-инференс (`--npu-nb`) — свой runtime без float-конверсий

Режим, в котором **вся нейросеть** исполняется на NPU, а вход и выход остаются
в `UINT8` прямо в памяти тензоров. Конверсий `float → uint8` и `uint8 → float`
на весь кадр больше нет: де-квантование делается только для тех значений,
которые реально участвуют в постобработке.

## Запуск

```bash
cd /home/pi/opencv_projects/test_dnn_prj
./run_npu.sh --npu-nb --view-width 960
```

Без окна (для робота):

```bash
./run_npu.sh --npu-nb --no-gui --capture-fps 30 --fast-resize
```

Полезные ключи:

- `--capture-fps 30` — ограничить захват. При 110–120 fps поток захвата
  занимает память и замедляет NPU; 30 fps даёт ~+20 % к скорости инференса.
- `--fast-resize` — `INTER_NEAREST` вместо `INTER_LINEAR`: ещё ~5 мс, но
  боксы немного смещаются (детекции те же).

## MobileNet-SSD 300 (`--ssd`) — самый быстрый вариант

```bash
./run_npu.sh --ssd --no-gui --capture-fps 60 --fast-resize
```

Это полноценный NPU-путь на другой модели: MobileNet-SSD v1, вход 300×300.
Граф тот же парсер/TIM-VX, но постобработка — SSD (якоря + NMS).

| Параметр | Значение |
|---|---|
| Модель | `tools/npu/mobilenet_ssd_uint8.nb` |
| Вход | 300×300×3 UINT8, HWC RGB |
| Выходы | `4×1917` (боксы) и `91×1917` (классы), UINT8 |
| Граф на NPU | ~14 мс |
| Полный кадр (noGUI) | **17–20 мс** |
| Частота (noGUI, `--capture-fps 60 --fast-resize`) | **~43 fps** |
| Частота (noGUI, LINEAR) | ~39 fps |

Проверка на `src/image.jpg`: dog 0.925, car 0.684, bicycle 0.595.
Классы — COCO SSD (91, с пропусками), файл `tools/npu/ssd_labels.txt`;
якоря — `tools/npu/box_priors.txt`.

Минус: SSD MobileNet v1 заметно слабее YOLOv8n по качеству и мелким объектам.
Если важна точность — остаётся `--npu-nb` (YOLOv8n), если скорость — `--ssd`.

## Как устроено

```text
MIPI 1920x1080 → поток захвата (~110–120 fps)
                       │ последний кадр
                       ▼
        resize YUYV сразу до 640x640 → BGR → UINT8 planar (CHW)  ← без float
                       │
                       ▼
        NBG-парсер → граф TIM-VX → VIPNano-Q (все слои)
                       │ 3 выхода UINT8: 80x80x144, 40x40x144, 20x20x144
                       ▼
        CPU: dequantize по месту + sigmoid + DFL + NMS → боксы → окно
```

- `.nb` читается парсером NBG (`tools/npu/nbg_parser_patched.c`).
- Граф строится через TIM-VX (`tim::vx::ops::NBG`) и компилируется за ~10 мс.
- Вход модели: `640x640x3 UINT8`, planar BGR, `scale=1/255`, `zp=0`.
- Выходы: `UINT8`, affine `scale/zp` берутся из `.nb` (0.2306/150, 0.2793/189,
  0.3217/203).
- `float`-тензоры на весь кадр не создаются вообще.

Вендорная обёртка `libnn_yolov8n_uint8.so` не используется и удалена из проекта.

## Проверка корректности

Один и тот же кадр `src/image.jpg`:

| Путь | Результат |
|---|---|
| CPU OpenCV | bicycle 0.80, dog 0.83, truck 0.58 |
| OpenCV TIM-VX (151/431 слоёв) | bicycle 0.84, dog 0.84, truck 0.50 |
| Старая обёртка `.nb` | bicycle 0.833202, dog 0.873271, truck 0.724141 |
| **Свой runtime (TIM-VX)** | bicycle 0.833202, dog 0.873271, truck 0.724141 |

Совпадение с обёрткой — до шестого знака, боксы идентичны. На пустой/размытой
сцене камеры — 0 объектов.

## Производительность

| Этап | Время |
|---|---|
| Компиляция графа (один раз) | ~10 мс |
| Исполнение графа на NPU | ~24 мс |
| Полный кадр, отдельный тест (LINEAR) | **~38 мс** |
| Полный кадр, отдельный тест (`--fast-resize`) | **~34 мс** |
| В приложении: захват без лимита, LINEAR | ~61 мс (14 fps) |
| В приложении: `--capture-fps 30` (быстрый путь YUYV) | **~39 мс (19 fps)** |
| В приложении: `--capture-fps 60 --fast-resize` | ~45 мс (19 fps) |

Для сравнения: путь через вендорную обёртку давал ~190 мс (≈5 fps), а первый
вариант своего runtime — ~57 мс. Итог: **20 fps против 5 fps**, без единой
float-конверсии на кадр.

Дальнейшее ускорение — только уменьшением входа модели: см.
`tools/npu/CONVERT-416.md` (ожидаемо ~18–22 мс, 45–55 fps для 416).

Поток захвата при этом держит 110–120 fps — изображение не замерзает,
обновляются только результаты детекции.

## Сборка

Нужны:

- `tools/npu/timvx/lib/libtim-vx.a` — статическая библиотека TIM-VX (ARM);
- `tools/npu/timvx/include` — заголовки TIM-VX;
- `tools/npu/nbg_parser_patched.c` — парсер NBG (поддерживает формат 1.0.15);
- `/opt/tim-vx/lib` — OpenVX/GAL runtime (уже на плате).

```bash
cd /home/pi/opencv_projects/test_dnn_prj
gcc -O2 -c tools/npu/nbg_parser_patched.c \
    -Itools/npu/timvx/include -o build/nbg_parser.o

cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/usr/local/opencv-4.12.0/lib/cmake/opencv4 \
  -DCMAKE_CXX_FLAGS="-pthread -I$PWD/../tools/npu/timvx/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-pthread -Wl,-rpath,/opt/tim-vx/lib" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="$PWD/nbg_parser.o $PWD/../tools/npu/timvx/lib/libtim-vx.a -L/opt/tim-vx/lib -lOpenVX -lGAL -lCLC -lArchModelSw -lNNArchPerf -ldl -pthread" \
  ..
make -j4
```

`-Wl,-rpath,/opt/tim-vx/lib` нужен, чтобы бинарник запускался и в обычном
CPU-режиме без `LD_LIBRARY_PATH`.

## Файлы

```text
tools/npu/yolov8n_uint8.nb              модель Acuity (все слои на NPU)
tools/npu/nbg_parser_patched.c          парсер NBG (формат 1.0.15)
tools/npu/timvx/lib/libtim-vx.a         TIM-VX для ARM
tools/npu/timvx/include/tim/...         заголовки TIM-VX
```

## Ограничения

- `.nb` должен быть собран Acuity под этот NPU (VIM3 = A311D). Для другой
  сцены/калибровки модель пересобирается на x86-ПК.
- Парсер NBG из TIM-VX официально поддерживает формат до 1.0.14; здесь
  включён формат 1.0.15 (`.nb` от DDK 6.4.8.7), проверено на этой модели.
- Пред-/постобработка (resize, sigmoid, DFL, NMS) остаётся на CPU — это не
  слои сети.
- Если нужно ещё быстрее: собрать `.nb` с входом 416 или 320 на x86-ПК —
  инструкция в `tools/npu/CONVERT-416.md`. Runtime подхватит новый размер
  автоматически.
