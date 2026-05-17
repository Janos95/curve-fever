# Curve Fever RL Agent

This project explores building an agent that learns to play Curve Fever through reinforcement learning. The goal is to train and evaluate policies that can control a curve in real time, survive longer, and improve through self-play or simulated experience.

## WebGPU demo

The browser demo keeps game logic and rendering in C++/WGSL, compiled to WebAssembly with Emscripten's Dawn WebGPU port. The page shell only forwards keyboard, pointer, and reset input.

Shared simulation structs, WebGPU helpers, and shader sources live in `cpp/curve_engine.hpp` and `cpp/curve_shaders.hpp`. The web demo, native frame demo, and native benchmark are intentionally thin runners over those shared pieces.

Build it locally with:

```bash
./scripts/build_web.sh
```

If the Emscripten SDK is not at `/Users/janosmeny/Projects/emsdk`, set `EMSDK_DIR` before running the script.

## Atari action pretraining

`experiments/train_atari_act_tinygrad.py` ports the reference PyTorch Atari action-chunk model from `../WorldModel` to tinygrad/WebGPU. It defaults to the stratified Atari-HEAD sample split and writes local checkpoints under ignored `runs/`.

```bash
.venv/bin/python experiments/train_atari_act_tinygrad.py
```

Reference PyTorch checkpoint on `../WorldModel/data/atari-act-stratified`: `43.12%` token accuracy, `77.27%` top-3 accuracy. The tinygrad/WebGPU 3-epoch run reached `42.75%` token accuracy and `77.04%` top-3 accuracy.
