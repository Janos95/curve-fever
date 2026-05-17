# Curve Fever RL Agent

This project explores building an agent that learns to play Curve Fever through reinforcement learning. The goal is to train and evaluate policies that can control a curve in real time, survive longer, and improve through self-play or simulated experience.

## WebGPU demo

The browser demo keeps game logic and rendering in C++/WGSL, compiled to WebAssembly with Emscripten's Dawn WebGPU port. The page shell only forwards keyboard, pointer, and reset input.

Build it locally with:

```bash
./scripts/build_web.sh
```

If the Emscripten SDK is not at `/Users/janosmeny/Projects/emsdk`, set `EMSDK_DIR` before running the script.
