import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_WEBGPU_PATH = Path(
  "/Users/janosmeny/Projects/dawn-7069/out/Release/src/dawn/native/libwebgpu_dawn_tinygrad.dylib"
)

if DEFAULT_WEBGPU_PATH.exists():
  os.environ.setdefault("WEBGPU_PATH", str(DEFAULT_WEBGPU_PATH))
os.environ.setdefault("WEBGPU_BACKEND", "WGPUBackendType_Metal")

from tinygrad import Tensor, dtypes, nn  # noqa: E402
from tinygrad.device import Device  # noqa: E402
from tinygrad.nn.optim import AdamW  # noqa: E402
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save  # noqa: E402
from tinygrad.runtime.ops_webgpu import WebGPUProgram  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_atari_act_tinygrad import TransformerBlock  # noqa: E402


def webgpu_buf(tensor: Tensor):
  tensor.realize()
  buf = tensor.uop.base.buffer
  buf.ensure_allocated()
  return buf._buf


class CurveActorCritic:
  def __init__(
    self,
    frame_stack: int = 4,
    image_size: int = 84,
    action_vocab: int = 3,
    chunk_len: int = 8,
    d_model: int = 256,
    nhead: int = 8,
    layers: int = 4,
    dropout: float = 0.0,
  ):
    self.chunk_len = chunk_len
    self.action_vocab = action_vocab
    self.d_model = d_model
    self.c1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
    self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    flat = self._cnn(Tensor.zeros(1, frame_stack, image_size, image_size)).shape[-1]
    self.state_proj = nn.Linear(flat, d_model)
    self.state_ln = nn.LayerNorm(d_model)
    self.state_type = Tensor.zeros(1, 1, d_model, requires_grad=True)
    self.query = Tensor.randn(1, chunk_len, d_model, requires_grad=True) * 0.02
    self.pos = Tensor.randn(1, chunk_len + 1, d_model, requires_grad=True) * 0.02
    self.blocks = [TransformerBlock(d_model, nhead, dropout) for _ in range(layers)]
    self.head = nn.Linear(d_model, action_vocab)
    self.value = nn.Linear(d_model, 1)

  def _cnn(self, x: Tensor) -> Tensor:
    x = self.c1(x).relu()
    x = self.c2(x).relu()
    x = self.c3(x).relu()
    return x.flatten(1)

  def encode(self, x: Tensor) -> Tensor:
    batch = x.shape[0]
    state = self.state_ln(self.state_proj(self._cnn(x))).reshape(batch, 1, self.d_model) + self.state_type
    queries = self.query.expand(batch, self.chunk_len, self.d_model)
    tokens = state.cat(queries, dim=1) + self.pos
    for block in self.blocks:
      tokens = block(tokens)
    return tokens[:, 1]

  def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
    token = self.encode(x)
    return self.head(token), self.value(token).reshape(-1)


def move_model(model: CurveActorCritic, device: str) -> None:
  Tensor.realize(*[tensor.to_(device) for tensor in get_state_dict(model).values()])


def load_atari_backbone(model: CurveActorCritic, checkpoint: Path, device: str) -> int:
  loaded = safe_load(str(checkpoint))
  state = get_state_dict(model)
  assigned = []
  loaded_count = 0
  for name, tensor in loaded.items():
    if name not in state or name.startswith("head."):
      continue
    if state[name].shape != tensor.shape:
      continue
    assigned.append(state[name].assign(tensor.to(device)))
    loaded_count += 1
  if assigned:
    Tensor.realize(*assigned)
  return loaded_count


def make_env_wgsl(width: int, height: int) -> bytes:
  cells = width * height
  return f"""
const MAX_PLAYERS: u32 = 2u;
const WORKGROUP_SIZE: u32 = 256u;
const WIDTH: u32 = {width}u;
const HEIGHT: u32 = {height}u;
const CELLS: u32 = {cells}u;
const FRAME_STACK: u32 = 4u;
const SEGMENT_SAMPLES: u32 = 10u;
const RADIUS_CELLS: i32 = 2i;
const DISK_DIAM: u32 = 5u;
const PIXELS_PER_SAMPLE: u32 = DISK_DIAM * DISK_DIAM;
const MAX_FRAGMENTS: u32 = SEGMENT_SAMPLES * PIXELS_PER_SAMPLE;
const INVALID_CELL: u32 = 0xffffffffu;
const SELF_GRACE_FRAMES: u32 = 8u;
const BOT_LOOKAHEAD_STEPS: u32 = 22u;
const BOT_SENSOR_STEPS: u32 = 42u;
const BOT_SENSOR_STEP: f32 = 2.0;
const PI: f32 = 3.141592653589793;
const SPEED: f32 = 0.88;
const TURN_RATE: f32 = 0.135;
const RADIUS: f32 = 1.65;
const SURVIVE_REWARD: f32 = 0.001;

@group(0) @binding(1) var<storage, read_write> players: array<f32>;
@group(0) @binding(2) var<storage, read_write> occupancy: array<u32>;
@group(0) @binding(3) var<storage, read_write> obs: array<f32>;
@group(0) @binding(4) var<storage, read_write> actions: array<i32>;
@group(0) @binding(5) var<storage, read_write> rewards: array<f32>;
@group(0) @binding(6) var<storage, read_write> dones: array<i32>;
@group(0) @binding(7) var<storage, read_write> stats: array<u32>;

var<workgroup> next_pos: array<vec2<f32>, MAX_PLAYERS>;
var<workgroup> next_heading: array<f32, MAX_PLAYERS>;
var<workgroup> chosen_action: array<i32, MAX_PLAYERS>;
var<workgroup> was_alive: array<u32, MAX_PLAYERS>;
var<workgroup> dead: array<atomic<u32>, MAX_PLAYERS>;
var<workgroup> touched_cell: array<u32, MAX_PLAYERS * MAX_FRAGMENTS>;
var<workgroup> touched_time: array<u32, MAX_PLAYERS * MAX_FRAGMENTS>;

fn player_base(env: u32, player: u32) -> u32 {{
  return (env * MAX_PLAYERS + player) * 8u;
}}

fn stat_base(env: u32) -> u32 {{
  return env * 8u;
}}

fn occ_index(env: u32, x: i32, y: i32) -> u32 {{
  return env * CELLS + u32(y) * WIDTH + u32(x);
}}

fn obs_index(env: u32, channel: u32, cell: u32) -> u32 {{
  return (env * FRAME_STACK + channel) * CELLS + cell;
}}

fn in_bounds(x: i32, y: i32) -> bool {{
  return x >= 0i && y >= 0i && x < i32(WIDTH) && y < i32(HEIGHT);
}}

fn direction(heading: f32) -> vec2<f32> {{
  return vec2<f32>(cos(heading), sin(heading));
}}

fn wrap_angle(a0: f32) -> f32 {{
  var a = a0;
  if (a > PI) {{
    a = a - 2.0 * PI;
  }}
  if (a < -PI) {{
    a = a + 2.0 * PI;
  }}
  return a;
}}

fn hash32(x0: u32) -> u32 {{
  var x = x0;
  x = (x ^ (x >> 16u)) * 0x7feb352du;
  x = (x ^ (x >> 15u)) * 0x846ca68bu;
  return x ^ (x >> 16u);
}}

fn random01(seed: u32, stream: u32) -> f32 {{
  let h = hash32(seed ^ (stream * 0x9e3779b9u));
  return f32(h & 0x00ffffffu) * (1.0 / 16777216.0);
}}

fn pos_of(env: u32, player: u32) -> vec2<f32> {{
  let b = player_base(env, player);
  return vec2<f32>(players[b + 0u], players[b + 1u]);
}}

fn heading_of(env: u32, player: u32) -> f32 {{
  return players[player_base(env, player) + 4u];
}}

fn alive_of(env: u32, player: u32) -> u32 {{
  return select(0u, 1u, players[player_base(env, player) + 5u] > 0.5);
}}

fn own_recent(token: u32, player: u32, frame: u32) -> bool {{
  if ((token & 255u) != player + 1u) {{
    return false;
  }}
  let trail_frame = token >> 8u;
  return frame <= trail_frame + SELF_GRACE_FRAMES;
}}

fn blocked_disk(env: u32, pos: vec2<f32>, player: u32, frame: u32) -> bool {{
  let cx = i32(floor(pos.x));
  let cy = i32(floor(pos.y));
  for (var oy = -RADIUS_CELLS; oy <= RADIUS_CELLS; oy = oy + 1i) {{
    for (var ox = -RADIUS_CELLS; ox <= RADIUS_CELLS; ox = ox + 1i) {{
      let r2 = f32(ox * ox + oy * oy);
      if (r2 > RADIUS * RADIUS + 0.25) {{
        continue;
      }}
      let x = cx + ox;
      let y = cy + oy;
      if (!in_bounds(x, y)) {{
        return true;
      }}
      let token = occupancy[occ_index(env, x, y)];
      if (token != 0u && !own_recent(token, player, frame)) {{
        return true;
      }}
    }}
  }}
  return false;
}}

fn ray_clearance(env: u32, pos: vec2<f32>, heading: f32, player: u32, frame: u32) -> f32 {{
  let dir = direction(heading);
  var probe = pos;
  for (var i = 0u; i < BOT_SENSOR_STEPS; i = i + 1u) {{
    probe = probe + dir * BOT_SENSOR_STEP;
    if (blocked_disk(env, probe, player, frame)) {{
      return f32(i);
    }}
  }}
  return f32(BOT_SENSOR_STEPS);
}}

fn rollout_clearance(env: u32, player: u32, action: i32, frame: u32) -> f32 {{
  var pos = pos_of(env, player);
  var heading = heading_of(env, player);
  var survived = 0.0;
  for (var i = 0u; i < BOT_LOOKAHEAD_STEPS; i = i + 1u) {{
    if (i < 8u) {{
      heading = wrap_angle(heading + f32(action) * TURN_RATE);
    }}
    pos = pos + direction(heading) * SPEED;
    if (blocked_disk(env, pos, player, frame)) {{
      return survived;
    }}
    survived = survived + 1.0;
  }}
  return survived;
}}

fn bot_candidate_score(env: u32, player: u32, action: i32, frame: u32) -> f32 {{
  let heading = wrap_angle(heading_of(env, player) + f32(action) * TURN_RATE);
  let pos = pos_of(env, player) + direction(heading) * SPEED;
  if (blocked_disk(env, pos, player, frame)) {{
    return -10000.0;
  }}

  let survival = rollout_clearance(env, player, action, frame);
  let open =
    ray_clearance(env, pos, heading, player, frame) * 1.15 +
    ray_clearance(env, pos, wrap_angle(heading - 0.45), player, frame) * 0.85 +
    ray_clearance(env, pos, wrap_angle(heading + 0.45), player, frame) * 0.85 +
    ray_clearance(env, pos, wrap_angle(heading - 0.90), player, frame) * 0.42 +
    ray_clearance(env, pos, wrap_angle(heading + 0.90), player, frame) * 0.42;

  let edge_margin = min(min(pos.x, f32(WIDTH) - pos.x), min(pos.y, f32(HEIGHT) - pos.y));
  let edge_bonus = min(edge_margin, 24.0) * 0.12;
  let to_human = pos_of(env, 0u) - pos;
  let target_heading = atan2(to_human.y, to_human.x);
  let target_alignment = 1.0 - min(abs(wrap_angle(target_heading - heading)) / PI, 1.0);
  let attack_bonus = target_alignment * 3.0;
  let previous_action = i32(players[player_base(env, player) + 6u]) - 1i;
  let smooth_bonus = select(0.0, 2.0, previous_action == action);
  let turn_cost = select(0.25, 0.0, action == 0i);
  return survival * 7.0 + open + edge_bonus + attack_bonus + smooth_bonus - turn_cost;
}}

fn bot_action(env: u32, player: u32, frame: u32) -> i32 {{
  var best_action = 0i;
  var best_score = -1000000.0;
  for (var i = 0u; i < 3u; i = i + 1u) {{
    let action = i32(i) - 1i;
    let score = bot_candidate_score(env, player, action, frame);
    if (score > best_score) {{
      best_score = score;
      best_action = action;
    }}
  }}
  return best_action;
}}

fn reset_player(env: u32, player: u32, seed: u32) {{
  let arena = vec2<f32>(f32(WIDTH), f32(HEIGHT));
  let center = arena * 0.5;
  let margin = max(10.0, RADIUS * 8.0 + SPEED * 12.0);
  let axis_angle = random01(seed, 11u) * 2.0 * PI - PI;
  let axis = direction(axis_angle);
  let max_half_x = (arena.x * 0.5 - margin) / max(abs(axis.x), 0.001);
  let max_half_y = (arena.y * 0.5 - margin) / max(abs(axis.y), 0.001);
  let max_half = max(12.0, min(max_half_x, max_half_y) - 2.0);
  let half_sep = min(15.0 + random01(seed, 12u) * 10.0, max_half);
  let side = select(-1.0, 1.0, player == 1u);
  let start = center + axis * side * half_sep;
  let jitter = (random01(seed, 13u) * 2.0 - 1.0) * 0.65;
  let toward_center = select(axis_angle, axis_angle + PI, player == 1u);
  let mirrored_jitter = select(jitter, -jitter, player == 1u);
  let heading = wrap_angle(toward_center + mirrored_jitter);
  let b = player_base(env, player);
  players[b + 0u] = start.x;
  players[b + 1u] = start.y;
  players[b + 2u] = start.x;
  players[b + 3u] = start.y;
  players[b + 4u] = heading;
  players[b + 5u] = 1.0;
  players[b + 6u] = 1.0;
  players[b + 7u] = 0.0;
}}

fn fragment_index(player: u32, fragment: u32) -> u32 {{
  return player * MAX_FRAGMENTS + fragment;
}}

fn compute_fragment(env: u32, player: u32, fragment: u32) {{
  let sample_id = fragment / PIXELS_PER_SAMPLE;
  let pixel_id = fragment % PIXELS_PER_SAMPLE;
  let ox = i32(pixel_id % DISK_DIAM) - RADIUS_CELLS;
  let oy = i32(pixel_id / DISK_DIAM) - RADIUS_CELLS;
  let idx = fragment_index(player, fragment);
  touched_cell[idx] = INVALID_CELL;
  touched_time[idx] = sample_id + 1u;
  if (was_alive[player] == 0u) {{
    return;
  }}
  let r2 = f32(ox * ox + oy * oy);
  if (r2 > RADIUS * RADIUS + 0.25) {{
    return;
  }}
  let t = f32(sample_id + 1u) / f32(SEGMENT_SAMPLES);
  let center = pos_of(env, player) + (next_pos[player] - pos_of(env, player)) * t;
  let x = i32(floor(center.x)) + ox;
  let y = i32(floor(center.y)) + oy;
  if (!in_bounds(x, y)) {{
    atomicStore(&dead[player], 1u);
    return;
  }}
  touched_cell[idx] = occ_index(env, x, y);
}}

fn clear_env_cell(env: u32, lane: u32) {{
  for (var i = lane; i < CELLS; i = i + WORKGROUP_SIZE) {{
    occupancy[env * CELLS + i] = 0u;
    for (var c = 0u; c < FRAME_STACK; c = c + 1u) {{
      obs[obs_index(env, c, i)] = 0.0;
    }}
  }}
}}

fn current_pixel(env: u32, cell: u32) -> f32 {{
  let x = cell % WIDTH;
  let y = cell / WIDTH;
  if (x < 2u || y < 2u || x >= WIDTH - 2u || y >= HEIGHT - 2u) {{
    return 0.35;
  }}
  let token = occupancy[env * CELLS + cell];
  if ((token & 255u) == 1u) {{
    return 1.0;
  }}
  if ((token & 255u) == 2u) {{
    return -1.0;
  }}
  return 0.0;
}}

fn mark_player_in_obs(env: u32, player: u32) {{
  let p = pos_of(env, player);
  let cx = i32(floor(p.x));
  let cy = i32(floor(p.y));
  for (var oy = -RADIUS_CELLS; oy <= RADIUS_CELLS; oy = oy + 1i) {{
    for (var ox = -RADIUS_CELLS; ox <= RADIUS_CELLS; ox = ox + 1i) {{
      let r2 = f32(ox * ox + oy * oy);
      if (r2 > RADIUS * RADIUS + 0.25) {{
        continue;
      }}
      let x = cx + ox;
      let y = cy + oy;
      if (in_bounds(x, y)) {{
        let value = select(1.0, -1.0, player == 1u);
        obs[obs_index(env, 3u, u32(y) * WIDTH + u32(x))] = value;
      }}
    }}
  }}
}}

fn update_obs(env: u32, lane: u32) {{
  for (var i = lane; i < CELLS; i = i + WORKGROUP_SIZE) {{
    obs[obs_index(env, 0u, i)] = obs[obs_index(env, 1u, i)];
    obs[obs_index(env, 1u, i)] = obs[obs_index(env, 2u, i)];
    obs[obs_index(env, 2u, i)] = obs[obs_index(env, 3u, i)];
    obs[obs_index(env, 3u, i)] = current_pixel(env, i);
  }}
}}

@compute @workgroup_size(256)
fn step_env(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {{
  let env = wg.x;
  let lane = lid3.x;
  let sb = stat_base(env);
  let prev_frame = stats[sb + 0u];
  let frame = prev_frame + 1u;
  let needs_reset = alive_of(env, 0u) == 0u || alive_of(env, 1u) == 0u || prev_frame == 0u;

  if (lane == 0u) {{
    stats[sb + 0u] = frame;
    rewards[env] = 0.0;
    dones[env] = 0i;
  }}

  if (needs_reset) {{
    clear_env_cell(env, lane);
  }}
  workgroupBarrier();

  if (needs_reset && lane < MAX_PLAYERS) {{
    reset_player(env, lane, hash32(env * 1103515245u + frame * 12345u + stats[sb + 5u] * 2654435761u));
  }}
  workgroupBarrier();

  if (needs_reset && lane == 0u) {{
    stats[sb + 5u] = stats[sb + 5u] + 1u;
  }}
  workgroupBarrier();

  if (lane < MAX_PLAYERS) {{
    let p = lane;
    was_alive[p] = alive_of(env, p);
    atomicStore(&dead[p], 0u);
    var action = 0i;
    if (p == 0u) {{
      action = clamp(actions[env] - 1i, -1i, 1i);
    }} else {{
      action = bot_action(env, p, frame);
    }}
    chosen_action[p] = action;
    let heading = wrap_angle(heading_of(env, p) + f32(action) * TURN_RATE);
    next_heading[p] = heading;
    next_pos[p] = pos_of(env, p) + direction(heading) * SPEED;
  }}
  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {{
    compute_fragment(env, i / MAX_FRAGMENTS, i % MAX_FRAGMENTS);
  }}
  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {{
    let p = i / MAX_FRAGMENTS;
    let c = touched_cell[i];
    if (c == INVALID_CELL || was_alive[p] == 0u) {{
      continue;
    }}
    let old = occupancy[c];
    if (old != 0u && !own_recent(old, p, frame)) {{
      atomicStore(&dead[p], 1u);
    }}
    let my_time = touched_time[i];
    for (var q = 0u; q < MAX_PLAYERS; q = q + 1u) {{
      if (q == p || was_alive[q] == 0u) {{
        continue;
      }}
      for (var other_fragment = 0u; other_fragment < MAX_FRAGMENTS; other_fragment = other_fragment + 1u) {{
        let other_idx = fragment_index(q, other_fragment);
        if (touched_cell[other_idx] == c) {{
          let other_time = touched_time[other_idx];
          if (other_time <= my_time) {{
            atomicStore(&dead[p], 1u);
          }}
        }}
      }}
    }}
  }}
  workgroupBarrier();

  let human_dead = atomicLoad(&dead[0]) != 0u;
  let bot_dead = atomicLoad(&dead[1]) != 0u;
  let terminal = human_dead || bot_dead;

  if (!terminal) {{
    for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {{
      let p = i / MAX_FRAGMENTS;
      let c = touched_cell[i];
      if (c != INVALID_CELL && was_alive[p] != 0u) {{
        occupancy[c] = (frame << 8u) | (p + 1u);
      }}
    }}
  }}
  workgroupBarrier();

  if (lane < MAX_PLAYERS) {{
    let p = lane;
    let b = player_base(env, p);
    if (!terminal) {{
      players[b + 2u] = players[b + 0u];
      players[b + 3u] = players[b + 1u];
      players[b + 0u] = next_pos[p].x;
      players[b + 1u] = next_pos[p].y;
      players[b + 4u] = next_heading[p];
      players[b + 5u] = 1.0;
      players[b + 6u] = f32(chosen_action[p] + 1i);
    }} else {{
      players[b + 5u] = 0.0;
    }}
  }}
  workgroupBarrier();

  if (terminal) {{
    if (lane == 0u) {{
      dones[env] = 1i;
      stats[sb + 1u] = stats[sb + 1u] + 1u;
      if (human_dead && !bot_dead) {{
        rewards[env] = -1.0;
        stats[sb + 3u] = stats[sb + 3u] + 1u;
      }} else if (bot_dead && !human_dead) {{
        rewards[env] = 1.0;
        stats[sb + 2u] = stats[sb + 2u] + 1u;
      }} else {{
        rewards[env] = 0.0;
        stats[sb + 4u] = stats[sb + 4u] + 1u;
      }}
    }}
    clear_env_cell(env, lane);
  }} else if (lane == 0u) {{
    rewards[env] = SURVIVE_REWARD;
  }}
  workgroupBarrier();

  if (terminal && lane < MAX_PLAYERS) {{
    reset_player(env, lane, hash32(env * 747796405u + frame * 2891336453u + stats[sb + 5u] * 277803737u));
  }}
  workgroupBarrier();

  update_obs(env, lane);
  workgroupBarrier();

  if (lane == 0u) {{
    mark_player_in_obs(env, 0u);
    mark_player_in_obs(env, 1u);
  }}
}}
""".encode()


class CurveWebGPUEnv:
  def __init__(self, batch_size: int, width: int, height: int, device: str):
    self.batch_size = batch_size
    self.width = width
    self.height = height
    self.device = device
    self.players = Tensor.zeros(batch_size * 2 * 8, dtype=dtypes.float32, device=device).contiguous().realize()
    self.occupancy = Tensor.zeros(batch_size * width * height, dtype=dtypes.uint32, device=device).contiguous().realize()
    self.obs = Tensor.zeros(batch_size, 4, height, width, dtype=dtypes.float32, device=device).contiguous().realize()
    self.rewards = Tensor.zeros(batch_size, dtype=dtypes.float32, device=device).contiguous().realize()
    self.dones = Tensor.zeros(batch_size, dtype=dtypes.int32, device=device).contiguous().realize()
    self.stats = Tensor.zeros(batch_size * 8, dtype=dtypes.uint32, device=device).contiguous().realize()
    self.zero_actions = Tensor.zeros(batch_size, dtype=dtypes.int32, device=device).contiguous().realize()
    self.program = WebGPUProgram((Device[device].device_res, False), "step_env", make_env_wgsl(width, height))
    self.step(self.zero_actions)

  def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    actions = actions.cast(dtypes.int32).contiguous().realize()
    self.program(
      webgpu_buf(self.players),
      webgpu_buf(self.occupancy),
      webgpu_buf(self.obs),
      webgpu_buf(actions),
      webgpu_buf(self.rewards),
      webgpu_buf(self.dones),
      webgpu_buf(self.stats),
      global_size=(self.batch_size, 1, 1),
      local_size=(256, 1, 1),
    )
    return self.obs, self.rewards, self.dones

  def counters(self) -> dict:
    Device[self.device].synchronize()
    stats = self.stats.numpy().reshape(self.batch_size, 8).astype(np.uint64)
    episodes = int(stats[:, 1].sum())
    wins = int(stats[:, 2].sum())
    losses = int(stats[:, 3].sum())
    draws = int(stats[:, 4].sum())
    total = max(1, wins + losses + draws)
    return {
      "episodes": episodes,
      "wins": wins,
      "losses": losses,
      "draws": draws,
      "win_rate": wins / total,
      "loss_rate": losses / total,
      "draw_rate": draws / total,
      "resets": int(stats[:, 5].sum()),
    }


def sample_actions(logits: Tensor, device: str) -> tuple[Tensor, Tensor, Tensor]:
  log_probs = logits.log_softmax(axis=-1)
  probs = log_probs.exp()
  uniform = Tensor.rand(*logits.shape, device=device).clamp(1e-6, 1.0 - 1e-6)
  gumbel = -(-uniform.log()).log()
  actions = (logits + gumbel).argmax(axis=-1).cast(dtypes.int32).contiguous().realize()
  one_hot = actions.one_hot(logits.shape[-1]).cast(dtypes.float32)
  action_logp = (log_probs * one_hot).sum(axis=-1)
  entropy = -(probs * log_probs).sum(axis=-1)
  return actions, action_logp, entropy


def action_logp_entropy(logits: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
  log_probs = logits.log_softmax(axis=-1)
  probs = log_probs.exp()
  one_hot = actions.one_hot(logits.shape[-1]).cast(dtypes.float32)
  return (log_probs * one_hot).sum(axis=-1), -(probs * log_probs).sum(axis=-1)


def stack_tensors(items: list[Tensor]) -> Tensor:
  return Tensor.cat(*[x.reshape(1, *x.shape) for x in items], dim=0)


def estimated_obs_gib(batch_size: int, steps: int, image_size: int, dtype: str) -> float:
  bytes_per_value = 2 if dtype == "float16" else 4
  values = batch_size * steps * 4 * image_size * image_size
  return values * bytes_per_value / (1024**3)


def maybe_store_obs(obs: Tensor, dtype: str) -> Tensor:
  obs = obs.detach()
  if dtype == "float16":
    return obs.cast(dtypes.float16).contiguous().realize().detach()
  return obs.clone().realize().detach()


def ppo_train(args: argparse.Namespace) -> None:
  if args.rollout_steps <= 0 or args.min_rollout_steps <= 0 or args.max_rollout_steps <= 0:
    raise ValueError("rollout step counts must be positive")
  if args.rollout_check_interval <= 0:
    raise ValueError("--rollout-check-interval must be positive")
  if not 0.0 < args.rollout_completion_frac <= 1.0:
    raise ValueError("--rollout-completion-frac must be in (0, 1]")
  if args.rollout_mode == "until-complete" and args.min_rollout_steps > args.max_rollout_steps:
    raise ValueError("--min-rollout-steps must be <= --max-rollout-steps")

  Tensor.manual_seed(args.seed)
  args.out.mkdir(parents=True, exist_ok=True)
  metrics_path = args.out / "metrics.jsonl"
  model_path = args.out / "model.safetensors"

  model = CurveActorCritic(
    frame_stack=4,
    image_size=args.image_size,
    action_vocab=3,
    chunk_len=8,
    d_model=args.d_model,
    nhead=args.heads,
    layers=args.layers,
    dropout=0.0,
  )
  move_model(model, args.device)
  loaded = 0
  if args.pretrained and args.pretrained.exists():
    loaded = load_atari_backbone(model, args.pretrained, args.device)
  opt = AdamW(get_parameters(model), lr=args.lr, weight_decay=args.weight_decay, fused=args.fused_optim)
  env = CurveWebGPUEnv(args.batch_size, args.image_size, args.image_size, args.device)

  max_rollout_steps = args.rollout_steps if args.rollout_mode == "fixed" else args.max_rollout_steps
  max_obs_gib = estimated_obs_gib(args.batch_size, max_rollout_steps, args.image_size, args.store_obs_dtype)
  if max_obs_gib > args.max_rollout_obs_gib:
    raise ValueError(
      f"max rollout observation storage would be ~{max_obs_gib:.1f} GiB. "
      f"Lower --batch-size/--max-rollout-steps or raise --max-rollout-obs-gib."
    )

  print(
    f"device={args.device} batch={args.batch_size} rollout_mode={args.rollout_mode} "
    f"rollout_steps={args.rollout_steps} min_rollout_steps={args.min_rollout_steps} "
    f"max_rollout_steps={args.max_rollout_steps} completion_frac={args.rollout_completion_frac} "
    f"store_obs_dtype={args.store_obs_dtype} max_obs_gib={max_obs_gib:.2f} "
    f"pretrained_tensors={loaded} target_win_rate={args.target_win_rate}",
    flush=True,
  )
  last_counters = env.counters()
  last_improve_check = time.time()
  best_win_rate = 0.0
  start_time = time.time()
  total_env_steps = 0

  for update in range(1, args.max_updates + 1):
    Tensor.training = False
    obs_buf: list[Tensor] = []
    action_buf: list[Tensor] = []
    logp_buf: list[Tensor] = []
    value_buf: list[Tensor] = []
    reward_buf: list[Tensor] = []
    done_buf: list[Tensor] = []
    done_seen = Tensor.zeros(args.batch_size, dtype=dtypes.int32, device=args.device).contiguous().realize()
    completed_envs = 0
    completed_fraction = 0.0

    rollout_limit = args.rollout_steps if args.rollout_mode == "fixed" else args.max_rollout_steps
    for step_idx in range(rollout_limit):
      obs_t = env.obs.clone().realize()
      logits, value = model(obs_t)
      actions, logp, _ = sample_actions(logits, args.device)
      env.step(actions)
      done_t = env.dones.clone().realize().detach()
      obs_buf.append(maybe_store_obs(obs_t, args.store_obs_dtype))
      action_buf.append(actions.detach())
      logp_buf.append(logp.detach())
      value_buf.append(value.detach())
      reward_buf.append(env.rewards.clone().realize().detach())
      done_buf.append(done_t)

      if args.rollout_mode == "until-complete":
        done_seen = (done_seen + done_t).clip(0, 1).cast(dtypes.int32).contiguous().realize()
        rollout_step = step_idx + 1
        should_check = rollout_step >= args.min_rollout_steps and rollout_step % args.rollout_check_interval == 0
        if should_check:
          completed_envs = int(done_seen.sum().numpy())
          completed_fraction = completed_envs / args.batch_size
          if completed_fraction >= args.rollout_completion_frac:
            break

    with_next = model(env.obs.clone().realize())[1].detach()
    rollout_steps = len(obs_buf)
    total_env_steps += rollout_steps * args.batch_size
    rewards = reward_buf
    dones = done_buf
    values = value_buf
    advantages: list[Tensor] = []
    returns: list[Tensor] = []
    gae = Tensor.zeros(args.batch_size, dtype=dtypes.float32, device=args.device).contiguous().realize()
    next_value = with_next
    for t in reversed(range(rollout_steps)):
      mask = 1.0 - dones[t].cast(dtypes.float32)
      delta = rewards[t] + args.gamma * next_value * mask - values[t]
      gae = delta + args.gamma * args.gae_lambda * mask * gae
      advantages.insert(0, gae.detach())
      returns.insert(0, (gae + values[t]).detach())
      next_value = values[t]

    obs = stack_tensors(obs_buf).reshape(rollout_steps * args.batch_size, 4, args.image_size, args.image_size)
    actions = stack_tensors(action_buf).reshape(rollout_steps * args.batch_size)
    old_logp = stack_tensors(logp_buf).reshape(rollout_steps * args.batch_size)
    adv = stack_tensors(advantages).reshape(rollout_steps * args.batch_size)
    ret = stack_tensors(returns).reshape(rollout_steps * args.batch_size)
    adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).detach()

    Tensor.training = True
    train_size = rollout_steps * args.batch_size
    policy_loss_acc = 0.0
    value_loss_acc = 0.0
    entropy_acc = 0.0
    minibatches = 0
    for _epoch in range(args.ppo_epochs):
      for start in range(0, train_size, args.minibatch_size):
        end = min(start + args.minibatch_size, train_size)
        mb_obs = obs[start:end].cast(dtypes.float32)
        mb_actions = actions[start:end]
        mb_old_logp = old_logp[start:end]
        mb_adv = adv[start:end]
        mb_ret = ret[start:end]

        logits, value = model(mb_obs)
        logp, entropy = action_logp_entropy(logits, mb_actions)
        ratio = (logp - mb_old_logp).exp()
        clipped = ratio.clip(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
        policy_loss = -(ratio * mb_adv).minimum(clipped * mb_adv).mean()
        value_loss = ((value - mb_ret) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        policy_loss_acc += float(policy_loss.numpy())
        value_loss_acc += float(value_loss.numpy())
        entropy_acc += float(entropy_loss.numpy())
        minibatches += 1

    if update % args.log_interval == 0 or update == 1:
      counters = env.counters()
      elapsed = time.time() - start_time
      delta_episodes = counters["episodes"] - last_counters["episodes"]
      delta_wins = counters["wins"] - last_counters["wins"]
      delta_losses = counters["losses"] - last_counters["losses"]
      delta_draws = counters["draws"] - last_counters["draws"]
      delta_total = max(1, delta_wins + delta_losses + delta_draws)
      interval_win_rate = delta_wins / delta_total
      action_counts = np.bincount(stack_tensors(action_buf).numpy().reshape(-1), minlength=3).astype(np.int64)
      best_win_rate = max(best_win_rate, interval_win_rate)
      row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "update": update,
        "env_steps": total_env_steps,
        "rollout_steps": rollout_steps,
        "rollout_completed_envs": completed_envs,
        "rollout_completed_fraction": completed_fraction,
        "episodes": counters["episodes"],
        "wins": counters["wins"],
        "losses": counters["losses"],
        "draws": counters["draws"],
        "win_rate": counters["win_rate"],
        "interval_episodes": delta_episodes,
        "interval_win_rate": interval_win_rate,
        "best_interval_win_rate": best_win_rate,
        "policy_loss": policy_loss_acc / max(1, minibatches),
        "value_loss": value_loss_acc / max(1, minibatches),
        "entropy": entropy_acc / max(1, minibatches),
        "action_counts": action_counts.tolist(),
      }
      print(json.dumps(row), flush=True)
      with metrics_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
      last_counters = counters

      if counters["episodes"] >= args.min_target_episodes and interval_win_rate >= args.target_win_rate:
        print(f"target_reached interval_win_rate={interval_win_rate:.4f}", flush=True)
        break

      if time.time() - last_improve_check >= args.improve_check_seconds:
        if best_win_rate < args.min_reasonable_win_rate:
          print(
            f"debug_trigger no_reasonable_improvement best_interval_win_rate={best_win_rate:.4f} "
            f"threshold={args.min_reasonable_win_rate:.4f}",
            flush=True,
          )
          break
        last_improve_check = time.time()

  state = {name: tensor.detach().to("CPU").realize() for name, tensor in get_state_dict(model).items()}
  safe_save(state, str(model_path))
  print(f"saved={model_path}", flush=True)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, default=Path("runs/curve-rl-tinygrad-webgpu"))
  parser.add_argument("--pretrained", type=Path, default=Path("runs/atari-act-tinygrad-webgpu/model.safetensors"))
  parser.add_argument("--device", default="WEBGPU")
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--image-size", type=int, default=84)
  parser.add_argument("--rollout-mode", choices=("fixed", "until-complete"), default="until-complete")
  parser.add_argument("--rollout-steps", type=int, default=64)
  parser.add_argument("--min-rollout-steps", type=int, default=16)
  parser.add_argument("--max-rollout-steps", type=int, default=128)
  parser.add_argument("--rollout-completion-frac", type=float, default=0.80)
  parser.add_argument("--rollout-check-interval", type=int, default=8)
  parser.add_argument("--store-obs-dtype", choices=("float32", "float16"), default="float32")
  parser.add_argument("--max-rollout-obs-gib", type=float, default=32.0)
  parser.add_argument("--minibatch-size", type=int, default=1024)
  parser.add_argument("--ppo-epochs", type=int, default=1)
  parser.add_argument("--max-updates", type=int, default=100000)
  parser.add_argument("--target-win-rate", type=float, default=0.90)
  parser.add_argument("--min-target-episodes", type=int, default=1000)
  parser.add_argument("--min-reasonable-win-rate", type=float, default=0.02)
  parser.add_argument("--improve-check-seconds", type=float, default=600.0)
  parser.add_argument("--gamma", type=float, default=0.995)
  parser.add_argument("--gae-lambda", type=float, default=0.95)
  parser.add_argument("--clip-ratio", type=float, default=0.2)
  parser.add_argument("--value-coef", type=float, default=0.5)
  parser.add_argument("--entropy-coef", type=float, default=0.08)
  parser.add_argument("--lr", type=float, default=3e-5)
  parser.add_argument("--weight-decay", type=float, default=1e-5)
  parser.add_argument("--d-model", type=int, default=256)
  parser.add_argument("--layers", type=int, default=4)
  parser.add_argument("--heads", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--log-interval", type=int, default=10)
  parser.add_argument("--fused-optim", action="store_true")
  return parser.parse_args()


if __name__ == "__main__":
  ppo_train(parse_args())
