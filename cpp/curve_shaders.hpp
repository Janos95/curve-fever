#pragma once

namespace curve::shaders {

inline constexpr char kInteractiveCompute[] = R"wgsl(
const WORKGROUP_SIZE: u32 = 256u;
const SEGMENT_SAMPLES: u32 = 18u;
const RADIUS_CELLS: i32 = 2i;
const DISK_DIAM: u32 = 5u;
const PIXELS_PER_SAMPLE: u32 = DISK_DIAM * DISK_DIAM;
const MAX_FRAGMENTS: u32 = SEGMENT_SAMPLES * PIXELS_PER_SAMPLE;
const INVALID_CELL: u32 = 0xffffffffu;
const SELF_GRACE_FRAMES: u32 = 10u;
const PI: f32 = 3.141592653589793;
const BG: u32 = 0xff12100cu;
const WALL: u32 = 0xff5b5045u;
const DEAD_COLOR: u32 = 0xff3d3068u;

struct Params {
  frame: u32,
  width: u32,
  height: u32,
  action: i32,
  reset: u32,
  speed: f32,
  turn_rate: f32,
  radius: f32,
};

struct PlayerState {
  pos: vec2<f32>,
  prev_pos: vec2<f32>,
  heading: f32,
  alive: u32,
  color: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> player: PlayerState;
@group(0) @binding(2) var<storage, read_write> occupancy: array<u32>;
@group(0) @binding(3) var<storage, read_write> image: array<u32>;

var<workgroup> next_pos: vec2<f32>;
var<workgroup> next_heading: f32;
var<workgroup> was_alive: u32;
var<workgroup> dead: atomic<u32>;
var<workgroup> touched_cell: array<u32, MAX_FRAGMENTS>;
var<workgroup> touched_time: array<u32, MAX_FRAGMENTS>;

fn cell(x: u32, y: u32) -> u32 {
  return y * params.width + x;
}

fn flatten_cell(x: i32, y: i32) -> u32 {
  return u32(y) * params.width + u32(x);
}

fn in_bounds(x: i32, y: i32) -> bool {
  return x >= 0i && y >= 0i && x < i32(params.width) && y < i32(params.height);
}

fn own_recent(token: u32) -> bool {
  if ((token & 255u) != 1u) {
    return false;
  }
  let trail_frame = token >> 8u;
  return params.frame <= trail_frame + SELF_GRACE_FRAMES;
}

fn wrap_angle(a0: f32) -> f32 {
  var a = a0;
  if (a > PI) {
    a = a - 2.0 * PI;
  }
  if (a < -PI) {
    a = a + 2.0 * PI;
  }
  return a;
}

fn reset_player() {
  let start = vec2<f32>(f32(params.width) * 0.5, f32(params.height) * 0.5);
  player.pos = start;
  player.prev_pos = start;
  player.heading = -0.35;
  player.alive = 1u;
  player.color = 0xffffdc3cu;
  player.pad = 0u;
}

fn compute_fragment(fragment: u32) {
  let sample_id = fragment / PIXELS_PER_SAMPLE;
  let pixel_id = fragment % PIXELS_PER_SAMPLE;
  let ox = i32(pixel_id % DISK_DIAM) - RADIUS_CELLS;
  let oy = i32(pixel_id / DISK_DIAM) - RADIUS_CELLS;

  touched_cell[fragment] = INVALID_CELL;
  touched_time[fragment] = sample_id + 1u;

  if (was_alive == 0u) {
    return;
  }

  let r2 = f32(ox * ox + oy * oy);
  if (r2 > params.radius * params.radius + 0.25) {
    return;
  }

  let t = f32(sample_id + 1u) / f32(SEGMENT_SAMPLES);
  let center = player.pos + (next_pos - player.pos) * t;
  let x = i32(floor(center.x)) + ox;
  let y = i32(floor(center.y)) + oy;

  if (!in_bounds(x, y)) {
    atomicStore(&dead, 1u);
    return;
  }

  touched_cell[fragment] = flatten_cell(x, y);
}

@compute @workgroup_size(256)
fn step_env(@builtin(local_invocation_id) lid3: vec3<u32>) {
  let lane = lid3.x;

  if (params.reset != 0u) {
    for (var i = lane; i < params.width * params.height; i = i + WORKGROUP_SIZE) {
      let x = i % params.width;
      let y = i / params.width;
      occupancy[i] = 0u;
      if (x < 2u || y < 2u || x >= params.width - 2u || y >= params.height - 2u) {
        image[i] = WALL;
      } else {
        image[i] = BG;
      }
    }
  }

  storageBarrier();
  workgroupBarrier();

  if (lane == 0u) {
    if (params.reset != 0u) {
      reset_player();
    }

    was_alive = player.alive;
    atomicStore(&dead, 0u);

    var heading = player.heading;
    if (params.action < 0i) {
      heading = heading - params.turn_rate;
    } else if (params.action > 0i) {
      heading = heading + params.turn_rate;
    }
    heading = wrap_angle(heading);
    next_heading = heading;
    next_pos = player.pos + vec2<f32>(cos(heading), sin(heading)) * params.speed;
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    compute_fragment(i);
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    let c = touched_cell[i];
    if (c == INVALID_CELL || was_alive == 0u) {
      continue;
    }
    let old = occupancy[c];
    if (old != 0u && !own_recent(old)) {
      atomicStore(&dead, 1u);
    }
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    let c = touched_cell[i];
    if (c != INVALID_CELL && was_alive != 0u && atomicLoad(&dead) == 0u) {
      occupancy[c] = (params.frame << 8u) | 1u;
      image[c] = player.color;
    }
  }

  workgroupBarrier();

  if (lane == 0u && was_alive != 0u) {
    if (atomicLoad(&dead) == 0u) {
      player.prev_pos = player.pos;
      player.pos = next_pos;
      player.heading = next_heading;
      player.alive = 1u;
    } else {
      player.alive = 0u;
      let x = clamp(i32(player.pos.x), 0i, i32(params.width) - 1i);
      let y = clamp(i32(player.pos.y), 0i, i32(params.height) - 1i);
      image[flatten_cell(x, y)] = DEAD_COLOR;
    }
  }
}
)wgsl";

inline constexpr char kInteractiveRender[] = R"wgsl(
struct Uniforms {
  frame: u32,
  width: u32,
  height: u32,
  action: i32,
  reset: u32,
  speed: f32,
  turn_rate: f32,
  radius: f32,
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> image: array<u32>;

struct VertexOut {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out: VertexOut;
  out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
  return out;
}

fn unpack_rgba(c: u32) -> vec4<f32> {
  let r = f32(c & 255u) / 255.0;
  let g = f32((c >> 8u) & 255u) / 255.0;
  let b = f32((c >> 16u) & 255u) / 255.0;
  let a = f32((c >> 24u) & 255u) / 255.0;
  return vec4<f32>(r, g, b, a);
}

@fragment
fn fs_main(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
  let x = min(u32(coord.x), params.width - 1u);
  let y = min(u32(coord.y), params.height - 1u);
  return unpack_rgba(image[y * params.width + x]);
}
)wgsl";

inline constexpr char kNativeDemoCompute[] = R"wgsl(
const MAX_PLAYERS: u32 = 2u;
const WORKGROUP_SIZE: u32 = 256u;
const SEGMENT_SAMPLES: u32 = 14u;
const RADIUS_CELLS: i32 = 2i;
const DISK_DIAM: u32 = 5u;
const PIXELS_PER_SAMPLE: u32 = DISK_DIAM * DISK_DIAM;
const MAX_FRAGMENTS: u32 = SEGMENT_SAMPLES * PIXELS_PER_SAMPLE;
const INVALID_CELL: u32 = 0xffffffffu;
const SELF_GRACE_FRAMES: u32 = 8u;
const PI: f32 = 3.141592653589793;

struct Params {
  frame: u32,
  width: u32,
  height: u32,
  players: u32,
  speed: f32,
  turn_rate: f32,
  radius: f32,
  pad: f32,
};

struct PlayerState {
  pos: vec2<f32>,
  prev_pos: vec2<f32>,
  heading: f32,
  alive: u32,
  color: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> players: array<PlayerState>;
@group(0) @binding(2) var<storage, read> actions: array<u32>;
@group(0) @binding(3) var<storage, read_write> occupancy: array<u32>;
@group(0) @binding(4) var<storage, read_write> image: array<u32>;

var<workgroup> next_pos: array<vec2<f32>, MAX_PLAYERS>;
var<workgroup> next_heading: array<f32, MAX_PLAYERS>;
var<workgroup> was_alive: array<u32, MAX_PLAYERS>;
var<workgroup> dead: array<atomic<u32>, MAX_PLAYERS>;
var<workgroup> touched_cell: array<u32, MAX_PLAYERS * MAX_FRAGMENTS>;
var<workgroup> touched_time: array<u32, MAX_PLAYERS * MAX_FRAGMENTS>;

fn flatten_cell(x: i32, y: i32) -> u32 {
  return u32(y) * params.width + u32(x);
}

fn in_bounds(x: i32, y: i32) -> bool {
  return x >= 0i && y >= 0i && x < i32(params.width) && y < i32(params.height);
}

fn own_recent(token: u32, player: u32) -> bool {
  if ((token & 255u) != player + 1u) {
    return false;
  }
  let trail_frame = token >> 8u;
  return params.frame <= trail_frame + SELF_GRACE_FRAMES;
}

fn wrap_angle(a: f32) -> f32 {
  if (a > PI) {
    return a - 2.0 * PI;
  }
  if (a < -PI) {
    return a + 2.0 * PI;
  }
  return a;
}

fn fragment_index(player: u32, fragment: u32) -> u32 {
  return player * MAX_FRAGMENTS + fragment;
}

fn compute_fragment(player: u32, fragment: u32) {
  let sample_id = fragment / PIXELS_PER_SAMPLE;
  let pixel_id = fragment % PIXELS_PER_SAMPLE;
  let ox = i32(pixel_id % DISK_DIAM) - RADIUS_CELLS;
  let oy = i32(pixel_id / DISK_DIAM) - RADIUS_CELLS;
  let idx = fragment_index(player, fragment);

  touched_cell[idx] = INVALID_CELL;
  touched_time[idx] = sample_id + 1u;

  if (was_alive[player] == 0u) {
    return;
  }

  let r2 = f32(ox * ox + oy * oy);
  if (r2 > params.radius * params.radius + 0.25) {
    return;
  }

  let t = f32(sample_id + 1u) / f32(SEGMENT_SAMPLES);
  let p0 = players[player].pos;
  let p1 = next_pos[player];
  let center = p0 + (p1 - p0) * t;
  let x = i32(floor(center.x)) + ox;
  let y = i32(floor(center.y)) + oy;

  if (!in_bounds(x, y)) {
    atomicStore(&dead[player], 1u);
    return;
  }

  touched_cell[idx] = flatten_cell(x, y);
}

@compute @workgroup_size(256)
fn step_env(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let lane = lid3.x;

  if (lane < MAX_PLAYERS) {
    let p = lane;
    was_alive[p] = players[p].alive;
    atomicStore(&dead[p], 0u);

    var heading = players[p].heading;
    let action = actions[p];
    if (action == 0u) {
      heading = heading - params.turn_rate;
    } else if (action == 2u) {
      heading = heading + params.turn_rate;
    }
    heading = wrap_angle(heading);
    next_heading[p] = heading;
    next_pos[p] = players[p].pos + vec2<f32>(cos(heading), sin(heading)) * params.speed;
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    compute_fragment(i / MAX_FRAGMENTS, i % MAX_FRAGMENTS);
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    let p = i / MAX_FRAGMENTS;
    let f = i % MAX_FRAGMENTS;
    let cell = touched_cell[i];

    if (cell == INVALID_CELL || was_alive[p] == 0u) {
      continue;
    }

    let old = occupancy[cell];
    if (old != 0u && !own_recent(old, p)) {
      atomicStore(&dead[p], 1u);
    }

    let my_time = touched_time[i];
    for (var q = 0u; q < MAX_PLAYERS; q = q + 1u) {
      if (q == p || was_alive[q] == 0u) {
        continue;
      }
      for (var other_fragment = 0u; other_fragment < MAX_FRAGMENTS; other_fragment = other_fragment + 1u) {
        let other_idx = fragment_index(q, other_fragment);
        if (touched_cell[other_idx] == cell) {
          let other_time = touched_time[other_idx];
          if (other_time < my_time || other_time == my_time) {
            atomicStore(&dead[p], 1u);
          }
        }
      }
    }
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    let p = i / MAX_FRAGMENTS;
    let cell = touched_cell[i];
    if (cell != INVALID_CELL && was_alive[p] != 0u && atomicLoad(&dead[p]) == 0u) {
      occupancy[cell] = (params.frame << 8u) | (p + 1u);
      image[cell] = players[p].color;
    }
  }

  workgroupBarrier();

  if (lane < MAX_PLAYERS && was_alive[lane] != 0u) {
    if (atomicLoad(&dead[lane]) == 0u) {
      players[lane].prev_pos = players[lane].pos;
      players[lane].pos = next_pos[lane];
      players[lane].heading = next_heading[lane];
    } else {
      players[lane].alive = 0u;
    }
  }
}
)wgsl";

inline constexpr char kBatchBenchmarkCompute[] = R"wgsl(
const MAX_PLAYERS: u32 = 2u;
const WORKGROUP_SIZE: u32 = 256u;
const SEGMENT_SAMPLES: u32 = 14u;
const RADIUS_CELLS: i32 = 2i;
const DISK_DIAM: u32 = 5u;
const PIXELS_PER_SAMPLE: u32 = DISK_DIAM * DISK_DIAM;
const MAX_FRAGMENTS: u32 = SEGMENT_SAMPLES * PIXELS_PER_SAMPLE;
const INVALID_CELL: u32 = 0xffffffffu;
const SELF_GRACE_FRAMES: u32 = 8u;
const PI: f32 = 3.141592653589793;

struct Params {
  frame: u32,
  width: u32,
  height: u32,
  players: u32,
  envs: u32,
  pad0: u32,
  speed: f32,
  turn_rate: f32,
  radius: f32,
  pad1: f32,
};

struct PlayerState {
  pos: vec2<f32>,
  prev_pos: vec2<f32>,
  heading: f32,
  alive: u32,
  color: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> players: array<PlayerState>;
@group(0) @binding(2) var<storage, read_write> occupancy: array<u32>;
@group(0) @binding(3) var<storage, read_write> image: array<u32>;

var<workgroup> next_pos: array<vec2<f32>, MAX_PLAYERS>;
var<workgroup> next_heading: array<f32, MAX_PLAYERS>;
var<workgroup> was_alive: array<u32, MAX_PLAYERS>;
var<workgroup> dead: array<atomic<u32>, MAX_PLAYERS>;
var<workgroup> touched_cell: array<u32, MAX_PLAYERS * MAX_FRAGMENTS>;
var<workgroup> touched_time: array<u32, MAX_PLAYERS * MAX_FRAGMENTS>;

fn cells_per_env() -> u32 {
  return params.width * params.height;
}

fn player_index(env: u32, player: u32) -> u32 {
  return env * MAX_PLAYERS + player;
}

fn flatten_cell(env: u32, x: i32, y: i32) -> u32 {
  return env * cells_per_env() + u32(y) * params.width + u32(x);
}

fn in_bounds(x: i32, y: i32) -> bool {
  return x >= 0i && y >= 0i && x < i32(params.width) && y < i32(params.height);
}

fn own_recent(token: u32, player: u32) -> bool {
  if ((token & 255u) != player + 1u) {
    return false;
  }
  let trail_frame = token >> 8u;
  return params.frame <= trail_frame + SELF_GRACE_FRAMES;
}

fn wrap_angle(a: f32) -> f32 {
  if (a > PI) {
    return a - 2.0 * PI;
  }
  if (a < -PI) {
    return a + 2.0 * PI;
  }
  return a;
}

fn hash_u32(x0: u32) -> u32 {
  var x = x0;
  x = (x ^ 61u) ^ (x >> 16u);
  x = x * 9u;
  x = x ^ (x >> 4u);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return x;
}

fn action_for(env: u32, player: u32) -> u32 {
  return hash_u32(params.frame * 747796405u + env * 2891336453u + player * 277803737u) % 3u;
}

fn reset_player(env: u32, player: u32, frame: u32) -> PlayerState {
  let h = hash_u32(env * 1664525u + player * 1013904223u + frame * 22695477u);
  let x = 24.0 + f32(h & 63u);
  let y = 24.0 + f32((h >> 8u) & 63u);
  let heading = f32((h >> 16u) & 1023u) / 1023.0 * 2.0 * PI - PI;
  let color = select(0xffffdc3cu, 0xff785cffu, player == 1u);
  return PlayerState(vec2<f32>(x, y), vec2<f32>(x, y), heading, 1u, color, 0u);
}

fn fragment_index(player: u32, fragment: u32) -> u32 {
  return player * MAX_FRAGMENTS + fragment;
}

fn compute_fragment(env: u32, player: u32, fragment: u32) {
  let sample_id = fragment / PIXELS_PER_SAMPLE;
  let pixel_id = fragment % PIXELS_PER_SAMPLE;
  let ox = i32(pixel_id % DISK_DIAM) - RADIUS_CELLS;
  let oy = i32(pixel_id / DISK_DIAM) - RADIUS_CELLS;
  let idx = fragment_index(player, fragment);

  touched_cell[idx] = INVALID_CELL;
  touched_time[idx] = sample_id + 1u;

  if (was_alive[player] == 0u) {
    return;
  }

  let r2 = f32(ox * ox + oy * oy);
  if (r2 > params.radius * params.radius + 0.25) {
    return;
  }

  let t = f32(sample_id + 1u) / f32(SEGMENT_SAMPLES);
  let p0 = players[player_index(env, player)].pos;
  let p1 = next_pos[player];
  let center = p0 + (p1 - p0) * t;
  let x = i32(floor(center.x)) + ox;
  let y = i32(floor(center.y)) + oy;

  if (!in_bounds(x, y)) {
    atomicStore(&dead[player], 1u);
    return;
  }

  touched_cell[idx] = flatten_cell(env, x, y);
}

@compute @workgroup_size(256)
fn step_env(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
  let env = wg.x;
  let lane = lid3.x;

  if (lane < MAX_PLAYERS) {
    let p = lane;
    let idx = player_index(env, p);
    was_alive[p] = 1u;
    if (players[idx].alive == 0u) {
      players[idx] = reset_player(env, p, params.frame);
    }
    atomicStore(&dead[p], 0u);

    var heading = players[idx].heading;
    let action = action_for(env, p);
    if (action == 0u) {
      heading = heading - params.turn_rate;
    } else if (action == 2u) {
      heading = heading + params.turn_rate;
    }
    heading = wrap_angle(heading);
    next_heading[p] = heading;
    next_pos[p] = players[idx].pos + vec2<f32>(cos(heading), sin(heading)) * params.speed;
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    compute_fragment(env, i / MAX_FRAGMENTS, i % MAX_FRAGMENTS);
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    let p = i / MAX_FRAGMENTS;
    let cell = touched_cell[i];

    if (cell == INVALID_CELL) {
      continue;
    }

    let old = occupancy[cell];
    if (old != 0u && !own_recent(old, p)) {
      atomicStore(&dead[p], 1u);
    }

    let my_time = touched_time[i];
    for (var q = 0u; q < MAX_PLAYERS; q = q + 1u) {
      if (q == p) {
        continue;
      }
      for (var other_fragment = 0u; other_fragment < MAX_FRAGMENTS; other_fragment = other_fragment + 1u) {
        let other_idx = fragment_index(q, other_fragment);
        if (touched_cell[other_idx] == cell) {
          let other_time = touched_time[other_idx];
          if (other_time <= my_time) {
            atomicStore(&dead[p], 1u);
          }
        }
      }
    }
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    let p = i / MAX_FRAGMENTS;
    let cell = touched_cell[i];
    if (cell != INVALID_CELL && atomicLoad(&dead[p]) == 0u) {
      occupancy[cell] = (params.frame << 8u) | (p + 1u);
      image[cell] = players[player_index(env, p)].color;
    }
  }

  workgroupBarrier();

  if (lane < MAX_PLAYERS) {
    let p = lane;
    let idx = player_index(env, p);
    if (atomicLoad(&dead[p]) == 0u) {
      players[idx].prev_pos = players[idx].pos;
      players[idx].pos = next_pos[p];
      players[idx].heading = next_heading[p];
      players[idx].alive = 1u;
    } else {
      players[idx] = reset_player(env, p, params.frame);
    }
  }
}
)wgsl";

}  // namespace curve::shaders
