#pragma once

namespace curve::shaders {

inline constexpr char kInteractiveCompute[] = R"wgsl(
const MAX_PLAYERS: u32 = 2u;
const WORKGROUP_SIZE: u32 = 256u;
const SEGMENT_SAMPLES: u32 = 18u;
const RADIUS_CELLS: i32 = 2i;
const DISK_DIAM: u32 = 5u;
const PIXELS_PER_SAMPLE: u32 = DISK_DIAM * DISK_DIAM;
const MAX_FRAGMENTS: u32 = SEGMENT_SAMPLES * PIXELS_PER_SAMPLE;
const INVALID_CELL: u32 = 0xffffffffu;
const SELF_GRACE_FRAMES: u32 = 10u;
const BOT_LOOKAHEAD_STEPS: u32 = 34u;
const BOT_SENSOR_STEPS: u32 = 70u;
const BOT_SENSOR_STEP: f32 = 3.0;
const PI: f32 = 3.141592653589793;
const BG: u32 = 0xff12100cu;
const WALL: u32 = 0xff5b5045u;
const HUMAN_DEAD: u32 = 0xff3d3068u;
const BOT_DEAD: u32 = 0xff30544au;

struct Params {
  frame: u32,
  width: u32,
  height: u32,
  action: i32,
  reset: u32,
  reset_seed: u32,
  speed: f32,
  turn_rate: f32,
  radius: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

struct PlayerState {
  pos: vec2<f32>,
  prev_pos: vec2<f32>,
  heading: f32,
  alive: u32,
  color: u32,
  pad: u32,
};

struct GameState {
  round_over: u32,
  winner: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> players: array<PlayerState>;
@group(0) @binding(2) var<storage, read_write> occupancy: array<u32>;
@group(0) @binding(3) var<storage, read_write> image: array<u32>;
@group(0) @binding(4) var<storage, read_write> game: GameState;

var<workgroup> next_pos: array<vec2<f32>, MAX_PLAYERS>;
var<workgroup> next_heading: array<f32, MAX_PLAYERS>;
var<workgroup> chosen_action: array<i32, MAX_PLAYERS>;
var<workgroup> was_alive: array<u32, MAX_PLAYERS>;
var<workgroup> dead: array<atomic<u32>, MAX_PLAYERS>;
var<workgroup> round_over: atomic<u32>;
var<workgroup> touched_cell: array<u32, MAX_PLAYERS * MAX_FRAGMENTS>;
var<workgroup> touched_time: array<u32, MAX_PLAYERS * MAX_FRAGMENTS>;

fn cell(x: u32, y: u32) -> u32 {
  return y * params.width + x;
}

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

fn direction(heading: f32) -> vec2<f32> {
  return vec2<f32>(cos(heading), sin(heading));
}

fn hash32(x0: u32) -> u32 {
  var x = x0;
  x = (x ^ (x >> 16u)) * 0x7feb352du;
  x = (x ^ (x >> 15u)) * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn random01(seed: u32, stream: u32) -> f32 {
  let h = hash32(seed ^ (stream * 0x9e3779b9u));
  return f32(h & 0x00ffffffu) * (1.0 / 16777216.0);
}

fn fragment_index(player: u32, fragment: u32) -> u32 {
  return player * MAX_FRAGMENTS + fragment;
}

fn reset_player(player: u32) {
  let seed = select(0x6d2b79f5u, params.reset_seed, params.reset_seed != 0u);
  let arena = vec2<f32>(f32(params.width), f32(params.height));
  let center = arena * 0.5;
  let margin = max(56.0, params.radius * 10.0 + params.speed * 16.0);
  let axis_angle = random01(seed, 11u) * 2.0 * PI - PI;
  let axis = direction(axis_angle);
  let max_half_x = (arena.x * 0.5 - margin) / max(abs(axis.x), 0.001);
  let max_half_y = (arena.y * 0.5 - margin) / max(abs(axis.y), 0.001);
  let max_half = max(72.0, min(max_half_x, max_half_y) - 10.0);
  let half_sep = min(92.0 + random01(seed, 12u) * 54.0, max_half);
  let side = select(-1.0, 1.0, player == 1u);
  let start = center + axis * side * half_sep;
  let jitter = (random01(seed, 13u) * 2.0 - 1.0) * 0.65;
  let toward_center = select(axis_angle, axis_angle + PI, player == 1u);
  let mirrored_jitter = select(jitter, -jitter, player == 1u);
  let heading = wrap_angle(toward_center + mirrored_jitter);
  let color = select(0xffffdc3cu, 0xffff5c9fu, player == 1u);
  players[player].pos = start;
  players[player].prev_pos = start;
  players[player].heading = heading;
  players[player].alive = 1u;
  players[player].color = color;
  players[player].pad = 1u;
}

fn blocked_disk(pos: vec2<f32>, player: u32) -> bool {
  let cx = i32(floor(pos.x));
  let cy = i32(floor(pos.y));

  for (var oy = -RADIUS_CELLS; oy <= RADIUS_CELLS; oy = oy + 1i) {
    for (var ox = -RADIUS_CELLS; ox <= RADIUS_CELLS; ox = ox + 1i) {
      let r2 = f32(ox * ox + oy * oy);
      if (r2 > params.radius * params.radius + 0.25) {
        continue;
      }

      let x = cx + ox;
      let y = cy + oy;
      if (!in_bounds(x, y)) {
        return true;
      }

      let token = occupancy[flatten_cell(x, y)];
      if (token != 0u && !own_recent(token, player)) {
        return true;
      }
    }
  }

  return false;
}

fn ray_clearance(pos: vec2<f32>, heading: f32, player: u32) -> f32 {
  let dir = direction(heading);
  var probe = pos;
  for (var i = 0u; i < BOT_SENSOR_STEPS; i = i + 1u) {
    probe = probe + dir * BOT_SENSOR_STEP;
    if (blocked_disk(probe, player)) {
      return f32(i);
    }
  }
  return f32(BOT_SENSOR_STEPS);
}

fn rollout_clearance(player: u32, action: i32) -> f32 {
  var pos = players[player].pos;
  var heading = players[player].heading;
  var survived = 0.0;
  for (var i = 0u; i < BOT_LOOKAHEAD_STEPS; i = i + 1u) {
    if (i < 9u) {
      heading = wrap_angle(heading + f32(action) * params.turn_rate);
    }
    pos = pos + direction(heading) * params.speed;
    if (blocked_disk(pos, player)) {
      return survived;
    }
    survived = survived + 1.0;
  }
  return survived;
}

fn bot_candidate_score(player: u32, action: i32) -> f32 {
  let heading = wrap_angle(players[player].heading + f32(action) * params.turn_rate);
  let pos = players[player].pos + direction(heading) * params.speed;
  if (blocked_disk(pos, player)) {
    return -10000.0;
  }

  let survival = rollout_clearance(player, action);
  let open =
    ray_clearance(pos, heading, player) * 1.15 +
    ray_clearance(pos, wrap_angle(heading - 0.45), player) * 0.85 +
    ray_clearance(pos, wrap_angle(heading + 0.45), player) * 0.85 +
    ray_clearance(pos, wrap_angle(heading - 0.90), player) * 0.42 +
    ray_clearance(pos, wrap_angle(heading + 0.90), player) * 0.42;

  let edge_margin = min(min(pos.x, f32(params.width) - pos.x), min(pos.y, f32(params.height) - pos.y));
  let edge_bonus = min(edge_margin, 96.0) * 0.07;

  let to_human = players[0].pos - pos;
  let target_heading = atan2(to_human.y, to_human.x);
  let target_alignment = 1.0 - min(abs(wrap_angle(target_heading - heading)) / PI, 1.0);
  let attack_bonus = target_alignment * 4.0;

  let previous_action = i32(players[player].pad) - 1i;
  let smooth_bonus = select(0.0, 3.5, previous_action == action);
  let turn_cost = select(0.35, 0.0, action == 0i);

  return survival * 7.0 + open + edge_bonus + attack_bonus + smooth_bonus - turn_cost;
}

fn bot_action(player: u32) -> i32 {
  var best_action = 0i;
  var best_score = -1000000.0;
  for (var i = 0u; i < 3u; i = i + 1u) {
    let action = i32(i) - 1i;
    let score = bot_candidate_score(player, action);
    if (score > best_score) {
      best_score = score;
      best_action = action;
    }
  }
  return best_action;
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
  let center = players[player].pos + (next_pos[player] - players[player].pos) * t;
  let x = i32(floor(center.x)) + ox;
  let y = i32(floor(center.y)) + oy;

  if (!in_bounds(x, y)) {
    atomicStore(&dead[player], 1u);
    return;
  }

  touched_cell[idx] = flatten_cell(x, y);
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
    atomicStore(&round_over, 0u);
    if (params.reset != 0u) {
      game.round_over = 0u;
      game.winner = 0u;
      game.pad0 = 0u;
      game.pad1 = 0u;
    }
  }

  if (lane < MAX_PLAYERS) {
    if (params.reset != 0u) {
      reset_player(lane);
    }
  }

  storageBarrier();
  workgroupBarrier();

  if (lane < MAX_PLAYERS) {
    let p = lane;
    was_alive[p] = players[p].alive;
    atomicStore(&dead[p], 0u);

    var action = 0i;
    if (p == 0u) {
      action = params.action;
    } else if (players[p].alive != 0u) {
      action = bot_action(p);
    }
    chosen_action[p] = action;

    var heading = players[p].heading;
    if (action < 0i) {
      heading = heading - params.turn_rate;
    } else if (action > 0i) {
      heading = heading + params.turn_rate;
    }
    heading = wrap_angle(heading);
    next_heading[p] = heading;
    next_pos[p] = players[p].pos + direction(heading) * params.speed;
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    compute_fragment(i / MAX_FRAGMENTS, i % MAX_FRAGMENTS);
  }

  workgroupBarrier();

  if (lane < MAX_PLAYERS && was_alive[lane] != 0u && atomicLoad(&dead[lane]) != 0u) {
    atomicStore(&round_over, 1u);
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    let p = i / MAX_FRAGMENTS;
    let c = touched_cell[i];
    if (c == INVALID_CELL || was_alive[p] == 0u) {
      continue;
    }
    let old = occupancy[c];
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
        if (touched_cell[other_idx] == c) {
          let other_time = touched_time[other_idx];
          if (other_time <= my_time) {
            atomicStore(&dead[p], 1u);
          }
        }
      }
    }
  }

  workgroupBarrier();

  if (lane < MAX_PLAYERS && was_alive[lane] != 0u && atomicLoad(&dead[lane]) != 0u) {
    atomicStore(&round_over, 1u);
  }

  workgroupBarrier();

  if (lane == 0u && atomicLoad(&round_over) != 0u && game.round_over == 0u) {
    let human_dead = atomicLoad(&dead[0]) != 0u;
    let bot_dead = atomicLoad(&dead[1]) != 0u;
    game.round_over = 1u;
    if (human_dead && !bot_dead) {
      game.winner = 2u;
    } else if (bot_dead && !human_dead) {
      game.winner = 1u;
    } else {
      game.winner = 3u;
    }
  }

  workgroupBarrier();

  for (var i = lane; i < MAX_PLAYERS * MAX_FRAGMENTS; i = i + WORKGROUP_SIZE) {
    let p = i / MAX_FRAGMENTS;
    let c = touched_cell[i];
    if (c != INVALID_CELL && was_alive[p] != 0u && atomicLoad(&dead[p]) == 0u && atomicLoad(&round_over) == 0u) {
      occupancy[c] = (params.frame << 8u) | (p + 1u);
      image[c] = players[p].color;
    }
  }

  workgroupBarrier();

  if (lane < MAX_PLAYERS && was_alive[lane] != 0u) {
    if (atomicLoad(&dead[lane]) == 0u && atomicLoad(&round_over) == 0u) {
      players[lane].prev_pos = players[lane].pos;
      players[lane].pos = next_pos[lane];
      players[lane].heading = next_heading[lane];
      players[lane].alive = 1u;
      players[lane].pad = u32(chosen_action[lane] + 1i);
    } else {
      players[lane].alive = 0u;
      if (atomicLoad(&dead[lane]) != 0u) {
        let x = clamp(i32(players[lane].pos.x), 0i, i32(params.width) - 1i);
        let y = clamp(i32(players[lane].pos.y), 0i, i32(params.height) - 1i);
        image[flatten_cell(x, y)] = select(HUMAN_DEAD, BOT_DEAD, lane == 1u);
      }
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
  reset_seed: u32,
  speed: f32,
  turn_rate: f32,
  radius: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
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
