#include <dawn/webgpu_cpp.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 128;
constexpr uint32_t kPlayers = 2;
constexpr uint32_t kImageBytesPerEnv = kWidth * kHeight * 4;
constexpr uint32_t kCellsPerEnv = kWidth * kHeight;

struct Params {
    uint32_t frame;
    uint32_t width;
    uint32_t height;
    uint32_t players;
    uint32_t envs;
    uint32_t pad0;
    float speed;
    float turnRate;
    float radius;
    float pad1;
};

struct PlayerState {
    float pos[2];
    float prevPos[2];
    float heading;
    uint32_t alive;
    uint32_t color;
    uint32_t pad;
};

struct Args {
    uint32_t batchSize = 1024;
    uint32_t warmupSteps = 20;
    uint32_t steps = 200;
};

static_assert(sizeof(PlayerState) == 32);

uint32_t Rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + flag);
        }
        const auto value = static_cast<uint32_t>(std::stoul(argv[++i]));
        if (flag == "--batch-size") {
            args.batchSize = value;
        } else if (flag == "--warmup-steps") {
            args.warmupSteps = value;
        } else if (flag == "--steps") {
            args.steps = value;
        } else {
            throw std::runtime_error("unknown argument " + flag);
        }
    }
    return args;
}

wgpu::ShaderModule CreateShaderModule(const wgpu::Device& device, const char* source) {
    wgpu::ShaderSourceWGSL wgsl;
    wgsl.code = source;
    wgpu::ShaderModuleDescriptor desc;
    desc.nextInChain = &wgsl;
    return device.CreateShaderModule(&desc);
}

wgpu::Buffer CreateBuffer(const wgpu::Device& device, uint64_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor desc;
    desc.size = size;
    desc.usage = usage;
    return device.CreateBuffer(&desc);
}

wgpu::BindGroupLayout CreateBindGroupLayout(const wgpu::Device& device) {
    std::array<wgpu::BindGroupLayoutEntry, 4> entries{};

    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Compute;
    entries[0].buffer.type = wgpu::BufferBindingType::Uniform;

    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Compute;
    entries[1].buffer.type = wgpu::BufferBindingType::Storage;

    entries[2].binding = 2;
    entries[2].visibility = wgpu::ShaderStage::Compute;
    entries[2].buffer.type = wgpu::BufferBindingType::Storage;

    entries[3].binding = 3;
    entries[3].visibility = wgpu::ShaderStage::Compute;
    entries[3].buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor desc;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    return device.CreateBindGroupLayout(&desc);
}

wgpu::BindGroup CreateBindGroup(const wgpu::Device& device,
                                const wgpu::BindGroupLayout& layout,
                                const wgpu::Buffer& params,
                                const wgpu::Buffer& players,
                                const wgpu::Buffer& occupancy,
                                const wgpu::Buffer& image,
                                uint64_t playerBytes,
                                uint64_t gridBytes,
                                uint64_t imageBytes) {
    std::array<wgpu::BindGroupEntry, 4> entries{};
    entries[0].binding = 0;
    entries[0].buffer = params;
    entries[0].size = sizeof(Params);

    entries[1].binding = 1;
    entries[1].buffer = players;
    entries[1].size = playerBytes;

    entries[2].binding = 2;
    entries[2].buffer = occupancy;
    entries[2].size = gridBytes;

    entries[3].binding = 3;
    entries[3].buffer = image;
    entries[3].size = imageBytes;

    wgpu::BindGroupDescriptor desc;
    desc.layout = layout;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    return device.CreateBindGroup(&desc);
}

wgpu::ComputePipeline CreatePipeline(const wgpu::Device& device,
                                     const wgpu::BindGroupLayout& bindGroupLayout,
                                     const wgpu::ShaderModule& shader) {
    wgpu::PipelineLayoutDescriptor layoutDesc;
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipelineDescriptor desc;
    desc.layout = pipelineLayout;
    desc.compute.module = shader;
    desc.compute.entryPoint = "step_env";
    return device.CreateComputePipeline(&desc);
}

wgpu::Instance CreateInstance() {
    wgpu::InstanceDescriptor desc;
    desc.capabilities.timedWaitAnyEnable = true;
    return wgpu::CreateInstance(&desc);
}

wgpu::Adapter RequestAdapter(const wgpu::Instance& instance) {
    wgpu::RequestAdapterOptions options;
    options.backendType = wgpu::BackendType::Metal;
    options.powerPreference = wgpu::PowerPreference::HighPerformance;

    wgpu::Adapter adapter;
    bool ok = false;
    auto future = instance.RequestAdapter(
        &options, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter found, wgpu::StringView message) {
            ok = status == wgpu::RequestAdapterStatus::Success;
            if (!ok) {
                std::cerr << "RequestAdapter failed: "
                          << std::string(message.data ? message.data : "", message.length) << "\n";
                return;
            }
            adapter = found;
        });
    instance.WaitAny(future, UINT64_MAX);
    if (!ok || adapter == nullptr) {
        throw std::runtime_error("failed to request WebGPU adapter");
    }
    return adapter;
}

wgpu::Device RequestDevice(const wgpu::Instance& instance, const wgpu::Adapter& adapter) {
    wgpu::DeviceDescriptor desc;
    wgpu::Limits requiredLimits;
    adapter.GetLimits(&requiredLimits);
    desc.requiredLimits = &requiredLimits;
    desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView message) {
            std::cerr << "Device lost: "
                      << std::string(message.data ? message.data : "", message.length) << "\n";
        });
    desc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            std::cerr << "WebGPU error: "
                      << std::string(message.data ? message.data : "", message.length) << "\n";
        });

    wgpu::Device device;
    bool ok = false;
    auto future = adapter.RequestDevice(
        &desc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device found, wgpu::StringView message) {
            ok = status == wgpu::RequestDeviceStatus::Success;
            if (!ok) {
                std::cerr << "RequestDevice failed: "
                          << std::string(message.data ? message.data : "", message.length) << "\n";
                return;
            }
            device = found;
        });
    instance.WaitAny(future, UINT64_MAX);
    if (!ok || device == nullptr) {
        throw std::runtime_error("failed to request WebGPU device");
    }
    return device;
}

void WaitForGpu(const wgpu::Instance& instance,
                const wgpu::Device& device,
                const wgpu::Queue& queue,
                const wgpu::Buffer& source) {
    wgpu::Buffer readback = CreateBuffer(device, 4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(source, 0, readback, 0, 4);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    bool ok = false;
    auto future = readback.MapAsync(
        wgpu::MapMode::Read, 0, 4, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            ok = status == wgpu::MapAsyncStatus::Success;
            if (!ok) {
                std::cerr << "MapAsync failed: "
                          << std::string(message.data ? message.data : "", message.length) << "\n";
            }
        });
    instance.WaitAny(future, UINT64_MAX);
    if (!ok) {
        throw std::runtime_error("failed waiting for GPU");
    }
    readback.Unmap();
}

constexpr char kShader[] = R"wgsl(
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

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = ParseArgs(argc, argv);

        wgpu::Instance instance = CreateInstance();
        wgpu::Adapter adapter = RequestAdapter(instance);
        wgpu::Device device = RequestDevice(instance, adapter);
        wgpu::Queue queue = device.GetQueue();

        const uint64_t playerBytes = uint64_t(args.batchSize) * kPlayers * sizeof(PlayerState);
        const uint64_t gridBytes = uint64_t(args.batchSize) * kCellsPerEnv * sizeof(uint32_t);
        const uint64_t imageBytes = uint64_t(args.batchSize) * kImageBytesPerEnv;

        std::vector<PlayerState> initialPlayers(args.batchSize * kPlayers);
        for (uint32_t env = 0; env < args.batchSize; ++env) {
            initialPlayers[env * kPlayers + 0] =
                {{30.0f, 64.0f}, {30.0f, 64.0f}, 0.0f, 1u, Rgba(60, 220, 255), 0u};
            initialPlayers[env * kPlayers + 1] =
                {{98.0f, 64.0f}, {98.0f, 64.0f}, 3.14159265f, 1u, Rgba(255, 92, 120), 0u};
        }

        wgpu::Buffer paramsBuffer =
            CreateBuffer(device, sizeof(Params), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer playerBuffer =
            CreateBuffer(device, playerBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer occupancyBuffer =
            CreateBuffer(device, gridBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        wgpu::Buffer imageBuffer =
            CreateBuffer(device, imageBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);

        queue.WriteBuffer(playerBuffer, 0, initialPlayers.data(), playerBytes);

        wgpu::ShaderModule shader = CreateShaderModule(device, kShader);
        wgpu::BindGroupLayout layout = CreateBindGroupLayout(device);
        wgpu::BindGroup bindGroup = CreateBindGroup(
            device, layout, paramsBuffer, playerBuffer, occupancyBuffer, imageBuffer, playerBytes, gridBytes, imageBytes);
        wgpu::ComputePipeline pipeline = CreatePipeline(device, layout, shader);

        auto submit_step = [&](uint32_t frame) {
            Params params = {frame, kWidth, kHeight, kPlayers, args.batchSize, 0u, 1.65f, 0.155f, 1.75f, 0.0f};
            queue.WriteBuffer(paramsBuffer, 0, &params, sizeof(params));

            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, bindGroup);
            pass.DispatchWorkgroups(args.batchSize);
            pass.End();
            wgpu::CommandBuffer commands = encoder.Finish();
            queue.Submit(1, &commands);
        };

        for (uint32_t i = 1; i <= args.warmupSteps; ++i) {
            submit_step(i);
        }
        WaitForGpu(instance, device, queue, occupancyBuffer);

        auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 1; i <= args.steps; ++i) {
            submit_step(args.warmupSteps + i);
        }
        WaitForGpu(instance, device, queue, occupancyBuffer);
        auto end = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration<double>(end - start).count();
        const double envSteps = double(args.batchSize) * double(args.steps);
        const double throughput = envSteps / seconds;
        std::cout << "batch_size,steps,seconds,env_steps_per_second\n"
                  << args.batchSize << "," << args.steps << "," << seconds << "," << throughput << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
