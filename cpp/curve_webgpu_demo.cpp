#include <dawn/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kWidth = 256;
constexpr uint32_t kHeight = 256;
constexpr uint32_t kPlayers = 2;
constexpr uint32_t kFrames = 300;
constexpr uint32_t kWorkgroupSize = 256;
constexpr uint32_t kImageBytes = kWidth * kHeight * 4;

struct Params {
    uint32_t frame;
    uint32_t width;
    uint32_t height;
    uint32_t players;
    float speed;
    float turnRate;
    float radius;
    float pad;
};

struct PlayerState {
    float pos[2];
    float prevPos[2];
    float heading;
    uint32_t alive;
    uint32_t color;
    uint32_t pad;
};

static_assert(sizeof(Params) == 32);
static_assert(sizeof(PlayerState) == 32);

uint32_t Rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
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
    std::array<wgpu::BindGroupLayoutEntry, 5> entries{};

    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Compute;
    entries[0].buffer.type = wgpu::BufferBindingType::Uniform;

    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Compute;
    entries[1].buffer.type = wgpu::BufferBindingType::Storage;

    entries[2].binding = 2;
    entries[2].visibility = wgpu::ShaderStage::Compute;
    entries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    entries[3].binding = 3;
    entries[3].visibility = wgpu::ShaderStage::Compute;
    entries[3].buffer.type = wgpu::BufferBindingType::Storage;

    entries[4].binding = 4;
    entries[4].visibility = wgpu::ShaderStage::Compute;
    entries[4].buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor desc;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    return device.CreateBindGroupLayout(&desc);
}

wgpu::BindGroup CreateBindGroup(const wgpu::Device& device,
                                const wgpu::BindGroupLayout& layout,
                                const wgpu::Buffer& params,
                                const wgpu::Buffer& players,
                                const wgpu::Buffer& actions,
                                const wgpu::Buffer& occupancy,
                                const wgpu::Buffer& image) {
    std::array<wgpu::BindGroupEntry, 5> entries{};
    entries[0].binding = 0;
    entries[0].buffer = params;
    entries[0].size = sizeof(Params);

    entries[1].binding = 1;
    entries[1].buffer = players;
    entries[1].size = sizeof(PlayerState) * kPlayers;

    entries[2].binding = 2;
    entries[2].buffer = actions;
    entries[2].size = sizeof(uint32_t) * kPlayers;

    entries[3].binding = 3;
    entries[3].buffer = occupancy;
    entries[3].size = sizeof(uint32_t) * kWidth * kHeight;

    entries[4].binding = 4;
    entries[4].buffer = image;
    entries[4].size = kImageBytes;

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

void WaitForReadback(const wgpu::Instance& instance, const wgpu::Buffer& readback) {
    bool ok = false;
    auto future = readback.MapAsync(
        wgpu::MapMode::Read, 0, kImageBytes, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            ok = status == wgpu::MapAsyncStatus::Success;
            if (!ok) {
                std::cerr << "MapAsync failed: "
                          << std::string(message.data ? message.data : "", message.length) << "\n";
            }
        });
    instance.WaitAny(future, UINT64_MAX);
    if (!ok) {
        throw std::runtime_error("failed to map readback buffer");
    }
}

void WritePpm(const std::filesystem::path& path, const uint8_t* rgba) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << kWidth << " " << kHeight << "\n255\n";
    for (uint32_t i = 0; i < kWidth * kHeight; ++i) {
        out.put(static_cast<char>(rgba[i * 4 + 0]));
        out.put(static_cast<char>(rgba[i * 4 + 1]));
        out.put(static_cast<char>(rgba[i * 4 + 2]));
    }
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

}  // namespace

int main() {
    try {
        const auto outputDir = std::filesystem::path("out/curve_demo_frames");
        std::filesystem::create_directories(outputDir);

        wgpu::Instance instance = CreateInstance();
        wgpu::Adapter adapter = RequestAdapter(instance);
        wgpu::AdapterInfo info;
        adapter.GetInfo(&info);
        std::cout << "Adapter: " << std::string(info.device.data, info.device.length) << "\n";

        wgpu::Device device = RequestDevice(instance, adapter);
        wgpu::Queue queue = device.GetQueue();

        std::array<PlayerState, kPlayers> initialPlayers{};
        initialPlayers[0] = {{62.0f, 128.0f}, {62.0f, 128.0f}, 0.0f, 1u, Rgba(60, 220, 255), 0u};
        initialPlayers[1] = {{194.0f, 128.0f}, {194.0f, 128.0f}, 3.14159265f, 1u, Rgba(255, 92, 120), 0u};

        std::vector<uint32_t> occupancy(kWidth * kHeight, 0);
        std::vector<uint32_t> image(kWidth * kHeight, Rgba(12, 14, 20));
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                if (x < 2 || y < 2 || x >= kWidth - 2 || y >= kHeight - 2) {
                    image[y * kWidth + x] = Rgba(70, 76, 92);
                }
            }
        }

        wgpu::Buffer paramsBuffer = CreateBuffer(device, sizeof(Params),
                                                 wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer playerBuffer = CreateBuffer(device, sizeof(PlayerState) * kPlayers,
                                                 wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer actionBuffer = CreateBuffer(device, sizeof(uint32_t) * kPlayers,
                                                 wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer occupancyBuffer = CreateBuffer(device, sizeof(uint32_t) * occupancy.size(),
                                                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer imageBuffer = CreateBuffer(device, kImageBytes,
                                                wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst |
                                                    wgpu::BufferUsage::CopySrc);
        wgpu::Buffer readbackBuffer = CreateBuffer(device, kImageBytes,
                                                   wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);

        queue.WriteBuffer(playerBuffer, 0, initialPlayers.data(), sizeof(PlayerState) * kPlayers);
        queue.WriteBuffer(occupancyBuffer, 0, occupancy.data(), sizeof(uint32_t) * occupancy.size());
        queue.WriteBuffer(imageBuffer, 0, image.data(), kImageBytes);

        wgpu::ShaderModule shader = CreateShaderModule(device, kShader);
        wgpu::BindGroupLayout layout = CreateBindGroupLayout(device);
        wgpu::BindGroup bindGroup =
            CreateBindGroup(device, layout, paramsBuffer, playerBuffer, actionBuffer, occupancyBuffer, imageBuffer);
        wgpu::ComputePipeline pipeline = CreatePipeline(device, layout, shader);

        std::mt19937 rng(42);
        std::discrete_distribution<uint32_t> actionDist({1.0, 2.0, 1.0});

        for (uint32_t frame = 1; frame <= kFrames; ++frame) {
            Params params = {frame, kWidth, kHeight, kPlayers, 1.65f, 0.155f, 1.75f, 0.0f};
            std::array<uint32_t, kPlayers> actions = {actionDist(rng), actionDist(rng)};

            queue.WriteBuffer(paramsBuffer, 0, &params, sizeof(params));
            queue.WriteBuffer(actionBuffer, 0, actions.data(), sizeof(uint32_t) * actions.size());

            wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
            {
                wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
                pass.SetPipeline(pipeline);
                pass.SetBindGroup(0, bindGroup);
                pass.DispatchWorkgroups(1);
                pass.End();
            }
            encoder.CopyBufferToBuffer(imageBuffer, 0, readbackBuffer, 0, kImageBytes);
            wgpu::CommandBuffer commands = encoder.Finish();
            queue.Submit(1, &commands);

            WaitForReadback(instance, readbackBuffer);
            const auto* rgba = static_cast<const uint8_t*>(readbackBuffer.GetConstMappedRange(0, kImageBytes));
            char name[64];
            std::snprintf(name, sizeof(name), "frame_%04u.ppm", frame);
            WritePpm(outputDir / name, rgba);
            readbackBuffer.Unmap();
        }

        std::cout << "Wrote " << kFrames << " frames to " << outputDir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
