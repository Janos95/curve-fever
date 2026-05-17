#include <webgpu/webgpu_cpp.h>

#include <emscripten/emscripten.h>

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kWidth = 512;
constexpr uint32_t kHeight = 512;
constexpr uint32_t kWorkgroupSize = 256;
constexpr uint32_t kCells = kWidth * kHeight;
constexpr uint32_t kImageBytes = kCells * 4;

struct Params {
    uint32_t frame;
    uint32_t width;
    uint32_t height;
    int32_t action;
    uint32_t reset;
    float speed;
    float turnRate;
    float radius;
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

wgpu::Instance gInstance = wgpuCreateInstance(nullptr);
wgpu::Adapter gAdapter;
wgpu::Device gDevice;
wgpu::Queue gQueue;
wgpu::Surface gSurface;
wgpu::TextureFormat gSurfaceFormat = wgpu::TextureFormat::Undefined;
wgpu::Buffer gParamsBuffer;
wgpu::Buffer gPlayerBuffer;
wgpu::Buffer gOccupancyBuffer;
wgpu::Buffer gImageBuffer;
wgpu::BindGroup gComputeBindGroup;
wgpu::BindGroup gRenderBindGroup;
wgpu::ComputePipeline gComputePipeline;
wgpu::RenderPipeline gRenderPipeline;

uint32_t gFrame = 0;
int32_t gAction = 0;
bool gResetQueued = true;
bool gReady = false;

uint32_t Rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
}

wgpu::Buffer CreateBuffer(uint64_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor desc;
    desc.size = size;
    desc.usage = usage;
    return gDevice.CreateBuffer(&desc);
}

wgpu::ShaderModule CreateShaderModule(const char* source) {
    wgpu::ShaderSourceWGSL wgsl;
    wgsl.code = source;
    wgpu::ShaderModuleDescriptor desc;
    desc.nextInChain = &wgsl;
    return gDevice.CreateShaderModule(&desc);
}

constexpr char kComputeShader[] = R"wgsl(
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

constexpr char kRenderShader[] = R"wgsl(
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

wgpu::BindGroupLayout CreateComputeBindGroupLayout() {
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
    return gDevice.CreateBindGroupLayout(&desc);
}

wgpu::BindGroupLayout CreateRenderBindGroupLayout() {
    std::array<wgpu::BindGroupLayoutEntry, 2> entries{};
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Fragment;
    entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Fragment;
    entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutDescriptor desc;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    return gDevice.CreateBindGroupLayout(&desc);
}

void CreateBindGroupsAndPipelines() {
    gParamsBuffer = CreateBuffer(sizeof(Params), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    gPlayerBuffer = CreateBuffer(sizeof(PlayerState), wgpu::BufferUsage::Storage);
    gOccupancyBuffer = CreateBuffer(kCells * sizeof(uint32_t), wgpu::BufferUsage::Storage);
    gImageBuffer = CreateBuffer(kImageBytes, wgpu::BufferUsage::Storage);

    wgpu::ShaderModule computeModule = CreateShaderModule(kComputeShader);
    wgpu::BindGroupLayout computeLayout = CreateComputeBindGroupLayout();
    {
        std::array<wgpu::BindGroupEntry, 4> entries{};
        entries[0].binding = 0;
        entries[0].buffer = gParamsBuffer;
        entries[0].size = sizeof(Params);
        entries[1].binding = 1;
        entries[1].buffer = gPlayerBuffer;
        entries[1].size = sizeof(PlayerState);
        entries[2].binding = 2;
        entries[2].buffer = gOccupancyBuffer;
        entries[2].size = kCells * sizeof(uint32_t);
        entries[3].binding = 3;
        entries[3].buffer = gImageBuffer;
        entries[3].size = kImageBytes;
        wgpu::BindGroupDescriptor desc;
        desc.layout = computeLayout;
        desc.entryCount = entries.size();
        desc.entries = entries.data();
        gComputeBindGroup = gDevice.CreateBindGroup(&desc);
    }
    {
        wgpu::PipelineLayoutDescriptor layoutDesc;
        layoutDesc.bindGroupLayoutCount = 1;
        layoutDesc.bindGroupLayouts = &computeLayout;
        wgpu::PipelineLayout layout = gDevice.CreatePipelineLayout(&layoutDesc);

        wgpu::ComputePipelineDescriptor desc;
        desc.layout = layout;
        desc.compute.module = computeModule;
        desc.compute.entryPoint = "step_env";
        gComputePipeline = gDevice.CreateComputePipeline(&desc);
    }

    wgpu::ShaderModule renderModule = CreateShaderModule(kRenderShader);
    wgpu::BindGroupLayout renderLayout = CreateRenderBindGroupLayout();
    {
        std::array<wgpu::BindGroupEntry, 2> entries{};
        entries[0].binding = 0;
        entries[0].buffer = gParamsBuffer;
        entries[0].size = sizeof(Params);
        entries[1].binding = 1;
        entries[1].buffer = gImageBuffer;
        entries[1].size = kImageBytes;
        wgpu::BindGroupDescriptor desc;
        desc.layout = renderLayout;
        desc.entryCount = entries.size();
        desc.entries = entries.data();
        gRenderBindGroup = gDevice.CreateBindGroup(&desc);
    }
    {
        wgpu::PipelineLayoutDescriptor layoutDesc;
        layoutDesc.bindGroupLayoutCount = 1;
        layoutDesc.bindGroupLayouts = &renderLayout;
        wgpu::PipelineLayout layout = gDevice.CreatePipelineLayout(&layoutDesc);

        wgpu::ColorTargetState colorTarget;
        colorTarget.format = gSurfaceFormat;

        wgpu::FragmentState fragment;
        fragment.module = renderModule;
        fragment.entryPoint = "fs_main";
        fragment.targetCount = 1;
        fragment.targets = &colorTarget;

        wgpu::RenderPipelineDescriptor desc;
        desc.layout = layout;
        desc.vertex.module = renderModule;
        desc.vertex.entryPoint = "vs_main";
        desc.fragment = &fragment;
        desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
        gRenderPipeline = gDevice.CreateRenderPipeline(&desc);
    }
}

void ConfigureSurface() {
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvasDesc;
    canvasDesc.selector = "#canvas";

    wgpu::SurfaceDescriptor surfaceDesc;
    surfaceDesc.nextInChain = &canvasDesc;
    gSurface = gInstance.CreateSurface(&surfaceDesc);

    wgpu::SurfaceCapabilities capabilities;
    gSurface.GetCapabilities(gAdapter, &capabilities);
    gSurfaceFormat = capabilities.formats[0];

    wgpu::SurfaceConfiguration config;
    config.device = gDevice;
    config.format = gSurfaceFormat;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = kWidth;
    config.height = kHeight;
    config.alphaMode = wgpu::CompositeAlphaMode::Auto;
    config.presentMode = wgpu::PresentMode::Fifo;
    gSurface.Configure(&config);
}

void Frame() {
    if (!gReady) {
        return;
    }

    gFrame += 1;
    Params params = {
        gFrame,
        kWidth,
        kHeight,
        gAction,
        gResetQueued ? 1u : 0u,
        1.85f,
        0.099f,
        2.0f,
    };
    gResetQueued = false;
    gQueue.WriteBuffer(gParamsBuffer, 0, &params, sizeof(params));

    wgpu::SurfaceTexture surfaceTexture;
    gSurface.GetCurrentTexture(&surfaceTexture);
    if (surfaceTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        surfaceTexture.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        return;
    }

    wgpu::CommandEncoder encoder = gDevice.CreateCommandEncoder();
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(gComputePipeline);
        pass.SetBindGroup(0, gComputeBindGroup);
        pass.DispatchWorkgroups(1);
        pass.End();
    }
    {
        wgpu::RenderPassColorAttachment attachment;
        attachment.view = surfaceTexture.texture.CreateView();
        attachment.loadOp = wgpu::LoadOp::Clear;
        attachment.storeOp = wgpu::StoreOp::Store;
        attachment.clearValue = {0.0, 0.0, 0.0, 1.0};

        wgpu::RenderPassDescriptor renderPass;
        renderPass.colorAttachmentCount = 1;
        renderPass.colorAttachments = &attachment;

        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass);
        pass.SetPipeline(gRenderPipeline);
        pass.SetBindGroup(0, gRenderBindGroup);
        pass.Draw(3);
        pass.End();
    }

    wgpu::CommandBuffer commands = encoder.Finish();
    gQueue.Submit(1, &commands);
}

void Start() {
    gQueue = gDevice.GetQueue();
    ConfigureSurface();
    CreateBindGroupsAndPipelines();
    gReady = true;
    EM_ASM({
      if (globalThis.curveFeverReady) globalThis.curveFeverReady();
    });
    emscripten_set_main_loop(Frame, 0, false);
}

void RequestDevice() {
    wgpu::DeviceDescriptor desc;
    desc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView message) {
            std::printf("WebGPU error (%u): %.*s\n", static_cast<unsigned>(type),
                        static_cast<int>(message.length), message.data ? message.data : "");
        });

    gAdapter.RequestDevice(
        &desc,
        wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
            if (status != wgpu::RequestDeviceStatus::Success) {
                std::printf("RequestDevice failed: %.*s\n", static_cast<int>(message.length),
                            message.data ? message.data : "");
                return;
            }
            gDevice = device;
            Start();
        });
}

void RequestAdapter() {
    gInstance.RequestAdapter(
        nullptr,
        wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                std::printf("RequestAdapter failed: %.*s\n", static_cast<int>(message.length),
                            message.data ? message.data : "");
                return;
            }
            gAdapter = adapter;
            RequestDevice();
        });
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE void curve_set_turn(int turn) {
    if (turn < 0) {
        gAction = -1;
    } else if (turn > 0) {
        gAction = 1;
    } else {
        gAction = 0;
    }
}

EMSCRIPTEN_KEEPALIVE void curve_reset_game() {
    gResetQueued = true;
}

}

int main() {
    RequestAdapter();
    return 0;
}
