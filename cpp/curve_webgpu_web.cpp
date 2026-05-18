#include "curve_engine.hpp"
#include "curve_shaders.hpp"

#include <emscripten/emscripten.h>

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kWidth = 512;
constexpr uint32_t kHeight = 512;
constexpr uint32_t kCells = kWidth * kHeight;
constexpr uint32_t kImageBytes = kCells * 4;

using Params = curve::InteractiveParams;
using PlayerState = curve::PlayerState;

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

wgpu::Buffer CreateBuffer(uint64_t size, wgpu::BufferUsage usage) {
    return curve::CreateBuffer(gDevice, size, usage);
}



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
    gPlayerBuffer = CreateBuffer(sizeof(PlayerState) * curve::kMaxPlayers, wgpu::BufferUsage::Storage);
    gOccupancyBuffer = CreateBuffer(kCells * sizeof(uint32_t), wgpu::BufferUsage::Storage);
    gImageBuffer = CreateBuffer(kImageBytes, wgpu::BufferUsage::Storage);

    wgpu::ShaderModule computeModule = curve::CreateShaderModule(gDevice, curve::shaders::kInteractiveCompute);
    wgpu::BindGroupLayout computeLayout = CreateComputeBindGroupLayout();
    {
        std::array<wgpu::BindGroupEntry, 4> entries{};
        entries[0].binding = 0;
        entries[0].buffer = gParamsBuffer;
        entries[0].size = sizeof(Params);
        entries[1].binding = 1;
        entries[1].buffer = gPlayerBuffer;
        entries[1].size = sizeof(PlayerState) * curve::kMaxPlayers;
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

    wgpu::ShaderModule renderModule = curve::CreateShaderModule(gDevice, curve::shaders::kInteractiveRender);
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
        1.30f,
        0.069f,
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
