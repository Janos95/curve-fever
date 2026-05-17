#include "curve_engine.hpp"
#include "curve_shaders.hpp"

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
constexpr uint32_t kPlayers = curve::kMaxPlayers;
constexpr uint32_t kImageBytesPerEnv = kWidth * kHeight * 4;
constexpr uint32_t kCellsPerEnv = kWidth * kHeight;

struct Args {
    uint32_t batchSize = 1024;
    uint32_t warmupSteps = 20;
    uint32_t steps = 200;
};

using Params = curve::BatchParams;
using PlayerState = curve::PlayerState;
using curve::Rgba;

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
    wgpu::Buffer readback = curve::CreateBuffer(device, 4, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);
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
            curve::CreateBuffer(device, sizeof(Params), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer playerBuffer =
            curve::CreateBuffer(device, playerBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer occupancyBuffer = curve::CreateBuffer(
            device, gridBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        wgpu::Buffer imageBuffer =
            curve::CreateBuffer(device, imageBytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);

        queue.WriteBuffer(playerBuffer, 0, initialPlayers.data(), playerBytes);

        wgpu::ShaderModule shader = curve::CreateShaderModule(device, curve::shaders::kBatchBenchmarkCompute);
        wgpu::BindGroupLayout layout = CreateBindGroupLayout(device);
        wgpu::BindGroup bindGroup = CreateBindGroup(
            device, layout, paramsBuffer, playerBuffer, occupancyBuffer, imageBuffer, playerBytes, gridBytes, imageBytes);
        wgpu::ComputePipeline pipeline = curve::CreateComputePipeline(device, layout, shader);

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
