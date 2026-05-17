#include "curve_engine.hpp"
#include "curve_shaders.hpp"

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
constexpr uint32_t kPlayers = curve::kMaxPlayers;
constexpr uint32_t kFrames = 300;
constexpr uint32_t kImageBytes = kWidth * kHeight * 4;

using Params = curve::NativeDemoParams;
using PlayerState = curve::PlayerState;
using curve::Rgba;

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

        wgpu::Buffer paramsBuffer = curve::CreateBuffer(device, sizeof(Params),
                                                        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer playerBuffer = curve::CreateBuffer(device, sizeof(PlayerState) * kPlayers,
                                                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer actionBuffer = curve::CreateBuffer(device, sizeof(uint32_t) * kPlayers,
                                                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer occupancyBuffer = curve::CreateBuffer(device, sizeof(uint32_t) * occupancy.size(),
                                                           wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        wgpu::Buffer imageBuffer =
            curve::CreateBuffer(device, kImageBytes,
                                wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
        wgpu::Buffer readbackBuffer =
            curve::CreateBuffer(device, kImageBytes, wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst);

        queue.WriteBuffer(playerBuffer, 0, initialPlayers.data(), sizeof(PlayerState) * kPlayers);
        queue.WriteBuffer(occupancyBuffer, 0, occupancy.data(), sizeof(uint32_t) * occupancy.size());
        queue.WriteBuffer(imageBuffer, 0, image.data(), kImageBytes);

        wgpu::ShaderModule shader = curve::CreateShaderModule(device, curve::shaders::kNativeDemoCompute);
        wgpu::BindGroupLayout layout = CreateBindGroupLayout(device);
        wgpu::BindGroup bindGroup =
            CreateBindGroup(device, layout, paramsBuffer, playerBuffer, actionBuffer, occupancyBuffer, imageBuffer);
        wgpu::ComputePipeline pipeline = curve::CreateComputePipeline(device, layout, shader);

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
