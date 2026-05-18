#pragma once

#if __has_include(<webgpu/webgpu_cpp.h>)
#include <webgpu/webgpu_cpp.h>
#else
#include <dawn/webgpu_cpp.h>
#endif

#include <array>
#include <cstdint>

namespace curve {

constexpr uint32_t kMaxPlayers = 2;
constexpr uint32_t kWorkgroupSize = 256;

struct PlayerState {
    float pos[2];
    float prevPos[2];
    float heading;
    uint32_t alive;
    uint32_t color;
    uint32_t pad;
};

struct InteractiveParams {
    uint32_t frame;
    uint32_t width;
    uint32_t height;
    int32_t action;
    uint32_t reset;
    float speed;
    float turnRate;
    float radius;
};

struct GameState {
    uint32_t roundOver;
    uint32_t winner;
    uint32_t pad0;
    uint32_t pad1;
};

struct NativeDemoParams {
    uint32_t frame;
    uint32_t width;
    uint32_t height;
    uint32_t players;
    float speed;
    float turnRate;
    float radius;
    float pad;
};

struct BatchParams {
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

static_assert(sizeof(PlayerState) == 32);
static_assert(sizeof(InteractiveParams) == 32);
static_assert(sizeof(GameState) == 16);
static_assert(sizeof(NativeDemoParams) == 32);
static_assert(sizeof(BatchParams) == 40);

inline uint32_t Rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
}

inline wgpu::ShaderModule CreateShaderModule(const wgpu::Device& device, const char* source) {
    wgpu::ShaderSourceWGSL wgsl;
    wgsl.code = source;
    wgpu::ShaderModuleDescriptor desc;
    desc.nextInChain = &wgsl;
    return device.CreateShaderModule(&desc);
}

inline wgpu::Buffer CreateBuffer(const wgpu::Device& device, uint64_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor desc;
    desc.size = size;
    desc.usage = usage;
    return device.CreateBuffer(&desc);
}

inline wgpu::ComputePipeline CreateComputePipeline(const wgpu::Device& device,
                                                   const wgpu::BindGroupLayout& bindGroupLayout,
                                                   const wgpu::ShaderModule& shader,
                                                   const char* entryPoint = "step_env") {
    wgpu::PipelineLayoutDescriptor layoutDesc;
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = &bindGroupLayout;
    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipelineDescriptor desc;
    desc.layout = pipelineLayout;
    desc.compute.module = shader;
    desc.compute.entryPoint = entryPoint;
    return device.CreateComputePipeline(&desc);
}

}  // namespace curve
