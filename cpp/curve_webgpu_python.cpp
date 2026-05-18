#include "curve_engine.hpp"
#include "curve_shaders.hpp"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

constexpr const char* kRunnerCapsuleName = "curve_fever.BatchRunner";
// tinygrad's vendored WebGPU ctypes bindings currently use the older native
// enum values without WGPUBufferBindingType_Undefined. The local Dawn headers
// include that newer enum value, so use the ABI values expected by tinygrad's
// libwebgpu_dawn_tinygrad.dylib when creating layouts for a tinygrad device.
constexpr auto kTinygradUniformBinding = static_cast<WGPUBufferBindingType>(1);
constexpr auto kTinygradStorageBinding = static_cast<WGPUBufferBindingType>(2);

WGPUStringView StringView(const char* text) {
    return WGPUStringView{ text, std::strlen(text) };
}

WGPUBuffer AsBuffer(unsigned long long ptr) {
    return reinterpret_cast<WGPUBuffer>(static_cast<uintptr_t>(ptr));
}

WGPUDevice AsDevice(unsigned long long ptr) {
    return reinterpret_cast<WGPUDevice>(static_cast<uintptr_t>(ptr));
}

struct BatchRunner {
    WGPUDevice device = nullptr;
    WGPUBuffer paramsBuffer = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUShaderModule shaderModule = nullptr;
    WGPUComputePipeline pipeline = nullptr;
    uint32_t batchSize = 0;
    uint32_t width = 0;
    uint32_t height = 0;

    BatchRunner(WGPUDevice borrowedDevice, uint32_t batch, uint32_t w, uint32_t h)
        : device(borrowedDevice), batchSize(batch), width(w), height(h) {
        if (device == nullptr) {
            throw std::runtime_error("device pointer is null");
        }
        wgpuDeviceAddRef(device);
        CreateParamsBuffer();
        CreatePipeline();
    }

    ~BatchRunner() {
        if (pipeline != nullptr) {
            wgpuComputePipelineRelease(pipeline);
        }
        if (shaderModule != nullptr) {
            wgpuShaderModuleRelease(shaderModule);
        }
        if (pipelineLayout != nullptr) {
            wgpuPipelineLayoutRelease(pipelineLayout);
        }
        if (bindGroupLayout != nullptr) {
            wgpuBindGroupLayoutRelease(bindGroupLayout);
        }
        if (paramsBuffer != nullptr) {
            wgpuBufferDestroy(paramsBuffer);
            wgpuBufferRelease(paramsBuffer);
        }
        if (device != nullptr) {
            wgpuDeviceRelease(device);
        }
    }

    void CreateParamsBuffer() {
        WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        desc.label = StringView("curve batch params");
        desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        desc.size = sizeof(curve::BatchParams);
        paramsBuffer = wgpuDeviceCreateBuffer(device, &desc);
        if (paramsBuffer == nullptr) {
            throw std::runtime_error("failed to create params buffer");
        }
    }

    static void OnPopErrorScope(WGPUPopErrorScopeStatus status,
                                WGPUErrorType type,
                                WGPUStringView message,
                                void* userdata1,
                                void*) {
        auto* out = static_cast<std::string*>(userdata1);
        if (status != WGPUPopErrorScopeStatus_Success) {
            *out = "error scope pop failed";
            return;
        }
        if (type != WGPUErrorType_NoError) {
            out->assign(message.data == nullptr ? "" : message.data, message.length);
        }
    }

    void CheckErrorScope(const char* where) {
        std::string error;
        WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
        callback.mode = WGPUCallbackMode_WaitAnyOnly;
        callback.callback = OnPopErrorScope;
        callback.userdata1 = &error;
        WGPUFuture future = wgpuDevicePopErrorScope(device, callback);
        WGPUAdapter adapter = wgpuDeviceGetAdapter(device);
        WGPUInstance instance = wgpuAdapterGetInstance(adapter);
        WGPUFutureWaitInfo waitInfo = WGPU_FUTURE_WAIT_INFO_INIT;
        waitInfo.future = future;
        WGPUWaitStatus waitStatus = wgpuInstanceWaitAny(instance, 1, &waitInfo, UINT64_MAX);
        wgpuInstanceRelease(instance);
        wgpuAdapterRelease(adapter);
        if (waitStatus != WGPUWaitStatus_Success || !waitInfo.completed) {
            throw std::runtime_error(std::string(where) + " failed while waiting for WebGPU validation");
        }
        if (!error.empty()) {
            throw std::runtime_error(std::string(where) + " WebGPU validation error: " + error);
        }
    }

    void CreatePipeline() {
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
        std::array<WGPUBindGroupLayoutEntry, 4> layoutEntries{};
        for (uint32_t i = 0; i < layoutEntries.size(); ++i) {
            layoutEntries[i] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            layoutEntries[i].binding = i;
            layoutEntries[i].visibility = WGPUShaderStage_Compute;
            layoutEntries[i].buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
            layoutEntries[i].buffer.type = i == 0 ? kTinygradUniformBinding : kTinygradStorageBinding;
        }

        WGPUBindGroupLayoutDescriptor layoutDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        layoutDesc.label = StringView("curve batch layout");
        layoutDesc.entryCount = layoutEntries.size();
        layoutDesc.entries = layoutEntries.data();
        bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &layoutDesc);
        if (bindGroupLayout == nullptr) {
            throw std::runtime_error("failed to create bind group layout");
        }

        WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgsl.code = StringView(curve::shaders::kBatchBenchmarkCompute);
        WGPUShaderModuleDescriptor shaderDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        shaderDesc.label = StringView("curve batch shader");
        shaderDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl);
        shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);
        if (shaderModule == nullptr) {
            throw std::runtime_error("failed to create shader module");
        }

        WGPUPipelineLayoutDescriptor pipelineLayoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        pipelineLayoutDesc.label = StringView("curve batch pipeline layout");
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;
        pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);
        if (pipelineLayout == nullptr) {
            throw std::runtime_error("failed to create pipeline layout");
        }

        WGPUComputePipelineDescriptor pipelineDesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        pipelineDesc.label = StringView("curve batch pipeline");
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.compute.module = shaderModule;
        pipelineDesc.compute.entryPoint = StringView("step_env");
        pipeline = wgpuDeviceCreateComputePipeline(device, &pipelineDesc);
        if (pipeline == nullptr) {
            throw std::runtime_error("failed to create compute pipeline");
        }
        CheckErrorScope("create batch pipeline");
    }

    WGPUBindGroup CreateBindGroup(WGPUBuffer players, WGPUBuffer occupancy, WGPUBuffer image) {
        if (players == nullptr || occupancy == nullptr || image == nullptr) {
            throw std::runtime_error("one or more buffer pointers are null");
        }

        std::array<WGPUBindGroupEntry, 4> entries{};
        entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[0].binding = 0;
        entries[0].buffer = paramsBuffer;
        entries[0].size = sizeof(curve::BatchParams);
        entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[1].binding = 1;
        entries[1].buffer = players;
        entries[1].size = wgpuBufferGetSize(players);
        entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[2].binding = 2;
        entries[2].buffer = occupancy;
        entries[2].size = wgpuBufferGetSize(occupancy);
        entries[3] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[3].binding = 3;
        entries[3].buffer = image;
        entries[3].size = wgpuBufferGetSize(image);

        WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        desc.label = StringView("curve batch bind group");
        desc.layout = bindGroupLayout;
        desc.entryCount = entries.size();
        desc.entries = entries.data();
        WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &desc);
        if (bindGroup == nullptr) {
            throw std::runtime_error("failed to create bind group");
        }
        return bindGroup;
    }

    void Step(uint32_t frame,
              WGPUBuffer players,
              WGPUBuffer occupancy,
              WGPUBuffer image,
              float speed,
              float turnRate,
              float radius) {
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
        curve::BatchParams params = {
            frame,
            width,
            height,
            curve::kMaxPlayers,
            batchSize,
            0u,
            speed,
            turnRate,
            radius,
            0.0f,
        };

        WGPUQueue queue = wgpuDeviceGetQueue(device);
        wgpuQueueWriteBuffer(queue, paramsBuffer, 0, &params, sizeof(params));

        WGPUBindGroup bindGroup = CreateBindGroup(players, occupancy, image);
        WGPUCommandEncoderDescriptor encoderDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoderDesc);
        WGPUComputePassDescriptor passDesc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
        wgpuComputePassEncoderSetPipeline(pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, batchSize, 1, 1);
        wgpuComputePassEncoderEnd(pass);

        WGPUCommandBufferDescriptor commandDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &commandDesc);
        wgpuQueueSubmit(queue, 1, &commands);

        wgpuCommandBufferRelease(commands);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(bindGroup);
        wgpuQueueRelease(queue);
        CheckErrorScope("batch step");
    }
};

BatchRunner* GetRunner(PyObject* capsule) {
    auto* runner = static_cast<BatchRunner*>(PyCapsule_GetPointer(capsule, kRunnerCapsuleName));
    if (runner == nullptr) {
        throw std::runtime_error("invalid BatchRunner capsule");
    }
    return runner;
}

void RunnerCapsuleDestructor(PyObject* capsule) {
    delete static_cast<BatchRunner*>(PyCapsule_GetPointer(capsule, kRunnerCapsuleName));
}

PyObject* CreateBatchRunner(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = { "device_ptr", "batch_size", "width", "height", nullptr };
    unsigned long long devicePtr = 0;
    unsigned int batchSize = 0;
    unsigned int width = 128;
    unsigned int height = 128;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "KI|II", const_cast<char**>(kwlist), &devicePtr, &batchSize, &width,
                                     &height)) {
        return nullptr;
    }

    try {
        auto* runner = new BatchRunner(AsDevice(devicePtr), batchSize, width, height);
        return PyCapsule_New(runner, kRunnerCapsuleName, RunnerCapsuleDestructor);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyObject* BatchStep(PyObject*, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "runner", "frame", "players_ptr", "occupancy_ptr", "image_ptr", "speed", "turn_rate", "radius", nullptr
    };
    PyObject* capsule = nullptr;
    unsigned int frame = 0;
    unsigned long long playersPtr = 0;
    unsigned long long occupancyPtr = 0;
    unsigned long long imagePtr = 0;
    double speed = 1.65;
    double turnRate = 0.155;
    double radius = 1.75;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OIKKK|ddd", const_cast<char**>(kwlist), &capsule, &frame,
                                     &playersPtr, &occupancyPtr, &imagePtr, &speed, &turnRate, &radius)) {
        return nullptr;
    }

    try {
        GetRunner(capsule)->Step(frame, AsBuffer(playersPtr), AsBuffer(occupancyPtr), AsBuffer(imagePtr),
                                 static_cast<float>(speed), static_cast<float>(turnRate), static_cast<float>(radius));
        Py_RETURN_NONE;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyObject* BufferSize(PyObject*, PyObject* args) {
    unsigned long long bufferPtr = 0;
    if (!PyArg_ParseTuple(args, "K", &bufferPtr)) {
        return nullptr;
    }
    WGPUBuffer buffer = AsBuffer(bufferPtr);
    if (buffer == nullptr) {
        PyErr_SetString(PyExc_ValueError, "buffer pointer is null");
        return nullptr;
    }
    return PyLong_FromUnsignedLongLong(wgpuBufferGetSize(buffer));
}

PyMethodDef kMethods[] = {
    { "create_batch_runner", reinterpret_cast<PyCFunction>(CreateBatchRunner), METH_VARARGS | METH_KEYWORDS,
      "Create a C++ WebGPU runner that dispatches the shared batch env shader." },
    { "batch_step", reinterpret_cast<PyCFunction>(BatchStep), METH_VARARGS | METH_KEYWORDS,
      "Dispatch one batch env step against tinygrad-owned WebGPU buffers." },
    { "buffer_size", BufferSize, METH_VARARGS, "Return a WGPUBuffer size from its raw handle." },
    { nullptr, nullptr, 0, nullptr },
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "_curve_webgpu_native",
    "Manual CPython bindings for dispatching Curve Fever WebGPU kernels on tinygrad-owned buffers.",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__curve_webgpu_native() {
    return PyModule_Create(&kModule);
}
