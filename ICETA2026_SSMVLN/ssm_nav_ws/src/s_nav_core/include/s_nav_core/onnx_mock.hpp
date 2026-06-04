#pragma once
#include <vector>
#include <string>
#include <iostream>

// ONNX Runtime Mock for Build Completion
namespace Ort {
    enum MemoryInfoType { OrtArenaAllocator };
    enum OrtMemoryType { OrtMemTypeDefault };
    enum LogLevel { ORT_LOGGING_LEVEL_WARNING };

    struct MemoryInfo {
        static MemoryInfo CreateCpu(MemoryInfoType, OrtMemoryType) { return {}; }
    };

    struct Env {
        Env(LogLevel, const char*) {}
    };

    struct SessionOptions {
        SessionOptions() {}
    };

    struct Session {
        Session(Env&, const char*, SessionOptions&) {}
        // Mock Run or other methods if needed
    };

    struct AllocatorWithDefaultOptions {
        AllocatorWithDefaultOptions() {}
    };

    struct Exception : public std::exception {
        std::string msg;
        Exception(const char* m) : msg(m) {}
        const char* what() const noexcept override { return msg.c_str(); }
    };
}
