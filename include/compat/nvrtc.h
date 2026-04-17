// Compatibility header: drop-in for <nvrtc.h> (NVIDIA Runtime Compilation)
// Compiles CUDA source to PTX using nvcc (Docker or local).
#pragma once

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <functional>

// ---------------------------------------------------------------------------
// NVRTC types and error codes
// ---------------------------------------------------------------------------

typedef enum {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
} nvrtcResult;

inline const char* nvrtcGetErrorString(nvrtcResult result) {
    switch (result) {
        case NVRTC_SUCCESS: return "NVRTC_SUCCESS";
        case NVRTC_ERROR_OUT_OF_MEMORY: return "NVRTC_ERROR_OUT_OF_MEMORY";
        case NVRTC_ERROR_PROGRAM_CREATION_FAILURE: return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
        case NVRTC_ERROR_INVALID_INPUT: return "NVRTC_ERROR_INVALID_INPUT";
        case NVRTC_ERROR_INVALID_PROGRAM: return "NVRTC_ERROR_INVALID_PROGRAM";
        case NVRTC_ERROR_COMPILATION: return "NVRTC_ERROR_COMPILATION";
        case NVRTC_ERROR_INTERNAL_ERROR: return "NVRTC_ERROR_INTERNAL_ERROR";
        default: return "NVRTC_ERROR_UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// Program object
// ---------------------------------------------------------------------------

struct _nvrtcProgram {
    std::string source;
    std::string name;
    std::string ptx;           // compiled PTX output
    std::string compile_log;
    std::vector<std::string> name_expressions;  // kernel names to track
    std::vector<std::pair<std::string, std::string>> lowered_names;  // expr → lowered
};

typedef _nvrtcProgram* nvrtcProgram;

// ---------------------------------------------------------------------------
// NVRTC API functions
// ---------------------------------------------------------------------------

inline nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog,
                                       const char* src,
                                       const char* name,
                                       int numHeaders,
                                       const char* const* headers,
                                       const char* const* includeNames) {
    (void)numHeaders; (void)headers; (void)includeNames;
    auto* p = new _nvrtcProgram();
    p->source = src ? src : "";
    p->name = name ? name : "default_program.cu";
    *prog = p;
    return NVRTC_SUCCESS;
}

inline nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression) {
    if (!prog || !name_expression) return NVRTC_ERROR_INVALID_INPUT;
    prog->name_expressions.push_back(name_expression);
    return NVRTC_SUCCESS;
}

inline nvrtcResult nvrtcCompileProgram(nvrtcProgram prog,
                                        int numOptions,
                                        const char* const* options) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;

    // Write source to temp file
    size_t hash = std::hash<std::string>{}(prog->source);
    std::string base = "/tmp/cuda_sim_nvrtc/" + std::to_string(hash);
    std::string mkdir_cmd = "mkdir -p /tmp/cuda_sim_nvrtc";
    (void)system(mkdir_cmd.c_str());

    std::string cu_path = base + ".cu";
    std::string ptx_path = base + ".ptx";
    std::string log_path = base + ".log";

    {
        std::ofstream out(cu_path);
        out << prog->source;
    }

    // Build nvcc command
    std::string cmd = "nvcc -ptx";
    for (int i = 0; i < numOptions; i++) {
        if (options[i]) cmd += std::string(" ") + options[i];
    }
    cmd += " " + cu_path + " -o " + ptx_path + " 2>" + log_path;

    int ret = system(cmd.c_str());

    // Read compile log
    {
        std::ifstream log_file(log_path);
        if (log_file.is_open()) {
            std::ostringstream ss;
            ss << log_file.rdbuf();
            prog->compile_log = ss.str();
        }
    }

    if (ret != 0) {
        fprintf(stderr, "[cuda_sim] NVRTC: nvcc compilation failed\n");
        if (!prog->compile_log.empty()) {
            fprintf(stderr, "%s\n", prog->compile_log.c_str());
        }
        return NVRTC_ERROR_COMPILATION;
    }

    // Read generated PTX
    {
        std::ifstream ptx_file(ptx_path);
        if (!ptx_file.is_open()) {
            return NVRTC_ERROR_INTERNAL_ERROR;
        }
        std::ostringstream ss;
        ss << ptx_file.rdbuf();
        prog->ptx = ss.str();
    }

    // Build lowered name mappings by searching PTX for .entry with matching names
    for (auto& expr : prog->name_expressions) {
        std::string clean = expr;
        if (!clean.empty() && clean[0] == '&') clean = clean.substr(1);

        // Search the compiled PTX for .entry lines containing the kernel name
        std::string lowered = clean;  // default: unmangled
        size_t pos = 0;
        while ((pos = prog->ptx.find(".entry", pos)) != std::string::npos) {
            // Find the kernel name on this line
            size_t line_end = prog->ptx.find('\n', pos);
            std::string entry_line = prog->ptx.substr(pos, line_end - pos);
            // Look for the clean name as a substring (mangled names contain the original)
            if (entry_line.find(clean) != std::string::npos) {
                // Extract the full mangled name: .entry MANGLED_NAME(
                size_t name_start = entry_line.find(' ', 6);  // after ".entry"
                if (name_start != std::string::npos) {
                    name_start++;  // skip space
                    size_t name_end = entry_line.find('(', name_start);
                    if (name_end == std::string::npos) name_end = entry_line.size();
                    lowered = entry_line.substr(name_start, name_end - name_start);
                    // Trim whitespace
                    while (!lowered.empty() && lowered.back() == ' ') lowered.pop_back();
                }
            }
            pos = line_end != std::string::npos ? line_end + 1 : prog->ptx.size();
        }
        prog->lowered_names.push_back({expr, lowered});
    }

    return NVRTC_SUCCESS;
}

inline nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
    *ptxSizeRet = prog->ptx.size() + 1;  // include null terminator
    return NVRTC_SUCCESS;
}

inline nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
    std::memcpy(ptx, prog->ptx.c_str(), prog->ptx.size() + 1);
    return NVRTC_SUCCESS;
}

inline nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
    *logSizeRet = prog->compile_log.size() + 1;
    return NVRTC_SUCCESS;
}

inline nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) {
    if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
    std::memcpy(log, prog->compile_log.c_str(), prog->compile_log.size() + 1);
    return NVRTC_SUCCESS;
}

inline nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                        const char* name_expression,
                                        const char** lowered_name) {
    if (!prog || !name_expression || !lowered_name) return NVRTC_ERROR_INVALID_INPUT;
    for (size_t i = 0; i < prog->lowered_names.size(); i++) {
        if (prog->lowered_names[i].first == name_expression) {
            *lowered_name = prog->lowered_names[i].second.c_str();
            return NVRTC_SUCCESS;
        }
    }
    return NVRTC_ERROR_INVALID_INPUT;
}

inline nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) {
    if (prog && *prog) {
        delete *prog;
        *prog = nullptr;
    }
    return NVRTC_SUCCESS;
}
