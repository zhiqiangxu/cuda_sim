/*
 * CUDAMiner_cuda_sim.cpp — cuda_sim version of CUDAMiner_cuda.cu
 *
 * This file replaces the original CUDAMiner_cuda.cu for CPU simulation.
 * Changes from original:
 *   1. __global__ kernel (ethash_calculate_dag_item) is removed — DAG generation
 *      runs via the ProgPow JIT kernel or is done in pure C++ here.
 *   2. <<<>>> kernel launch syntax replaced with direct function call.
 *   3. __constant__ variables are plain globals (managed by cudaMemcpyToSymbol).
 *   4. cuda_helper.h included with CUDA_SIM_HOST_DEVICE_STUBS already defined.
 */

#include "CUDAMiner_cuda.h"
#include <thread>
#include <vector>

#define ETHASH_HASH_BYTES 64
#define ETHASH_DATASET_PARENTS 512

// __constant__ variables — plain globals for CPU simulation
// cudaMemcpyToSymbol/FromSymbol access these via name.
static uint32_t d_dag_size;
static hash64_t* d_dag;
static uint32_t d_light_size;
static hash64_t* d_light;
static hash32_t d_header;
static uint64_t d_target;

// ---------------------------------------------------------------------------
// Keccak-f1600 (same as original, minus __device__ qualifier)
// ---------------------------------------------------------------------------

static const uint64_t keccakf_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

static inline void keccak_f1600_round(uint64_t st[25], const int r) {
    const uint32_t keccakf_rotc[24] = {
        1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
        27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
    };
    const uint32_t keccakf_piln[24] = {
        10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
    };

    uint64_t t, bc[5];
    for (int i = 0; i < 5; i++)
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
    for (int i = 0; i < 5; i++) {
        t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
        for (uint32_t j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
    t = st[1];
    for (int i = 0; i < 24; i++) {
        uint32_t j = keccakf_piln[i];
        bc[0] = st[j];
        st[j] = ROTL64(t, keccakf_rotc[i]);
        t = bc[0];
    }
    for (uint32_t j = 0; j < 25; j += 5) {
        for (int i = 0; i < 5; i++) bc[i] = st[j + i];
        for (int i = 0; i < 5; i++) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }
    st[0] ^= keccakf_rndc[r];
}

static inline void keccak_f1600(uint64_t st[25]) {
    for (int i = 8; i < 25; i++) st[i] = 0;
    st[8] = 0x8000000000000001;
    for (int r = 0; r < 24; r++) keccak_f1600_round(st, r);
}

// ---------------------------------------------------------------------------
// FNV hash
// ---------------------------------------------------------------------------

#define FNV_PRIME 0x01000193U
#define fnv(x, y) ((uint32_t(x) * FNV_PRIME) ^ uint32_t(y))

static inline uint4 fnv4(uint4 a, uint4 b) {
    uint4 c;
    c.x = a.x * FNV_PRIME ^ b.x;
    c.y = a.y * FNV_PRIME ^ b.y;
    c.z = a.z * FNV_PRIME ^ b.z;
    c.w = a.w * FNV_PRIME ^ b.w;
    return c;
}

// ---------------------------------------------------------------------------
// DAG item calculation (CPU version — sequential, no warp shuffles)
// ---------------------------------------------------------------------------

#define NODE_WORDS (ETHASH_HASH_BYTES / sizeof(uint32_t))

static void ethash_calculate_dag_item_cpu(
    uint32_t start, hash64_t* g_dag, uint64_t dag_bytes,
    hash64_t* g_light, uint32_t light_words,
    uint32_t num_threads)
{
    uint64_t num_nodes = dag_bytes / sizeof(hash64_t);
    uint64_t num_nodes_rounded = ((num_nodes + 3) / 4) * 4;

    for (uint32_t tid = 0; tid < num_threads; tid++) {
        uint64_t node_index = start + tid;
        if (node_index >= num_nodes_rounded) continue;

        hash200_t dag_node;
        for (int i = 0; i < 4; i++)
            dag_node.uint4s[i] = g_light[node_index % light_words].uint4s[i];
        dag_node.words[0] ^= (uint32_t)node_index;
        keccak_f1600(dag_node.uint64s);

        for (uint32_t i = 0; i < ETHASH_DATASET_PARENTS; ++i) {
            uint32_t parent_index = fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % light_words;
            for (int w = 0; w < 4; w++) {
                dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], g_light[parent_index].uint4s[w]);
            }
        }
        keccak_f1600(dag_node.uint64s);

        if (node_index < num_nodes) {
            for (int i = 0; i < 4; i++)
                g_dag[node_index].uint4s[i] = dag_node.uint4s[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side API (same interface as original)
// ---------------------------------------------------------------------------

void ethash_generate_dag(
    hash64_t* dag, uint64_t dag_bytes,
    hash64_t* light, uint32_t light_words,
    uint32_t blocks, uint32_t threads,
    cudaStream_t stream, int device)
{
    (void)stream; (void)device;
    uint64_t const work = dag_bytes / sizeof(hash64_t);

    // Multi-threaded DAG generation using std::thread
    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4;
    unsigned num_workers = hw_threads;

    uint64_t items_per_worker = (work + num_workers - 1) / num_workers;
    std::vector<std::thread> workers;
    workers.reserve(num_workers);

    for (unsigned w = 0; w < num_workers; w++) {
        uint64_t start = w * items_per_worker;
        uint64_t count = items_per_worker;
        if (start + count > work) count = (start < work) ? work - start : 0;
        if (count == 0) break;

        workers.emplace_back([=]() {
            ethash_calculate_dag_item_cpu(
                (uint32_t)start, dag, dag_bytes, light, light_words, (uint32_t)count);
        });
    }

    for (auto& t : workers) t.join();
}

void set_constants(hash64_t* _dag, uint32_t _dag_size, hash64_t* _light, uint32_t _light_size) {
    d_dag = _dag;
    d_dag_size = _dag_size;
    d_light = _light;
    d_light_size = _light_size;
}

void get_constants(hash64_t** _dag, uint32_t* _dag_size, hash64_t** _light, uint32_t* _light_size) {
    if (_dag) *_dag = d_dag;
    if (_dag_size) *_dag_size = d_dag_size;
    if (_light) *_light = d_light;
    if (_light_size) *_light_size = d_light_size;
}

void set_header(hash32_t _header) {
    d_header = _header;
}

void set_target(uint64_t _target) {
    d_target = _target;
}
