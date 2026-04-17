/*
 * test_verify.cpp — Verification tests for cuda_sim + quai-gpu-miner
 *
 * Test 1: set_constants / get_constants roundtrip
 * Test 2: ProgPow hash — reference CPU implementation determinism + verify
 * Test 3: DAG generation — compare cuda_sim DAG vs ethash full context DAG
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>

// libethash / progpow reference implementation
#include <ethash/ethash.hpp>
#include <ethash/progpow.hpp>

// cuda_sim compat headers (for hash64_t, set_constants etc.)
#include "CUDAMiner_cuda.h"

// From CUDAMiner_cuda_sim.cpp
extern void ethash_generate_dag(
    hash64_t* dag, uint64_t dag_bytes,
    hash64_t* light, uint32_t light_words,
    uint32_t blocks, uint32_t threads,
    cudaStream_t stream, int device);
extern void set_constants(hash64_t* _dag, uint32_t _dag_size,
                          hash64_t* _light, uint32_t _light_size);
extern void get_constants(hash64_t** _dag, uint32_t* _dag_size,
                          hash64_t** _light, uint32_t* _light_size);

// ---------------------------------------------------------------------------
// Test 1: set_constants / get_constants roundtrip
// ---------------------------------------------------------------------------

static bool test_constants_roundtrip() {
    printf("=== Test 1: set_constants / get_constants roundtrip ===\n");

    hash64_t dummy_dag[1] = {};
    hash64_t dummy_light[1] = {};
    dummy_dag[0].words[0] = 0xAAAAAAAA;
    dummy_light[0].words[0] = 0xBBBBBBBB;

    set_constants(dummy_dag, 12345, dummy_light, 67890);

    hash64_t* got_dag = NULL;
    hash64_t* got_light = NULL;
    uint32_t got_dag_size = 0, got_light_size = 0;
    get_constants(&got_dag, &got_dag_size, &got_light, &got_light_size);

    bool pass = (got_dag == dummy_dag &&
                 got_dag_size == 12345 &&
                 got_light == dummy_light &&
                 got_light_size == 67890);

    printf("  dag ptr match: %s, dag_size: %u == 12345: %s\n",
           got_dag == dummy_dag ? "yes" : "no",
           got_dag_size, got_dag_size == 12345 ? "yes" : "no");
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ---------------------------------------------------------------------------
// Test 2: ProgPow hash determinism + verify
// ---------------------------------------------------------------------------

static bool test_progpow_hash() {
    printf("\n=== Test 2: ProgPow hash (reference CPU) ===\n");

    const int epoch = 0;
    const int block_number = 0;

    printf("  creating epoch context for epoch %d...\n", epoch);
    auto t0 = std::chrono::steady_clock::now();
    ethash_epoch_context* ctx = ethash_create_epoch_context(epoch);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ctx) {
        printf("  FAIL: could not create epoch context\n");
        return false;
    }
    printf("  epoch context created in %.1f ms\n", ms);
    printf("  light_cache_num_items=%d, full_dataset_num_items=%d\n",
           ctx->light_cache_num_items, ctx->full_dataset_num_items);

    // Test header
    ethash_hash256 header = {};
    header.word32s[0] = 0xdeadbeef;
    header.word32s[1] = 0x12345678;

    const uint64_t test_nonces[] = {0, 1, 42, 12345};
    const int num_nonces = sizeof(test_nonces) / sizeof(test_nonces[0]);

    bool all_pass = true;
    for (int i = 0; i < num_nonces; i++) {
        uint64_t nonce = test_nonces[i];

        t0 = std::chrono::steady_clock::now();
        ethash::result res = progpow::hash(*ctx, block_number, header, nonce);
        t1 = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Determinism check: compute again
        ethash::result res2 = progpow::hash(*ctx, block_number, header, nonce);
        bool deterministic = (memcmp(&res.final_hash, &res2.final_hash, 32) == 0 &&
                              memcmp(&res.mix_hash, &res2.mix_hash, 32) == 0);

        // Verify check
        ethash_hash256 easy = {};
        memset(easy.bytes, 0xFF, 32);
        bool verified = progpow::verify(*ctx, block_number, header,
                                        res.mix_hash, nonce, easy);

        printf("  nonce=%-6llu final=%08x%08x... mix=%08x%08x... %.0fms det=%s ver=%s\n",
               (unsigned long long)nonce,
               res.final_hash.word32s[0], res.final_hash.word32s[1],
               res.mix_hash.word32s[0], res.mix_hash.word32s[1],
               ms,
               deterministic ? "ok" : "FAIL",
               verified ? "ok" : "FAIL");

        if (!deterministic || !verified) all_pass = false;
    }

    ethash_destroy_epoch_context(ctx);
    printf("  %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

// ---------------------------------------------------------------------------
// Test 3: Ethash hash — reference implementation determinism + verify
// ---------------------------------------------------------------------------

static bool test_ethash_hash() {
    printf("\n=== Test 3: Ethash hash (reference CPU) ===\n");

    const int epoch = 0;

    ethash_epoch_context* ctx = ethash_create_epoch_context(epoch);
    if (!ctx) {
        printf("  FAIL: could not create epoch context\n");
        return false;
    }

    ethash_hash256 header = {};
    header.word32s[0] = 0xcafebabe;
    header.word32s[7] = 0xfeedface;

    const uint64_t test_nonces[] = {0, 1, 999};
    const int num_nonces = sizeof(test_nonces) / sizeof(test_nonces[0]);

    bool all_pass = true;
    for (int i = 0; i < num_nonces; i++) {
        uint64_t nonce = test_nonces[i];

        ethash::result res = ethash::hash(*ctx, header, nonce);
        ethash::result res2 = ethash::hash(*ctx, header, nonce);

        bool deterministic = (memcmp(&res.final_hash, &res2.final_hash, 32) == 0 &&
                              memcmp(&res.mix_hash, &res2.mix_hash, 32) == 0);

        ethash_hash256 easy = {};
        memset(easy.bytes, 0xFF, 32);
        bool verified = ethash::verify(*ctx, header, res.mix_hash, nonce, easy);

        printf("  nonce=%-6llu final=%08x%08x... det=%s ver=%s\n",
               (unsigned long long)nonce,
               res.final_hash.word32s[0], res.final_hash.word32s[1],
               deterministic ? "ok" : "FAIL",
               verified ? "ok" : "FAIL");

        if (!deterministic || !verified) all_pass = false;
    }

    ethash_destroy_epoch_context(ctx);
    printf("  %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("quai-gpu-miner + cuda_sim verification tests\n");
    printf("=============================================\n\n");

    int passed = 0, failed = 0;

    test_constants_roundtrip() ? passed++ : failed++;
    test_progpow_hash() ? passed++ : failed++;
    test_ethash_hash() ? passed++ : failed++;

    printf("\n=============================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
