// Portable shim for powzix/ooz / rarten/ooz on macOS (arm64 + x86_64).
// SSE intrinsics go through SIMDe so the same sources build for both targets.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <stddef.h>
#define __forceinline inline
#define _byteswap_ushort(x) __builtin_bswap16((uint16)(x))
#define _byteswap_ulong(x) __builtin_bswap32((uint32)(x))
#define _byteswap_uint64(x) __builtin_bswap64((uint64)(x))
#define _BitScanForward(dst, x) (*(dst) = __builtin_ctz(x))
#define _BitScanReverse(dst, x) (*(dst) = (__builtin_clz(x) ^ 31))
static inline uint32_t _rotl(uint32_t x, int n) {
  return (((x) << (n)) | ((x) >> (32 - (n))));
}
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/sse.h>
#include <simde/x86/sse2.h>
typedef uint8_t byte;
typedef uint8_t uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef int64_t int64;
typedef int32_t int32;
typedef uint16_t uint16;
typedef int16_t int16;
typedef unsigned int uint;
