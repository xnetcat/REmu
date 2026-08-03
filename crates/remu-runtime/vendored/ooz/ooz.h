/* Public decompress entry for the vendored ooz Kraken decoder (plan P3). */
#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Returns bytes written to dst, or a negative value on failure. */
int ooz_kraken_decompress(const unsigned char *src, size_t src_len,
                          unsigned char *dst, size_t dst_len);
#ifdef __cplusplus
}
#endif
