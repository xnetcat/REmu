#include "stdafx.h"
#include "ooz.h"

// C++ definition from kraken.cpp (library cut).
int Kraken_Decompress(const byte *src, size_t src_len, byte *dst, size_t dst_len);

extern "C" int ooz_kraken_decompress(const unsigned char *src, size_t src_len,
                                     unsigned char *dst, size_t dst_len) {
  return Kraken_Decompress(src, src_len, dst, dst_len);
}
