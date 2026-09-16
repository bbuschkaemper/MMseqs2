#ifndef MMSEQS_LIN8MEMORY_H
#define MMSEQS_LIN8MEMORY_H

#include <cstddef>
#include <cstdint>

namespace Lin8Memory {
inline size_t bitmapBytes(uint64_t ranks) {
    return (ranks / 64 + (ranks % 64 != 0)) * sizeof(uint64_t);
}

// The streaming decider holds one bitmap and bounded row/text buffers. Node 0
// reserves this allowance before opening its aligner's cache.
inline size_t deciderBytes(uint64_t ranks) {
    return bitmapBytes(ranks) + (32u << 20);
}
}

#endif
