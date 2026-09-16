#ifndef MMSEQS_LIN8CHUNKCACHE_H
#define MMSEQS_LIN8CHUNKCACHE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#ifdef OPENMP
#include <omp.h>
#endif

// Fixed-size file chunks the reader keeps itself, for a filesystem whose page cache does not.
// Set associative, WAYS slots to a set picked by the chunk's hash; the victim is the set's least
// recently acquired slot that no one holds, so a chunk the current slices keep coming back to is
// not thrown out for one that arrived later. A caller acquires a chunk, which pins it, fills it
// after a MISS, and releases it after copying or aligning out of it. Nothing waits: a chunk still
// being filled comes back PENDING, pinned, and a set with every slot pinned answers BUSY.
class ChunkCache {
public:
    // one megabyte: what a parallel file system reads well in one go, sixteen times fewer
    // sequences cut in two by a chunk edge than at 64 KB, and sixteen times fewer acquires
    static const size_t CHUNK = 1u << 20;
    static const unsigned WAYS = 8;
    enum Outcome { HIT, MISS, PENDING, BUSY };

    ChunkCache() : bytes(NULL), stripes(NULL) {}
    ~ChunkCache() { close(); }

    bool open(size_t wanted);
    void close();
    size_t chunkCount() const { return slots.size(); }

    Outcome acquire(uint32_t file, uint64_t chunk, uint32_t &slot);
    char *memoryOf(uint32_t slot) const { return bytes + (size_t) slot * CHUNK; }
    void filled(uint32_t slot);
    // abandon empties a slot acquired with MISS that will not be filled after all
    void release(uint32_t slot, bool abandon);

    // Read after the workers have joined.
    uint64_t count(Outcome outcome) const;

private:
    enum State { EMPTY, FILLING, READY };
    struct Slot {
        uint64_t key;
        uint64_t stamp;
        uint32_t pins;
        uint8_t state;
    };
    static const unsigned STRIPES = 4096;
    struct alignas(64) Stripe {
        uint64_t tick = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t busy = 0;
#ifdef OPENMP
        omp_lock_t lock;
#endif
    };

    void lock(uint32_t slot);
    void unlock(uint32_t slot);

    char *bytes;
    std::vector<Slot> slots;
    // Explicitly aligned allocation also works when the enclosing reader is
    // heap-allocated by a C++11 caller (ordinary new need not honor alignas(64)).
    Stripe *stripes;
};

#endif
