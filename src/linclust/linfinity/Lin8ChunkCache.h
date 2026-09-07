#ifndef MMSEQS_LIN8CHUNKCACHE_H
#define MMSEQS_LIN8CHUNKCACHE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#ifdef OPENMP
#include <omp.h>
#endif

// Fixed-size file chunks the reader keeps itself, for a filesystem whose page cache does not.
// Two-way set associative: a chunk lives in one of two slots picked by its hash, a used bit
// gives a slot a second chance before it is taken. A caller acquires a chunk, which pins it,
// fills it after a MISS, and releases it after copying out. Nothing waits: a chunk still
// being filled comes back PENDING, pinned, for the caller to keep only if the fill is its own,
// and a pair with both slots pinned answers BUSY.
class ChunkCache {
public:
    static const size_t CHUNK = 64u << 10;
    enum Outcome { HIT, MISS, PENDING, BUSY };

    ChunkCache() : bytes(NULL), hits(0), misses(0), busy(0) {}
    ~ChunkCache() { close(); }

    bool open(size_t wanted);
    void close();
    size_t chunkCount() const { return slots.size(); }

    Outcome acquire(uint32_t file, uint64_t chunk, uint32_t &slot);
    char *memoryOf(uint32_t slot) const { return bytes + (size_t) slot * CHUNK; }
    void filled(uint32_t slot);
    // abandon empties a slot acquired with MISS that will not be filled after all
    void release(uint32_t slot, bool abandon);

    uint64_t hits, misses, busy;

private:
    enum State { EMPTY, FILLING, READY };
    struct Slot {
        uint64_t key;
        uint32_t pins;
        uint8_t state, used;
    };
    static const unsigned STRIPES = 256;

    void lock(uint32_t slot);
    void unlock(uint32_t slot);

    char *bytes;
    std::vector<Slot> slots;
#ifdef OPENMP
    omp_lock_t locks[STRIPES];
#endif
};

#endif
