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

    ChunkCache() : bytes(NULL), tick(0), hits(0), misses(0), busy(0) {}
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
        uint32_t stamp;
        uint32_t pins;
        uint8_t state;
    };
    static const unsigned STRIPES = 4096;

    void lock(uint32_t slot);
    void unlock(uint32_t slot);

    char *bytes;
    uint32_t tick;
    std::vector<Slot> slots;
#ifdef OPENMP
    omp_lock_t locks[STRIPES];
#endif
};

#endif
