#include "Lin8ChunkCache.h"

#include <sys/mman.h>

static const uint64_t NO_KEY = ~uint64_t(0);

// both slots of a pair share a stripe, so one lock covers an acquire
void ChunkCache::lock(uint32_t slot) {
#ifdef OPENMP
    omp_set_lock(&locks[(slot / 2) % STRIPES]);
#else
    (void) slot;
#endif
}

void ChunkCache::unlock(uint32_t slot) {
#ifdef OPENMP
    omp_unset_lock(&locks[(slot / 2) % STRIPES]);
#else
    (void) slot;
#endif
}

bool ChunkCache::open(size_t wanted) {
    close();
    const size_t chunks = wanted / CHUNK / 2 * 2;
    if (chunks < 4 * STRIPES) {
        return false;
    }
    void *at = mmap(NULL, chunks * CHUNK, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                    -1, 0);
    if (at == MAP_FAILED) {
        return false;
    }
    // chunks land at hashed positions, so a huge page would be faulted in for every 64 KB
    // touched and the cache would take 32 times what it holds
#ifdef MADV_NOHUGEPAGE
    madvise(at, chunks * CHUNK, MADV_NOHUGEPAGE);
#endif
    bytes = static_cast<char *>(at);
    const Slot empty = {NO_KEY, 0, EMPTY, 0};
    slots.assign(chunks, empty);
#ifdef OPENMP
    for (unsigned i = 0; i < STRIPES; i++) {
        omp_init_lock(&locks[i]);
    }
#endif
    return true;
}

void ChunkCache::close() {
    if (bytes == NULL) {
        return;
    }
    munmap(bytes, slots.size() * CHUNK);
    bytes = NULL;
    slots.clear();
#ifdef OPENMP
    for (unsigned i = 0; i < STRIPES; i++) {
        omp_destroy_lock(&locks[i]);
    }
#endif
}

ChunkCache::Outcome ChunkCache::acquire(uint32_t file, uint64_t chunk, uint32_t &slot) {
    const uint64_t key = (static_cast<uint64_t>(file) << 48) | chunk;
    uint64_t h = key * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    const uint32_t first = static_cast<uint32_t>((h % (slots.size() / 2)) * 2);
    lock(first);
    Outcome out = BUSY;
    uint32_t victim = ~0u;
    for (uint32_t i = first; i < first + 2; i++) {
        Slot &s = slots[i];
        if (s.key == key) {
            s.pins++;
            s.used = 1;
            slot = i;
            out = s.state == READY ? HIT : PENDING;
            victim = ~0u;
            break;
        }
        if (s.pins != 0 || s.state == FILLING) {
            continue;
        }
        // an empty slot, or one not used since the last time it was passed over
        if (s.state == EMPTY || s.used == 0) {
            if (victim == ~0u || slots[victim].state != EMPTY) {
                victim = i;
            }
        } else {
            s.used = 0;
            if (victim == ~0u) {
                victim = i;
            }
        }
    }
    if (victim != ~0u) {
        Slot &s = slots[victim];
        s.key = key;
        s.pins = 1;
        s.state = FILLING;
        s.used = 1;
        slot = victim;
        out = MISS;
    }
    __atomic_fetch_add(out == MISS ? &misses : out == BUSY ? &busy : &hits, 1, __ATOMIC_RELAXED);
    unlock(first);
    return out;
}

void ChunkCache::filled(uint32_t slot) {
    lock(slot);
    slots[slot].state = READY;
    unlock(slot);
}

void ChunkCache::release(uint32_t slot, bool abandon) {
    lock(slot);
    Slot &s = slots[slot];
    if (abandon) {
        s.key = NO_KEY;
        s.state = EMPTY;
    }
    s.pins--;
    unlock(slot);
}
