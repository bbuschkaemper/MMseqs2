#include "Lin8ChunkCache.h"

#include <sys/mman.h>

static const uint64_t NO_KEY = ~uint64_t(0);

// the slots of a set share a stripe, so one lock covers an acquire
void ChunkCache::lock(uint32_t slot) {
#ifdef OPENMP
    omp_set_lock(&locks[(slot / WAYS) % STRIPES]);
#else
    (void) slot;
#endif
}

void ChunkCache::unlock(uint32_t slot) {
#ifdef OPENMP
    omp_unset_lock(&locks[(slot / WAYS) % STRIPES]);
#else
    (void) slot;
#endif
}

bool ChunkCache::open(size_t wanted) {
    close();
    const size_t chunks = wanted / CHUNK / WAYS * WAYS;
    if (chunks < 16 * (size_t) WAYS) {
        return false;
    }
    void *at = mmap(NULL, chunks * CHUNK, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                    -1, 0);
    if (at == MAP_FAILED) {
        return false;
    }
    // chunks land at hashed positions, so a huge page would be faulted in for every chunk
    // touched and a sparsely used cache would take far more than it holds
#ifdef MADV_NOHUGEPAGE
    madvise(at, chunks * CHUNK, MADV_NOHUGEPAGE);
#endif
    bytes = static_cast<char *>(at);
    const Slot empty = {NO_KEY, 0, 0, EMPTY};
    slots.assign(chunks, empty);
    tick = 0;
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
    const uint32_t first = static_cast<uint32_t>((h % (slots.size() / WAYS)) * WAYS);
    lock(first);
    // pins and states are changed without the lock elsewhere, but only ever downward and
    // forward: a slot seen free here stays free until this acquire takes it
    const uint32_t now = __atomic_add_fetch(&tick, 1, __ATOMIC_RELAXED);
    Outcome out = BUSY;
    uint32_t victim = ~0u;
    for (uint32_t i = first; i < first + WAYS; i++) {
        Slot &s = slots[i];
        if (s.key == key) {
            __atomic_add_fetch(&s.pins, 1, __ATOMIC_RELAXED);
            s.stamp = now;
            slot = i;
            out = __atomic_load_n(&s.state, __ATOMIC_ACQUIRE) == READY ? HIT : PENDING;
            victim = ~0u;
            break;
        }
        if (__atomic_load_n(&s.pins, __ATOMIC_RELAXED) != 0 || __atomic_load_n(&s.state, __ATOMIC_ACQUIRE) == FILLING) {
            continue;
        }
        // an empty slot first, then the one acquired longest ago
        if (victim == ~0u || s.state == EMPTY
            || (slots[victim].state != EMPTY && (int32_t) (s.stamp - slots[victim].stamp) < 0)) {
            victim = i;
        }
    }
    if (victim != ~0u) {
        Slot &s = slots[victim];
        s.key = key;
        s.pins = 1;
        s.stamp = now;
        __atomic_store_n(&s.state, (uint8_t) FILLING, __ATOMIC_RELEASE);
        slot = victim;
        out = MISS;
    }
    __atomic_fetch_add(out == MISS ? &misses : out == BUSY ? &busy : &hits, 1, __ATOMIC_RELAXED);
    unlock(first);
    return out;
}

void ChunkCache::filled(uint32_t slot) {
    __atomic_store_n(&slots[slot].state, (uint8_t) READY, __ATOMIC_RELEASE);
}

void ChunkCache::release(uint32_t slot, bool abandon) {
    if (abandon) {
        lock(slot);
        Slot &s = slots[slot];
        s.key = NO_KEY;
        __atomic_store_n(&s.state, (uint8_t) EMPTY, __ATOMIC_RELEASE);
        __atomic_fetch_sub(&s.pins, 1, __ATOMIC_RELEASE);
        unlock(slot);
        return;
    }
    __atomic_fetch_sub(&slots[slot].pins, 1, __ATOMIC_RELEASE);
}
