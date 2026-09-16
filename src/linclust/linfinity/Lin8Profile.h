#ifndef MMSEQS_LIN8PROFILE_H
#define MMSEQS_LIN8PROFILE_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/resource.h>
#include <time.h>
#ifdef OPENMP
#include <omp.h>
#endif

// Opt-in, per-worker counters: no shared atomics in the measured loops.
// Phase scopes measure process CPU; worker scopes measure the calling thread.
// Nested scopes overlap and must not be added to estimate total wall time.
class Lin8Profile {
public:
    enum Metric {
        TOTAL, HASH_LENGTH, HASH_READ, HASH_COMPUTE, HASH_CONCAT, HASH_SORT,
        HASH_GROUP, HASH_COMPARE, HASH_BUCKETS, HASH_COMPARISONS, HASH_PAIR_SORT,
        HASH_DEDUP, HASH_CHAIN, HASH_WRITE, HASH_MERGE, HASH_PARTITION_READ,
        EXTRACT_READ, EXTRACT_SELECT, CHECKPOINT,
        BUCKET, WINDOW, SEGMENT_READ, SEGMENT_DECODE, SEGMENT_MERGE, PAIR_GROUP,
        WRITER_WAIT, WRITER_COPY, WRITER_SORT, WRITER_ENCODE, WRITER_WRITE,
        WRITER_INDEX, WRITER_INDEX_SYNC, WRITER_SYNC, METADATA,
        WINDOW_OUTPUT_ALLOC, WINDOW_READ_PHASE, WINDOW_DECODE_MERGE_PHASE,
        SEGMENT_BUFFER_RESIZE, SEGMENT_SCRATCH_SETUP, SEGMENT_SCRATCH_RESIZE,
        CHECKPOINT_FLUSH_PHASE, CHECKPOINT_SYNC_PHASE, WRITER_WRITEBACK, METRICS
    };
    struct Counter {
        double wall = 0, cpu = 0;
        uint64_t calls = 0, bytes = 0, items = 0, maximum = 0;
        long minor = 0, major = 0;
        std::array<uint64_t, 16> sizes = {};
    };

    Lin8Profile(const char *stage, unsigned int threads)
        : enabled(on("MMSEQS_LIN8_PROFILE")), detail(on("MMSEQS_LIN8_PROFILE_DETAIL")),
          stage(stage), serial(threads), counters(enabled ? threads + 1 : 0) {}

    bool enabled;
    bool detail;

    class Scope {
    public:
        Scope(Lin8Profile *owner, Metric metric, bool process = false,
              uint64_t bytes = 0, uint64_t items = 0, uint64_t key = UINT64_MAX)
            : owner(owner && owner->enabled ? owner : NULL), metric(metric), process(process),
              bytes(bytes), items(items), key(key) {
            if (!this->owner) return;
            counter = &this->owner->at(metric, process);
            wall = now(CLOCK_MONOTONIC);
            cpu = now(process ? CLOCK_PROCESS_CPUTIME_ID : CLOCK_THREAD_CPUTIME_ID);
            if (process) getrusage(RUSAGE_SELF, &usage);
        }
        ~Scope() { stop(); }
        void stop() {
            if (!owner) return;
            const double elapsed = now(CLOCK_MONOTONIC) - wall;
            const double spent = now(process ? CLOCK_PROCESS_CPUTIME_ID : CLOCK_THREAD_CPUTIME_ID) - cpu;
            counter->wall += elapsed; counter->cpu += spent;
            counter->calls++; counter->bytes += bytes; counter->items += items;
            counter->maximum = std::max(counter->maximum, items);
            if (process) {
                struct rusage after;
                getrusage(RUSAGE_SELF, &after);
                counter->minor += after.ru_minflt - usage.ru_minflt;
                counter->major += after.ru_majflt - usage.ru_majflt;
            }
            if (process && owner->detail && key != UINT64_MAX) {
                fprintf(stderr, "LIN8_PROFILE_DETAIL stage=%s metric=%s key=%llu items=%llu wall_s=%.6f cpu_s=%.6f\n",
                        owner->stage, name(metric), (unsigned long long) key, (unsigned long long) items, elapsed, spent);
            }
            owner = NULL;
        }
    private:
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
        Lin8Profile *owner;
        Counter *counter = NULL;
        Metric metric;
        bool process;
        uint64_t bytes, items, key;
        double wall = 0, cpu = 0;
        struct rusage usage = {};
    };

    void note(Metric metric, uint64_t items) {
        if (!enabled) return;
        Counter &c = at(metric, false);
        c.calls++; c.items += items; c.maximum = std::max(c.maximum, items);
        unsigned bin = 0;
        for (uint64_t n = items; n >= 4 && bin < 15; n >>= 2) bin++;
        c.sizes[bin]++;
    }

    void report() const {
        if (!enabled) return;
        fprintf(stderr, "LIN8_PROFILE stage=%s units=seconds,bytes note=nested_scopes_overlap worker_wall_is_summed\n", stage);
        for (int process = 0; process < 2; process++) {
            for (size_t metric = 0; metric < METRICS; metric++) {
                Counter sum;
                const size_t from = process ? serial : 0, until = process ? serial + 1 : serial;
                for (size_t t = from; t < until; t++) {
                    const Counter &c = counters[t][metric];
                    sum.wall += c.wall; sum.cpu += c.cpu; sum.calls += c.calls;
                    sum.bytes += c.bytes; sum.items += c.items; sum.maximum = std::max(sum.maximum, c.maximum);
                    sum.minor += c.minor; sum.major += c.major;
                    for (size_t b = 0; b < sum.sizes.size(); b++) sum.sizes[b] += c.sizes[b];
                }
                if (!sum.calls) continue;
                fprintf(stderr, "LIN8_PROFILE stage=%s scope=%s metric=%s wall_s=%.6f cpu_s=%.6f calls=%llu bytes=%llu items=%llu max_items=%llu minor_faults=%ld major_faults=%ld\n",
                        stage, process ? "phase" : "worker", name((Metric) metric), sum.wall, sum.cpu,
                        (unsigned long long) sum.calls, (unsigned long long) sum.bytes,
                        (unsigned long long) sum.items, (unsigned long long) sum.maximum, sum.minor, sum.major);
                for (size_t b = 0; b < sum.sizes.size(); b++) {
                    if (sum.sizes[b]) fprintf(stderr, "LIN8_PROFILE_SIZE stage=%s metric=%s bin=%zu count=%llu\n",
                                             stage, name((Metric) metric), b, (unsigned long long) sum.sizes[b]);
                }
            }
        }
    }

private:
    static bool on(const char *name) {
        const char *value = getenv(name);
        return value && strcmp(value, "1") == 0;
    }
    static double now(clockid_t clock) {
        struct timespec ts;
        clock_gettime(clock, &ts);
        return ts.tv_sec + ts.tv_nsec * 1e-9;
    }
    Counter &at(Metric metric, bool process) {
        unsigned int thread = 0;
#ifdef OPENMP
        thread = (unsigned int) omp_get_thread_num();
#endif
        return counters[process ? serial : thread][metric];
    }
    static const char *name(Metric metric) {
        static const char *names[] = {
            "total", "hash_length", "hash_read_submit_wait", "hash_compute", "hash_concat", "hash_sort",
            "hash_group", "hash_compare", "hash_buckets", "hash_comparisons", "hash_pair_sort",
            "hash_dedup", "hash_chain", "hash_write", "hash_merge", "hash_partition_read",
            "extract_read_submit_wait", "extract_select_route", "checkpoint",
            "bucket", "window", "segment_read", "segment_decode", "segment_merge", "pair_group",
            "writer_wait", "writer_copy", "writer_sort", "writer_encode", "writer_write",
            "writer_index", "writer_index_sync", "writer_data_sync", "metadata",
            "window_output_alloc", "window_read_phase", "window_decode_merge_phase",
            "segment_buffer_resize", "segment_scratch_setup", "segment_scratch_resize",
            "checkpoint_flush_phase", "checkpoint_sync_phase", "writer_writeback"
        };
        static_assert(sizeof(names) / sizeof(*names) == METRICS, "Missing profiling metric name");
        return names[metric];
    }
    const char *stage;
    size_t serial;
    std::vector<std::array<Counter, METRICS> > counters;
};

#endif
