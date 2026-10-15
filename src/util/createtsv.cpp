#include "Parameters.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "IndexReader.h"
#include "FileUtil.h"
#include "MMseqsNode.h"
#include "BatchReader.h"
#include "FastSort.h"
#include "Timer.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#ifdef OPENMP
#include <omp.h>

#endif

#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

// --createdb-mode 3: a record of the first shuffle; a rep record is (rep, cluster size), a member record (member, rep)
struct KeyPair {
    DBKeyType key;
    DBKeyType value;
};

// a record of the second shuffle, (rep, member, name), as it lies in a bucket that was read back
struct NamedMember {
    DBKeyType rep;
    DBKeyType member;
    const char *name;
};

// the rep line first, then the members in key order
static bool compareNamedMember(const NamedMember &a, const NamedMember &b) {
    if (a.rep != b.rep) {
        return a.rep < b.rep;
    }
    if ((a.member == a.rep) != (b.member == b.rep)) {
        return a.member == a.rep;
    }
    return a.member < b.member;
}

static void writeFully(int fd, const std::string &file, const char *data, size_t size, size_t offset) {
    for (size_t written = 0; written < size; ) {
        const ssize_t count = pwrite(fd, data + written, size - written, offset + written);
        if (count <= 0) {
            Debug(Debug::ERROR) << "Cannot write to " << file << "\n";
            EXIT(EXIT_FAILURE);
        }
        written += count;
    }
}

static void readFully(int fd, const std::string &file, char *data, size_t size, size_t offset) {
    for (size_t done = 0; done < size; ) {
        const ssize_t count = pread(fd, data + done, size - done, offset + done);
        if (count <= 0) {
            Debug(Debug::ERROR) << "Cannot read from " << file << "\n";
            EXIT(EXIT_FAILURE);
        }
        done += count;
    }
}

// moves records between ranks through files on the shared file system: every rank appends runs to its own file,
// each run grouped by destination, and indexes the destination, rank, offset and length of every non-empty piece
class Shuffle {
public:
    struct Segment {
        size_t destination;
        unsigned int rank;
        size_t offset;
        size_t length;
    };

    explicit Shuffle(const std::string &prefix) : prefix(prefix), file(prefix + "." + SSTR(MMseqsNode::rank)), written(0) {
        fd = open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd < 0) {
            Debug(Debug::ERROR) << "Cannot open " << file << "\n";
            EXIT(EXIT_FAILURE);
        }
    }

    // destination d is data[starts[d], starts[d + 1])
    void append(const char *data, const std::vector<size_t> &starts) {
        writeFully(fd, file, data, starts.back(), written);
        for (size_t d = 0; d + 1 < starts.size(); d++) {
            if (starts[d + 1] > starts[d]) {
                index += SSTR(d) + "\t" + SSTR(MMseqsNode::rank) + "\t" + SSTR(written + starts[d]) + "\t" + SSTR(starts[d + 1] - starts[d]) + "\n";
            }
        }
        written += starts.back();
    }

    // returns once every rank has appended all its runs
    void finish() {
        if (fsync(fd) != 0 || close(fd) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << file << "\n";
            EXIT(EXIT_FAILURE);
        }
        MMseqsNode::publishFile(prefix + ".index." + SSTR(MMseqsNode::rank), index);
        if (MMseqsNode::isMaster()) {
            MMseqsNode::publishFile(prefix + ".index", MMseqsNode::gatherFiles(prefix + ".index."));
        }
        std::istringstream all(MMseqsNode::waitForFile(prefix + ".index"));
        Segment segment;
        while (all >> segment.destination >> segment.rank >> segment.offset >> segment.length) {
            segments.push_back(segment);
        }
        std::sort(segments.begin(), segments.end(), [](const Segment &a, const Segment &b) {
            return a.destination < b.destination;
        });
    }

    size_t bytes(size_t from, size_t to) const {
        size_t sum = 0;
        for (size_t i = first(from); i < segments.size() && segments[i].destination < to; i++) {
            sum += segments[i].length;
        }
        return sum;
    }

    // reads the destinations [from, to) that every rank appended into data
    void read(size_t from, size_t to, std::vector<char> &data, BatchReader &reader) const {
        data.resize(bytes(from, to));
        std::vector<int> fds(MMseqsNode::count, -1);
        std::vector<BatchReader::Request> requests;
        size_t at = 0;
        for (size_t i = first(from); i < segments.size() && segments[i].destination < to; i++) {
            const Segment &segment = segments[i];
            if (fds[segment.rank] < 0) {
                const std::string name = prefix + "." + SSTR(segment.rank);
                fds[segment.rank] = open(name.c_str(), O_RDONLY);
                if (fds[segment.rank] < 0) {
                    Debug(Debug::ERROR) << "Cannot open " << name << "\n";
                    EXIT(EXIT_FAILURE);
                }
            }
            BatchReader::Request request = { fds[segment.rank], data.data() + at, segment.offset, segment.length, segment.length };
            requests.push_back(request);
            at += segment.length;
        }
        reader.read(requests, prefix.c_str());
        for (size_t rank = 0; rank < fds.size(); rank++) {
            if (fds[rank] >= 0) {
                close(fds[rank]);
            }
        }
    }

    void remove() const {
        for (unsigned int rank = 0; rank < MMseqsNode::count; rank++) {
            FileUtil::remove((prefix + "." + SSTR(rank)).c_str());
        }
        FileUtil::remove((prefix + ".index").c_str());
    }

private:
    size_t first(size_t destination) const {
        return std::lower_bound(segments.begin(), segments.end(), destination, [](const Segment &segment, size_t d) {
            return segment.destination < d;
        }) - segments.begin();
    }

    std::string prefix;
    std::string file;
    int fd;
    size_t written;
    std::string index;
    std::vector<Segment> segments;
};

// counting sort of key pairs by destination; starts are byte offsets for Shuffle::append
static void groupPairs(const std::vector<KeyPair> &records, const std::vector<unsigned int> &destinations, size_t destinationCount,
                       std::vector<KeyPair> &grouped, std::vector<size_t> &starts) {
    std::vector<size_t> at(destinationCount + 1, 0);
    for (size_t i = 0; i < destinations.size(); i++) {
        at[destinations[i] + 1]++;
    }
    for (size_t d = 0; d < destinationCount; d++) {
        at[d + 1] += at[d];
    }
    starts.resize(destinationCount + 1);
    for (size_t d = 0; d <= destinationCount; d++) {
        starts[d] = at[d] * sizeof(KeyPair);
    }
    grouped.resize(records.size());
    for (size_t i = 0; i < records.size(); i++) {
        grouped[at[destinations[i]]++] = records[i];
    }
}

// --createdb-mode 3 on every rank of a SLURM step: names are joined in three phases through the shared file system.
// 1. every cluster entry tells the owner of each key its rep, and the owner of the rep the cluster size
// 2. owners name their keys and send (rep, key, name) to the bucket of the rep
// 3. rank 0 places the buckets in the TSV, every rank sorts and writes its buckets
static int createtsvLinfinity(Parameters &par) {
    MMseqsNode::init();
    if (par.filenames.size() != 4 || par.db1 != par.db2) {
        Debug(Debug::ERROR) << "createtsv on a --createdb-mode 3 database needs <sequenceDB> <sequenceDB> <clusterDB> <tsvFile>, with the same sequence database twice\n";
        EXIT(EXIT_FAILURE);
    }
    if (par.fullHeader || par.firstSeqRepr || par.targetTsvColumn != 1 || par.idxSeqSrc != 0 || par.dbOut) {
        Debug(Debug::ERROR) << "--full-header, --first-seq-as-repr, --target-column, --idx-seq-src and --db-output are not supported for --createdb-mode 3 databases\n";
        EXIT(EXIT_FAILURE);
    }
    const int clusterType = FileUtil::parseDbType(par.db3.c_str());
    if (Parameters::isEqualDbtype(clusterType, Parameters::DBTYPE_CLUSTER_RES) == false
        || DBReader<DBKeyType>::getExtendedDbtype(clusterType) != Parameters::DBTYPE_EXTENDED_LINFINITY) {
        Debug(Debug::ERROR) << par.db3 << " is not a --createdb-mode 3 clustering\n";
        EXIT(EXIT_FAILURE);
    }

    const unsigned int rank = MMseqsNode::rank;
    const size_t nodes = MMseqsNode::count;
    const int threads = par.threads;
    DBReader<DBKeyType> headers(par.hdr1.c_str(), par.hdr1Index.c_str(), threads,
                                DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    headers.open(DBReader<DBKeyType>::NOSORT);
    const size_t keys = headers.getSize();
    const DBReader<DBKeyType>::Index *bins = headers.getIndex();
    const size_t binCount = headers.getIndexSize();
    BatchReader batchReader(threads);
    Timer timer;

    // rank r owns the keys [owners[r], owners[r + 1]): whole bins, about the same number of keys for every rank
    std::vector<size_t> owners(nodes + 1, keys);
    owners[0] = 0;
    for (size_t bin = 0, next = 1; bin < binCount && next < nodes; bin++) {
        while (next < nodes && bins[bin].id * nodes >= next * keys) {
            owners[next++] = bins[bin].id;
        }
    }
    if (MMseqsNode::isMaster()) {
        size_t maxKeys = 0;
        for (size_t r = 0; r < nodes; r++) {
            maxKeys = std::max(maxKeys, owners[r + 1] - owners[r]);
        }
        Debug(Debug::INFO) << "Largest rank key range: " << maxKeys << " keys, average " << keys / nodes << "\n";
    }
    const auto ownerOf = [&owners](size_t key) -> size_t {
        return std::upper_bound(owners.begin(), owners.end(), key) - owners.begin() - 1;
    };
    // buckets are key ranges of reps, so their order is the order of the TSV
    const size_t bucketCount = 64 * nodes;
    const size_t bucketKeys = std::max<size_t>(1, (keys + bucketCount - 1) / bucketCount);
    const size_t memory = Util::computeMemory(par.splitMemoryLimit);
    // the job and step id keep the files of a crashed run away from the next one
    const char *jobId = getenv("SLURM_JOB_ID");
    const char *stepId = getenv("SLURM_STEP_ID");
    const std::string prefix = par.db4 + ".linfinity." + (jobId != NULL ? jobId : "local") + "." + (stepId != NULL ? stepId : "0");

    // phase 1
    Shuffle pairs(prefix + ".pairs");
    {
        std::vector<size_t> rowOffset;
        std::vector<size_t> rowLength;
        std::vector<unsigned int> rowFile;
        size_t total = 0;
        std::ifstream index(par.db3Index.c_str());
        if (index.is_open() == false) {
            Debug(Debug::ERROR) << "Cannot open " << par.db3Index << "\n";
            EXIT(EXIT_FAILURE);
        }
        std::string line;
        while (std::getline(index, line)) {
            size_t offset;
            size_t length;
            unsigned int file;
            // a last line without its newline is a torn write
            if (index.eof() || sscanf(line.c_str(), "%zu %zu %u", &offset, &length, &file) != 3) {
                Debug(Debug::ERROR) << "Malformed line " << rowOffset.size() + 1 << " in " << par.db3Index << "\n";
                EXIT(EXIT_FAILURE);
            }
            rowOffset.push_back(offset);
            rowLength.push_back(length);
            rowFile.push_back(file);
            total += length;
        }
        // a key takes at least two bytes of text, then a record, its destination and its grouped copy
        const size_t chunk = std::max<size_t>(memory / 40, 4096);
        std::vector<char> text;
        std::vector<KeyPair> records;
        std::vector<unsigned int> destinations;
        std::vector<KeyPair> grouped;
        std::vector<size_t> starts;
        size_t at = 0;
        for (size_t row = 0; row < rowOffset.size(); at += rowLength[row], row++) {
            // a rank reads the rows that start in its share of all cluster bytes
            if (at * nodes < rank * total || at * nodes >= (rank + 1) * total) {
                continue;
            }
            const std::string dataFile = par.db3 + "." + SSTR(rowFile[row]);
            const int fd = open(dataFile.c_str(), O_RDONLY);
            if (fd < 0) {
                Debug(Debug::ERROR) << "Cannot open " << dataFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            for (size_t done = 0; done < rowLength[row]; ) {
                const size_t want = std::min(chunk, rowLength[row] - done);
                text.resize(want);
                readFully(fd, dataFile, text.data(), want, rowOffset[row] + done);
                // whole entries only, the rest is read again with the next chunk
                size_t end = want;
                while (end > 0 && text[end - 1] != '\0') {
                    end--;
                }
                if (end == 0) {
                    Debug(Debug::ERROR) << "The cluster entry at byte " << rowOffset[row] + done << " of " << dataFile
                                        << " is not terminated or longer than " << chunk << " bytes; raise --split-memory-limit\n";
                    EXIT(EXIT_FAILURE);
                }
                records.clear();
                destinations.clear();
                for (const char *entry = text.data(); entry < text.data() + end; ) {
                    const char *entryEnd = entry + strlen(entry);
                    DBKeyType rep = 0;
                    size_t members = 0;
                    for (const char *line = entry; line < entryEnd; ) {
                        const char *lineEnd = static_cast<const char *>(memchr(line, '\n', entryEnd - line));
                        if (lineEnd == NULL) {
                            lineEnd = entryEnd;
                        }
                        if (lineEnd > line) {
                            if (*line < '0' || *line > '9') {
                                Debug(Debug::ERROR) << "Malformed key in " << dataFile << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                            const size_t key = Util::fast_atoi<size_t>(line);
                            if (key >= keys) {
                                Debug(Debug::ERROR) << "Key " << key << " of " << par.db3 << " is not in " << par.db1 << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                            if (members == 0) {
                                rep = key;
                            } else {
                                KeyPair pair = { static_cast<DBKeyType>(key), rep };
                                records.push_back(pair);
                                destinations.push_back(2 * ownerOf(key) + 1);
                            }
                            members++;
                        }
                        line = lineEnd + 1;
                    }
                    if (members > 0) {
                        KeyPair pair = { rep, static_cast<DBKeyType>(members) };
                        records.push_back(pair);
                        destinations.push_back(2 * ownerOf(rep));
                    }
                    entry = entryEnd + 1;
                }
                groupPairs(records, destinations, 2 * nodes, grouped, starts);
                pairs.append(reinterpret_cast<const char *>(grouped.data()), starts);
                done += end;
            }
            close(fd);
        }
        pairs.finish();
    }
    Debug(Debug::INFO) << "Phase 1 moved " << pairs.bytes(0, 2 * nodes) << " bytes in " << timer.lap() << "\n";

    // phase 2
    const size_t ownFrom = owners[rank];
    const size_t ownTo = owners[rank + 1];
    const size_t ownerBytes = 2 * (ownTo - ownFrom) * sizeof(DBKeyType);
    const size_t incomingBytes = std::max(pairs.bytes(2 * rank, 2 * rank + 1), pairs.bytes(2 * rank + 1, 2 * rank + 2));
    if (ownerBytes + incomingBytes >= memory) {
        Debug(Debug::ERROR) << "The " << ownTo - ownFrom << " keys of rank " << rank << " need " << ownerBytes + incomingBytes << " bytes, more than --split-memory-limit; use more ranks\n";
        EXIT(EXIT_FAILURE);
    }
    std::vector<DBKeyType> repOf(ownTo - ownFrom, DB_KEY_INVALID);
    std::vector<DBKeyType> clusterSize(ownTo - ownFrom, 0);
    {
        std::vector<char> incoming;
        for (size_t kind = 0; kind < 2; kind++) {
            pairs.read(2 * rank + kind, 2 * rank + kind + 1, incoming, batchReader);
            for (size_t pos = 0; pos < incoming.size(); pos += sizeof(KeyPair)) {
                KeyPair pair;
                memcpy(&pair, incoming.data() + pos, sizeof(KeyPair));
                const size_t local = pair.key - ownFrom;
                if (repOf[local] != DB_KEY_INVALID) {
                    Debug(Debug::ERROR) << "Key " << pair.key << " is in two clusters of " << par.db3 << "\n";
                    EXIT(EXIT_FAILURE);
                }
                repOf[local] = (kind == 0) ? pair.key : pair.value;
                if (kind == 0) {
                    clusterSize[local] = pair.value;
                }
            }
        }
    }
    for (size_t local = 0; local < repOf.size(); local++) {
        if (repOf[local] == DB_KEY_INVALID) {
            repOf[local] = ownFrom + local;
            clusterSize[local] = 1;
        }
    }
    const size_t chunkBudget = memory - ownerBytes;
    Shuffle names(prefix + ".names");
    std::vector<size_t> tsvBytes(bucketCount, 0);
    size_t bin = 0;
    while (bin < binCount && bins[bin].id < ownFrom) {
        bin++;
    }
    while (bin < binCount && bins[bin].id < ownTo) {
        const size_t chunkFrom = bins[bin].id;
        size_t chunkTo = chunkFrom;
        size_t chunkBytes = 0;
        // names, their offsets and two copies of the records, with room for the thread buffers to grow
        while (bin < binCount && bins[bin].id < ownTo) {
            const size_t binTo = (bin + 1 < binCount) ? bins[bin + 1].id : keys;
            const size_t need = 4 * bins[bin].length + 64 * (binTo - bins[bin].id);
            if (chunkTo > chunkFrom && chunkBytes + need > chunkBudget) {
                break;
            }
            chunkBytes += need;
            chunkTo = binTo;
            bin++;
        }
        if (chunkBytes > chunkBudget) {
            Debug(Debug::ERROR) << "The names of keys " << chunkFrom << " to " << chunkTo << " need " << chunkBytes << " bytes, more than the "
                                << chunkBudget << " left; raise --split-memory-limit or use more ranks\n";
            EXIT(EXIT_FAILURE);
        }
        headers.loadData(chunkFrom, chunkTo - chunkFrom);
        std::vector<std::string> threadRecords(threads);
        std::vector<std::vector<size_t>> recordBytes(threads, std::vector<size_t>(bucketCount, 0));
        std::vector<std::vector<size_t>> threadTsvBytes(threads, std::vector<size_t>(bucketCount, 0));
#pragma omp parallel num_threads(threads)
        {
            unsigned int thread = 0;
#ifdef OPENMP
            thread = (unsigned int) omp_get_thread_num();
#endif
#pragma omp for schedule(static)
            for (size_t key = chunkFrom; key < chunkTo; key++) {
                const std::string name = Util::parseFastaHeader(headers.getData(key, thread));
                const DBKeyType rep = repOf[key - ownFrom];
                const DBKeyType member = key;
                const size_t bucket = rep / bucketKeys;
                std::string &out = threadRecords[thread];
                out.append(reinterpret_cast<const char *>(&rep), sizeof(DBKeyType));
                out.append(reinterpret_cast<const char *>(&member), sizeof(DBKeyType));
                out.append(name.c_str(), name.size() + 1);
                recordBytes[thread][bucket] += 2 * sizeof(DBKeyType) + name.size() + 1;
                // a line is "repName\tmemberName\n": the member part on every line, the rep part once per cluster at its rep
                threadTsvBytes[thread][bucket] += name.size() + 1;
                if (rep == member) {
                    threadTsvBytes[thread][bucket] += clusterSize[key - ownFrom] * (name.size() + 1);
                }
            }
        }
        std::vector<size_t> starts(bucketCount + 1, 0);
        for (size_t b = 0; b < bucketCount; b++) {
            starts[b + 1] = starts[b];
            for (int t = 0; t < threads; t++) {
                starts[b + 1] += recordBytes[t][b];
            }
        }
        // thread t places its records of bucket b after those of the threads before it
        for (size_t b = 0; b < bucketCount; b++) {
            size_t at = starts[b];
            for (int t = 0; t < threads; t++) {
                const size_t bytes = recordBytes[t][b];
                recordBytes[t][b] = at;
                at += bytes;
                tsvBytes[b] += threadTsvBytes[t][b];
            }
        }
        std::vector<char> grouped(starts.back());
#pragma omp parallel for schedule(static, 1) num_threads(threads)
        for (int t = 0; t < threads; t++) {
            const std::string &records = threadRecords[t];
            for (size_t pos = 0; pos < records.size(); ) {
                DBKeyType rep;
                memcpy(&rep, records.data() + pos, sizeof(DBKeyType));
                const size_t length = 2 * sizeof(DBKeyType) + strlen(records.data() + pos + 2 * sizeof(DBKeyType)) + 1;
                size_t &at = recordBytes[t][rep / bucketKeys];
                memcpy(grouped.data() + at, records.data() + pos, length);
                at += length;
                pos += length;
            }
        }
        names.append(grouped.data(), starts);
    }
    // the last loaded names are not needed in phase 3
    headers.close();
    std::vector<DBKeyType>().swap(repOf);
    std::vector<DBKeyType>().swap(clusterSize);
    std::string bytesText;
    for (size_t b = 0; b < bucketCount; b++) {
        if (tsvBytes[b] > 0) {
            bytesText += SSTR(b) + "\t" + SSTR(tsvBytes[b]) + "\n";
        }
    }
    MMseqsNode::publishFile(prefix + ".bytes." + SSTR(rank), bytesText);
    names.finish();
    Debug(Debug::INFO) << "Phase 2 moved " << names.bytes(0, bucketCount) << " bytes in " << timer.lap() << "\n";

    // phase 3
    const std::string planFile = prefix + ".plan";
    if (MMseqsNode::isMaster()) {
        std::vector<size_t> bucketBytes(bucketCount, 0);
        std::istringstream gathered(MMseqsNode::gatherFiles(prefix + ".bytes."));
        size_t bucket;
        size_t bytes;
        while (gathered >> bucket >> bytes) {
            bucketBytes[bucket] += bytes;
        }
        size_t total = 0;
        size_t maxBucketBytes = 0;
        for (size_t b = 0; b < bucketCount; b++) {
            total += bucketBytes[b];
            maxBucketBytes = std::max(maxBucketBytes, bucketBytes[b]);
        }
        Debug(Debug::INFO) << "Largest bucket: " << maxBucketBytes << " of " << total << " TSV bytes\n";
        std::string plan;
        size_t at = 0;
        for (size_t b = 0; b < bucketCount; b++) {
            // runs of buckets of about equal bytes keep the part of every rank in one piece of the file
            const size_t owner = (total == 0) ? 0 : std::min(nodes - 1, at * nodes / total);
            plan += SSTR(b) + "\t" + SSTR(at) + "\t" + SSTR(bucketBytes[b]) + "\t" + SSTR(owner) + "\n";
            at += bucketBytes[b];
        }
        const int fd = open(par.db4.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd < 0 || ftruncate(fd, total) != 0 || fsync(fd) != 0 || close(fd) != 0) {
            Debug(Debug::ERROR) << "Cannot create " << par.db4 << "\n";
            EXIT(EXIT_FAILURE);
        }
        // every rank published its names index only after it had read its pairs
        pairs.remove();
        MMseqsNode::publishFile(planFile, plan);
    }
    std::vector<size_t> myBuckets;
    std::vector<size_t> myOffsets;
    std::vector<size_t> myBytes;
    {
        std::istringstream plan(MMseqsNode::waitForFile(planFile));
        size_t bucket;
        size_t offset;
        size_t bytes;
        size_t owner;
        while (plan >> bucket >> offset >> bytes >> owner) {
            if (owner == rank && bytes > 0) {
                myBuckets.push_back(bucket);
                myOffsets.push_back(offset);
                myBytes.push_back(bytes);
            }
        }
    }
    const int out = open(par.db4.c_str(), O_WRONLY);
    if (out < 0) {
        Debug(Debug::ERROR) << "Cannot open " << par.db4 << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::vector<char> data;
    std::vector<NamedMember> members;
    std::string lines;
    for (size_t i = 0; i < myBuckets.size(); i++) {
        const size_t bucket = myBuckets[i];
        const size_t recordBytes = names.bytes(bucket, bucket + 1);
        // shortcut: one bucket must fit in memory; split buckets by rep range if a single cluster region outgrows a rank
        // the records, an entry of at most 24 bytes for every record of at least 17, and the lines
        if (3 * recordBytes > memory) {
            Debug(Debug::ERROR) << "Bucket " << bucket << " needs about " << 3 * recordBytes << " bytes, more than --split-memory-limit; use more ranks\n";
            EXIT(EXIT_FAILURE);
        }
        names.read(bucket, bucket + 1, data, batchReader);
        members.clear();
        members.reserve(recordBytes / (2 * sizeof(DBKeyType) + 1));
        for (size_t pos = 0; pos < data.size(); ) {
            NamedMember member;
            memcpy(&member.rep, data.data() + pos, sizeof(DBKeyType));
            memcpy(&member.member, data.data() + pos + sizeof(DBKeyType), sizeof(DBKeyType));
            member.name = data.data() + pos + 2 * sizeof(DBKeyType);
            pos += 2 * sizeof(DBKeyType) + strlen(member.name) + 1;
            members.push_back(member);
        }
        SORT_PARALLEL(members.begin(), members.end(), compareNamedMember);
        size_t written = 0;
        const auto flush = [&]() {
            if (written + lines.size() > myBytes[i]) {
                Debug(Debug::ERROR) << "Bucket " << bucket << " has more than its " << myBytes[i] << " bytes\n";
                EXIT(EXIT_FAILURE);
            }
            writeFully(out, par.db4, lines.data(), lines.size(), myOffsets[i] + written);
            written += lines.size();
            lines.clear();
        };
        for (size_t first = 0; first < members.size(); ) {
            const NamedMember &rep = members[first];
            if (rep.member != rep.rep) {
                Debug(Debug::ERROR) << "The cluster of key " << rep.rep << " has no line for its rep\n";
                EXIT(EXIT_FAILURE);
            }
            size_t next = first;
            for (; next < members.size() && members[next].rep == rep.rep; next++) {
                lines.append(rep.name);
                lines.push_back('\t');
                lines.append(members[next].name);
                lines.push_back('\n');
                if (lines.size() >= (64 << 20)) {
                    flush();
                }
            }
            first = next;
        }
        flush();
        if (written != myBytes[i]) {
            Debug(Debug::ERROR) << "Bucket " << bucket << " wrote " << written << " bytes instead of " << myBytes[i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    if (fsync(out) != 0 || close(out) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << par.db4 << "\n";
        EXIT(EXIT_FAILURE);
    }
    Debug(Debug::INFO) << "Phase 3 wrote " << myBuckets.size() << " buckets in " << timer.lap() << "\n";

    MMseqsNode::publishFile(prefix + ".done." + SSTR(rank), "");
    if (MMseqsNode::isMaster()) {
        MMseqsNode::gatherFiles(prefix + ".done.");
        names.remove();
        FileUtil::remove(planFile.c_str());
    }
    return EXIT_SUCCESS;
}

int createtsv(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, Parameters::PARSE_VARIADIC, 0);
    if (DBReader<DBKeyType>::getExtendedDbtype(FileUtil::parseDbType(par.db1.c_str())) & Parameters::DBTYPE_EXTENDED_LINFINITY) {
        return createtsvLinfinity(par);
    }

    bool queryNucs = Parameters::isEqualDbtype(FileUtil::parseDbType(par.db1.c_str()), Parameters::DBTYPE_NUCLEOTIDES);
    bool targetNucs = Parameters::isEqualDbtype(FileUtil::parseDbType(par.db2.c_str()), Parameters::DBTYPE_NUCLEOTIDES);
    const bool touch = (par.preloadMode != Parameters::PRELOAD_MODE_MMAP);
    int queryHeaderType = (queryNucs) ? IndexReader::SRC_HEADERS : IndexReader::HEADERS;
    queryHeaderType = (par.idxSeqSrc == 0) ? queryHeaderType :  (par.idxSeqSrc == 1) ?  IndexReader::HEADERS : IndexReader::SRC_HEADERS;
    IndexReader qDbrHeader(par.db1, par.threads, queryHeaderType, (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0);
    IndexReader * tDbrHeader=NULL;
    DBReader<DBKeyType> * queryDB = qDbrHeader.sequenceReader;
    DBReader<DBKeyType> * targetDB = NULL;
    bool sameDB = (par.db2.compare(par.db1) == 0);
    const bool hasTargetDB = par.filenames.size() > 3;
    DBReader<DBKeyType>::Index * qHeaderIndex = qDbrHeader.sequenceReader->getIndex();
    DBReader<DBKeyType>::Index * tHeaderIndex = NULL;

    if (hasTargetDB) {
        if (sameDB) {
            tDbrHeader = &qDbrHeader;
            tHeaderIndex = qHeaderIndex;
            targetDB = queryDB;
        } else {

            int targetHeaderType = (targetNucs) ? IndexReader::SRC_HEADERS : IndexReader::HEADERS;
            targetHeaderType = (par.idxSeqSrc == 0) ? targetHeaderType :  (par.idxSeqSrc == 1) ?  IndexReader::HEADERS : IndexReader::SRC_HEADERS;

            tDbrHeader = new IndexReader(par.db2, par.threads, targetHeaderType, touch);
            tHeaderIndex = tDbrHeader->sequenceReader->getIndex();
            targetDB = tDbrHeader->sequenceReader;
        }
    }

    DBReader<DBKeyType> *reader;
    if (hasTargetDB) {

        reader = new DBReader<DBKeyType>(par.db3.c_str(), par.db3Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA);
    } else {

        reader = new DBReader<DBKeyType>(par.db2.c_str(), par.db2Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA);
    }
    reader->open(DBReader<DBKeyType>::LINEAR_ACCCESS);

    uint16_t extended = DBReader<DBKeyType>::getExtendedDbtype(reader->getDbtype());
    bool needSET = false;
    std::map<DBKeyType, std::string> qSetToSource, tSetToSource;
    if (extended & Parameters::DBTYPE_EXTENDED_SET) {
        needSET = true;
        if (hasTargetDB) {
            qSetToSource = Util::readLookup((par.db1 + ".source"), 2);
            tSetToSource = Util::readLookup((par.db2 + ".source"), 2);
        } else {
            qSetToSource = Util::readLookup((par.db1 + ".source"), 2);
        }
    }

    const std::string& dataFile = hasTargetDB ? par.db4 : par.db3;
    const std::string& indexFile = hasTargetDB ? par.db4Index : par.db3Index;
    const bool shouldCompress = par.dbOut == true && par.compressed == true;
    const int dbType = par.dbOut == true ? Parameters::DBTYPE_GENERIC_DB : Parameters::DBTYPE_OMIT_FILE;
    DBWriter writer(dataFile.c_str(), indexFile.c_str(), par.threads, shouldCompress, dbType);
    writer.open();

    const size_t targetColumn = (par.targetTsvColumn == 0) ? SIZE_T_MAX :  par.targetTsvColumn - 1;
#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = (unsigned int) omp_get_thread_num();
#endif

        const char *columnPointer[255];
        char *dbKey = new char[par.maxSeqLen + 1];

        std::string outputBuffer;
        outputBuffer.reserve(10 * 1024);

#pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < reader->getSize(); ++i) {
            DBKeyType queryKey = reader->getDbKey(i);
            size_t queryIndex;
            char *headerData;
            if(needSET == false) {
                queryIndex = queryDB->getId(queryKey);
                headerData = queryDB->getData(queryIndex, thread_idx);
                if (headerData == NULL) {
                    Debug(Debug::WARNING) << "Invalid header entry in query " << queryKey << "!\n";
                    continue;
                }
            }

            std::string queryHeader;
            if (needSET == true) {
                queryHeader = qSetToSource[queryKey];
            } else if (par.fullHeader) {
                queryHeader = "\"";
                queryHeader.append(headerData, qHeaderIndex[queryIndex].length - 2);
                queryHeader.append("\"");
            } else {
                queryHeader = Util::parseFastaHeader(headerData);
            }

            size_t entryIndex = 0;

            char *data = reader->getData(i, thread_idx);
            while (*data != '\0') {
                if(targetColumn != SIZE_T_MAX){
                    size_t foundElements = Util::getWordsOfLine(data, columnPointer, 255);
                    if (foundElements < targetColumn) {
                        Debug(Debug::WARNING) << "Not enough columns!" << "\n";
                        continue;
                    }
                    Util::parseKey(columnPointer[targetColumn], dbKey);
                }
                std::string targetAccession;
                if(targetColumn == SIZE_T_MAX){
                    targetAccession = "";
                } else if (hasTargetDB) {
                    DBKeyType targetKey = Util::fast_atoi<DBKeyType>(dbKey);
                    size_t targetIndex = targetDB->getId(targetKey);
                    char *targetData;
                    if(needSET == false) {
                        targetData = targetDB->getData(targetIndex, thread_idx);
                        if (targetData == NULL) {
                            Debug(Debug::WARNING) << "Invalid header entry in query " << queryKey << " and target " << targetKey << "!\n";
                            continue;
                        }
                        if (par.fullHeader) {
                            targetAccession = "\"";
                            targetAccession.append(targetData, tHeaderIndex[targetIndex].length - 2);
                            targetAccession.append("\"");
                        } else {
                            targetAccession = Util::parseFastaHeader(targetData);
                        }
                    } else {
                        targetAccession = tSetToSource[targetKey];
                    }
                } else {
                    targetAccession = dbKey;
                }

                if (par.firstSeqRepr && !entryIndex) {
                    queryHeader = targetAccession;
                }

                outputBuffer.append(queryHeader);
                outputBuffer.append("\t");
                outputBuffer.append(targetAccession);

                size_t offset = 0;
                if (targetColumn != 0) {
                    outputBuffer.append("\t");
                    offset = 0;
                } else {
                    offset = strlen(dbKey);
                }

                char *nextLine = Util::skipLine(data);
                outputBuffer.append(data + offset, (nextLine - (data + offset)) - 1);
                outputBuffer.append("\n");
                data = nextLine;
                entryIndex++;
            }
            writer.writeData(outputBuffer.c_str(), outputBuffer.length(), queryKey, thread_idx, par.dbOut);
            outputBuffer.clear();
        }
        delete[] dbKey;
    }
    writer.close(par.dbOut == false);

    if (par.dbOut == false) {
        if (hasTargetDB) {
            FileUtil::remove(par.db4Index.c_str());
        } else {
            FileUtil::remove(par.db3Index.c_str());
        }
    }

    reader->close();
    delete reader;
    if (hasTargetDB) {
        if (sameDB == false) {
            delete tDbrHeader;
        }
    }
    qSetToSource.clear();
    tSetToSource.clear();

    return EXIT_SUCCESS;
}
#undef SIZE_T_MAX
