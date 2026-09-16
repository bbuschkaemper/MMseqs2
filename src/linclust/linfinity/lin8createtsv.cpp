#include "Lin8DbReader.h"
#include "Parameters.h"
#include "Debug.h"
#include "DBReader.h"
#include "FileUtil.h"
#include "Util.h"
#include "Timer.h"
#include "MemoryMapped.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

static void writeAt(int fd, const std::string &path, std::string &line, uint64_t &at) {
    if (line.empty()) {
        return;
    }
    const ssize_t wrote = pwrite(fd, line.c_str(), line.size(), (off_t) at);
    if (wrote < 0 || (size_t) wrote != line.size()) {
        Debug(Debug::ERROR) << "Cannot write " << line.size() << " byte to " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    at += line.size();
    line.clear();
}

static const unsigned int NAME_CHUNK_BITS = 30;
static const uint64_t NAME_CHUNK = uint64_t(1) << NAME_CHUNK_BITS;

struct PackedNames {
    std::vector<std::vector<char> > chunk;
    std::vector<uint64_t> at;
    PackedNames(size_t ranks) : at(ranks + 1, 0) {}

    // room for total bytes of names, so that threads can place theirs side by side
    void reserve(uint64_t total, unsigned int threads) {
        chunk.resize((size_t) ((total + NAME_CHUNK - 1) >> NAME_CHUNK_BITS));
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (size_t which = 0; which < chunk.size(); which++) {
            const size_t size = (size_t) std::min<uint64_t>(NAME_CHUNK, total - which * NAME_CHUNK);
            std::vector<char>(size).swap(chunk[which]);
        }
    }

    void place(uint64_t at, const char *name, size_t length) {
        size_t wrote = 0;
        while (wrote < length) {
            const size_t which = (size_t) (at >> NAME_CHUNK_BITS);
            const size_t off = (size_t) (at & (NAME_CHUNK - 1));
            const size_t take = std::min<uint64_t>(length - wrote, NAME_CHUNK - off);
            memcpy(&chunk[which][off], name + wrote, take);
            wrote += take;
            at += take;
        }
    }

    void appendTo(std::string &line, uint64_t rank) const {
        uint64_t from = at[rank];
        const uint64_t until = at[rank + 1];
        while (from < until) {
            const size_t which = (size_t) (from >> NAME_CHUNK_BITS);
            const size_t off = (size_t) (from & (NAME_CHUNK - 1));
            const size_t take = std::min<uint64_t>(until - from, NAME_CHUNK - off);
            line.append(&chunk[which][off], take);
            from += take;
        }
    }

    size_t lengthOf(uint64_t rank) const { return (size_t) (at[rank + 1] - at[rank]); }
    size_t bytes() const {
        size_t total = at.size() * sizeof(uint64_t);
        for (size_t i = 0; i < chunk.size(); i++) total += chunk[i].size();
        return total;
    }
};

// the name Util::parseFastaHeader cuts from a header line, without a string per header: the scratch
// string keeps its room from one line to the next
static size_t nameOf(const char *header, std::string &scratch, const char *&name) {
    const size_t length = Util::skipNoneWhitespace(header);
    scratch.assign(header, length);
    const std::pair<ssize_t, ssize_t> pos = Util::getFastaHeaderPosition(scratch);
    if (pos.first == -1 && pos.second == -1) {
        name = header;
        return 0;
    }
    name = header + pos.first;
    return (size_t) (pos.second - pos.first);
}

int lin8createtsv(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    Lin8DbReader reader(par.db1, true);
    reader.open();

    const size_t budget = Util::computeMemory(par.splitMemoryLimit);
    const unsigned int threads = std::max<unsigned int>(1, par.threads);
    const auto skip = [&]() {
        Debug(Debug::WARNING) << "Names, headers and the cluster index exceed the " << (budget >> 20)
                              << " MiB naming budget; no TSV was written. Keep the sequence database "
                              << par.db1 << " to preserve the rank-to-name mapping\n";
        reader.close();
        return EXIT_SUCCESS;
    };
    // Count the index without allocating DBReader's per-cluster table. Include
    // mapped files, index sorting/mappings and per-thread header/output scratch.
    const size_t indexBytes = FileUtil::getFileSize(par.db2Index);
    size_t need = (reader.getSize() + 1) * sizeof(uint64_t) + reader.keptBytes();
    const auto reserve = [&](size_t bytes) {
        if (need > budget || bytes > budget - need) return false;
        need += bytes;
        return true;
    };
    if (!reserve(indexBytes) || !reserve(reader.headerBytes())
        || !reserve((size_t) threads * (32u << 20))) return skip();
    {
        MemoryMapped index(par.db2Index.c_str(), MemoryMapped::WholeFile, MemoryMapped::SequentialScan);
        if (!index.isValid()) {
            Debug(Debug::ERROR) << "Cannot open " << par.db2Index << "\n";
            EXIT(EXIT_FAILURE);
        }
        const size_t clusters = Util::ompCountLines((char *) index.getData(), index.size(), threads);
        const size_t perCluster = sizeof(DBReader<DBKeyType>::Index) + 2 * sizeof(DBLocalId)
                                 + 2 * sizeof(size_t);
        if (clusters > budget / perCluster || !reserve(clusters * perCluster)) return skip();
    }

    Timer timer;
    PackedNames nameOfRank(reader.getSize());
    // a length range is a run of headers a stream can start at on its own, so the ranges are the
    // units of work: a first pass measures the names of every range, a prefix sum places the ranges,
    // and a second pass parses again and writes each name where it belongs. Largest ranges first
    // keeps the threads even, one thread streaming a header at a time kept all the others waiting
    const Lin8DbIndex &index = reader.getIndex();
    const size_t rangeCount = index.rangeCount();
    std::vector<size_t> order(rangeCount);
    for (size_t r = 0; r < rangeCount; r++) {
        order[r] = r;
    }
    std::sort(order.begin(), order.end(), [&index](size_t a, size_t b) {
        return index.rankAfter(a) - index[a].firstRank() > index.rankAfter(b) - index[b].firstRank();
    });
    std::vector<uint64_t> startOfRange(rangeCount + 1, 0);
    Debug(Debug::INFO) << "Naming " << reader.getSize() << " sequences\n";
    for (int pass = 0; pass < 2; pass++) {
        Debug::Progress nameProgress(rangeCount);
#pragma omp parallel num_threads(threads)
        {
            std::string scratch;
            const char *begin = NULL;
            size_t length = 0;
#pragma omp for schedule(dynamic, 1)
            for (size_t k = 0; k < rangeCount; k++) {
                const size_t r = order[k];
                Lin8DbReader::HeaderStream headers(reader, r, r + 1);
                uint64_t rank = index[r].firstRank();
                uint64_t at = pass == 0 ? 0 : startOfRange[r];
                while (headers.next(begin, length)) {
                    const char *name = NULL;
                    const size_t nameLength = nameOf(begin, scratch, name);
                    if (pass == 1) {
                        nameOfRank.at[rank] = at;
                        nameOfRank.place(at, name, nameLength);
                    }
                    at += nameLength;
                    rank++;
                }
                if (rank != index.rankAfter(r)) {
                    Debug(Debug::ERROR) << "The headers of length range " << r << " hold "
                                        << (rank - index[r].firstRank()) << " entries and the index names "
                                        << (index.rankAfter(r) - index[r].firstRank()) << "\n";
                    EXIT(EXIT_FAILURE);
                }
                if (pass == 0) {
                    startOfRange[r + 1] = at;
                }
                nameProgress.updateProgress();
            }
        }
        if (pass == 0) {
            for (size_t r = 0; r < rangeCount; r++) {
                startOfRange[r + 1] += startOfRange[r];
            }
            if (!reserve(startOfRange[rangeCount])) return skip();
            nameOfRank.reserve(startOfRange[rangeCount], threads);
        }
    }
    nameOfRank.at[reader.getSize()] = startOfRange[rangeCount];
    Debug(Debug::INFO) << "Read " << reader.getSize() << " names in " << timer.lap() << ", " << (nameOfRank.bytes() >> 30) << " GB\n";

    DBReader<DBKeyType> clusters(par.db2.c_str(), par.db2Index.c_str(), par.threads,
                                 DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA);
    clusters.open(DBReader<DBKeyType>::LINEAR_ACCCESS);

    std::vector<size_t> edge(threads + 1, 0);
    for (unsigned int t = 0; t <= threads; t++) {
        edge[t] = clusters.getSize() * t / threads;
    }
    std::vector<uint64_t> bytesOf(threads, 0);
    std::vector<uint64_t> rowsOf(threads, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (unsigned int t = 0; t < threads; t++) {
        uint64_t sum = 0;
        uint64_t mine = 0;
        for (size_t i = edge[t]; i < edge[t + 1]; i++) {
            const uint64_t rep = clusters.getDbKey(i);
            if (rep >= reader.getSize()) {
                Debug(Debug::ERROR) << "The clustering names rank " << rep << ", past the database\n";
                EXIT(EXIT_FAILURE);
            }
            char *data = clusters.getData(i, t);
            while (data != NULL && *data != '\0') {
                const uint64_t member = strtoull(data, NULL, 10);
                if (member >= reader.getSize()) {
                    Debug(Debug::ERROR) << "The clustering names rank " << member
                                        << ", past the database\n";
                    EXIT(EXIT_FAILURE);
                }
                sum += nameOfRank.lengthOf(rep) + nameOfRank.lengthOf(member) + 2;
                mine++;
                data = Util::skipLine(data);
            }
        }
        bytesOf[t] = sum;
        rowsOf[t] = mine;
    }
    std::vector<uint64_t> startOf(threads + 1, 0);
    for (unsigned int t = 0; t < threads; t++) {
        startOf[t + 1] = startOf[t] + bytesOf[t];
    }

    const std::string tmp = par.db3 + ".tmp";
    const int out = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    Debug(Debug::INFO) << "Writing " << (startOf[threads] >> 20) << " MB of rows in "
                       << threads << " parts\n";
    Debug::Progress writeProgress(threads);
    if (out < 0 || ftruncate(out, (off_t) startOf[threads]) != 0) {
        Debug(Debug::ERROR) << "Cannot make " << tmp << " of " << startOf[threads] << " byte\n";
        EXIT(EXIT_FAILURE);
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (unsigned int t = 0; t < threads; t++) {
        uint64_t at = startOf[t];
        std::string line;
        line.reserve(1u << 20);
        for (size_t i = edge[t]; i < edge[t + 1]; i++) {
            const uint64_t rep = clusters.getDbKey(i);
            char *data = clusters.getData(i, t);
            while (data != NULL && *data != '\0') {
                const uint64_t member = strtoull(data, NULL, 10);
                nameOfRank.appendTo(line, rep);
                line.push_back('\t');
                nameOfRank.appendTo(line, member);
                line.push_back('\n');
                data = Util::skipLine(data);
                if (line.size() >= (1u << 20)) {
                    writeAt(out, tmp, line, at);
                }
            }
        }
        writeAt(out, tmp, line, at);
        if (at != startOf[t + 1]) {
            Debug(Debug::ERROR) << "Thread " << t << " wrote to " << at << " and was sized to "
                                << startOf[t + 1] << "\n";
            EXIT(EXIT_FAILURE);
        }
        writeProgress.updateProgress();
    }
    if (::close(out) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(tmp, par.db3);
    uint64_t rows = 0;
    for (unsigned int t = 0; t < threads; t++) {
        rows += rowsOf[t];
    }

    clusters.close();
    reader.close();
    Debug(Debug::INFO) << "Wrote " << rows << " rows in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}
