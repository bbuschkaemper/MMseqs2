#include "DBReader.h"
#include "Util.h"

#include <cstdio>
#include <cstdlib>
#include <string>

const char* binary_name = "test_dbreaderlinfinity";

// prints key, name and entry length for the keys [from, from + count) of a --createdb-mode 3 header database
int main(int argc, const char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <db_h> <from> <count> <threads>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string db = argv[1];
    const size_t from = strtoull(argv[2], NULL, 10);
    const size_t count = strtoull(argv[3], NULL, 10);
    DBReader<DBKeyType> reader(db.c_str(), (db + ".index").c_str(), atoi(argv[4]),
                               DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    reader.open(DBReader<DBKeyType>::NOSORT);
    reader.loadData(from, count);
    for (size_t id = from; id < from + count; id++) {
        const size_t length = reader.getEntryLen(id);
        printf("%zu\t%.*s\t%zu\n", id, (int) (length - 2), reader.getData(id, 0), length);
    }
    reader.close();
    return EXIT_SUCCESS;
}
