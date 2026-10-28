// Exact MV 13-word hashing. Python supplies Unicode-lowercased, whitespace-joined
// UTF-8 words. Only compact matches are returned; no approximate membership test.
#include <openssl/md5.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

struct Match { uint64_t start, end; uint8_t eligible; };

extern "C" size_t prelude_scan(const char* text, size_t length,
                               const uint64_t* index, size_t index_size,
                               unsigned n, unsigned min_distinct, size_t capacity,
                               uint64_t* hashes, uint64_t* starts, uint64_t* ends,
                               uint8_t* eligible, uint64_t* evaluated) {
    try {
        std::vector<uint64_t> start, end;
        for (size_t p = 0; p < length;) {
            const size_t a = p;
            while (p < length && text[p] != ' ') ++p;
            start.push_back(a); end.push_back(p);
            if (p < length) ++p;
        }
        std::unordered_map<uint64_t, Match> found;
        *evaluated = start.size() >= n ? start.size() - n + 1 : 0;
        for (size_t i = 0; i < *evaluated; ++i) {
            unsigned char digest[16];
            MD5(reinterpret_cast<const unsigned char*>(text + start[i]), end[i+n-1]-start[i], digest);
            uint64_t hash = 0;
            for (int j = 0; j < 8; ++j) hash = (hash << 8) | digest[j];
            if (!std::binary_search(index, index + index_size, hash)) continue;
            unsigned distinct = 0;
            for (size_t j = i; j < i+n; ++j) {
                bool seen = false;
                for (size_t k = i; k < j; ++k) {
                    if (end[j]-start[j] == end[k]-start[k] &&
                        std::memcmp(text + start[j], text + start[k], end[j]-start[j]) == 0) {
                        seen = true; break;
                    }
                }
                if (!seen) ++distinct;
            }
            const bool valid = distinct >= min_distinct;
            auto previous = found.find(hash);
            if (previous == found.end() || (!previous->second.eligible && valid))
                found[hash] = {start[i], end[i+n-1], static_cast<uint8_t>(valid)};
        }
        size_t at = 0;
        for (const auto& item : found) {
            if (at < capacity) {
                hashes[at] = item.first; starts[at] = item.second.start;
                ends[at] = item.second.end; eligible[at] = item.second.eligible;
            }
            ++at;
        }
        return found.size();
    } catch (...) { return std::numeric_limits<size_t>::max(); }
}
