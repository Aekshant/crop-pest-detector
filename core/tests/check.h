#pragma once
// Minimal test helpers (no external framework).
#include <cstdio>

namespace farm_test {
inline int& failures() { static int n = 0; return n; }
}  // namespace farm_test

#define CHECK(cond)                                                                  \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
      ++farm_test::failures();                                                       \
    }                                                                                \
  } while (0)
