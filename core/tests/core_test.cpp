#include "farm/core.h"

#include <cstdio>

int main() {
  if (farm::version().rfind("farm-core", 0) != 0) {
    std::fprintf(stderr, "unexpected version: %s\n", farm::version().c_str());
    return 1;
  }
  std::puts("ok");
  return 0;
}
