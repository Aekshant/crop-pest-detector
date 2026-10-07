#include <cstdio>
#include <string>

#include "check.h"
#include "farm/core.h"

void RunVisionTests();

int main() {
  CHECK(farm::version().rfind("farm-core", 0) == 0);
  RunVisionTests();
  if (farm_test::failures()) {
    std::fprintf(stderr, "%d check(s) failed\n", farm_test::failures());
    return 1;
  }
  std::puts("all tests passed");
  return 0;
}
