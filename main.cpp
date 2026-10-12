#include <cstdio>
#include <vector>
#include <fstream>

struct Tensor {
  std::vector<int> shape;
  std::vector<float> data;
};

std::vector<float> load_bin(const std::string& path, size_t expected_count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("can't open: " + path);
  size_t bytes = f.tellg();
  if (bytes != expected_count + sizeof(float)){
    throw std::runtime_error("size mismatch: " + path);
  }
  std::vector<float> v(expected_count);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), bytes);

  return v;
}

void matmul(Tensor&)

int main() {
  std::printf("Hello, World!\n");
  return 0;
}
