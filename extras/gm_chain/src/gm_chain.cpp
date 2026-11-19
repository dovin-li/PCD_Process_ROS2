// 用 grid_map 官方插件链算坡度/粗糙度（参数结构完全照抄 grid_map_demos 的 filters_demo）
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <grid_map_core/GridMap.hpp>
#include <filters/filter_chain.hpp>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: gm_chain <grid.bin> <layers.bin> [--ros-args ...]\n");
    return 64;
  }
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("gm_chain");

  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::fprintf(stderr, "cannot open input: %s\n", argv[1]); return 1; }
  int32_t shape[2]; double meta[3];
  if (std::fread(shape, sizeof(int32_t), 2, f) != 2 ||
      std::fread(meta, sizeof(double), 3, f) != 3) {
    std::fprintf(stderr, "truncated grid header\n");
    std::fclose(f);
    return 1;
  }
  int rows = shape[0], cols = shape[1];
  double res = meta[0], ox = meta[1], oy = meta[2];
  if (rows <= 0 || cols <= 0 || !(res > 0.0) ||
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::fprintf(stderr, "invalid grid geometry\n");
    std::fclose(f);
    return 1;
  }
  std::vector<float> heights(static_cast<size_t>(rows) * cols);
  if (std::fread(heights.data(), sizeof(float), heights.size(), f) != heights.size()) {
    std::fprintf(stderr, "truncated grid payload\n");
    std::fclose(f);
    return 1;
  }
  std::fclose(f);

  grid_map::GridMap map({"elevation"});
  map.setFrameId("map");
  map.setGeometry(grid_map::Length(cols * res, rows * res), res,
                  grid_map::Position(ox + cols * res / 2.0, oy + rows * res / 2.0));
  // 用 atPosition 让 grid_map 自己把世界坐标换成索引，避免手写 Index 的行列顺序错误
  // （手写会越界写坏堆，方形图上不暴露、长条图上直接 malloc 崩溃）
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const double x = ox + (c + 0.5) * res;
      const double y = oy + (r + 0.5) * res;
      grid_map::Index index;
      if (!map.getIndex(grid_map::Position(x, y), index)) continue;
      map.at("elevation", index) = heights[static_cast<size_t>(r) * cols + c];
    }
  }

  // 插件链参数全部由 --params-file 提供（FilterChain 只认参数覆盖）
  filters::FilterChain<grid_map::GridMap> chain("grid_map::GridMap");
  if (!chain.configure("filters", node->get_node_logging_interface(),
                       node->get_node_parameters_interface())) {
    fprintf(stderr, "chain configure failed\n");
    return 2;
  }
  grid_map::GridMap out = map;
  if (!chain.update(map, out)) { fprintf(stderr, "chain update failed\n"); return 3; }

  printf("输出层：");
  for (const auto& name : out.getLayers()) printf("%s ", name.c_str());
  printf("\n");
  const char* layers[] = {"elevation_inpainted", "slope", "step", "step_large", "roughness"};
  for (const char* layer : layers) {
    if (!out.exists(layer)) {
      fprintf(stderr, "缺层：%s\n", layer);
      return 4;
    }
  }
  FILE* g = std::fopen(argv[2], "wb");
  if (!g) { std::fprintf(stderr, "cannot open output: %s\n", argv[2]); return 5; }
  if (std::fwrite(&rows, sizeof(int32_t), 1, g) != 1 ||
      std::fwrite(&cols, sizeof(int32_t), 1, g) != 1) {
    std::fprintf(stderr, "cannot write output header\n");
    std::fclose(g);
    return 5;
  }
  for (const char* layer : layers) {
    const auto& m = out[layer];
    std::vector<float> v(m.data(), m.data() + m.size());
    if (std::fwrite(v.data(), sizeof(float), v.size(), g) != v.size()) {
      std::fprintf(stderr, "cannot write layer: %s\n", layer);
      std::fclose(g);
      return 5;
    }
    printf("  %-14s 最大 %.3f\n", layer, v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()));
  }
  std::fclose(g);
  printf("grid_map 链完成：%d×%d\n", rows, cols);
  rclcpp::shutdown();
  return 0;
}
