/***************************************************************************************************
 * Ex02 — Layout print 8 connaisseur. 看 8 个 layout 的物理形状。
 *
 * 流程（**严格按这个顺序**）：
 *   1) 看每个 case 的 Shape / Stride
 *   2) 在脑子里 / 纸上预测：
 *      - cosize 是多少（最大 idx + 1）
 *      - 是 row-major / col-major / 别的？
 *      - (0,0) (0,1) (1,0) 这三个位置的索引各是多少
 *   3) 跑程序，对照 print_layout 的图
 *   4) 不一致的，写到 PROGRESS.md
 **************************************************************************************************/
#include <cstdio>
#include <cute/layout.hpp>
#include <cute/util/print_tensor.hpp>  // for print_layout

using namespace cute;

template <class L>
void show(char const* tag, L const& L_obj) {
  printf("---- %s ----\n", tag);
  printf("cosize = %d, size = %d\n", int(cosize(L_obj)), int(size(L_obj)));
  print_layout(L_obj);
  printf("\n");
}

int main() {
  // Case 1 — 经典 row-major 4x4
  show("C1 row-major 4x4",
       make_layout(Shape<_4,_4>{}, Stride<_4,_1>{}));

  // Case 2 — col-major 4x4
  show("C2 col-major 4x4",
       make_layout(Shape<_4,_4>{}, Stride<_1,_4>{}));

  // Case 3 — 第一维分层（inner 2x2 形成 4 行），外层 4 列
  show("C3 hierarchical row split",
       make_layout(make_shape (make_shape(_2{}, _2{}), _4{}),
                   make_stride(make_stride(_8{}, _4{}), _1{})));

  // Case 4 — 全部分层（看是否在内存里完全连续）
  show("C4 hierarchical all",
       make_layout(make_shape (_2{}, make_shape(_2{}, _2{})),
                   make_stride(_1{}, make_stride(_2{}, _4{}))));

  // Case 5 — 含 size-1 维（broadcast-like）
  show("C5 size-1 dim",
       make_layout(Shape<_4,_1>{}, Stride<_1,_0>{}));

  // Case 6 — 跨行有 gap 的 stride（row stride > col size）
  show("C6 strided row",
       make_layout(Shape<_4,_4>{}, Stride<_8,_1>{}));

  // Case 7 — 第 0 维 stride 为 0（broadcast）
  show("C7 broadcast row",
       make_layout(Shape<_4,_4>{}, Stride<_0,_1>{}));

  // Case 8 — 列 stride 大（行紧凑、列稀疏）
  show("C8 sparse columns",
       make_layout(Shape<_4,_4>{}, Stride<_1,_8>{}));

  printf("Predict checklist:\n");
  printf("  [ ] cosize 都对吗？\n");
  printf("  [ ] (0,0)/(0,1)/(1,0) 三个位置的索引你都猜对了吗？\n");
  printf("  [ ] C5 cosize 是 4 还是 5？为什么？\n");
  printf("  [ ] C7 的 cosize 跟 C1 比谁大？意味着什么？\n");
  return 0;
}
