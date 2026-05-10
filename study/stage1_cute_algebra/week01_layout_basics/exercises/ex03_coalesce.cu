/***************************************************************************************************
 * Ex03 — coalesce 6 题。给定 layout，预测 coalesce 后的形状再跑。
 *
 * coalesce 4 条规则（include/cute/layout.hpp:768-880, bw_coalesce 从右到左）：
 *   s0:d0      _1:d1   =>  continue        // size=1 跳过
 *   _1:d0      s1:d1   =>  replace_front   // 替换 size-1
 *   s0:s1*d1   s1:d1   =>  merge           // prev_stride == next_size * next_stride，合并
 *   s0:d0      s1:d1   =>  prepend         // 不连续，新加一维
 *
 * 流程：
 *   1) 看 case，自己心算 coalesce 后的形状
 *   2) 跑程序对照
 *   3) 错的回 layout.hpp:768 看是哪一步出了
 **************************************************************************************************/
#include <cstdio>
#include <cute/layout.hpp>

using namespace cute;

template <class L>
void show(char const* tag, L const& layout) {
  printf("%s\n", tag);
  printf("  before: "); print(layout);          printf("\n");
  printf("  after : "); print(coalesce(layout)); printf("\n\n");
}

int main() {
  // C1 — 已经合并到极致的 row-major，应该不动 / 简化
  show("C1: <4,4>:<4,1>  (row-major)",
       make_layout(Shape<_4,_4>{}, Stride<_4,_1>{}));

  // C2 — 跨行有 gap，不连续
  show("C2: <4,4>:<8,1>  (strided rows)",
       make_layout(Shape<_4,_4>{}, Stride<_8,_1>{}));

  // C3 — 两维内存连续（prev_stride == next_size * next_stride）
  show("C3: <2,4>:<4,1>  (perfectly contiguous)",
       make_layout(Shape<_2,_4>{}, Stride<_4,_1>{}));

  // C4 — 含 size-1 维
  show("C4: <4,1>:<1,0>  (size-1 dim)",
       make_layout(Shape<_4,_1>{}, Stride<_1,_0>{}));

  // C5 — 嵌套 hierarchical 全部连续
  show("C5: <(2,2),4>:<(_1,_2),_4>  (hierarchical all-contig)",
       make_layout(make_shape (make_shape(_2{}, _2{}), _4{}),
                   make_stride(make_stride(_1{}, _2{}), _4{})));

  // C6 — 三维：左两维不连续 / 右两维连续
  show("C6: <4,2,4>:<8,4,1>  (mixed 3D)",
       make_layout(Shape<_4,_2,_4>{}, Stride<_8,_4,_1>{}));

  printf("Predict checklist:\n");
  printf("  [ ] C1 你预测的 rank 是几？为什么 row-major 不能合到 1D？\n");
  printf("  [ ] C2 vs C3 的差异——记住 merge 条件是 prev_stride == next_size * next_stride\n");
  printf("  [ ] C4 中那条 size-1 dim 是 'continue 跳过' 还是 'replace_front'？为什么？\n");
  printf("  [ ] C6 最后能不能合成 1 维？\n");
  return 0;
}
