/***************************************************************************************************
 * Ex01 verify — print actuals for the 5 hand-calc problems in ex01_crd2idx_paper.md
 *
 * Run AFTER you wrote your paper answers. Don't peek before.
 **************************************************************************************************/
#include <cstdio>
#include <cute/layout.hpp>

using namespace cute;

template <class L, class C>
void show(char const* tag, L const& layout, C const& coord) {
  printf("%-4s ", tag);
  print(layout); printf("  @ "); print(coord); printf("  =>  %d\n", int(layout(coord)));
}

int main() {
  printf("== Ex01 actuals ==\n\n");

  // Q1: Layout<Shape<_8>, Stride<_2>> @ 5
  {
    auto L = make_layout(Shape<_8>{}, Stride<_2>{});
    show("Q1", L, 5);
  }

  // Q2: Layout<Shape<_4,_3>, Stride<_3,_1>> @ 7  (1D integer coord)
  {
    auto L = make_layout(Shape<_4,_3>{}, Stride<_3,_1>{});
    show("Q2", L, 7);
  }

  // Q3: Layout<Shape<_2,_3>, Stride<_3,_1>> @ 4
  {
    auto L = make_layout(Shape<_2,_3>{}, Stride<_3,_1>{});
    show("Q3", L, 4);
  }

  // Q4: hierarchical shape, 2D coord (2, 1)
  {
    auto L = make_layout(make_shape (make_shape(_2{}, _4{}), _3{}),
                         make_stride(make_stride(_3{}, _6{}), _1{}));
    show("Q4", L, make_coord(2, 1));
  }

  // Q5: nested tuple coord ((1, 0), 2)
  {
    auto L = make_layout(make_shape (make_shape(_2{}, _2{}), _3{}),
                         make_stride(make_stride(_3{}, _1{}), _6{}));
    show("Q5", L, make_coord(make_coord(1, 0), 2));
  }

  printf("\n");
  printf("Cross-check yourself: did rule-2's divmod direction match yours?\n");
  printf("(c %% prod(s) goes to inner d, c / prod(s) goes to outer D)\n");
  return 0;
}
