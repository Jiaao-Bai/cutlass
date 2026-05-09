/***************************************************************************************************
 * Ex04 verify — actuals for composition + complement problems in ex04_composition_paper.md
 **************************************************************************************************/
#include <cstdio>
#include <cute/layout.hpp>

using namespace cute;

template <class L>
void show(char const* tag, L const& layout) {
  printf("%-30s => ", tag);
  print(layout);
  printf("\n");
}

int main() {
  printf("== Ex04 actuals ==\n\n");

  // composition
  printf("-- composition --\n");

  show("Q1: A=(_8):(_2) o B=(_4):(_1)",
       composition(Layout<_8, _2>{}, Layout<_4, _1>{}));

  show("Q2: A=(_8):(_1) o B=(_4):(_2)",
       composition(Layout<_8, _1>{}, Layout<_4, _2>{}));

  show("Q3: A=(_24):(_1) o B=(_4,_6):(_6,_1)",
       composition(Layout<_24, _1>{},
                   make_layout(Shape<_4,_6>{}, Stride<_6,_1>{})));

  show("Q4: A=(_4,_8):(_8,_1) o B=(_4):(_8)",
       composition(make_layout(Shape<_4,_8>{}, Stride<_8,_1>{}),
                   Layout<_4, _8>{}));

  // complement
  printf("\n-- complement --\n");

  show("Q5: complement((_4):(_3), _12)",
       complement(Layout<_4, _3>{}, _12{}));

  show("Q6: complement((_2):(_4), _12)",
       complement(Layout<_2, _4>{}, _12{}));

  printf("\n");
  printf("Cross-checks:\n");
  printf("  - composition: result size === B's size?\n");
  printf("  - composition: result stride === A's stride 'sampled' along B?\n");
  printf("  - complement: size(A) * size(complement) >= M?\n");
  return 0;
}
