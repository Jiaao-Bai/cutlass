# Week 1 — Layout basics

预计 ~15h
> **硬件**：🟢 5060 Ti（CPU 思考 + 任意 GPU `print` 验证）

## 目标
- 能用纸笔算出任意 `Layout` 的 `crd2idx`
- 能解释 `Layout` 为什么继承 `cute::tuple`
- 能用 `cute::print_layout` 看出 layout 的几何形状

## 读
- `include/cute/stride.hpp:47-170` — `crd2idx` 4 条递归规则
  ```
  op(c,         s,     d)         => c * d                              // 整数：直接乘
  op(c,    (s,S),(d,D))           => op(c%prod(s),s,d) + op(c/prod(s),S,D)  // 整数+tuple shape：divmod
  op((c,C),(s,S),(d,D))           => op(c,s,d) + op(C,S,D)              // tuple 坐标：逐 mode 求和
  ```
- `include/cute/layout.hpp:95-220` — `Layout` struct 设计
  - 继承 `cute::tuple<Shape,Stride>`，EBO 让静态 layout 零运行时开销
  - `operator()(Coord)` 用 `if constexpr (has_underscore<Coord>)` 同时支持坐标映射和切片
- `include/cute/layout.hpp:768-880` — `bw_coalesce`（栈式从右到左合并）
  ```
  s0:d0      _1:d1   =>  continue       // size=1 跳过
  _1:d0      s1:d1   =>  replace_front   // 替换 size-1
  s0:s1*d1   s1:d1   =>  merge           // 连续内存，合并
  s0:d0      s1:d1   =>  prepend         // 不连续，新加一维
  ```
- `media/docs/cpp/cute/01_layout.md` — 官方层级化解释

## 写
- `exercises/ex01_crd2idx_paper.md` — 5 个手算题，写过程
- `exercises/ex02_layout_print.cu` — 用 `make_layout` + `print_layout` 打 8 个不同 layout，肉眼对照
- `exercises/ex03_coalesce.cu` — 给 6 个 layout 调 `coalesce`，预测结果再跑

## 跑
```bash
make study_stage1_w01_ex02_layout_print -j && ./study_stage1_w01_ex02_layout_print
make study_stage1_w01_ex03_coalesce -j && ./study_stage1_w01_ex03_coalesce
```

## 自检
1. `Layout<Shape<_8,_4>, Stride<_4,_1>>` 是 row-major 还是 col-major？为什么？
2. `Layout<Shape<Shape<_2,_4>,_3>, Stride<Stride<_3,_6>,_1>>` 在 `(2,1)` 的索引是？
3. EBO（empty base optimization）在静态 layout 里省了多少字节？为什么动态 layout 省不了？
4. `coalesce` 后 layout 总是更短或相等吗？给一个反例（如果有）。
5. `print_layout` 打出来的图，行/列分别对应 shape 的哪个 mode？
