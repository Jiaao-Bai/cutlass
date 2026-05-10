# Ex01 — `crd2idx` 手算

不许跑代码、不许 GPT，全在纸上算。算完再跑 `ex01_verify` 对答案。

## 4 条规则速记（出自 `include/cute/stride.hpp:47-170`）

```
op(c,         s,      d)        => c * d                                 // 1. 整数 + 整数 shape
op(c,    (s,S),(d,D))           => op(c%prod(s),  s, d)                  // 2. 整数 c + tuple shape
                                 + op(c/prod(s),  S, D)
op((c,C),(s,S),(d,D))           => op(c, s, d) + op(C, S, D)             // 3. tuple coord + tuple shape
```

`prod(s)` 是 s 这个 mode 的 size（如果 s 是 tuple 就是嵌套乘积）。

---

## 题目

### Q1（规则 1，热身）

```cpp
Layout<Shape<_8>, Stride<_2>>
```

求坐标 `5` 的索引。

**你的答案**：`____`

---

### Q2（规则 2，2D row-major-ish）

```cpp
Layout<Shape<_4,_3>, Stride<_3,_1>>
```

求 1D 整数坐标 `7` 的索引（**不是** `(2,1)` 那种 2D 坐标，是把它当 1D 索引到 size=12 的 layout 里）。

**你的答案**：`____`

提示：用规则 2，先 `7 % 4 = ?`、`7 / 4 = ?`。

---

### Q3（规则 2，col-major-ish）

```cpp
Layout<Shape<_2,_3>, Stride<_3,_1>>
```

求 1D 整数坐标 `4` 的索引。

**你的答案**：`____`

---

### Q4（规则 3，hierarchical shape，2D 坐标）

```cpp
Layout<Shape<Shape<_2,_4>, _3>, Stride<Stride<_3,_6>, _1>>
```

求 2D 坐标 `(2, 1)` 的索引。

注意：第 0 个 mode 是嵌套 `Shape<_2,_4>`，但你给的坐标 `2` 是整数 → 用规则 2 处理这个 mode；第 1 个 mode 直接规则 1。最后两 mode 求和。

**你的答案**：`____`

---

### Q5（规则 3，全 tuple 坐标）

```cpp
Layout<Shape<Shape<_2,_2>, _3>, Stride<Stride<_3,_1>, _6>>
```

求嵌套 tuple 坐标 `((1, 0), 2)` 的索引。

**你的答案**：`____`

---

## 验证

```bash
cmake -DCUTLASS_ENABLE_STUDY=ON -DCUTLASS_NVCC_ARCHS=90a ..  # 或 100a
make study_stage1_w01_ex01_verify -j
./study_stage1_w01_ex01_verify
```

把 5 个答案对照程序输出。错的回 `include/cute/stride.hpp:47-170` 找具体哪条规则没绕清楚，重写到 PROGRESS.md 的"自检题失败记录"里。
