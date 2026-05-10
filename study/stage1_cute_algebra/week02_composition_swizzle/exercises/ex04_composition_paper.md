# Ex04 — `composition` + `complement` 手算

锚点：`include/cute/layout.hpp:1021-1260`、`media/docs/cpp/cute/02_layout_algebra.md`

---

## A. composition 速记

```
(A ∘ B)(c) = A(B(c))
```
- B 决定**哪些坐标进**（B 的 size 是结果的 size）
- A 决定**坐标到地址的映射**（A 的 stride 主导）
- 形状对 B 的形状走，stride 用 A 在 B 的 stride 处的 stride

直觉：B 是"选择器"（在 A 的 1D 索引空间里挑），A 是"地址表"。

## B. complement 速记

`complement(A, M)` 找出 `[0, M)` 里 **A 没用到** 的地址，组成一个新 layout，size = M / size(A)。
满足：`A` 与 `complement(A, M)` 的乘积可以 cover `[0, M)`。

---

## 题目（4 + 2）

### Q1. 基础 stride pickup

```cpp
A = Layout<_8, _2>      // (_8):(_2)，8 元素，stride 2，cosize 16
B = Layout<_4, _1>      // (_4):(_1)，4 元素，stride 1
composition(A, B) = ?
```

提示：B 的 stride 1 在 A 里"走" stride 2 → 结果 stride 是 ?；size 是 B 的 size。

**你的答案**：`(__):(__)`

---

### Q2. 取偶数项

```cpp
A = Layout<_8, _1>      // identity 0..7
B = Layout<_4, _2>      // size 4, stride 2 → 选出 0,2,4,6
composition(A, B) = ?
```

**你的答案**：`(__):(__)`

---

### Q3. 1D → 2D reshape（README 自检题原题）

```cpp
A = Layout<_8, _1>
B = Layout<_4, _2>
```
（同 Q2，README 答案应该是 `(_4):(_2)`）

如果换成下面这个：

```cpp
A = Layout<_24, _1>                                  // identity, size 24
B = Layout<Shape<_4,_6>, Stride<_6,_1>>              // 4x6 row-major
composition(A, B) = ?
```

**你的答案**：`(_,_):(_,_)`

---

### Q4. 走在 hierarchical A 上

```cpp
A = Layout<Shape<_4,_8>, Stride<_8,_1>>              // 4x8 row-major, cosize 32
B = Layout<_4, _8>                                   // size 4, stride 8
composition(A, B) = ?
```

提示：B 在 A 的 1D 索引空间里走 stride 8。A(0)=0, A(8)=? A 的 1D 索引到 (m,n)：8%4=0,8/4=2 → A(0,2) = 0*8+2*1=2。所以 B 选出的是 A 的索引 0,8,16,24 → 各自地址 ?

**你的答案**：`(__):(__)`

---

### Q5. complement 入门（README 自检题）

```cpp
complement(Layout<_4, _3>, _12)
```

A = `(_4):(_3)` 用了地址 `{0, 3, 6, 9}`。`[0,12)` 里剩下的是 `{1,2,4,5,7,8,10,11}`，共 8 个 = 12/4*2。
排好序找规律：`(1,2),(4,5),(7,8),(10,11)` → 内层 size 2 stride 1，外层 size 4 stride 3。

**你的答案**：`(__,__):(__,__)`

---

### Q6. complement 不整除

```cpp
complement(Layout<_2, _4>, _12)
```

A = `(_2):(_4)` 用 `{0, 4}`，剩下 `{1,2,3,5,6,7,8,9,10,11}` —— 不规则。
complement 只保证 size = ceil_div + 一些规则化。提示：CuTe 的 complement 会用 stride 1 + stride 8 拼出 size 6 的 layout（=12/2）。

**你的答案**：`(__,__):(__,__)`（如果不确定就跑 verify，这题难点是看实现的舍入约定）

---

## 验证

```bash
make study_stage1_w02_ex04_verify -j
./study_stage1_w02_ex04_verify
```

错的回 `include/cute/layout.hpp:1021-1260` 看具体是 fold 哪一步出问题，或者打开 `media/docs/cpp/cute/02_layout_algebra.md` 找对应章节。
