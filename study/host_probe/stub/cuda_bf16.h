#pragma once
#include <cstdint>
struct __nv_bfloat16_raw { unsigned short x; };
struct __nv_bfloat16 { unsigned short __x;
  __nv_bfloat16()=default; __nv_bfloat16(const __nv_bfloat16_raw& r):__x(r.x){}
  operator __nv_bfloat16_raw() const { __nv_bfloat16_raw r; r.x=__x; return r; } };
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
