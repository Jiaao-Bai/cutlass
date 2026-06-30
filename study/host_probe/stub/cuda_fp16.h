#pragma once
#include <cstdint>
struct __half_raw { unsigned short x; };
struct __half { unsigned short __x;
  __half()=default; __half(const __half_raw& r):__x(r.x){}
  operator __half_raw() const { __half_raw r; r.x=__x; return r; } };
struct __half2 { __half x, y; };
struct __half2_raw { unsigned short x, y; };
typedef __half half;
inline float __half2float(__half h){ (void)h; return 0.f; }
inline __half __float2half(float f){ (void)f; __half h; h.__x=0; return h; }
inline __half __float2half_rn(float f){ return __float2half(f); }
typedef __half2 half2;
