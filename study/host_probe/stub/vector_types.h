#pragma once
// Minimal host stub of CUDA vector types for layout-algebra probes.
struct float1{float x;}; struct float2{float x,y;};
struct float3{float x,y,z;}; struct float4{float x,y,z,w;};
struct int1{int x;}; struct int2{int x,y;}; struct int3{int x,y,z;}; struct int4{int x,y,z,w;};
struct uint1{unsigned x;}; struct uint2{unsigned x,y;}; struct uint3{unsigned x,y,z;}; struct uint4{unsigned x,y,z,w;};
struct uchar1{unsigned char x;}; struct uchar2{unsigned char x,y;}; struct uchar4{unsigned char x,y,z,w;};
struct char1{signed char x;}; struct char2{signed char x,y;}; struct char4{signed char x,y,z,w;};
struct short1{short x;}; struct short2{short x,y;}; struct short4{short x,y,z,w;};
struct ushort1{unsigned short x;}; struct ushort2{unsigned short x,y;}; struct ushort4{unsigned short x,y,z,w;};
struct longlong1{long long x;}; struct longlong2{long long x,y;};
struct double1{double x;}; struct double2{double x,y;};
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
