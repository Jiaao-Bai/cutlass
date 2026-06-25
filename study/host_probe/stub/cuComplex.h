#pragma once
struct cuFloatComplex { float x, y; };
struct cuDoubleComplex { double x, y; };
typedef cuFloatComplex cuComplex;
inline cuFloatComplex  make_cuFloatComplex(float x,float y){ return {x,y}; }
inline cuDoubleComplex make_cuDoubleComplex(double x,double y){ return {x,y}; }
inline float  cuCrealf(cuFloatComplex z){ return z.x; }
inline float  cuCimagf(cuFloatComplex z){ return z.y; }
inline double cuCreal (cuDoubleComplex z){ return z.x; }
inline double cuCimag (cuDoubleComplex z){ return z.y; }
