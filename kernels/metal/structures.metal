#include <metal_stdlib>
using namespace metal;
struct Ray{
    float3 origin;
    float3 direction;
    uint depth;
};
struct Interval {
    float min;
    float max;
};
struct HitRecord{
    bool hit;
    float t;
    float3 p;
    float3 normal;
    bool front_face;
    float debug;
};