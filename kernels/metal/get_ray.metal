#include <metal_stdlib>
using namespace metal;
struct Ray{
    float3 origin;
    float3 direction;
    uint depth;
};

Ray get_ray(float2 uv, float3 camera_center, float3 pixel00_loc, float3 pixel_delta_u, float3 pixel_delta_v){
    Ray ray;
    float3 pixel_center = pixel00_loc + uv.x * pixel_delta_u + uv.y * pixel_delta_v;
    ray.depth = 0;
    ray.origin = camera_center;
    ray.direction = pixel_center - ray.origin;
    return ray;
}