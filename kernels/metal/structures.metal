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
class random {
    private:
    thread float seed;
    unsigned rstep(const unsigned z, const int s1, const int s2, const int s3, const unsigned M) {
        unsigned b=(((z << s1) ^ z) >> s2);
        return (((z & M) << s3) ^ b);
    }
    public:
    thread random(const unsigned seed1, const unsigned seed2, const unsigned seed3) {
        unsigned seed  = seed1 * 1099087573UL;
        unsigned seedb = seed2 * 1099087573UL;
        unsigned seedc = seed3 * 1099087573UL;
        unsigned z1 = rstep(seed,13,19,12,429496729UL);
        unsigned z2 = rstep(seed,2,25,4,4294967288UL);
        unsigned z3 = rstep(seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*seed + 1013904223UL);
        unsigned r1 = (z1^z2^z3^z4^seedb);
        z1 = rstep(r1,13,19,12,429496729UL);
        z2 = rstep(r1,2,25,4,4294967288UL);
        z3 = rstep(r1,3,11,17,429496280UL);
        z4 = (1664525*r1 + 1013904223UL);
        r1 = (z1^z2^z3^z4^seedc);
        z1 = rstep(r1,13,19,12,429496729UL);
        z2 = rstep(r1,2,25,4,4294967288UL);
        z3 = rstep(r1,3,11,17,429496280UL);
        z4 = (1664525*r1 + 1013904223UL);
        this->seed = (z1^z2^z3^z4) * 2.3283064365387e-10;
    }
    thread float rand() {
        unsigned hashed_seed = this->seed * 1099087573UL;
        unsigned z1 = rstep(hashed_seed,13,19,12,429496729UL);
        unsigned z2 = rstep(hashed_seed,2,25,4,4294967288UL);
        unsigned z3 = rstep(hashed_seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*hashed_seed + 1013904223UL);
        thread float old_seed = this->seed;
        this->seed = (z1^z2^z3^z4) * 2.3283064365387e-10;
        return old_seed;
    }
};