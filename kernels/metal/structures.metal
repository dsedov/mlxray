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
class MetalRandom {
private:
    thread uint state;
    thread uint a = 1664525u;
    thread uint c = 1013904223u;

public:
    MetalRandom(uint seed) {
        state = seed;
    }

    // Generate a random uint
    thread uint rand_uint() {
        state = a * state + c;
        return state;
    }

    // Generate a random float in [0, 1)
    thread float rand_float() {
        return float(rand_uint()) / float(0xFFFFFFFFu);
    }

    // Generate a random float in [min, max)
    thread float rand_range(float min, float max) {
        return min + (max - min) * rand_float();
    }

    // Generate a random float3 with components in [0, 1)
    thread float3 rand_float3() {
        return float3(rand_float(), rand_float(), rand_float());
    }

    // Generate a random point in a unit sphere
    thread float3 rand_in_unit_sphere() {
        while (true) {
            float3 p = float3(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));
            if (length_squared(p) < 1.0) return p;
        }
    }

    // Generate a random unit vector
    thread float3 rand_unit_vector() {
        return normalize(rand_in_unit_sphere());
    }

    // Generate a random point on a hemisphere oriented along the normal
    thread float3 rand_on_hemisphere(float3 normal) {
        float3 on_unit_sphere = rand_unit_vector();
        if (dot(on_unit_sphere, normal) > 0.0) {
            return on_unit_sphere;
        } else {
            return -on_unit_sphere;
        }
    }
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
    thread float3 random_in_unit_sphere() {
        float3 p;
        do {
            p = 2.0 * float3(this->rand(), this->rand(), this->rand()) - 1.0;
        } while (dot(p, p) >= 1.0);
        return p;
    }
    thread float3 random_unit_vector() {
        return normalize(random_in_unit_sphere());
    }
    thread float3 random_on_hemisphere(float3 normal) {
        float3 on_unit_sphere = random_unit_vector();
        if (dot(on_unit_sphere, normal) > 0.0) {
            return on_unit_sphere;
        } else {
            return -on_unit_sphere;
        }
    }
};

class BlueNoiseRandom {
private:
    thread const device float* array;
    thread uint len;
    thread uint index;
    thread uint state;
    thread const device float* blue_noise_2d;
    thread uint blue_noise_index;
    thread bool has_2d_texture;

public:
    // Existing constructor
    BlueNoiseRandom(const device float* input_array, uint input_len, uint offset) {
        array = input_array;
        len = input_len;
        index = offset % len;
        state = offset;
        has_2d_texture = false;
    }

    // New constructor with 2D blue noise texture
    BlueNoiseRandom(const device float* input_array, uint input_len, uint offset, const device float* blue_noise_texture) {
        array = input_array;
        len = input_len;
        index = offset % len;
        state = offset;
        blue_noise_2d = blue_noise_texture;
        blue_noise_index = offset % (128 * 128);
        has_2d_texture = true;
    }

    thread float rand() {
        float result = array[index];
        index = (index + 1) % len;
        
        // Simple scrambling using xorshift
        state ^= (state << 13);
        state ^= (state >> 17);
        state ^= (state << 5);
        
        // Combine blue noise with scrambled value
        result = fract(result + float(state) * 2.3283064365387e-10);
        
        return result;
    }

    thread float3 random_in_unit_sphere() {
        float3 p;
        do {
            p = 2.0 * float3(this->rand(), this->rand(), this->rand()) - 1.0;
        } while (dot(p, p) >= 1.0);
        return p;
    }
    thread float3 random_unit_vector() {
        return normalize(random_in_unit_sphere());
    }
    thread float3 random_on_hemisphere(float3 normal) {
        float3 on_unit_sphere = random_unit_vector();
        if (dot(on_unit_sphere, normal) > 0.0) {
            return on_unit_sphere;
        } else {
            return -on_unit_sphere;
        }
    }

    thread float3 better_random_on_hemisphere(float3 normal) {
        if (!has_2d_texture) {
            return random_on_hemisphere(normal); // Fall back to the original method if 2D texture is not available
        }

        // Sample two values from the 2D blue noise texture
        float u = blue_noise_2d[blue_noise_index];
        float v = blue_noise_2d[(blue_noise_index + 1) % (128 * 128)];
        blue_noise_index = (blue_noise_index + 2) % (128 * 128);

        // Apply some scrambling to u and v
        u = fract(u + float(state) * 2.3283064365387e-10);
        v = fract(v + float(state ^ 0x12345678) * 2.3283064365387e-10);

        // Generate spherical coordinates
        float theta = 2 * M_PI_F * u;
        float phi = acos(2 * v - 1) / 2; // Divide by 2 to map to hemisphere

        // Convert spherical coordinates to Cartesian
        float3 direction;
        direction.x = cos(theta) * sin(phi);
        direction.y = sin(theta) * sin(phi);
        direction.z = cos(phi);

        // Ensure the direction is in the correct hemisphere
        if (dot(direction, normal) < 0) {
            direction = -direction;
        }

        return direction;
    }
};