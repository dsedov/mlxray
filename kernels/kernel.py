import mlx.core as mx

def gaussian_blur(a: mx.array, r: float):
    source = """
        uint elem = (thread_position_in_grid.x + thread_position_in_grid.y * threads_per_grid.x) * 3;
        int radius = int(ceil(3 * sigma));
       
        float sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        float sum_weights = 0.0;

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                uint nx = thread_position_in_grid.x + dx;
                uint ny = thread_position_in_grid.y + dy;
                float2 offset = float2(dx,dy);
                float distance_squared = dot(offset, offset);
                float weight = exp(-distance_squared / (2.0 * sigma * sigma ));
                
                if (nx >= 0 && nx < threads_per_grid.x && ny >= 0 && ny < threads_per_grid.y) {
                    uint neighbor_elem = (nx + ny * threads_per_grid.x) * 3;
                    sum_r += inp[neighbor_elem] * weight;
                    sum_g += inp[neighbor_elem + 1] * weight;
                    sum_b += inp[neighbor_elem + 2] * weight;
                    sum_weights += weight;
                }
            }
        }

        
        // Calculate the average and store in b1
        if (sum_weights > 0) {
            out[elem]     = (sum_r / sum_weights);
            out[elem + 1] = (sum_g / sum_weights);
            out[elem + 2] = (sum_b / sum_weights);
        }

        out[elem]      = max( min(out[elem], 1.0), 0.0);
        out[elem + 1]  = max( min(out[elem + 1], 1.0), 0.0);
        out[elem + 2]  = max( min(out[elem + 2], 1.0), 0.0);

    """
    kernel = mx.fast.metal_kernel(
        name="gaussian_blur",
        source=source
    )
    outputs = kernel(
        inputs={"inp": a, "sigma": r}, 
        template={"T": mx.float32}, 
        grid=(a.shape[0], a.shape[1], 1), 
        threadgroup=(256,1, 1), 
        output_shapes={"out": a.shape},
        output_dtypes={"out": a.dtype},
    )
    return outputs["out"]
def blur(a: mx.array, r: float):
    source = """
        uint elem = (thread_position_in_grid.x + thread_position_in_grid.y * threads_per_grid.x) * 3;
        int blur_r = int(r);
       
        float sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        int count = 0;

        for (int dy = -blur_r; dy <= blur_r; dy++) {
            for (int dx = -blur_r; dx <= blur_r; dx++) {
                uint nx = thread_position_in_grid.x + dx;
                uint ny = thread_position_in_grid.y + dy;
                
                if (nx >= 0 && nx < threads_per_grid.x && ny >= 0 && ny < threads_per_grid.y) {
                    uint neighbor_elem = (nx + ny * threads_per_grid.x) * 3;
                    sum_r += inp[neighbor_elem];
                    sum_g += inp[neighbor_elem + 1];
                    sum_b += inp[neighbor_elem + 2];
                    count++;
                }
            }
        }

        
        // Calculate the average and store in b1
        if (count > 0) {
            out[elem]     = (sum_r / float(count));
            out[elem + 1] = (sum_g / float(count));
            out[elem + 2] = (sum_b / float(count));
        }

        out[elem]      = max( min(out[elem], 1.0), 0.0);
        out[elem + 1]  = max( min(out[elem + 1], 1.0), 0.0);
        out[elem + 2]  = max( min(out[elem + 2], 1.0), 0.0);

    """
    kernel = mx.fast.metal_kernel(
        name="blur",
        source=source
    )
    outputs = kernel(
        inputs={"inp": a, "r": r}, 
        template={"T": mx.float32}, 
        grid=(a.shape[0], a.shape[1], 1), 
        threadgroup=(256,1, 1), 
        output_shapes={"out": a.shape},
        output_dtypes={"out": a.dtype},
    )
    return outputs["out"]
def reaction_diffusion(a: mx.array, b1: mx.array, b2: mx.array):
    source = """
        uint elem = (thread_position_in_grid.x + thread_position_in_grid.y * threads_per_grid.x) * 3;
       
        float m = 4.0;

        // Calculate the average and store in b1
        
        out[elem]     = (b1[elem]     - b2[elem]) * m;
        out[elem + 1] = (b1[elem + 1] - b2[elem + 1]) * m;
        out[elem + 2] = (b1[elem + 2] - b2[elem + 2]) * m;
        
        if(out[elem]   > 0.5) {out[elem]   = 1.0;}
        if(out[elem+1] > 0.5) {out[elem+1] = 1.0;}
        if(out[elem+2] > 0.5) {out[elem+2] = 1.0;}

        out[elem]      = max( min(out[elem], 1.0), 0.0);
        out[elem + 1]  = max( min(out[elem + 1], 1.0), 0.0);
        out[elem + 2]  = max( min(out[elem + 2], 1.0), 0.0);

    """
    kernel = mx.fast.metal_kernel(
        name="reaction_diffusion",
        source=source
    )
    outputs = kernel(
        inputs={"inp": a, "b1": b1, "b2": b2}, 
        template={"T": mx.float32}, 
        grid=(a.shape[0], a.shape[1], 1), 
        threadgroup=(256,1, 1), 
        output_shapes={"out": a.shape},
        output_dtypes={"out": a.dtype},
    )
    return outputs["out"]
def noise(a: mx.array, p:float):
    header = """
        #include <metal_stdlib>
        using namespace metal;
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
       
    """
    source = """

        uint elem = (thread_position_in_grid.x + thread_position_in_grid.y * threads_per_grid.x) * 3;
        random rand = random(thread_position_in_grid.x, thread_position_in_grid.y + 1,  thread_position_in_grid.y * threads_per_grid.x);

        if(rand.rand() > p){
            out[elem]     = 1.0;
            out[elem + 1] = 1.0;
            out[elem + 2] = 1.0;
        } else {
            out[elem]     = inp[elem];
            out[elem + 1] = inp[elem + 1];
            out[elem + 2] = inp[elem + 2];
        }
    """
    kernel = mx.fast.metal_kernel(
        name="noise",
        source=source,
        header=header,
    )
    outputs = kernel(
        inputs={"inp": a, "p": p}, 
        template={"T": mx.float32}, 
        grid=(a.shape[0], a.shape[1], 1), 
        threadgroup=(256,1, 1), 
        output_shapes={"out": a.shape},
        output_dtypes={"out": a.dtype},
    )
    return outputs["out"]

