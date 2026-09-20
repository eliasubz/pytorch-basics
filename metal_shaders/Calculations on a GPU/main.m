void add_arrays(const float* inA,
                const float* inB,
                float* result,
                int length)
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    MetalAdder* adder = [[MetalAdder alloc] initWithDevice:device];

    for (int index = 0; index < length ; index ++)
    {
        result[index] = inA[index] + inB[index];
    }
}