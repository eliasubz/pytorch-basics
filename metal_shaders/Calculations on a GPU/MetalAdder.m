#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface MetalAdder : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)verifyResults;

@end

@implementation MetalAdder {
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _addPipeline;
    id<MTLCommandQueue> _commandQueue;
    id<MTLBuffer> _bufferA;
    id<MTLBuffer> _bufferB;
    id<MTLBuffer> _resultBuffer;
    NSUInteger _arrayLength;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
{
    self = [super init];
    if (!self) {
        return nil;
    }

    _device = device;
    _arrayLength = 1 << 20;

    NSError *error = nil;
    NSURL *libraryURL = [NSURL fileURLWithPath:@"default.metallib"];
    id<MTLLibrary> library = [_device newLibraryWithURL:libraryURL error:&error];
    if (!library) {
        NSLog(@"Failed to load %@: %@", libraryURL.path, error);
        return nil;
    }

    id<MTLFunction> addFunction = [library newFunctionWithName:@"add_arrays"];
    if (!addFunction) {
        NSLog(@"Failed to find add_arrays.");
        return nil;
    }

    _addPipeline = [_device newComputePipelineStateWithFunction:addFunction error:&error];
    if (!_addPipeline) {
        NSLog(@"Failed to create pipeline state: %@", error);
        return nil;
    }

    _commandQueue = [_device newCommandQueue];
    NSUInteger bufferSize = _arrayLength * sizeof(float);
    _bufferA = [_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _bufferB = [_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _resultBuffer = [_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];

    [self generateRandomFloatData:_bufferA];
    [self generateRandomFloatData:_bufferB];

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        NSLog(@"GPU kernel execution time: %.3f ms",
              (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.0);
    }];

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_addPipeline];
    [encoder setBuffer:_bufferA offset:0 atIndex:0];
    [encoder setBuffer:_bufferB offset:0 atIndex:1];
    [encoder setBuffer:_resultBuffer offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(_arrayLength, 1, 1);
    NSUInteger width = MIN(_addPipeline.maxTotalThreadsPerThreadgroup, _arrayLength);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if (commandBuffer.status == MTLCommandBufferStatusError) {
        NSLog(@"GPU command failed: %@", commandBuffer.error);
        return nil;
    }

    return self;
}

- (void)generateRandomFloatData:(id<MTLBuffer>)buffer
{
    float *values = buffer.contents;
    for (NSUInteger index = 0; index < _arrayLength; index++) {
        values[index] = (float)arc4random() / (float)UINT32_MAX;
    }
}

- (void)verifyResults
{
    float *a = _bufferA.contents;
    float *b = _bufferB.contents;
    float *result = _resultBuffer.contents;

    for (NSUInteger index = 0; index < _arrayLength; index++) {
        if (result[index] != a[index] + b[index]) {
            NSLog(@"Mismatch at %lu: %g != %g", (unsigned long)index,
                  result[index], a[index] + b[index]);
            return;
        }
    }

    NSLog(@"Computed %lu additions correctly.", (unsigned long)_arrayLength);
}

@end

int main(void)
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is unavailable on this Mac.");
            return 1;
        }

        MetalAdder *adder = [[MetalAdder alloc] initWithDevice:device];
        if (!adder) {
            return 1;
        }

        [adder verifyResults];
    }

    return 0;
}
