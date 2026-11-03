#include <metal_stdlib>
using namespace metal;

kernel void gaussianBlur(device const uchar* x [[buffer(0)]],
                         device uchar* x_out [[buffer(1)]],
                         constant uint& blurSize [[buffer(2)]],
                         constant uint& height [[buffer(3)]],
                         constant uint& width [[buffer(4)]],
                         constant uint& channels [[buffer(5)]],
                         constant float& sigma [[buffer(6)]],
                         uint i [[thread_position_in_grid]]) {
    // The gaussian Blur; blurSize has to be odd
    // Instead of worrying about edgecases I apply gaussian blur with clipping the edges
    // (out of bounds pixels are clamped to the nearest edge pixel).
    // w(x, y) = e ^(-(x^2+y^2) / (2*sigma^2)), normalized by the sum of all weights in the window
    if (i >= channels * height * width) {
        return;
    }

    // image layout is (C, H, W) -> i = c * H * W + y * W + x
    int c = i / (height * width);
    int y = (i / width) % height;
    int x_pos = i % width;

    int r = blurSize / 2;
    float sum = 0;
    float weightSum = 0;
    for (int row = -r; row <= r; row++) {
        for (int col = -r; col <= r; col++) {
            int yy = clamp(y + row, 0, int(height) - 1);
            int xx = clamp(x_pos + col, 0, int(width) - 1);
            float w = exp(-float(row * row + col * col) / (2 * sigma * sigma));
            sum += x[c * height * width + yy * width + xx] * w;
            weightSum += w;
        }
    }
    x_out[i] = uchar(sum / weightSum + 0.5f);
}
