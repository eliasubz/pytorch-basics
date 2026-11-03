import torch

lib = torch.mps.compile_shader("""
kernel void whoami(device uint* out [[buffer(0)]],
                   constant uint& n [[buffer(1)]],
                   uint2 id [[thread_position_in_grid]]
                   ) {
    out[id.x * n+ id.y] = id.x * n + id.y;
    // out[id] = id;   // try: uint2 id, then write id.x / id.y into separate slots
}
""")
out = torch.full((16,), -1, dtype=torch.int32, device="mps")
lib.whoami(out, 4, threads=(4,4), group_size=(2,2))
print(out)
