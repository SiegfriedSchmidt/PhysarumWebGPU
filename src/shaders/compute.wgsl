@group(0) @binding(2) var<uniform> fieldRes: vec2f;
@group(0) @binding(3) var<storage> fieldStateIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> fieldStateOut: array<f32>;

fn hash(s: f32) -> f32 {
    var state = u32(s);
    state ^= 2747636418;
    state *= 2654435769;
    state ^= state >> 16;
    state *= 2654435769;
    state ^= state >> 16;
    return f32(state) / 4294967295;
}

fn getPos(pos: vec3u) -> u32 {
    return u32(pos.x + pos.y * u32(fieldRes.x));
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
    if (f32(cell.x) >= fieldRes.x && f32(cell.y) >= fieldRes.y) {
        return;
    }

    if (cell.x == 10 && cell.y == 31) {
        let pos = getPos(cell);
        fieldStateOut[pos] = 1;
    }
}
