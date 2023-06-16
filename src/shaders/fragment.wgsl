@group(0) @binding(0) var<uniform> time: f32;
@group(0) @binding(1) var<uniform> res: vec2u;
@group(0) @binding(2) var<uniform> fieldRes: vec2u;
@group(0) @binding(3) var<storage> fieldState: array<f32>;

fn getCellPos(pos: vec4f) -> u32 {
    let cell = floor(pos.xy / vec2f((res / fieldRes)));
    return u32(cell.x + cell.y * f32(fieldRes.x));
}

@fragment
fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let cellPos = getCellPos(pos);
    let color = f32(fieldState[cellPos]);
    return vec4f(color, color, color, 1);
}
