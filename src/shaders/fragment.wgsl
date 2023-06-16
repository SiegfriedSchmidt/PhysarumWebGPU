@group(0) @binding(0) var<uniform> time: f32;
@group(0) @binding(1) var<uniform> res: vec2u;
@group(0) @binding(2) var<uniform> fieldRes: vec2u;
@group(0) @binding(3) var<storage> fieldState: array<f32>;

fn getCellPos(pos: vec4f) -> u32 {
    let cell = floor(pos.xy / vec2f((res / fieldRes)));
    return u32(cell.x + cell.y * f32(fieldRes.x));
}

fn palette1(t: f32) -> vec3f {
    let a = vec3f(0.5, 0.5, 0.5);
    let b = vec3f(0.5, 0.5, 0.5);
    let c = vec3f(0.5, 0.5, 0.5);
    let d = vec3f(0.5, 0.5, 0.5);

    return a + b * cos(6.28318 * (c * t + d));
}

fn palette2(t: f32) -> vec3f {
    let a = vec3f(0, 0.5, 0.5);
    let b = vec3f(0, 0.5, 0.5);
    let c = vec3f(0, 0.5, 0.333);
    let d = vec3f(0, 0.5, 0.478);

    return a + b * cos(6.28318 * (c * t + d));
}

@fragment
fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let cellPos = getCellPos(pos);
    let color = f32(fieldState[cellPos]);
    return vec4f(palette2(color), 1);
}
