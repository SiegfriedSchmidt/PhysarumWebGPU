@group(0) @binding(1) var<uniform> res: vec2f;
@group(0) @binding(2) var<uniform> fieldRes: vec2f;
@group(0) @binding(3) var<storage> fieldState: array<f32>;

fn getCellPos(pos: vec4f) -> u32 {
    let cell = floor(pos.xy / (res / fieldRes));
    return u32(cell.x + cell.y * fieldRes.x);
}

@fragment
fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let cellPos = getCellPos(pos);
  let color = fieldState[cellPos];
  return vec4f(color, color, color, 1);
}
