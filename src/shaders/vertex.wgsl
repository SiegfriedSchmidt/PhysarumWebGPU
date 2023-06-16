@vertex
fn vertexMain(@location(0) pos: vec2f, @builtin(instance_index) instance: u32) -> @builtin(position) vec4f {
  return vec4f(pos, 0, 1);
}