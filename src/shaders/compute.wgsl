struct Agent {
    pos: vec2f,
    speed: f32,
    angle: f32,
}
const PI = 3.14159265359;

@group(0) @binding(0) var<uniform> time: f32;
@group(0) @binding(2) var<uniform> fieldRes: vec2u;
@group(0) @binding(3) var<storage> fieldStateIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> fieldStateOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(6) var<uniform> evaporateSpeed: f32;
@group(0) @binding(7) var<uniform> diffuseSpeed: f32;
@group(0) @binding(8) var<uniform> numAgents: u32;


fn hash(state: f32) -> f32 {
    var s = u32(state);
    s ^= 2747636419;
    s *= 2654435769;
    s ^= s >> 16;
    s *= 2654435769;
    s ^= s >> 16;
    s *= 2654435769;
    return f32(s) / 4294967295;
}

fn getPos(pos: vec2u) -> u32 {
    return u32(pos.x + pos.y * u32(fieldRes.x));
}

fn checkBorder(x: f32, y: f32) -> bool {
    return (x < 0 || x >= f32(fieldRes.x) || y < 0 || y >= f32(fieldRes.y));
}

fn getVal(x: u32, y: u32) -> f32 {
    if (checkBorder(f32(x), f32(y))) {
        return 0;
    }
    return fieldStateOut[getPos(vec2u(x, y))];
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    return start * (1 - t) + end * t;
}

@compute @workgroup_size(8, 8)
fn processField(@builtin(global_invocation_id) cell: vec3u) {
    if (checkBorder(f32(cell.x), f32(cell.y))) {
        return;
    }

    let sum = getVal(cell.x, cell.y) +
              getVal(cell.x + 1, cell.y + 1) +
              getVal(cell.x + 1, cell.y) +
              getVal(cell.x + 1, cell.y - 1) +
              getVal(cell.x, cell.y - 1) +
              getVal(cell.x - 1, cell.y - 1) +
              getVal(cell.x - 1, cell.y) +
              getVal(cell.x - 1, cell.y + 1) +
              getVal(cell.x, cell.y + 1);

    let pos = getPos(cell.xy);

    let diffusedValue = lerp(fieldStateIn[pos], sum / 9, diffuseSpeed);
    fieldStateOut[pos] = max(0, diffusedValue - evaporateSpeed);
}

@compute @workgroup_size(64)
fn updateAgents(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= numAgents) {
        return;
    }
    var agent = agents[id.x];
    let direction = vec2f(cos(agent.angle), sin(agent.angle));
    var newPos = agent.pos + direction * agent.speed;
    if (checkBorder(newPos.x, newPos.y)) {
        newPos.x = min(f32(fieldRes.x) - 2, max(1, newPos.x));
        newPos.y = min(f32(fieldRes.y) - 2, max(1, newPos.y));
        let random = hash(agent.pos.y * f32(fieldRes.x) + agent.pos.x + hash(f32(id.x)) * time);
        agents[id.x].angle = random * 2 * PI;
    }
    agents[id.x].pos = newPos;
    fieldStateOut[getPos(vec2u(agent.pos))] = 1;
}
