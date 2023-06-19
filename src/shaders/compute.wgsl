const PI = 3.14159265359;

struct Agent {
    pos: vec2f,
    angle: f32,
    speed: f32,
    sensorLength: f32,
    sensorSize: f32,
    turnAngles: vec3f,
    sensorAngles: vec3f,
}

struct Global {
    evaporateSpeed: f32,
    diffuseSpeed: f32,
    numAgents: f32,
    wobbling: f32,
    pheromone: f32,
    maxPheromone: f32,
    twistingAngle: f32,
}

@group(0) @binding(0) var<uniform> time: f32;
@group(0) @binding(2) var<uniform> fieldRes: vec2u;
@group(0) @binding(3) var<storage> fieldStateIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> fieldStateOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(6) var<uniform> global: Global;


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

fn checkBorder(x: i32, y: i32) -> bool {
    return (x < 0 || x >= i32(fieldRes.x) || y < 0 || y >= i32(fieldRes.y));
}

fn getVal(x: u32, y: u32) -> f32 {
    if (checkBorder(i32(x), i32(y))) {
        return 0;
    }
    return fieldStateIn[getPos(vec2u(x, y))];
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    return start * (1 - t) + end * t;
}

@compute @workgroup_size(8, 8)
fn processField(@builtin(global_invocation_id) cell: vec3u) {
    if (checkBorder(i32(cell.x), i32(cell.y))) {
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

    let diffusedValue = lerp(fieldStateIn[pos], sum / 9, global.diffuseSpeed);
    fieldStateOut[pos] = min(global.maxPheromone, diffusedValue * (1 - global.evaporateSpeed));
}

fn sense(agent: Agent, sensorAngleOffset: f32) -> f32 {
    let sensorAngle = agent.angle + sensorAngleOffset;
    let sensorCentre = agent.pos + vec2f(cos(sensorAngle), sin(sensorAngle)) * agent.sensorLength;
    var sum: f32 = 0;
    for (var x: i32 = i32(-agent.sensorSize); x <= i32(agent.sensorSize); x++) {
        for (var y: i32 = i32(-agent.sensorSize); y <= i32(agent.sensorSize); y++) {
            let pos = vec2i(sensorCentre) + vec2i(x, y);
            if (!checkBorder(pos.x, pos.y)) {
                sum += fieldStateIn[getPos(vec2u(pos))];
            }
        }
    }
    return sum;
}

@compute @workgroup_size(64)
fn updateAgents(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= u32(global.numAgents)) {
        return;
    }
    var agent = agents[id.x];
    let random = hash(agent.pos.y * f32(fieldRes.x) + agent.pos.x + hash(f32(id.x)) * time);

    let sensor1 = sense(agent, agent.sensorAngles.x);
    let sensor2 = sense(agent, agent.sensorAngles.y);
    let sensor3 = sense(agent, agent.sensorAngles.z);

    var moveAngle: f32 = 0;
    if (sensor1 > sensor2 && sensor1 > sensor3) {
        moveAngle = agent.turnAngles.x;
    } else if (sensor2 > sensor1 && sensor2 > sensor3) {
        moveAngle = agent.turnAngles.y;
    } else if (sensor3 > sensor1 && sensor2 > sensor2){
        moveAngle = agent.turnAngles.z;
    }

    moveAngle += agent.angle + (random * 2 - 1) * global.wobbling + global.twistingAngle;
    agents[id.x].angle = moveAngle;
    let direction = vec2f(cos(moveAngle), sin(moveAngle));
    var newPos = agent.pos + direction * agent.speed;

    if (checkBorder(i32(newPos.x), i32(newPos.y))) {
        newPos.x = min(f32(fieldRes.x) - 2, max(1, newPos.x));
        newPos.y = min(f32(fieldRes.y) - 2, max(1, newPos.y));
        agents[id.x].angle = random * 2 * PI;
    }

    agents[id.x].pos = newPos;
    fieldStateOut[getPos(vec2u(agent.pos))] += global.pheromone;
}
