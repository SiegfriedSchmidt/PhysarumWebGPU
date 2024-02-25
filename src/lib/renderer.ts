import vertexShader from '../shaders/vertex.wgsl'
import fragmentShader from '../shaders/fragment.wgsl'
import computeShader from '../shaders/compute.wgsl'
import {InfoInterface} from "../index";

function getTime() {
    return (new Date()).getMilliseconds()
}

function getRandomValue(v1: number, v2 = 0) {
    const max = Math.max(v1, v2)
    const min = Math.min(v1, v2)
    return Math.random() * (max - min) + min;
}

function getInRange(range: [number, number]) {
    return getRandomValue(...range)
}

function radians(angle: number) {
    return angle / 180 * Math.PI
}

export default class {
    canvas: HTMLCanvasElement;
    info: InfoInterface
    step: number
    resolution: [number, number]
    fieldResolution: [number, number]
    numAgents: number
    workgroupSize: number
    workgroupProcessFieldCount: [number, number]
    workgroupUpdateAgentsCount: number
    evaporateSpeed: number
    diffuseSpeed: number
    agentParamsCount: number
    wobbling: number
    pheromone: number
    maxPheromone: number
    twistingAngle: number
    speedRange: [number, number]
    sensorLengthRange: [number, number]
    sensorSizeRange: [number, number]
    turnAnglesRange: [number, number][]
    sensorAnglesRange: [number, number][]

    // API Data Structures
    adapter: GPUAdapter;
    device: GPUDevice;
    queue: GPUQueue;

    // Frame Backings
    context: GPUCanvasContext;
    canvasFormat: GPUTextureFormat;

    // Arrays
    vertexArray: Float32Array
    uniformTimeArray: Float32Array
    uniformResolutionArray: Uint32Array
    uniformFieldResolutionArray: Uint32Array
    fieldStateArray: Float32Array
    agentsArray: Float32Array
    uniformGlobalParamsArray: Float32Array

    // Buffers
    vertexBuffer: GPUBuffer
    uniformTimeBuffer: GPUBuffer
    uniformResolutionBuffer: GPUBuffer
    uniformFieldResolutionBuffer: GPUBuffer
    cellStateBuffers: GPUBuffer[]
    agentsBuffer: GPUBuffer
    uniformGlobalParamsBuffer: GPUBuffer

    // Layouts
    vertexBufferLayout: GPUVertexBufferLayout
    bindGroupLayout: GPUBindGroupLayout
    pipelineLayout: GPUPipelineLayout

    // Bind groups
    bindGroups: GPUBindGroup[]

    // Pipelines
    processFieldPipeline: GPUComputePipeline
    updateAgentsPipeline: GPUComputePipeline
    renderPipeline: GPURenderPipeline

    constructor(canvas: HTMLCanvasElement, info: InfoInterface, numAgents: number) {
        this.canvas = canvas
        this.info = info
        this.step = 0
        this.fieldResolution = [canvas.width, canvas.height];
        this.resolution = [canvas.width, canvas.height];
        this.workgroupSize = 8;

        this.agentParamsCount = 16
        this.numAgents = numAgents
        this.evaporateSpeed = 0.05
        this.diffuseSpeed = 0.1
        this.wobbling = 10
        this.pheromone = 0.1
        this.maxPheromone = 3
        this.twistingAngle = 0

        this.speedRange = [1, 1]
        this.sensorLengthRange = [10, 10]
        this.sensorSizeRange = [2, 2]
        this.turnAnglesRange = [
            [-90, -90],
            [0, 0],
            [90, 90]
        ]
        this.sensorAnglesRange = [
            [-90, -90],
            [0, 0],
            [90, 90]
        ]

        this.wobbling = radians(this.wobbling)
        this.twistingAngle = radians(this.twistingAngle)
        this.turnAnglesRange = this.turnAnglesRange.map(([x, y]) => [radians(x), radians(y)])
        this.sensorAnglesRange = this.sensorAnglesRange.map(([x, y]) => [radians(x), radians(y)])

        this.workgroupProcessFieldCount = [Math.ceil(this.fieldResolution[0] / this.workgroupSize),
            Math.ceil(this.fieldResolution[1] / this.workgroupSize)];
        this.workgroupUpdateAgentsCount = Math.ceil(this.numAgents / (this.workgroupSize * this.workgroupSize))
    }

    update() {
        const t = getTime()

        const encoder = this.device.createCommandEncoder();
        this.processField(encoder)
        this.updateAgents(encoder)
        this.step++
        this.render(encoder)
        this.queue.submit([encoder.finish()]);

        const dt = getTime() - t
        this.info.renderTime.innerText = `${dt} ms`
        this.uniformTimeArray[0] += dt;
        this.writeBuffer(this.uniformTimeBuffer, this.uniformTimeArray)
        requestAnimationFrame(() => this.update())
    }

    initAgents() {
        for (let i = 0; i < this.numAgents; i++) {
            const id = i * this.agentParamsCount
            //     pos: vec2f
            // const r = getRandomValue(500)
            // const a = radians(getRandomValue(360))
            // this.agentsArray[id] = this.fieldResolution[0] / 2 + Math.cos(a) * r
            // this.agentsArray[id + 1] = this.fieldResolution[1] / 2 + Math.sin(a) * r

            const r = Math.min(this.fieldResolution[0], this.fieldResolution[1]) / 3
            let x, y;
            do {
                x = getRandomValue(-r, r)
                y = getRandomValue(-r, r)
                this.agentsArray[id] = this.fieldResolution[0] / 2 + x
                this.agentsArray[id + 1] = this.fieldResolution[1] / 2 + y
            } while (x * x + y * y > r * r)

            // this.agentsArray[id] = this.fieldResolution[0] / 2
            // this.agentsArray[id + 1] = this.fieldResolution[1] / 2

            // this.agentsArray[id] = getInRange([100, this.fieldResolution[0] - 100])
            // this.agentsArray[id + 1] = getInRange([100, this.fieldResolution[1] - 100])
            //     angle: f32
            this.agentsArray[id + 2] = Math.random() * Math.PI * 2

            //     speed: f32
            this.agentsArray[id + 3] = getInRange(this.speedRange)

            //     sensorLength: f32
            this.agentsArray[id + 4] = getInRange(this.sensorLengthRange)

            //     sensorSize: f32
            this.agentsArray[id + 5] = getInRange(this.sensorSizeRange)

            // alignment padding
            this.agentsArray[id + 6] = 0
            this.agentsArray[id + 7] = 0

            //     turnAngles: vec3f
            this.agentsArray[id + 8] = getInRange(this.turnAnglesRange[0])
            this.agentsArray[id + 9] = getInRange(this.turnAnglesRange[1])
            this.agentsArray[id + 10] = getInRange(this.turnAnglesRange[2])

            // alignment padding
            this.agentsArray[id + 11] = 0

            //     sensorAngles: vec3f
            this.agentsArray[id + 12] = getInRange(this.sensorAnglesRange[0])
            this.agentsArray[id + 13] = getInRange(this.sensorAnglesRange[1])
            this.agentsArray[id + 14] = getInRange(this.sensorAnglesRange[2])
        }
    }

    processField(encoder: GPUCommandEncoder) {
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.processFieldPipeline)
        computePass.setBindGroup(0, this.bindGroups[this.step % 2]);
        computePass.dispatchWorkgroups(this.workgroupProcessFieldCount[0], this.workgroupProcessFieldCount[1]);
        computePass.end();
    }

    updateAgents(encoder: GPUCommandEncoder) {
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.updateAgentsPipeline)
        computePass.setBindGroup(0, this.bindGroups[this.step % 2]);
        computePass.dispatchWorkgroups(this.workgroupUpdateAgentsCount);
        computePass.end();
    }

    render(encoder: GPUCommandEncoder) {
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: {r: 0, g: 0, b: 0.4, a: 1.0},
                storeOp: "store",
            }]
        });
        pass.setPipeline(this.renderPipeline);
        pass.setBindGroup(0, this.bindGroups[this.step % 2]);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.draw(this.vertexArray.length / 2);
        pass.end();
    }

    async init() {
        if (await this.initApi()) {
            console.log(this.resolution, this.fieldResolution)
            this.initCanvas()
            this.createArrays()
            this.createBuffers()
            this.writeBuffers()
            this.createLayouts()
            this.createBindings()
            this.createPipelines()
            return true
        } else {
            return false
        }
    }

    createArrays() {
        this.vertexArray = new Float32Array([
            -1, -1,
            1, -1,
            1, 1,
            -1, -1,
            1, 1,
            -1, 1,
        ]);
        this.uniformTimeArray = new Float32Array([0]);
        this.uniformResolutionArray = new Uint32Array(this.resolution);
        this.uniformFieldResolutionArray = new Uint32Array(this.fieldResolution);
        this.fieldStateArray = new Float32Array(this.fieldResolution[0] * this.fieldResolution[1]);
        this.agentsArray = new Float32Array(this.numAgents * this.agentParamsCount);
        this.uniformGlobalParamsArray = new Float32Array([
            this.evaporateSpeed,
            this.diffuseSpeed,
            this.numAgents,
            this.wobbling,
            this.pheromone,
            this.maxPheromone,
            this.twistingAngle
        ])
        this.initAgents()
        console.log(this.agentsArray.byteLength)
    }

    createBuffers() {
        this.vertexBuffer = this.createBuffer('vertices', this.vertexArray, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST)
        this.uniformTimeBuffer = this.createBuffer('uniform time', this.uniformTimeArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.uniformResolutionBuffer = this.createBuffer('uniform resolution', this.uniformResolutionArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.uniformFieldResolutionBuffer = this.createBuffer('uniform field resolution', this.uniformFieldResolutionArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.cellStateBuffers = [
            this.createBuffer('Field state A', this.fieldStateArray, GPUBufferUsage.STORAGE),
            this.createBuffer('Field state B', this.fieldStateArray, GPUBufferUsage.STORAGE)
        ];
        this.agentsBuffer = this.createBuffer('Agents buffer', this.agentsArray, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
        this.uniformGlobalParamsBuffer = this.createBuffer('Global params buffer', this.uniformGlobalParamsArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
    }

    writeBuffers() {
        this.writeBuffer(this.vertexBuffer, this.vertexArray)
        this.writeBuffer(this.uniformTimeBuffer, this.uniformTimeArray)
        this.writeBuffer(this.uniformResolutionBuffer, this.uniformResolutionArray)
        this.writeBuffer(this.uniformFieldResolutionBuffer, this.uniformFieldResolutionArray)
        this.writeBuffer(this.agentsBuffer, this.agentsArray)
        this.writeBuffer(this.uniformGlobalParamsBuffer, this.uniformGlobalParamsArray)
    }

    createLayouts() {
        this.vertexBufferLayout = this.createVertexLayout(this.vertexArray.BYTES_PER_ELEMENT * 2, 'float32x2')
        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: "Cell Bind Group Layout",
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {type: "uniform"}
            }, {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 3,
                visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                buffer: {type: "read-only-storage"}
            }, {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "storage"}
            }, {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "storage"}
            }, {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {type: "uniform"}
            }]
        });
        this.pipelineLayout = this.device.createPipelineLayout({
            label: "Cell Pipeline Layout",
            bindGroupLayouts: [this.bindGroupLayout],
        });
    }

    createBindings() {
        const entries: GPUBindGroupEntry[] = [{
            binding: 0,
            resource: {buffer: this.uniformTimeBuffer}
        }, {
            binding: 1,
            resource: {buffer: this.uniformResolutionBuffer}
        }, {
            binding: 2,
            resource: {buffer: this.uniformFieldResolutionBuffer}
        }, {
            binding: 3,
            resource: {buffer: this.cellStateBuffers[0]}
        }, {
            binding: 4,
            resource: {buffer: this.cellStateBuffers[1]}
        }, {
            binding: 5,
            resource: {buffer: this.agentsBuffer}
        }, {
            binding: 6,
            resource: {buffer: this.uniformGlobalParamsBuffer}
        }]

        const entries2: GPUBindGroupEntry[] = [...entries]
        entries2[3] = {binding: 3, resource: {buffer: this.cellStateBuffers[1]}}
        entries2[4] = {binding: 4, resource: {buffer: this.cellStateBuffers[0]}}

        this.bindGroups = [
            this.device.createBindGroup({
                label: "Cell renderer bind group A",
                layout: this.bindGroupLayout,
                entries: entries,
            }),
            this.device.createBindGroup({
                label: "Cell renderer bind group B",
                layout: this.bindGroupLayout,
                entries: entries2,
            }),
        ];
    }

    createPipelines() {
        const fragmentModule = this.device.createShaderModule({code: fragmentShader});
        const vertexModule = this.device.createShaderModule({code: vertexShader});
        const computeModule = this.device.createShaderModule({code: computeShader});

        this.processFieldPipeline = this.device.createComputePipeline({
            label: "Process field pipeline",
            layout: this.pipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: "processField",
            }
        });

        this.updateAgentsPipeline = this.device.createComputePipeline({
            label: "Update agents pipeline",
            layout: this.pipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: "updateAgents",
            }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            label: "Render pipeline",
            layout: this.pipelineLayout,
            vertex: {
                module: vertexModule,
                entryPoint: "vertexMain",
                buffers: [this.vertexBufferLayout]
            },
            fragment: {
                module: fragmentModule,
                entryPoint: "fragmentMain",
                targets: [{
                    format: this.canvasFormat
                }]
            }
        });
    }

    async initApi() {
        try {
            this.adapter = await navigator.gpu.requestAdapter();
            this.device = await this.adapter.requestDevice();
            this.queue = this.device.queue
            console.log('Adapter: ', this.adapter)
            console.log('Device: ', this.device)
        } catch (e) {
            console.log(e)
            return false
        }
        return true
    }

    initCanvas() {
        this.context = this.canvas.getContext("webgpu");
        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.canvasFormat,
        });
    }

    createBuffer(label: string, array: BufferSource, usage: GPUBufferUsageFlags) {
        return this.device.createBuffer({
            label: label,
            size: array.byteLength,
            usage: usage,
        });
    }

    writeBuffer(gpuBuffer: GPUBuffer, data: BufferSource | SharedArrayBuffer) {
        this.queue.writeBuffer(gpuBuffer, /*bufferOffset=*/0, data);
    }

    createVertexLayout(arrayStride: number, format: GPUVertexFormat): GPUVertexBufferLayout {
        return {
            arrayStride: arrayStride,
            attributes: [{
                format: format,
                offset: 0,
                shaderLocation: 0, // Position, see vertex shader
            }],
        };
    }
}