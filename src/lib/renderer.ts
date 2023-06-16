import vertexShader from '../shaders/vertex.wgsl'
import fragmentShader from '../shaders/fragment.wgsl'
import computeShader from '../shaders/compute.wgsl'

function getTime() {
    return (new Date()).getMilliseconds()
}

function getRandomValue(v1: number, v2 = 0) {
    const max = Math.max(v1, v2)
    const min = Math.min(v1, v2)
    return Math.random() * (max - min) + min;
}

function radians(angle: number) {
    return angle / 180 * Math.PI
}

export default class {
    canvas: HTMLCanvasElement;
    step: number
    resolution: [number, number]
    fieldResolution: [number, number]
    numAgents: number
    workgroupSize: number
    workgroupProcessFieldCount: [number, number]
    workgroupUpdateAgentsCount: number
    evaporateSpeed: number
    deffuseSpeed: number
    agentParamsCount: number
    wobbling: number

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
    uniformEvaporateSpeedArray: Float32Array
    uniformDiffuseSpeedArray: Float32Array
    uniformNumAgentsArray: Uint32Array
    uniformWobblingArray: Float32Array

    // Buffers
    vertexBuffer: GPUBuffer
    uniformTimeBuffer: GPUBuffer
    uniformResolutionBuffer: GPUBuffer
    uniformFieldResolutionBuffer: GPUBuffer
    cellStateBuffers: GPUBuffer[]
    agentsBuffer: GPUBuffer
    uniformEvaporateSpeedBuffer: GPUBuffer
    uniformDiffuseSpeedBuffer: GPUBuffer
    uniformNumAgentsBuffer: GPUBuffer
    uniformWobblingBuffer: GPUBuffer

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

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas
        this.step = 0
        this.fieldResolution = [canvas.width, canvas.height];
        this.resolution = [canvas.width, canvas.height];
        this.workgroupSize = 8;

        this.numAgents = 500000
        this.workgroupProcessFieldCount = [Math.ceil(this.fieldResolution[0] / this.workgroupSize),
            Math.ceil(this.fieldResolution[1] / this.workgroupSize)];
        this.workgroupUpdateAgentsCount = Math.ceil(this.numAgents / (this.workgroupSize * this.workgroupSize))
        this.evaporateSpeed = 0.01
        this.deffuseSpeed = 0.3
        this.agentParamsCount = 16
        this.wobbling = 0
    }

    update() {
        this.uniformTimeArray[0] += 1000 / 60;
        this.writeBuffer(this.uniformTimeBuffer, this.uniformTimeArray)
        const encoder = this.device.createCommandEncoder();
        this.processField(encoder)
        this.updateAgents(encoder)
        this.step++
        this.render(encoder)
        this.queue.submit([encoder.finish()]);
        requestAnimationFrame(() => this.update())
    }

    initAgents() {
        for (let i = 0; i < this.numAgents; i++) {
            const id = i * this.agentParamsCount
            //     pos: vec2f
            this.agentsArray[id] = this.fieldResolution[0] / 2
            this.agentsArray[id + 1] = this.fieldResolution[1] / 2

            //     angle: f32
            this.agentsArray[id + 2] = Math.random() * Math.PI * 2

            //     speed: f32
            this.agentsArray[id + 3] = 1

            //     sensorLength: f32
            this.agentsArray[id + 4] = 4

            //     sensorSize: f32
            this.agentsArray[id + 5] = 3

            //     turnAngles: vec3f
            this.agentsArray[id + 8] = radians(-60)
            this.agentsArray[id + 9] = radians(0)
            this.agentsArray[id + 10] = radians(60)

            //     sensorAngles: vec3f
            this.agentsArray[id + 12] = radians(-60)
            this.agentsArray[id + 13] = radians(0)
            this.agentsArray[id + 14] = radians(60)
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
        this.agentsArray = new Float32Array(length = this.numAgents * this.agentParamsCount);
        this.initAgents()
        console.log(this.agentsArray.byteLength)
        this.uniformEvaporateSpeedArray = new Float32Array([this.evaporateSpeed])
        this.uniformDiffuseSpeedArray = new Float32Array([this.deffuseSpeed])
        this.uniformNumAgentsArray = new Uint32Array([this.numAgents])
        this.uniformWobblingArray = new Float32Array([this.wobbling])
    }

    createBuffers() {
        this.vertexBuffer = this.createBuffer('vertices', this.vertexArray, GPUBufferUsage.VERTEX)
        this.uniformTimeBuffer = this.createBuffer('uniform time', this.uniformTimeArray, GPUBufferUsage.UNIFORM);
        this.uniformResolutionBuffer = this.createBuffer('uniform resolution', this.uniformResolutionArray, GPUBufferUsage.UNIFORM);
        this.uniformFieldResolutionBuffer = this.createBuffer('uniform field resolution', this.uniformFieldResolutionArray, GPUBufferUsage.UNIFORM);
        this.cellStateBuffers = [
            this.createBuffer('Field state A', this.fieldStateArray, GPUBufferUsage.STORAGE),
            this.createBuffer('Field state B', this.fieldStateArray, GPUBufferUsage.STORAGE)
        ];
        this.agentsBuffer = this.createBuffer('Agents buffer', this.agentsArray, GPUBufferUsage.STORAGE)
        this.uniformEvaporateSpeedBuffer = this.createBuffer('Evaporate speed buffer', this.uniformEvaporateSpeedArray, GPUBufferUsage.UNIFORM)
        this.uniformDiffuseSpeedBuffer = this.createBuffer('Diffuse speed buffer', this.uniformDiffuseSpeedArray, GPUBufferUsage.UNIFORM)
        this.uniformNumAgentsBuffer = this.createBuffer('Diffuse speed buffer', this.uniformDiffuseSpeedArray, GPUBufferUsage.UNIFORM)
        this.uniformWobblingBuffer = this.createBuffer('Wobbling buffer', this.uniformWobblingArray, GPUBufferUsage.UNIFORM)
    }

    writeBuffers() {
        this.writeBuffer(this.vertexBuffer, this.vertexArray)
        this.writeBuffer(this.uniformTimeBuffer, this.uniformTimeArray)
        this.writeBuffer(this.uniformResolutionBuffer, this.uniformResolutionArray)
        this.writeBuffer(this.uniformFieldResolutionBuffer, this.uniformFieldResolutionArray)
        this.writeBuffer(this.cellStateBuffers[0], this.fieldStateArray)
        this.writeBuffer(this.cellStateBuffers[1], this.fieldStateArray)
        this.writeBuffer(this.agentsBuffer, this.agentsArray)
        this.writeBuffer(this.uniformEvaporateSpeedBuffer, this.uniformEvaporateSpeedArray)
        this.writeBuffer(this.uniformDiffuseSpeedBuffer, this.uniformDiffuseSpeedArray)
        this.writeBuffer(this.uniformNumAgentsBuffer, this.uniformNumAgentsArray)
        this.writeBuffer(this.uniformWobblingBuffer, this.uniformWobblingArray)
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
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 8,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 9,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }]
        });
        this.pipelineLayout = this.device.createPipelineLayout({
            label: "Cell Pipeline Layout",
            bindGroupLayouts: [this.bindGroupLayout],
        });
    }

    createBindings() {
        this.bindGroups = [
            this.device.createBindGroup({
                label: "Cell renderer bind group A",
                layout: this.bindGroupLayout,
                entries: [{
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
                    resource: {buffer: this.uniformEvaporateSpeedBuffer}
                }, {
                    binding: 7,
                    resource: {buffer: this.uniformDiffuseSpeedBuffer}
                }, {
                    binding: 8,
                    resource: {buffer: this.uniformNumAgentsBuffer}
                }, {
                    binding: 9,
                    resource: {buffer: this.uniformWobblingBuffer}
                }],
            }),
            this.device.createBindGroup({
                label: "Cell renderer bind group B",
                layout: this.bindGroupLayout,
                entries: [{
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
                    resource: {buffer: this.cellStateBuffers[1]}
                }, {
                    binding: 4,
                    resource: {buffer: this.cellStateBuffers[0]}
                }, {
                    binding: 5,
                    resource: {buffer: this.agentsBuffer}
                }, {
                    binding: 6,
                    resource: {buffer: this.uniformEvaporateSpeedBuffer}
                }, {
                    binding: 7,
                    resource: {buffer: this.uniformDiffuseSpeedBuffer}
                }, {
                    binding: 8,
                    resource: {buffer: this.uniformNumAgentsBuffer}
                }, {
                    binding: 9,
                    resource: {buffer: this.uniformWobblingBuffer}
                }],
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
            usage: usage | GPUBufferUsage.COPY_DST,
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