import vertexShader from '../shaders/vertex.wgsl'
import fragmentShader from '../shaders/fragment.wgsl'
import computeShader from '../shaders/compute.wgsl'

export default class {
    canvas: HTMLCanvasElement;
    step: number
    resolution: [number, number]
    fieldResolution: [number, number]
    workgroupSize: number
    workgroupCount: [number, number]

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
    uniformResolutionArray: Float32Array
    uniformFieldResolutionArray: Float32Array
    fieldStateArray: Float32Array

    // Buffers
    vertexBuffer: GPUBuffer
    uniformTimeBuffer: GPUBuffer
    uniformResolutionBuffer: GPUBuffer
    uniformFieldResolutionBuffer: GPUBuffer
    cellStateBuffers: GPUBuffer[]

    // Layouts
    vertexBufferLayout: GPUVertexBufferLayout
    bindGroupLayout: GPUBindGroupLayout
    pipelineLayout: GPUPipelineLayout

    // Bind groups
    bindGroups: GPUBindGroup[]

    // Pipelines
    computePipeline: GPUComputePipeline
    renderPipeline: GPURenderPipeline

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas
        this.step = 0
        this.fieldResolution = [canvas.width / 32, canvas.height / 32];
        this.resolution = [canvas.width, canvas.height];
        this.workgroupSize = 8;
        this.workgroupCount = [Math.ceil(this.fieldResolution[0] / this.workgroupSize), Math.ceil(this.fieldResolution[1] / this.workgroupSize)];
    }

    update() {
        this.uniformTimeArray[0] += 1 / 60;
        this.writeBuffer(this.uniformTimeBuffer, this.uniformTimeArray)
        const encoder = this.device.createCommandEncoder();
        this.compute(encoder)
        this.step++
        this.render(encoder)
        this.queue.submit([encoder.finish()]);
        requestAnimationFrame(() => this.update())
    }

    compute(encoder: GPUCommandEncoder) {
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.computePipeline)
        computePass.setBindGroup(0, this.bindGroups[this.step % 2]);
        computePass.dispatchWorkgroups(this.workgroupCount[0], this.workgroupCount[1]);
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
        this.uniformResolutionArray = new Float32Array(this.resolution);
        this.uniformFieldResolutionArray = new Float32Array(this.fieldResolution);
        this.fieldStateArray = new Float32Array(this.fieldResolution[0] * this.fieldResolution[1]);
        // for (let i = 0; i < this.fieldStateArray.length; ++i) {
        //     this.fieldStateArray[i] = Math.random()
        // }
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
    }

    writeBuffers() {
        this.writeBuffer(this.vertexBuffer, this.vertexArray)
        this.writeBuffer(this.uniformTimeBuffer, this.uniformTimeArray)
        this.writeBuffer(this.uniformResolutionBuffer, this.uniformResolutionArray)
        this.writeBuffer(this.uniformFieldResolutionBuffer, this.uniformFieldResolutionArray)
        this.writeBuffer(this.cellStateBuffers[0], this.fieldStateArray)
        this.writeBuffer(this.cellStateBuffers[1], this.fieldStateArray)
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
                }],
            }),
        ];
    }

    createPipelines() {
        const fragmentModule = this.device.createShaderModule({code: fragmentShader});
        const vertexModule = this.device.createShaderModule({code: vertexShader});
        const computeModule = this.device.createShaderModule({code: computeShader});

        this.computePipeline = this.device.createComputePipeline({
            label: "Compute pipeline",
            layout: this.pipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: "computeMain",
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