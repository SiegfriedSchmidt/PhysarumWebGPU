import vertexShader from '../shaders/vertex.wgsl'
import fragmentShader from '../shaders/fragment.wgsl'
import computeShader from '../shaders/compute.wgsl'

export default class {
    canvas: HTMLCanvasElement;
    step: number
    gridSize: number
    workgroupSize: number
    workgroupCount: number

    // API Data Structures
    adapter: GPUAdapter;
    device: GPUDevice;
    queue: GPUQueue;

    // Frame Backings
    context: GPUCanvasContext;
    canvasFormat: GPUTextureFormat;

    // Arrays
    vertexArray: Float32Array
    uniformArray: Float32Array
    cellStateArray: Uint32Array

    // Buffers
    vertexBuffer: GPUBuffer
    uniformBuffer: GPUBuffer
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
        this.gridSize = 16;
        this.workgroupSize = 8;
        this.workgroupCount = Math.ceil(this.gridSize / this.workgroupSize);
    }

    update() {
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
        computePass.dispatchWorkgroups(this.workgroupCount, this.workgroupCount);
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
        pass.draw(this.vertexArray.length / 2, this.gridSize * this.gridSize);
        pass.end();
    }

    async init() {
        if (await this.initApi()) {
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
        this.uniformArray = new Float32Array([this.gridSize, this.gridSize]);
        this.cellStateArray = new Uint32Array(this.gridSize * this.gridSize);
        for (let i = 0; i < this.cellStateArray.length; ++i) {
            this.cellStateArray[i] = Math.random() > 0.7 ? 1 : 0;
        }
    }

    createBuffers() {
        this.vertexBuffer = this.createBuffer('vertices', this.vertexArray, GPUBufferUsage.VERTEX)
        this.uniformBuffer = this.createBuffer('uniform grid', this.uniformArray, GPUBufferUsage.UNIFORM);
        this.cellStateBuffers = [
            this.createBuffer('Cells state A', this.cellStateArray, GPUBufferUsage.STORAGE),
            this.createBuffer('Cells state B', this.cellStateArray, GPUBufferUsage.STORAGE)
        ];
    }

    writeBuffers() {
        this.writeBuffer(this.vertexBuffer, this.vertexArray)
        this.writeBuffer(this.uniformBuffer, this.uniformArray)
        this.writeBuffer(this.cellStateBuffers[0], this.cellStateArray)
        this.writeBuffer(this.cellStateBuffers[1], this.cellStateArray)
    }

    createLayouts() {
        this.vertexBufferLayout = this.createVertexLayout(this.vertexArray.BYTES_PER_ELEMENT * 2, 'float32x2')
        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: "Cell Bind Group Layout",
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"} // Grid uniform buffer
            }, {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
                buffer: {type: "read-only-storage"} // Cell state input buffer
            }, {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "storage"} // Cell state output buffer
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
                    resource: {buffer: this.uniformBuffer}
                }, {
                    binding: 1,
                    resource: {buffer: this.cellStateBuffers[0]}
                }, {
                    binding: 2, // New Entry
                    resource: {buffer: this.cellStateBuffers[1]}
                }],
            }),
            this.device.createBindGroup({
                label: "Cell renderer bind group B",
                layout: this.bindGroupLayout,
                entries: [{
                    binding: 0,
                    resource: {buffer: this.uniformBuffer}
                }, {
                    binding: 1,
                    resource: {buffer: this.cellStateBuffers[1]}
                }, {
                    binding: 2, // New Entry
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