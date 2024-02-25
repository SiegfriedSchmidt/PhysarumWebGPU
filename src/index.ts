import "./styles/main.css"
import Renderer from "./lib/renderer";

export interface InfoInterface {
    renderTime: HTMLParagraphElement
}

const infoRenderTime = document.getElementById('renderTime') as HTMLParagraphElement
const info: InfoInterface = {renderTime: infoRenderTime}

let num_agents = Number(prompt('Select the number of particles', '10000'))
if (!(num_agents >= 1 && num_agents <= 10000000)) {
    num_agents = 10000
}

const canvas = document.getElementById('root') as HTMLCanvasElement
canvas.width = window.innerWidth
canvas.height = window.innerHeight
const renderer = new Renderer(canvas, info, num_agents)
if (await renderer.init()) {
    renderer.update()
} else {
    document.body.innerHTML = '<div class="not-supported"><h1>WebGPU not supported!</h1></div>'
}
