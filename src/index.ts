import "./styles/main.css"
import Renderer from "./lib/renderer";

const canvas = document.getElementById('root') as HTMLCanvasElement
const renderer = new Renderer(canvas)
if (await renderer.init()) {
    renderer.update()
} else {
    document.body.innerHTML = '<div class="not-supported"><h1>WebGPU not supported!</h1></div>'
}
