import "./styles/main.css"
import Renderer from "./lib/renderer";

const canvas = document.getElementById('root') as HTMLCanvasElement
const renderer = new Renderer(canvas)
if (await renderer.init()) {
    renderer.update()
} else {

}
