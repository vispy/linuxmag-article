from vispy import gloo
from vispy import app
import numpy as np

n = 100
position = np.zeros((2*n, 2)).astype(np.float32)
position[:,0] = np.repeat(np.linspace(-1, 1, n), 2)
position[::2,1] = -.2
position[1::2,1] = .2

VERT_SHADER = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

FRAG_SHADER = """
void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, close_keys='escape')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = position

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
        self.program.draw('triangle_strip')

c = Canvas()
c.show()
app.run()
