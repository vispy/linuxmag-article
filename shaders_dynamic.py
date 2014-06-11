from vispy import gloo
from vispy import app
import numpy as np

n = 100
position = np.zeros((2*n, 2)).astype(np.float32)
position[:,0] = np.repeat(np.linspace(-1, 1, n), 2)
position[::2,1] = -.2
position[1::2,1] = .2
color = np.linspace(0., 1., 2*n).astype(np.float32)

VERT_SHADER = """
const float M_PI = 3.14159265358979323846;

attribute vec2 a_position;

attribute float a_color;
varying float v_color;

uniform float u_time;

void main (void) {
    float x = a_position.x;
    float y = a_position.y + .1 * cos(2.0*M_PI*(u_time-.5*x));

    gl_Position = vec4(x, y, 0.0, 1.0);
    v_color = a_color;
}
"""

FRAG_SHADER = """

uniform float u_time;

varying float v_color;

void main()
{
    gl_FragColor = vec4(1.0, v_color, 0.0, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, close_keys='escape')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = gloo.VertexBuffer(position)
        self.program['a_color'] = gloo.VertexBuffer(color)

        self._timer = app.Timer(1.0 / 60)
        self._timer.connect(self.on_timer)
        self._timer.start()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
        self.program.draw('triangle_strip')

    def on_timer(self, event):
        self.program['u_time'] = event.iteration * 1./60
        self.update()

c = Canvas()
c.show()
app.run()
