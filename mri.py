import numpy as np

from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
from vispy.util import get_data_file
from vispy.util.cube import cube

# def checkerboard(grid_num=8, grid_size=32):
    # row_even = grid_num / 2 * [0, 1]
    # row_odd = grid_num / 2 * [1, 0]
    # Z = np.row_stack(grid_num / 2 * (row_even, row_odd)).astype(np.uint8)
    # return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)


mri = np.load(get_data_file('brain/mri.npz'))
mri = mri['data']

V, I, _ = cube()
vertices = gloo.VertexBuffer(V)
indices = gloo.IndexBuffer(I)

VERT_SHADER = """

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform sampler3D texture;

attribute vec3 position;
attribute vec3 texcoord;

varying vec3 v_texcoord;

attribute vec4 color;
varying vec4 v_color;

void main()
{
    v_color = color;
    gl_Position = projection * view * model * vec4(position,1.0);
    v_texcoord = texcoord;
}
"""

FRAG_SHADER = """

varying vec4 v_color;
uniform sampler3D texture;
varying vec3 v_texcoord;

void main()
{
    gl_FragColor = texture3D(texture, v_texcoord);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, close_keys='escape')
        self.size = 800, 600

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        
        self.theta, self.phi = 0, 0
        self.translate = 5

        self.program.bind(vertices)
        self.program['texture'] = mri
        
        self.update_matrices()

    def update_matrices(self):
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        
        rotate(self.model, self.theta, 1, 0, 0)
        rotate(self.model, self.phi, 0, 1, 0)
        
        translate(self.view, 0, 0, -self.translate)
        
        self.program['model'] = self.model
        self.program['view'] = self.view
        
    def on_initialize(self, event):
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)

    def on_mouse_move(self, event):
        if event.is_dragging:
            x0, y0 = event.press_event.pos
            x1, y1 = event.last_event.pos
            x, y = event.pos
            dx, dy = x - x1, y - y1
            self.phi += dx
            self.theta += -dy
            self.update_matrices()
            self.update()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(45.0, width / float(height), 1.0, 20.0)
        self.program['projection'] = self.projection

    def on_mouse_wheel(self, event):
        self.translate += -event.delta[1]/5.
        self.translate = max(2, self.translate)
        self.update_matrices()
        self.update()

    def on_paint(self, event):
        gloo.clear()
        self.program.draw('triangles', indices=indices)

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
