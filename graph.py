from vispy import gloo
from vispy import app

import numpy as np
from math import exp
import networkx as nx




n = 50
g = nx.watts_strogatz_graph(n, 4, 0.25)
M = nx.adjacency_matrix(g)
Ms = M.sum(axis=1)
positions = nx.spring_layout(g)
positions = np.array([positions[k] for k in sorted(positions.keys())], dtype=np.float32)-.5
edges = np.vstack(g.edges()).astype(np.uint32)
color = np.random.rand(len(positions), 3)
size = np.random.randint(low=3, high=10, size=n)


start = positions[edges[:,0],:]
end = positions[edges[:,1],:]

normal = end - start
normal = np.fliplr(normal)
normal[:,0] *= -1
normal /= np.sqrt((normal**2).sum(axis=1)).reshape((-1,1))

h = .02

positions_edges = np.dstack((
    start - h*normal,
    end - h*normal,
    start + h*normal,

    end + h*normal,
    start + h*normal,
    end - h*normal,
)).transpose((0, 2, 1)).ravel().reshape((-1, 2))

index_edges = np.array([
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [1., 1.],
    [0., 1.],
    [1., 0.],
], dtype=np.float32)
index_edges = np.tile(index_edges, (edges.shape[0], 1))


data = np.zeros(n, dtype=[
        ('a_position', np.float32, 2),
        ('a_color', np.float32, 3),
        ('a_size', np.float32, 1),
    ])

data['a_position'] = positions
data['a_color'] = color
data['a_size'] = size


VERT_SHADER = """
attribute vec2 a_position;
attribute vec3 a_color;
attribute float a_size;

uniform vec2 u_pan;
uniform vec2 u_scale;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_radius;
varying float v_linewidth;
varying float v_antialias;

void main (void) {
    v_radius = a_size;
    v_linewidth = 1.0;
    v_antialias = 1.0;
    v_fg_color  = vec4(0.0,0.0,0.0,1.0);
    v_bg_color  = vec4(a_color,    1.0);

    vec2 position_tr = u_scale * (a_position + u_pan);
    gl_Position = vec4(position_tr, 0.0, 1.0);
    float z = sqrt(pow(u_scale.x,2) + pow(u_scale.y,2));
    gl_PointSize = 2.0*z*(v_radius + v_linewidth + 1.5*v_antialias);
}
"""

FRAG_SHADER = """

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_radius;
varying float v_linewidth;
varying float v_antialias;
void main()
{
    float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;
    float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
    float d = abs(r - v_radius) - t;
    if( d < 0.0 )
        gl_FragColor = v_fg_color;
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > v_radius)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}
"""



VERT_SHADER_EDGES = """
attribute vec2 a_position;
attribute vec2 a_index;
varying vec2 v_index;

uniform vec2 u_pan;
uniform vec2 u_scale;

void main (void) {
    vec2 position_tr = u_scale * (a_position + u_pan);
    gl_Position = vec4(position_tr, 0.0, 1.0);
    v_index = a_index;
}
"""

FRAG_SHADER_EDGES = """
varying vec2 v_index;
void main()
{
    vec4 v_fg_color = vec4(0., 0., 0., 1.);
    vec4 v_bg_color = vec4(0., 0., 0., 0.);

    float r = 10.0*(v_index.y - .5);
    float d = abs(r) - .1;
    if( d < 0.0 )
        gl_FragColor = v_fg_color;
    else
    {
        float alpha = d;
        alpha = exp(-alpha*alpha);
        if (abs(r) > 5.)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}
"""

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, close_keys='escape')

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program.bind(gloo.VertexBuffer(data))

        self.program_edges = gloo.Program(VERT_SHADER_EDGES, FRAG_SHADER_EDGES)
        self.program_edges['a_position'] = gloo.VertexBuffer(positions_edges)
        self.program_edges['a_index'] = gloo.VertexBuffer(index_edges)

        for p in (self.program, self.program_edges):
            p['u_pan'] = (0., 0.)
            p['u_scale'] = (1., 1.)

    def on_initialize(self, event):
        gloo.set_state(clear_color=(1, 1, 1, 1), blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_mouse_move(self, event):

        def _normalize((x, y)):
            w, h = float(self.width), float(self.height)
            return x/(w/2.)-1., y/(h/2.)-1.

        if event.is_dragging:
            for p in (self.program, self.program_edges):
                x0, y0 = _normalize(event.press_event.pos)
                x1, y1 = _normalize(event.last_event.pos)
                x, y = _normalize(event.pos)
                dx, dy = x - x1, -(y - y1)
                button = event.press_event.button

                pan_x, pan_y = p['u_pan']
                scale_x, scale_y = p['u_scale']

                if button == 1:
                    p['u_pan'] = (pan_x+dx/scale_x, pan_y+dy/scale_y)
                elif button == 2:
                    scale_x_new, scale_y_new = scale_x * exp(2.5*dx), scale_y * exp(2.5*dy)
                    p['u_scale'] = (scale_x_new, scale_y_new)
                    p['u_pan'] = (pan_x - x0 * (1./scale_x - 1./scale_x_new),
                                             pan_y + y0 * (1./scale_y - 1./scale_y_new))
            self.update()

    def on_resize(self, event):
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, self.width, self.height)

    def on_paint(self, event):
        gloo.clear()
        self.program_edges.draw('triangles')
        self.program.draw('points')

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
