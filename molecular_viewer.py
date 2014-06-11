# !/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 2014, Aurore Deschildre, Gael Goret, Cyrille Rossant, Nicolas P. Rougier.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------
import numpy as np

from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
from vispy.util import get_data_file

vertex = """

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;

attribute vec3  a_position;
attribute vec3  a_color;
attribute float a_radius;

varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;

varying float fog;

void main (void) {
    v_radius = a_radius;
    v_color = a_color;

    v_eye_position = u_view * u_model * vec4(a_position,1.0);
    v_light_direction = normalize(u_light_position);
    float dist = length(v_eye_position.xyz);

    gl_Position = u_projection * v_eye_position;

    // stackoverflow.com/questions/8608844/...
    //  ... resizing-point-sprites-based-on-distance-from-the-camera
    vec4  proj_corner = u_projection * vec4(a_radius, a_radius, v_eye_position.z, v_eye_position.w);  // # noqa
    gl_PointSize = 512.0 * proj_corner.x / proj_corner.w;
    
    float end_fog = 50.0;
    float fog_coord = abs(gl_Position.z);
    fog_coord = clamp(fog_coord, 0.0, end_fog);
    fog = 1.5*exp(-.001*fog_coord*fog_coord);
    fog = clamp(fog, 0.0, 1.0);
}
"""

fragment = """

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;

varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;
varying float fog; 

void main()
{
    // r^2 = (x - x0)^2 + (y - y0)^2 + (z - z0)^2
    vec2 texcoord = gl_PointCoord* 2.0 - vec2(1.0);
    float x = texcoord.x;
    float y = texcoord.y;
    float d = 1.0 - x*x - y*y;
    if (d <= 0.0)
        discard;

    float z = sqrt(d);
    vec4 pos = v_eye_position;
    pos.z += v_radius*z;
    vec3 pos2 = pos.xyz;
    pos = u_projection * pos;
    gl_FragDepth = 0.5*(pos.z / pos.w)+0.5;
    vec3 normal = vec3(x,y,z);
    float diffuse = clamp(dot(normal, v_light_direction), 0.0, 1.0);

    // Specular lighting.
    vec3 M = pos2.xyz;
    vec3 O = v_eye_position.xyz;
    vec3 L = vec3(-5., 5., -5.);
    vec3 K = normalize(normalize(L - M) + normalize(O - M));
    // WARNING: abs() is necessary, otherwise weird bugs may appear with some
    // GPU drivers...
    float specular = clamp(pow(abs(dot(normal, K)), 40.), 0.0, 1.0);
    vec3 v_light = vec3(1., 1., 1.);
    gl_FragColor.rgb = (.15*v_color + .55*diffuse * v_color
                        + .35*specular * v_light);
                        
    gl_FragColor = mix(vec4(0., 0., 0., 1.), gl_FragColor, fog);
                        
                        
}
"""


class MolecularViewerCanvas(app.Canvas):

    def __init__(self, fname):
        app.Canvas.__init__(self, close_keys='escape')
        self.size = 800, 600

        self.program = gloo.Program(vertex, fragment)

        self.load_molecule(fname)
        self.load_data()

        self.theta = 0
        self.phi = 0
        self.translate = 25
        
        self.update_matrices()

    def load_molecule(self, fname):
        molecule = np.load(fname)['molecule']
        self._nAtoms = molecule.shape[0]

        # The x,y,z values store in one array
        self.coords = molecule[:, :3]

        # The array that will store the color and alpha scale for all the atoms
        self.atomsColours = molecule[:, 3:6]

        # The array that will store the scale for all the atoms.
        self.atomsScales = molecule[:, 6]

    def load_data(self):
        n = self._nAtoms

        data = np.zeros(n, [('a_position', np.float32, 3),
                            ('a_color', np.float32, 3),
                            ('a_radius', np.float32, 1)])

        data['a_position'] = self.coords
        data['a_color'] = self.atomsColours
        data['a_radius'] = self.atomsScales

        self.program.bind(gloo.VertexBuffer(data))

        self.program['u_light_position'] = 0., 0., 2.

    def on_initialize(self, event):
        gloo.set_state(depth_test=True, clear_color=(0, 0, 0, 1))

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(25.0, width / float(height), 2.0, 100.0)
        self.program['u_projection'] = self.projection

    def on_mouse_wheel(self, event):
        self.translate += -event.delta[1]*2
        self.translate = max(2, self.translate)
        self.update_matrices()
        self.update()
        
    def on_mouse_move(self, event):
        if event.is_dragging:
            x0, y0 = event.press_event.pos
            x1, y1 = event.last_event.pos
            x, y = event.pos
            dx, dy = x - x1, y - y1
            self.phi += .1*dx
            self.theta += .1*dy
            self.update_matrices()
            self.update()

    def update_matrices(self):
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        
        rotate(self.model, self.theta, 1, 0, 0)
        rotate(self.model, self.phi, 0, 1, 0)
        
        translate(self.view, 0, 0, -self.translate)
        
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        
    def on_draw(self, event):
        gloo.clear()
        self.program.draw('points')

def main(fname):
    mvc = MolecularViewerCanvas(fname)
    mvc.show()
    app.run()

if __name__ == '__main__':
    fname = get_data_file('molecular_viewer/micelle.npz')
    main(fname)
