import numpy as np

from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate
from vispy.util import get_data_file

brain = np.load(get_data_file('brain/brain.npz'))
data = brain['vertex_buffer']
faces = brain['index_buffer']
    
VERT_SHADER = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec4 u_color;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec4 a_color;

varying vec3 v_position;
varying vec3 v_normal;
varying vec4 v_color;

void main()
{
    v_normal = a_normal;
    v_position = a_position;
    v_color = a_color * u_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

FRAG_SHADER = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_normal;

uniform vec3 u_light_intensity;
uniform vec3 u_light_position;

varying vec3 v_position;
varying vec3 v_normal;
varying vec4 v_color;

void main()
{
    // Calculate normal in world coordinates
    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

    // Calculate the location of this fragment (pixel) in world coordinates
    vec3 position = vec3(u_view*u_model * vec4(v_position, 1));

    // Calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = u_light_position - position;

    // Calculate the cosine of the angle of incidence (brightness)
    float brightness = dot(normal, surfaceToLight) /
                      (length(surfaceToLight) * length(normal));
    brightness = max(min(brightness,1.0),0.0);

    // Calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)

    // Specular lighting.
    vec3 surfaceToCamera = vec3(0.0, 0.0, 1.0) - position;
    vec3 K = normalize(normalize(surfaceToLight) + normalize(surfaceToCamera));
    float specular = clamp(pow(abs(dot(normal, K)), 40.), 0.0, 1.0);
    
    gl_FragColor = v_color * brightness * vec4(u_light_intensity, 1) +
                   specular * vec4(1.0, 1.0, 1.0, 1.0);
}
"""


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, close_keys='escape')
        self.size = 800, 600

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        
        self.theta, self.phi = -80, 180
        self.translate = 3

        self.faces = gloo.IndexBuffer(faces)
        self.program.bind(gloo.VertexBuffer(data))
        
        self.program['u_color'] = 1, 1, 1, 1
        self.program['u_light_position'] = (1., 1., 1.)
        self.program['u_light_intensity'] = (1., 1., 1.)
        
        self.update_matrices()

    def update_matrices(self):
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        
        rotate(self.model, self.theta, 1, 0, 0)
        rotate(self.model, self.phi, 0, 1, 0)
        
        translate(self.view, 0, 0, -self.translate)
        
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_normal'] = np.array(np.matrix(np.dot(self.view, 
                                                             self.model)).I.T)
        
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
        self.program['u_projection'] = self.projection

    def on_mouse_wheel(self, event):
        self.translate += -event.delta[1]/5.
        self.translate = max(2, self.translate)
        self.update_matrices()
        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('triangles', indices=self.faces)

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
