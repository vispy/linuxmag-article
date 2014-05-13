from vispy import app
from vispy import gloo

c = app.Canvas(show=True, close_keys='escape')

@c.connect
def on_mouse_move(event):
    x, y = event.pos
    w, h = c.size
    gloo.clear(color=(x/float(w), 0.0, y/float(h), 1.0))
    c.update()

app.run()

