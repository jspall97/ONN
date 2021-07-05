from contextlib import contextmanager
import numpy as np
import pycuda.driver
import pycuda.gl
from pycuda.gl import graphics_map_flags
from glumpy import gloo, gl
import cupy as cp


@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()


def create_shared_texture(w, h, c=4,
                          map_flags=graphics_map_flags.WRITE_DISCARD,
                          dtype=np.uint8):
    """Create and return a Texture2D with gloo and pycuda views."""
    tex = np.zeros((h, w, c), dtype).view(gloo.Texture2D)
    tex.activate()  # force gloo to create on GPU
    tex.deactivate()
    buffer = pycuda.gl.RegisteredImage(
        int(tex.handle), tex.target, map_flags)
    return tex, buffer


def setup(w, h):

    pycuda.driver.init()
    device = pycuda.driver.Device(0)
    context = device.make_context()
    cp.cuda.runtime.setDevice(0)

    # create a buffer with pycuda and gloo views
    tex, cuda_buffer = create_shared_texture(w, h, 4)
    # create a shader program to draw to the screen
    vertex = """
        uniform float scale;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            v_texcoord = texcoord;
            gl_Position = vec4(scale*position, 0.0, 1.0);
        }
    """
    fragment = """
        uniform sampler2D tex;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(tex, v_texcoord);
        } 
    """
    # Build the program and corresponding buffers (with 4 vertices)
    screen = gloo.Program(vertex, fragment, count=4)
    # Upload data into GPU
    screen['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    screen['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    screen['scale'] = 1.0
    screen['tex'] = tex

    return screen, cuda_buffer, context


def window_on_draw(window, screen, cuda_buffer, cp_arr):

    tex = screen['tex']
    h, w = tex.shape[:2]

    assert tex.nbytes == cp_arr.nbytes
    with cuda_activate(cuda_buffer) as ary:
        cpy = pycuda.driver.Memcpy2D()
        cpy.set_src_device(cp_arr.data.ptr)
        cpy.set_dst_array(ary)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes // h
        cpy.height = h
        cpy(aligned=False)

    window.clear()
    screen.draw(gl.GL_TRIANGLE_STRIP)
    # window.swap()
