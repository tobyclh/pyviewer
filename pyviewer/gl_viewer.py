# Imgui viewer that supports separate ui and compute threads, image uploads from torch tensors.
# Original by Pauli Kemppinen (https://github.com/msqrt)
# Modified by Erik Härkönen

import numpy as np
import multiprocessing as mp
from pathlib import Path
from urllib.request import urlretrieve
from threading import get_ident
import os
from sys import platform
from contextlib import contextmanager, nullcontext

import imgui.core
# import imgui.plot as implot
from imgui.integrations.glfw import GlfwRenderer
from .imgui_themes import theme_dark_overshifted, theme_deep_dark, theme_ps, theme_contrast
from .utils import normalize_image_data

import glfw
glfw.ERROR_REPORTING = 'raise' # make sure errors don't get swallowed

import OpenGL.GL as gl
import torch

cuda_synchronize = lambda : None
if torch.cuda.is_available():
    cuda_synchronize = torch.cuda.synchronize

has_pycuda = False
try:
    import pycuda
    import pycuda.gl as cuda_gl
    import pycuda.tools
    has_pycuda = True
except Exception:
    print('PyCUDA with GL support not available, images will be uploaded from RAM.')

class _texture:
    '''
    This class maps torch tensors to gl textures without a CPU roundtrip.
    '''
    def __init__(self, min_mag_filter=gl.GL_LINEAR):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex) # need to bind to modify
        # sets repeat and filtering parameters; change the second value of any tuple to change the value
        for params in ((gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT), (gl.GL_TEXTURE_MIN_FILTER, min_mag_filter), (gl.GL_TEXTURE_MAG_FILTER, min_mag_filter)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, *params)
        self.mapper = None
        self.shape = [0,0] # texture
        self._cuda_buffer = None

    # be sure to del textures if you create a forget them often (python doesn't necessarily call del on garbage collect)
    def __del__(self):
        gl.glDeleteTextures(1, [self.tex])
        if self.mapper is not None:
            self.mapper.unregister()

    def set_interp(self, key, val):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, key, val)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def upload_np(self, image):
        image = normalize_image_data(image, 'uint8')

        # support for shapes (h,w), (h,w,1), (h,w,3) and (h,w,4)
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=-1) #image.repeat(1,1,3)
        # if image.shape[2] == 3:
        #     image = np.concatenate([image, np.ones_like(image[:,:,0:1])*255], axis=-1)

        shape = image.shape
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        if shape[0] != self.shape[0] or shape[1] != self.shape[1]:
            # Reallocate
            self.shape = shape
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, shape[1], shape[0], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image)
        else:
            # Overwrite
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, shape[1], shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    @torch.no_grad()
    def upload_torch(self, img):
        assert img.device.type == "cuda", "Please provide a CUDA tensor"
        assert img.ndim == 3, "Please provide a HWC tensor"
        assert img.shape[2] < min(img.shape[0], img.shape[1]), "Please provide a HWC tensor"

        if img.dtype.is_floating_point:
            if img.max() <= 1.0:
                img *= 255
            img = (img).byte()

        if img.shape[2] == 1:
            img = img.repeat(1,1,3)
        if img.shape[2] == 3:
            if (self._cuda_buffer is None) or (img.shape[0] != self._cuda_buffer.shape[0] or img.shape[1] != self._cuda_buffer.shape[1]):
                self._cuda_buffer = torch.ones((img.shape[0], img.shape[1], 4), dtype=torch.uint8, device=img.device) * 255
                self._cuda_buffer.requires_grad = False
            self._cuda_buffer[..., :-1] = img
        elif img.shape[2] == 4:
            if (self._cuda_buffer is not None) and (img.shape == self._cuda_buffer.shape):
                self._cuda_buffer[:] = img
            else:
                self._cuda_buffer = img

        # img = img.contiguous()
        if has_pycuda:
            self.upload_ptr(self._cuda_buffer.data_ptr(), self._cuda_buffer.shape)
        else:
            self.upload_np(self._cuda_buffer.detach().cpu().numpy())
        # if has_pycuda:
        #     self.upload_ptr(img.data_ptr(), img.shape)
        # else:
        #     self.upload_np(img.detach().cpu().numpy())

    # Copy from cuda pointer
    def upload_ptr(self, ptr, shape):
        assert has_pycuda, 'PyCUDA-GL not available, cannot upload using raw pointer'
        # assert shape[-1] == 3, 'Data format not RGB'
        
        # reallocate if shape changed or data type changed from np to torch
        
        if shape[0] != self.shape[0] or shape[1] != self.shape[1] or self.mapper is None:
            self.shape = shape
            if self.mapper is not None:
                self.mapper.unregister()

            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA , shape[1], shape[0], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            self.mapper = cuda_gl.RegisteredImage(int(self.tex), gl.GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.WRITE_DISCARD)
        tex_data = self.mapper.map()
        tex_arr = tex_data.array(0, 0)
        ptr_int = int(ptr)
        assert ptr_int == ptr, 'Device pointer overflow'

        # copy from torch tensor to mapped gl texture (avoid cpu roundtrip)
        cpy = pycuda.driver.Memcpy2D()
        cpy.set_src_device(ptr_int)
        cpy.set_dst_array(tex_arr)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = 1*shape[1]*shape[2]
        # cpy.dst_pitch = int(cpy.dst_pitch / 3 * 4)
        # cpy.src_pitch = int(cpy.src_pitch / 3 * 4)
        cpy.height = shape[0]
        cpy(aligned=False)

        # cleanup
        tex_data.unmap()
        cuda_synchronize()
        # gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


class _editable:
    def __init__(self, name, ui_code = '', run_code = ''):
        self.name = name
        self.ui_code = ui_code if len(ui_code)>0 else 'imgui.begin(\'Test\')\nimgui.text(\'Example\')#your code here!\nimgui.end()'
        self.tentative_ui_code = self.ui_code
        self.run_code = run_code
        self.run_exception = ''
        self.ui_exception = ''
        self.ui_code_visible = False
    def try_execute(self, string, **kwargs):
        try:
            for key, value in kwargs.items():
                locals()[key] = value
            exec(string)
        except Exception as e: # while generally a bad idea, here we truly want to skip any potential error to not disrupt the worker threads
            return 'Exception: ' + str(e)
        return ''
    def loop(self, v):
        imgui.begin(self.name)
        
        self.run_code = imgui.input_text_multiline('run code', self.run_code, 2048)[1]
        if len(self.run_exception)>0:
            imgui.text(self.run_exception)

        _, self.ui_code_visible = imgui.checkbox('Show UI code', self.ui_code_visible)
        if self.ui_code_visible:
            self.tentative_ui_code = imgui.input_text_multiline('ui code', self.tentative_ui_code, 2048)[1]
            if imgui.button('Apply UI code'):
                self.ui_code = self.tentative_ui_code
            if len(self.ui_exception)>0:
                imgui.text(self.ui_exception)
                
        imgui.end()

        self.ui_exception = self.try_execute(self.ui_code, v=v)

    def run(self, **kwargs):
        self.run_exception = self.try_execute(self.run_code, **kwargs)


class viewer:
    def __init__(self, title, inifile=None, swap_interval=0, hidden=False, use_cuda=True):
        self.quit = False
        self.use_cuda = use_cuda
        self._images = {}
        self._editables = {}
        self.tex_interp_mode = gl.GL_LINEAR
        self.default_font_size = 36
        self._cuda_context = None
        
        fname = inifile or "".join(c for c in title.lower() if c.isalnum())
        self._inifile = Path(fname).with_suffix('.ini')

        if not glfw.init():
            raise RuntimeError('GLFW init failed')
        
        try:
            with open(self._inifile, 'r') as file:
                self._width, self._height = [int(i) for i in file.readline().split()]
                self.window_pos = [int(i) for i in file.readline().split()]
                start_maximized = int(file.readline().rstrip())
                self.ui_scale = float(file.readline().rstrip())
                self.fullscreen = bool(int(file.readline().rstrip()))
                key = file.readline().rstrip()
                while key is not None and len(key)>0:
                    code = [None, None]
                    for i in range(2):
                        lines = int(file.readline().rstrip())
                        code[i] = '\n'.join((file.readline().rstrip() for _ in range(lines)))
                    self._editables[key] = _editable(key, code[0], code[1])
                    key = file.readline().rstrip()
        except Exception as e:
            self._width, self._height = 1280, 720
            self.window_pos = (50, 50)
            self.ui_scale = 1.0
            self.fullscreen = False
            start_maximized = 0

        glfw.window_hint(glfw.MAXIMIZED, start_maximized)
        glfw.window_hint(glfw.VISIBLE, not hidden)
        
        # MacOS requires forward-compatible core profile
        if 'darwin' in platform:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        if self.fullscreen:
            monitor = glfw.get_monitors()[0]
            params = glfw.get_video_mode(monitor)
            self._window = glfw.create_window(params.size.width, params.size.height, title, monitor, None)
        else:
            self._window = glfw.create_window(self._width, self._height, title, None, None)
        
        if not self._window:
            raise RuntimeError('Could not create window')

        glfw.set_window_pos(self._window, *self.window_pos)
        
        glfw.make_context_current(self._window)
        print('GL context:', gl.glGetString(gl.GL_VERSION).decode('utf8'))
        
        if has_pycuda and use_cuda:
            # print(f'_cuda_context: {cuda_context}')
            try:
                import pycuda.gl.autoinit
            except:
                print('Failed to autoinit')
                pass
            
            try:
                import pycuda.gl.autoinit
                self._cuda_context = pycuda.gl.make_context(pycuda.driver.Device(0))
            except:
                print('Failed to make context')
                pass
                        
            # print(f'_cuda_context: {self._cuda_context}')
        glfw.swap_interval(swap_interval) # should increase on high refresh rate monitors
        glfw.make_context_current(None)

        self._imgui_context = imgui.create_context()
        # self._implot_context = implot.create_context()
        # implot.set_imgui_context(self._imgui_context)
        #implot.get_style().anti_aliased_lines = True # turn global AA on

        font = self.get_default_font()

        # MPLUSRounded1c-Medium.tff: no content for sizes >35
        font_sizes = range(8, 36, 2) if 'darwin' in platform else range(8, 36, 1) # Apple M1 has limit on GL texture count
        font_sizes = [int(s) for s in font_sizes]
        handle = imgui.get_io().fonts
        self._imgui_fonts = {
            size: handle.add_font_from_file_ttf(font, size,
                glyph_ranges=handle.get_glyph_ranges_chinese_full()) for size in font_sizes
        }

        self._context_lock = mp.Lock()
        self._context_tid = None # id of thread in critical section

    def get_default_font(self):
        return str(Path(__file__).parent / 'MPLUSRounded1c-Medium.ttf')
    
    def push_context(self):
        if has_pycuda and self.use_cuda:
            self._cuda_context.push()
    
    def pop_context(self):
        if has_pycuda and self.use_cuda:
            self._cuda_context.pop()

    @contextmanager
    def lock(self, strict=True):
        # Prevent double locks, e.g. when
        # calling upload_image() from UI thread
        tid = get_ident()
        if self._context_tid == tid:
            yield self._context_lock
            return
        
        context_manager = None
        
        try:
            self._context_lock.acquire()
            self._context_tid = tid
            glfw.make_context_current(self._window)
            context_manager = self._context_lock
        except glfw.GLFWError as e:
            reason = {65544: 'No monitor found'}.get(e.error_code, 'unknown')
            print(f'{str(e)} (code 0x{e.error_code:x}: "{reason}")')
            context_manager = nullcontext
            if strict:
                raise e
        finally:
            yield context_manager

            # Cleanup after caller is done
            glfw.make_context_current(None)
            self._context_tid = None
            self._context_lock.release()

    # Scales fonts and sliders/etc
    def set_ui_scale(self, scale):
        k = self.default_font_size
        self.set_font_size(k*scale)
        self.ui_scale = self.font_size / k

    def set_interp_linear(self, update_existing=True):
        if update_existing:
            for tex in self._images.values():
                tex.set_interp(gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                tex.set_interp(gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.tex_interp_mode = gl.GL_LINEAR

    def set_interp_nearest(self, update_existing=True):
        if update_existing:
            for tex in self._images.values():
                tex.set_interp(gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                tex.set_interp(gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        self.tex_interp_mode = gl.GL_NEAREST

    def editable(self, name, **kwargs):
        if name not in self._editables:
            self._editables[name] = _editable(name)
        self._editables[name].run(**kwargs)

    def keydown(self, key):
        return key in self._pressed_keys

    def keyhit(self, key):
        if key in self._hit_keys:
            self._hit_keys.remove(key)
            return True
        return False

    def draw_image(self, name, scale=1, width=None, pad_h=0, pad_v=0):
        if name in self._images:
            img = self._images[name]
            if width == 'fill':
                scale = imgui.get_window_content_region_width() / img.shape[1]
            elif width == 'fit':
                H, W = img.shape[0:2]
                cW, cH = [r-l for l,r in zip(
                    imgui.get_window_content_region_min(), imgui.get_window_content_region_max())]
                scale = min((cW-pad_h)/W, (cH-pad_v)/H)
            elif width is not None:
                scale = width / img.shape[1]
            imgui.image(img.tex, img.shape[1]*scale, img.shape[0]*scale)

    def close(self):
        glfw.set_window_should_close(self._window, True)

    @property
    def font_size(self):
        return self._cur_font_size

    @property
    def spacing(self):
        return round(self._cur_font_size * 0.3) # 0.4

    def set_font_size(self, target): # Applied on next frame.
        self._cur_font_size = min((abs(key - target), key) for key in self._imgui_fonts.keys())[1]

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.set_fullscreen(self.fullscreen)

    def set_fullscreen(self, value):
        monitor = glfw.get_monitors()[0]
        params = glfw.get_video_mode(monitor)
        if value:
            # Save size and pos
            self._width, self._height = glfw.get_window_size(self._window)
            self.window_pos = glfw.get_window_pos(self._window)
            glfw.set_window_monitor(self._window, monitor, \
                0, 0, params.size.width, params.size.height, params.refresh_rate)
        else:
            # Restore previous size and pos
            posy = max(10, self.window_pos[1]) # title bar at least partially visible
            glfw.set_window_monitor(self._window, None, \
                self.window_pos[0], posy, self._width, self._height, params.refresh_rate)

    def set_default_style(self, color_scheme='dark', spacing=9, indent=23, scrollbar=27):
        #theme_custom()        
        #theme_dark_overshifted()
        #theme_ps()
        theme_deep_dark()
        #theme_contrast()
        
        # Overrides based on UI scale / font size
        s = imgui.get_style()
        s.window_padding        = [spacing, spacing]
        s.item_spacing          = [spacing, spacing]
        s.item_inner_spacing    = [spacing, spacing]
        s.columns_min_spacing   = spacing
        s.indent_spacing        = indent
        s.scrollbar_size        = scrollbar

        c0 = s.colors[imgui.COLOR_MENUBAR_BACKGROUND]
        c1 = s.colors[imgui.COLOR_FRAME_BACKGROUND]
        s.colors[imgui.COLOR_POPUP_BACKGROUND] = [x * 0.7 + y * 0.3 for x, y in zip(c0, c1)][:3] + [1]

    def start(self, loopfunc, workers = (), glfw_init_callback = None):
        # allow single thread object
        print('Viewer start')
        if not hasattr(workers, '__len__'):
            workers = (workers,)

        for i in range(len(workers)):
            workers[i].start()

        self.set_ui_scale(self.ui_scale)        
        
        with self.lock():
            impl = GlfwRenderer(self._window)
        
        self._pressed_keys = set()
        self._hit_keys = set()

        def on_key(window, key, scan, pressed, mods):
            if pressed:
                if key not in self._pressed_keys:
                    self._hit_keys.add(key)
                self._pressed_keys.add(key)
            else:
                if key in self._pressed_keys:
                    self._pressed_keys.remove(key) # check seems to be needed over RDP sometimes
            if key != glfw.KEY_ESCAPE: # imgui erases text with escape (??)
                impl.keyboard_callback(window, key, scan, pressed, mods)

        glfw.set_key_callback(self._window, on_key)

        # For settings custom callbacks etc.
        if glfw_init_callback is not None:
            glfw_init_callback(self._window)

        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            impl.process_inputs()

            if self.keyhit(glfw.KEY_ESCAPE):
                glfw.set_window_should_close(self._window, 1)

            with self.lock(strict=False) as l:
                if l == nullcontext:
                    continue

                # Breaks on MacOS. Needed?
                #imgui.get_io().display_size = glfw.get_framebuffer_size(self._window)
                
                imgui.new_frame()

                # Tero viewer:
                imgui.push_font(self._imgui_fonts[self._cur_font_size])
                self.set_default_style(spacing=self.spacing, indent=self.font_size, scrollbar=self.font_size+4)
        
                loopfunc(self)

                for key in self._editables:
                    self._editables[key].loop(self)

                imgui.pop_font()
                imgui.render()
                
                gl.glClearColor(0, 0, 0, 1)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)

                impl.render(imgui.get_draw_data())
                
                # TODO: compute thread has to wait until sync is done
                # and lock is released if calling upload_image()?
                glfw.swap_buffers(self._window)
        
        # Update size and pos
        if not self.fullscreen:
            self._width, self._height = glfw.get_framebuffer_size(self._window)
            self.window_pos = glfw.get_window_pos(self._window)

        with open(self._inifile, 'w') as file:
            file.write('{} {}\n'.format(self._width, self._height))
            file.write('{} {}\n'.format(*self.window_pos))
            file.write('{}\n'.format(glfw.get_window_attrib(self._window, glfw.MAXIMIZED)))
            file.write('{}\n'.format(self.ui_scale))
            file.write('{}\n'.format(int(self.fullscreen)))
            for k, e in self._editables.items():
                file.write(k+'\n')
                for code in (e.ui_code, e.run_code):
                    lines = code.split('\n')
                    file.write(str(len(lines))+'\n')
                    for line in lines:
                        file.write(line+'\n')

        with self.lock():
            self.quit = True

        for i in range(len(workers)):
            workers[i].join()
            
        glfw.make_context_current(self._window)
        del self._images
        self._images = {}
        glfw.make_context_current(None)

        glfw.destroy_window(self._window)
        self.pop_context()
    
    @torch.no_grad()
    def upload_image(self, name, data):
        if torch.is_tensor(data):
            if data.device.type in ['mps', 'cpu'] or not self.use_cuda:
                # would require gl-metal interop or metal UI backend
                return self.upload_image_np(name, data.cpu().numpy())
            else:
                return self.upload_image_torch(name, data)
        else:
            return self.upload_image_np(name, data)

    # Upload image from PyTorch tensor
    @torch.no_grad()
    def upload_image_torch(self, name, tensor):
        assert isinstance(tensor, torch.Tensor)
        with self.lock(strict=False) as l:
            if l == nullcontext: # isinstance doesn't work
                return
            cuda_synchronize()
            if not self.quit:
                self.push_context() # set the context for whichever thread wants to upload
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode)
                self._images[name].upload_torch(tensor)
                self.pop_context()
    
    # Upload data from cuda pointer retrieved using custom TF op 
    def upload_image_TF_ptr(self, name, ptr, shape):
        with self.lock(strict=False) as l:
            if l == nullcontext: # isinstance doesn't work
                return
            cuda_synchronize()
            if not self.quit:
                self.push_context() # set the context for whichever thread wants to upload
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode)
                self._images[name].upload_ptr(ptr, shape)
                self.pop_context()

    def upload_image_np(self, name, data):
        assert isinstance(data, np.ndarray)
        with self.lock(strict=False) as l:
            if l == nullcontext: # isinstance doesn't work
                return
            # cuda_synchronize()
            if not self.quit:
                self.push_context() # set the context for whichever thread wants to upload
                if name not in self._images:
                    self._images[name] = _texture(self.tex_interp_mode)
                self._images[name].upload_np(data)
                self.pop_context()
