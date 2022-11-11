import multiprocessing as mp

from pathlib import Path
import numpy as np
import array
import pyviewer # pip install -e .
import imgui
import torch
import cv2
# Don't accidentally test different version
assert Path(pyviewer.__file__).parents[1] == Path(__file__).parents[1], \
    'Not running local editable install, please run "pip install --force-reinstall -e ."'

def toarr(a: np.ndarray):
    return array.array(a.dtype.char, a)

def demo():
    # siv = pyviewer.single_image_viewer.SingleImageViewer('Async viewer', hidden=False, use_cuda=False)
    class Test(pyviewer.toolbar_viewer.ToolbarViewer):
        
        def setup_state(self):
            self.state.seed = 0
            self.state.img = None
            self.state._backup_img = np.ascontiguousarray(cv2.imread('index.png')[..., ::-1])
        
        def compute(self):
            self.state.img = torch.from_numpy(self.state._backup_img).to('cuda:0')
            return self.state.img

        def draw_toolbar(self):
            self.state.seed = imgui.slider_int('Seed', self.state.seed, 0, 1000)[1]
            imgui.separator()
            imgui.text('Async viewer: separate process,\nwon\'t freeze if breakpoint is hit.')

    print('Test')
    _ = Test('test_viewer', use_cuda=True)
    print('Done')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    demo()