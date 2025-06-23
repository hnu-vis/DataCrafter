import os

class BackendModel(object):
    def __init__(self, dataname=None, step=0, case_mode=True):
        # Initialize parameters, including dataset name, step, case mode, etc.
        self.dataname = dataname
        self.step = step
        self.case_mode = case_mode
        self.nearest_actions = 3
        self.nearest_frames = 4
        self.window_size = 5
        if dataname is None:
            return 
        self._init()
    
    def update_data_root(self, dataname, step):
        # Update the dataset name and step
        self.step = step

    def _init(self):
        # Initialize any additional setup if necessary
        pass

    def reset(self, dataname, step):
        # Reset the model with a new dataset name and step
        self.dataname = dataname
        self.step = step
        self._init()

    
    def buffer_exist(self, path=None):
        # Check if the buffer path exists; use the provided path or default buffer path
        buffer_path = self.buffer_path
        if path:
            buffer_path = path
        if os.path.exists(buffer_path):
            return True
        else:
            return False
