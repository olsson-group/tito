import torch


class Module(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device


class DeviceTracker(Module):
    def __init__(self):
        super().__init__()
        self.device_tracker = torch.nn.Parameter(torch.tensor(1.0))
        #  self.register_buffer("device_tracker", torch.tensor(1))

    #  @property
    #  def device(self):
    #      return self.device_tracker.device
