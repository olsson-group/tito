def count_parameters(mod):
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)
