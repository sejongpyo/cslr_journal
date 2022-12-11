import torch

def load_chk_point(model, state_dict):
    model.load_state_dict(state_dict["model_state_dict"])

    return model