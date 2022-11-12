
import torch
def load_pretrain_model(path,model,load_keys = None):
    # Load network
    print("Loading pretrained model from ", path)
    checkpoint_dict = torch.load(path, map_location='cpu')
    net_type = type(model).__name__
    assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'
    if load_keys is None:
        missing_k, unexpected_k = model.load_state_dict(checkpoint_dict["net"], strict=False)
        #print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)
    else:
        state_dict = {}
        for k, v in checkpoint_dict["net"].items():
            for keys in load_keys:
                if keys in k:
                    state_dict [k] = v
        missing_k, unexpected_k = model.load_state_dict( state_dict , strict=False)
        #print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)


