import torch

def merge_template_search(inp_list, return_search=False, return_template=False):
    """NOTICE: search region related features must be in the last place"""
    seq_dict = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
                "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
                "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}
    if return_search:
        x = inp_list[-1]
        seq_dict.update({"feat_x": x["feat"], "mask_x": x["mask"], "pos_x": x["pos"]})
    if return_template:
        z = inp_list[0]
        seq_dict.update({"feat_z": z["feat"], "mask_z": z["mask"], "pos_z": z["pos"]})
    return seq_dict


def get_qkv(inp_list):
    """The 1st element of the inp_list is about the template,
    the 2nd (the last) element is about the search region,
    dict_x = inp_list[-1] means the last element is the search region"""
    dict_x = inp_list[-1]
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    q = dict_x["feat"] + dict_x["pos"]
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]

    """DEBUG BEGIN HERE"""
    # scr = dict_x["feat"] .clone().detach().unsqueeze(0)
    # plot_map(scr,_type= "T (H W) B C",T=1,H=20,W=20,B=2,C=128)
    # m, n = dict_x["mask"].shape
    # a = torch.zeros(m,n)
    # for i in range(m):
    #     for j in range(n):
    #         if not dict_x["mask"][i][j]:
    #             a[i][j] = 2.5
    # a = a.unsqueeze(0)
    # a = a.unsqueeze(0)
    # plot_map(a, _type="T C B (H W)", T=1, H=20, W=20, B=2, C=1)
    """DEBUG END HERE"""

    key_padding_mask = dict_c["mask"]
    return q, k, v, key_padding_mask

def get_space_time_qkv(inp_list):
    """ALL element of the inp_list is about the template/search
    region"""
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    q = dict_c["feat"] + dict_c["pos"]
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]

    """DEBUG BEGIN HERE"""
    # scr = dict_x["feat"] .clone().detach().unsqueeze(0)
    # plot_map(scr,_type= "T (H W) B C",T=1,H=20,W=20,B=2,C=128)
    # m, n = dict_x["mask"].shape
    # a = torch.zeros(m,n)
    # for i in range(m):
    #     for j in range(n):
    #         if not dict_x["mask"][i][j]:
    #             a[i][j] = 2.5
    # a = a.unsqueeze(0)
    # a = a.unsqueeze(0)
    # plot_map(a, _type="T C B (H W)", T=1, H=20, W=20, B=2, C=1)
    """DEBUG END HERE"""

    key_padding_mask = dict_c["mask"]
    return q, k, v, key_padding_mask,dict_c["pos"]

def get_FS_qkv(inp_list):
    """ALL element of the inp_list is about the template/search
    region"""
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    #q = dict_c["feat"] + dict_c["pos"] q is a learnable parameter in FS
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]

    key_padding_mask = dict_c["mask"]
    return k, v, key_padding_mask,dict_c["pos"]



def get_FS_qkv(inp_list):
    """ALL element of the inp_list is about the template/search
    region"""
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    #q = dict_c["feat"] + dict_c["pos"] q is a learnable parameter in FS
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]

    key_padding_mask = dict_c["mask"]
    return k, v, key_padding_mask,dict_c["pos"]

def get_encoder_qkv(inp_list):
    """The 1st element of the inp_list is about the template,
    the 2nd (the last) element is about the search region,
    dict_x = inp_list[-1] means the last element is the search region"""
    dict_x = inp_list[-1]
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    q = dict_x["feat"] + dict_x["pos"]
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]
    return v, dict_c["pos"],dict_c["mask"]


def get_temp_scr(feat,temp_len,scr_len):
    return feat[:temp_len],feat[:scr_len]