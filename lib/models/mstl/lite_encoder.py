"""
(2021.06.27)
Transformer encoder class (Lite version)
-- Only use one layer of encoder
-- search region as queries, "concat" as keys and values
-- only pass the search region to the FFN
-- functions takes standard pytorch Tensor as input (for TensorRT)
"""
from typing import Optional
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from lib.utils.merge import get_qkv,get_space_time_qkv,get_FS_qkv,get_encoder_qkv,get_temp_scr

class TransformerEncoderLayerLite(nn.Module):
    """One lite encoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        # from plot_features.visual712 import PLT_MAP
        # self.draw = PLT_MAP(2,2)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor] = None):
        """ q, k, v denote queries, keys, and values respectively """
        # s = time.time()
        src2 = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)[0]
        src = q + self.dropout1(src2)
        src = self.norm1(src)
        # e1 = time.time()
        #self.draw.plot(window_id=0,tensor=src.unsqueeze(0),_type='T (H W) B C',B=1,T=1,C=128,W=16,H=16 )
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        #self.draw.plot(window_id=1, tensor=src.unsqueeze(0), _type='T (H W) B C', B=1, T=1, C=128, W=16, H=16)
        # e2 = time.time()
        # print("self-attention time: %.1f" % ((e1-s) * 1000))
        # print("MLP time: %.1f" % ((e2-e1) * 1000))
        return src


class TransformerEncoderLite(nn.Module):
    """search feature as queries, concatenated feature as keys and values"""
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        self.encoder = TransformerEncoderLayerLite(d_model, nhead, dim_feedforward, dropout, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor] = None):
        memory = self.encoder(q, k, v, key_padding_mask=key_padding_mask)
        return memory


def build_lite_encoder(cfg):
    print("Building lite transformer encoder...")
    encoder = TransformerEncoderLite(d_model=cfg.MODEL.HIDDEN_DIM, dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
                                     nhead=cfg.MODEL.TRANSFORMER.NHEADS,
                                     dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD)
    return encoder

from lib.models.stark.transformer import Transformer
class TransformerEncodersLite(nn.Module):
    """search feature as queries, concatenated feature as keys and values"""
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",Encoders = None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        #use all as q,k,v
        self.encoders = Encoders
        #use query as qery
        self.decoder = TransformerEncoderLayerLite(d_model, nhead, dim_feedforward, dropout, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_dict_list, mode="encoder", return_encoder_output=False,scr_shape = 16 ):
        if self.encoders is not None:
            feat, pos, mask = get_encoder_qkv(feat_dict_list)
            memory = self.encoders(feat=feat, mask=mask, query_embed=None, pos_embed=pos, mode= mode)
            feat_pos = 0
            for i in range(len(feat_dict_list)):
                feat_len = feat_dict_list[i]['feat'].shape[0]
                feat_dict_list[i]['feat'] = memory[feat_pos : feat_pos+feat_len]
                feat_pos = feat_pos+feat_len
            #get q,k,v from memory of (template,previou,scr  )  -> ([template,previou],scr )
            """use the template and previous in the memory ,and pos&mask in the feat_dict_list
            to form the input of the q,k,v"""
        q,k,v,mask = get_qkv(feat_dict_list)
        memory =self.decoder(q, k, v, key_padding_mask=mask)
        return memory



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_lite_encoders(cfg):
    print("Building lite transformer encoder...")
    num_encoder_layers =  cfg.MODEL.TRANSFORMER.ENC_LAYERS
    if  num_encoder_layers> 1:
        Encoders = Transformer(
            d_model=cfg.MODEL.HIDDEN_DIM,
            dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
            nhead=cfg.MODEL.TRANSFORMER.NHEADS,
            dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
            num_encoder_layers=num_encoder_layers -1,
            num_decoder_layers=cfg.MODEL.TRANSFORMER.DEC_LAYERS,
            normalize_before=cfg.MODEL.TRANSFORMER.PRE_NORM,
            return_intermediate_dec=False,  # we use false to avoid DDP error,
            divide_norm=cfg.MODEL.TRANSFORMER.DIVIDE_NORM
            )
    else:
        Encoders = None
    encoder = TransformerEncodersLite(d_model=cfg.MODEL.HIDDEN_DIM, dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
                                     nhead=cfg.MODEL.TRANSFORMER.NHEADS,
                                     dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
                                      Encoders  = Encoders )
    return encoder


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
