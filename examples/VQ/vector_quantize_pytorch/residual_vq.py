from functools import partial
import torch
from torch import nn
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
import torch.nn.functional as F

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        shared_codebook = False,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])
        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))
        return quantized_out, all_indices, all_losses

class VariableRVQ(nn.Module):
    def __init__(
        self,
        *,
        dim,
        CB_size_list,
        **kwargs
    ):
        super().__init__()
        self.num_quantizers = len(CB_size_list)
        self.layers = nn.ModuleList([VectorQuantize(dim, CB_size_list[i], **kwargs) for i in range(self.num_quantizers)])
        self.dim = dim
        self.CB_size_list = CB_size_list        
    def forward(self, x):
        quantized_out = torch.zeros_like(x)
        mse_list = []
        residual = x
        for i, layer in enumerate(self.layers):
            quantized, indices, loss, usage = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            indices_cat = indices.unsqueeze(1) if i == 0 else torch.cat([indices_cat, indices.unsqueeze(1)], 1)
            loss_cat = loss if i == 0 else torch.cat([loss_cat, loss], 0)
            usage = usage.unsqueeze(0)
            perplexity_cat = usage if i == 0 else torch.cat([perplexity_cat, usage], 0)
            mse_list.append(F.mse_loss(x, quantized_out))
                        
        return quantized_out, indices_cat, loss_cat, perplexity_cat, mse_list # (b, Q, w, h), (b, Q, w, h), (b), (b)

class MultiLayerVQ(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])
        self.num_quantizers = num_quantizers
    def forward(self, x):
        cb_dim = self.layers[0].codebook_dim
        for i in range(self.num_quantizers):
            part_x = x[:, i*cb_dim:(i + 1)*cb_dim, :, :]
            quantized, indices, loss, usage = self.layers[i](part_x)   # (b, Q, w, h), (b, w, h), (b)
            quantized_cat = quantized if i == 0 else torch.cat([quantized_cat, quantized], 1)
            indices_cat = indices.unsqueeze(1) if i == 0 else torch.cat([indices_cat, indices.unsqueeze(1)], 1)
            loss_cat = loss if i == 0 else torch.cat([loss_cat, loss], 0)
            usage = usage.unsqueeze(0)
            perplexity_cat = usage if i == 0 else torch.cat([perplexity_cat, usage], 0)
#         perplexity = perplexity_cat.mean()
        return quantized_cat, indices_cat, loss_cat, perplexity_cat # (b, Q, w, h), (b, Q, w, h), (b), (b)

    def vq_recon(self, indices_cat):
        for i in range(self.num_quantizers):
            recon = self.layers[i].codebook[indices_cat[:, i]]  # (b, w, h, Q)
            recon = recon.permute(0, 3, 1, 2)                   # (b, Q, w, h)
            recon_cat = recon if i == 0 else torch.cat([recon_cat, recon], 1)
        return recon_cat    
class HierarchicalVQ(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers,
        dim_list,
        CB_size_list,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(dim_list[i], CB_size_list[i], **kwargs) for i in range(num_quantizers)])
        self.num_quantizers =  num_quantizers
        self.dim_list = dim_list
        self.CB_size_list = CB_size_list

    def forward(self, x):
        cb_dim = self.layers[0].codebook_dim
        for i in range(self.num_quantizers):
            start = 0 if i == 0 else sum(self.dim_list[:i])
            end = self.dim_list[0] if i == 0 else sum(self.dim_list[:i+1])            
            part_x = x[:, start:end, :, :]            
            quantized, indices, loss, usage = self.layers[i](part_x)   # (b, Q, w, h), (b, w, h), (b)
            quantized_cat = quantized if i == 0 else torch.cat([quantized_cat, quantized], 1)
            indices_cat = indices.unsqueeze(1) if i == 0 else torch.cat([indices_cat, indices.unsqueeze(1)], 1)
            loss_cat = loss if i == 0 else torch.cat([loss_cat, loss], 0)
            usage = usage.unsqueeze(0)
            perplexity_cat = usage if i == 0 else torch.cat([perplexity_cat, usage], 0)
#         perplexity = perplexity_cat.mean()
        return quantized_cat, indices_cat, loss_cat, perplexity_cat # (b, Q, w, h), (b, Q, w, h), (b), (b)
class BlockVQ(nn.Module):
    def __init__(
        self,
        *,
        block_len,
        CB_size_list,
        **kwargs
    ):
        super().__init__()
        self.dim = block_len * block_len
        self.BL = block_len
        self.layers = nn.ModuleList([VectorQuantize(self.dim, CB_size_list[i], **kwargs) for i in range(len(CB_size_list))])
        self.CB_size_list = CB_size_list
    def forward(self, x):
        b, c, h, w = x.shape
        x_hat = torch.zeros_like(x)
        for i, layer in enumerate(self.layers):
            vec_in_vq = block2vector(x[:, i], self.BL)
            quantized, indices, loss, usage = layer(vec_in_vq)
            x_hat[:, i] = vector2block(quantized, self.BL, h, w)
            indices_cat = indices.unsqueeze(1) if i == 0 else torch.cat([indices_cat, indices.unsqueeze(1)], 1)
            loss_cat = loss if i == 0 else torch.cat([loss_cat, loss], 0)
            usage = usage.unsqueeze(0)
            perplexity_cat = usage if i == 0 else torch.cat([perplexity_cat, usage], 0)                        
        return x_hat, indices_cat, loss_cat, perplexity_cat # (b, Q, w, h), (b, Q, w, h), (b), (b)
    
def block2vector(block, BL):
    b, h, w = block.shape
    device = block.device    
    
    h_times = h // BL
    w_times = w // BL    
    vec_num = h_times * w_times     # vector count in channel i
    vec_len = BL * BL # each block would flatten into vector, so dim = len^2    
    vector = torch.zeros(b, vec_num, vec_len).to(device)
    
    for j in range(h_times):
        for k in range(w_times):
            vec = block[:, j*BL: (j+1)*BL, k*BL: (k+1)*BL]
            vec_flat = vec.reshape(b, vec_len)
            vector[:, j*h_times+k] = vec_flat
    return vector
    
def vector2block(vector, BL, h, w):
    b, vec_num, vec_len = vector.shape
    device = vector.device    
    n_times = vec_num // BL
    block = torch.zeros(b, h, w).to(device)    
    for j in range(n_times):
        for k in range(n_times):
            tmp = vector[:, j*n_times+k].reshape(b, BL, BL)            
            block[:, j*BL: (j+1)*BL, k*BL: (k+1)*BL] = tmp
    return block

    
    
    
    
    
    