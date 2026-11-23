from config import *
from pytorch_msssim import ssim, ms_ssim
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
import torchac
import zlib
import shutil
import tempfile



def tensor2cuda(data, device):
    for k, v in data.items():
        if isinstance(v, list):
            data[k] = [elem.to(device) for elem in v]
        else:
            data[k] = v.to(device)
    return data


def torch2npimg(img, chans=3):
    img = img.detach().cpu()
    img = img.permute((1, 2, 0)).numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if chans == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


############### optimizer & scheduler ###############
def get_optimizer_scheduler(args, model, loader, param_group_fn=None):
    args = copy.deepcopy(args)
    args = easydict.EasyDict(args)
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(args), param_group_fn=param_group_fn)
    scheduler, _ = create_scheduler_v2(optimizer, **scheduler_kwargs(args), updates_per_epoch=len(loader))
    return optimizer, scheduler


############### loss function ###############
def loss_fn(pred, target, loss_type):
    target = target.detach()

    if 'L2' in loss_type:
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean()       
    elif 'L1' in loss_type:
        loss = torch.mean(torch.abs(pred - target))
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = 0.3 * torch.mean(torch.abs(pred - target)) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion4':
        loss = 0.5 * torch.mean(torch.abs(pred - target)) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion6':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * torch.mean(torch.abs(pred - target))
    elif loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * torch.mean(torch.abs(pred - target))
    elif loss_type == 'Fusion9':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion10':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion11':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion12':
        loss = 0.8 * torch.mean(torch.abs(pred - target)) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion13':
        loss = 0.7 * torch.mean(torch.abs(pred-target)) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss


############### metric ###############
def calculate_psnr(pred, gt):
    l2_loss = F.mse_loss(pred.detach(), gt.detach(), reduction='none')
    psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
    return psnr.cpu()

def calculate_msssim(pred, gt):
    msssim = ms_ssim(pred.float().detach(), gt.detach(), data_range=1, size_average=False)
    return msssim.cpu()


############### compression ###############

class QuantNoise(nn.Module):
    """
    Quant-Noise with optional STE.
    """
    def __init__(self, bitwidth, noise_ratio, ste, axis):
        super().__init__()
        self.register_buffer('bitwidth', torch.tensor(bitwidth, dtype=torch.float32))
        self.register_buffer('noise_ratio', torch.tensor(noise_ratio, dtype=torch.float32))
        self.ste = ste
        self.axis = axis

    def extra_repr(self):
        s = 'ste={ste}, axis={axis}'
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training:
            x_q, x_scale = _quantize_ste(x, self.bitwidth, self.axis)
            x_q = x_q if self.ste else x_q.detach()
            x_qr = x_q.to(x.dtype) * x_scale
            mask = (torch.rand_like(x) > self.noise_ratio).to(x.dtype)
            return x * mask + x_qr * (1. - mask)
        else:
            return x


def init_quantization(model, quant_level, quant_noise, quant_ste):
    """
    Initialize quantization for the model.
    """
    _quant_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ParameterList)
    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, _quant_target_cls):
                if isinstance(v, nn.ParameterList):
                    for n in range(len(v)):
                        quant_layer = QuantNoise(bitwidth=quant_level, noise_ratio=quant_noise, ste=quant_ste, axis=compute_best_quant_axis(v[n]))
                        quant_layer.to(list(v[n])[0].device)
                        nn.utils.parametrize.register_parametrization(v, str(n), quant_layer)
                else:
                    quant_layer = QuantNoise(bitwidth=quant_level, noise_ratio=quant_noise, ste=quant_ste, axis=compute_best_quant_axis(v.weight))
                    quant_layer.to(list(v.parameters())[0].device)
                    if 'head' in k:
                        nn.utils.parametrize.register_parametrization(v, 'weight', quant_layer)
                    else:
                        nn.utils.parametrize.register_parametrization(v, 'weight_orig', quant_layer)


# def set_quantization(model, quant_level, quant_noise, quant_ste):
#     """
#     Set quantization for the model.
#     """
#     with torch.no_grad():
#         for k, v in model.named_modules():
#             if isinstance(v, QuantNoise):
#                 v.bitwidth.copy_(quant_level)
#                 v.noise_ratio.copy_(quant_noise)
#                 v.ste = quant_ste

def remove_quantization(model):
    """
    remove quantization after fine-tuning is done
    """
    _quant_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ParameterList)
    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, _quant_target_cls):
                if isinstance(v, nn.ParameterList):
                    for n in range(len(v)):
                        nn.utils.parametrize.remove_parametrizations(v, str(n), False)
                else:
                    if 'head' in k:
                        nn.utils.parametrize.remove_parametrizations(v, 'weight', False)
                    else:
                        nn.utils.parametrize.remove_parametrizations(v, 'weight_orig', False)


class QuantSTE(nn.Module):
    """
    Quant-STE
    """
    def __init__(self, bitwidth):
        super().__init__()
        self.register_buffer('bitwidth', torch.tensor(bitwidth, dtype=torch.float32))
        # self.axis = axis

    def forward(self, x):
        weight = torch.tanh(x)
        weight_q = qfn.apply(weight, self.bitwidth)
        return weight_q

class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2 ** (k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out * torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def set_ste(model, bits):
    """
    Initialize quantization for the model.
    """
    _quant_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ParameterList)
    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, _quant_target_cls):
                # Grids
                if isinstance(v, nn.ParameterList):
                    for n in range(len(v)):
                        quant_layer = QuantSTE(bitwidth=bits)
                        quant_layer.to(list(v[n])[0].device)
                        nn.utils.parametrize.register_parametrization(v, str(n), quant_layer)
                # Convolutions
                else:
                    quant_layer = QuantSTE(bitwidth=bits)
                    quant_layer.to(list(v.parameters())[0].device)
                    if 'head' in k:
                        nn.utils.parametrize.register_parametrization(v, 'weight', quant_layer)
                    else:
                        nn.utils.parametrize.register_parametrization(v, 'weight_orig', quant_layer)

def remove_ste(model):
    """
    remove quantization after fine-tuning is done
    """
    _quant_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ParameterList)
    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, _quant_target_cls):
                if isinstance(v, nn.ParameterList):
                    for n in range(len(v)):
                        nn.utils.parametrize.remove_parametrizations(v, str(n), False)
                else:
                    if 'head' in k:
                        nn.utils.parametrize.remove_parametrizations(v, 'weight', False)
                    else:
                        nn.utils.parametrize.remove_parametrizations(v, 'weight_orig', False)


def compute_best_quant_axis(x, thres=0.05):
    """
    Compute the best quantization axis for a tensor. 
    Similar to the one used in HNeRV quantization: https://github.com/haochen-rye/HNeRV/blob/main/hnerv_utils.py#L26
    """
    best_axis = None
    best_axis_dim = 0
    for axis in range(x.ndim):
        dim = x.shape[axis]
        if x.numel() / dim >= x.numel() * thres:
            continue
        if dim > best_axis_dim:
            best_axis = axis
            best_axis_dim = dim
    return best_axis

def _ste(x):
    return (x.round() - x).detach() + x

def _quantize_ste(x, bits, axis=None):
    quant_range = 2. ** bits - 1.
    x_max = abs(x).max(dim=axis, keepdim=True)[0] if axis is not None else abs(x).max()
    x_scale = 2 * x_max / quant_range + 1e-6
    x_q = _ste(x / x_scale).clamp(-2**(bits - 1), 2**(bits - 1) - 1)
    return x_q, x_scale

def _quant_tensor(x, quant_level):
    """
    Quantize a tensor.
    """
    axis = compute_best_quant_axis(x)
    with torch.no_grad():
        x_q, x_scale = _quantize_ste(x, quant_level, axis)
        x_q = x_q.to(torch.int32)
        x_qr = x_q.to(x.dtype) * x_scale

    # Return the quantisied tensors (both int/float) and config
    meta = {
        'axis': axis,
        'scale': x_scale.half()
    }

    return x_q, x_qr, meta

def _dequant_tensor(quant, scale):
    new_x = quant * scale.expand_as(quant)
    return new_x



def _quant_model(model, quant_level):
    """
    Quantize the full model.
    """
    # Quantize
    qr_state_dict = copy.deepcopy(model.state_dict())
    q_state_dict = {}
    q_config = {'quant_level': quant_level}
    if quant_level == 32:
        pass
    elif quant_level <= 16:
        for k, v in model.state_dict().items():
            q_state_dict[k], qr_state_dict[k], q_config[k] = _quant_tensor(v, quant_level)
    else:
        raise ValueError

    # Load quantisied state dict
    model.load_state_dict(qr_state_dict)

    # Ruturn configs
    return q_state_dict, q_config

def _dequant_model(quant_ckt, quant_cfg):
    new_ckt = dict()
    for k in quant_ckt.keys():
        new_ckt[k] = _dequant_tensor(quant_ckt[k], quant_cfg[k]['scale'])
    return new_ckt


def _arithmetic_encoding(x):
    """ Arithmetic encoding for a tensor with torchac. """    
    with torch.no_grad():
        x = x.detach().view(-1).cpu()
        sym, inverse, counts = x.unique(return_inverse=True, return_counts=True)
        inverse = inverse.to(torch.int16)
        counts = torch.concat([torch.zeros([1], dtype=torch.int64, device=counts.device), counts])
        cdf = torch.cumsum(counts, dim=0).float() / counts.sum().float()
        byte_stream = torchac.encode_float_cdf(cdf[None].repeat(math.prod(x.shape), 1), inverse, check_input_bounds=True, needs_normalization=True)        
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(x.shape), 1), byte_stream).int()
        assert inverse_out.equal(inverse)
        x_out = sym[inverse_out.int()]
        assert x_out.equal(x)
        return sym, cdf, byte_stream

def _arithmetic_decoding(byte_stream, sym, cdf, shape):
    """ Arithmetic decoding for a tensor with torchac. """    
    with torch.no_grad():
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(shape), 1), byte_stream).int()
        x_out = sym[inverse_out.int()].view(shape)
        return x_out


def _compress_tensor(x, output_dir, name, dim=None):
    """ Compress a tensor and store its meta data. """
    # assert x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

    if x.numel() > 0:
        # arithmetic_encoding requires postive integers
        offset = x.min()
        x -= offset

        if dim is None:
            sym, cdf, bits = _arithmetic_encoding(x)
            bits_length = len(bits)
        else:
            sym = []
            cdf = []
            bits = None
            bits_length = []
            x_splits = torch.split(x, 1, dim=dim)
            for x_i in x_splits:
                sym_i, cdf_i, bits_i = _arithmetic_encoding(x_i.contiguous())
                sym.append(sym_i)
                cdf.append(cdf_i)
                bits = bits + bits_i if bits is not None else bits_i
                bits_length.append(len(bits_i))

        # meta data
        meta = {
            'dim': dim,
            'sym': sym,
            'cdf': cdf,
            'bits_length': bits_length,
            'shape': x.shape,
            'offset': offset,
        }
    else:
        # meta data
        meta = {
            'dim': None,
            'sym': None,
            'cdf': None,
            'bits_length': 0,
            'shape': None,
            'offset': None,
        }

    # Either compress the tensor and return the meta data, or return the tensor
    os.makedirs(output_dir, exist_ok=True)

    # byte_stream
    if x.numel() > 0:
        with open(os.path.join(output_dir, name + '.bit'), 'wb') as f:
            f.write(bits)

    return meta

def _decompress_tensor(state_dict, output_dir, name):
    """ Decompress a tensor. """
    # Read meta file
    dim = state_dict['dim']
    sym = state_dict['sym']
    cdf = state_dict['cdf']
    bits_length = state_dict['bits_length']
    shape = state_dict['shape']
    offset = state_dict['offset']

    # Read bitstreams
    if bits_length > 0:
        with open(os.path.join(output_dir, name + '.bit'), 'rb') as f:
            byte_stream = f.read()

        if dim is None:
            x = _arithmetic_decoding(byte_stream, sym, cdf, shape)
        else:
            x_splits = []
            bits_count = 0
            for i in range(shape[dim]):
                x_splits.append(_arithmetic_decoding(byte_stream[bits_count:bits_count + bits_length[i]], sym[i], cdf[i], list(shape[:dim]) + [1] + list(shape[dim+1:])))
                bits_count += bits_length[i]
            x = torch.concat(x_splits, dim=dim)

        x += offset.to(x.device)
    else:
        x = torch.zeros([0])

    return x


def _compress_model(model, bits, output_dir):
    # Quantization
    quant_ckt, quant_cfg = _quant_model(model, bits)

    # Arithmetic encoding
    comp_state_dict = dict()
    comp_state_dict['quant_cfg'] = quant_cfg
    for k in quant_ckt.keys():
        meta_k = _compress_tensor(quant_ckt[k], output_dir, k)
        comp_state_dict[k] = meta_k

    return comp_state_dict

def _decompress_model(comp_state_dict, output_dir):
    # Arithmetic decoding
    quant_state_dict = dict()
    quant_cfg = comp_state_dict.pop('quant_cfg')
    for k in comp_state_dict.keys():
        quant_v = _decompress_tensor(comp_state_dict[k], output_dir, k)
        quant_state_dict[k] = quant_v.to(quant_cfg[k]['scale'].device)

    # Dequantization
    decomp_state_dict = _dequant_model(quant_state_dict, quant_cfg)
    return decomp_state_dict


def compress_and_save_model(model, qbit, save_dir):
    output_dir = os.path.join(save_dir, f"Q{qbit}")
    temp_dir = tempfile.TemporaryDirectory(dir=save_dir)

    comp_state_dict = _compress_model(model, qbit, temp_dir.name)

    ckpt_path = os.path.join(temp_dir.name, 'model.pth.tar')
    compress_ckpt_path = os.path.join(temp_dir.name, 'model_compressed.pth.tar')

    # Save the state_dict
    torch.save(comp_state_dict, ckpt_path)
    # Compress the state_dict
    with open(ckpt_path, mode='rb') as f_in:
        with open(compress_ckpt_path, mode='wb') as f_out:
            f_out.write(zlib.compress(f_in.read(), zlib.Z_NO_COMPRESSION))
    os.remove(ckpt_path)
    
    shutil.make_archive(output_dir, 'zip', temp_dir.name)
    temp_dir.cleanup()

    num_bytes = os.path.getsize(output_dir + '.zip')
    return num_bytes

def decompress_and_load_model(model, qbit, save_dir):
    output_dir = os.path.join(save_dir, f"Q{qbit}")
    temp_dir = tempfile.TemporaryDirectory(dir=save_dir)

    ckpt_path = os.path.join(temp_dir.name, 'model.pth.tar')
    compress_ckpt_path = os.path.join(temp_dir.name, 'model_compressed.pth.tar')

    shutil.unpack_archive(output_dir + '.zip', temp_dir.name)
    # Decompress the state_dict
    with open(compress_ckpt_path, mode='rb') as f_in:
        with open(ckpt_path, mode='wb') as f_out:
            f_out.write(zlib.decompress(f_in.read()))

    comp_state_dict = torch.load(ckpt_path)
    decomp_state_dict = _decompress_model(comp_state_dict, temp_dir.name)

    temp_dir.cleanup()
    return decomp_state_dict
