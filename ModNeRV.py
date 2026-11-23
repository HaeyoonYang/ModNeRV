from config import *



class ModNeRV(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Encoder
        self.encoder = Encoder(cfg)
        # Decoder
        self.decoder = Decoder(cfg, input_chans=self.encoder.out_chans)

    def forward(self, data):
        dec_start = time.time()
        t = data['t']

        # Encoder
        input_enc = self.encoder(t)
        # Decoder
        out_img, _ = self.decoder(input_enc, t)
        dec_time = time.time() - dec_start
        return out_img, dec_time

    def count_params(self):
        enc_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        model_params = enc_params + dec_params
        return {
            'model_params': model_params / 1e6,
            'enc_params': enc_params / 1e6,
            'dec_params': dec_params / 1e6,
        }



class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """
        base_resolutions: [[c1, h1, w1], [c2, h2, w2], ...], h1 < h2, w1 < w2 (in ascending order in terms of resolution)
        """

        # hyperparameters
        base_resolutions = cfg['enc_base']
        base_resolutions.sort(key=lambda x: x[1])
        self.bases = nn.ParameterList()
        self.gammabetas = nn.ParameterList()

        # set base & gamma/beta parameters
        for br in base_resolutions:
            gb_resolution = (cfg['video_size'][0], 2, br[0])
            self.bases.append(nn.Parameter(nn.init.uniform_(torch.empty(br, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True))
            self.gammabetas.append(nn.Parameter(nn.init.uniform_(torch.empty(gb_resolution, dtype=torch.float32), -1e-3, 1e-3), requires_grad=True))

        self.out_chans = sum([base_resolutions[d][0] for d in range(len(base_resolutions))])
        self.out_size = [self.out_chans] + base_resolutions[-1][1:]

    def forward(self, t):
        out_feats = []
        for l_base, gb in zip(self.bases, self.gammabetas):
            # gamma, beta value for t-th frame
            gamma = gb[t, 0]
            beta = gb[t, 1]

            l_feat = l_base[None, ...] * gamma[..., None, None] + beta[..., None, None]

            # Concatenate multi-resolution bases along the channel dimension after matching the resolutions
            if l_feat.shape[-2:] != self.out_size[-2:]:
                l_feat = F.interpolate(l_feat, size=self.out_size[-2:], mode='bilinear', align_corners=True)
            out_feats.append(l_feat)

        out_feats = torch.concat(out_feats, dim=1)          # [B, C, H, W]
        return out_feats


class Decoder(nn.Module):
    def __init__(self, cfg, input_chans):
        super().__init__()

        # hyperparameters
        ngf = cfg['dec_dim']
        strides = cfg['dec_strides']
        depths = cfg['dec_depths']

        self.stem = nn.Conv2d(input_chans, ngf, 3, padding='same')

        self.blocks = nn.ModuleList()
        for i, stride in enumerate(strides):
            if i == 0:
                new_ngf = ngf
            else:
                reduction = np.sqrt(stride) if cfg['reduction']==-1 else cfg['reduction']
                new_ngf = int(max(ngf / (1 if stride==1 else reduction), cfg['lower_width']))

            self.blocks.append(ModNeRVBlock(ngf, new_ngf, stride, ai_resolution=(cfg['video_size'][0], cfg['at_dim']), depth=depths[i]))
            ngf = new_ngf

        self.head = nn.Sequential(
            nn.Conv2d(ngf, cfg['video_size'][1], kernel_size=1),
            nn.Sigmoid(),
        )

        # initialization
        self.apply(self._init_blocks)

    def _init_blocks(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_enc, t):
        dec_start = time.time()

        x = self.stem(input_enc)
        for i, block in enumerate(self.blocks):
            x = block(x, t)
        x = self.head(x)

        dec_time = time.time() - dec_start
        return x, dec_time


class ModNeRVBlock(nn.Module):
    def __init__(self, ngf, new_ngf, stride, ai_resolution, depth=1):
        super().__init__()

        # AdaIN
        self.norm = nn.InstanceNorm2d(ngf, affine=False)
        self.gammas = nn.Parameter(nn.init.xavier_uniform_(torch.empty(ai_resolution, dtype=torch.float32)))
        self.betas = nn.Parameter(nn.init.xavier_uniform_(torch.empty(ai_resolution, dtype=torch.float32)))
        self.g_linear = nn.Linear(ai_resolution[-1], ngf)
        self.b_linear = nn.Linear(ai_resolution[-1], ngf)

        # convolution
        self.conv = []
        for i in range(depth):
            self.conv.append(nn.Conv2d(ngf, ngf, 1))
            self.conv.append(nn.Conv2d(ngf, ngf, 3, padding='same', groups=ngf))
            self.conv.append(nn.GELU())
        self.conv = nn.Sequential(*self.conv)
        self.pconv = nn.Conv2d(ngf, new_ngf*stride**2, 1)
        self.ps = nn.PixelShuffle(stride) if stride !=1 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, input, t):
        # AdaIN
        x = self.norm(input)
        gamma = self.g_linear(self.gammas[t])
        beta = self.b_linear(self.betas[t])
        x = x * gamma[..., None, None] + beta[..., None, None]

        # convolution
        x = self.conv(x)
        x = x + input
        x = self.act(self.ps(self.pconv(x)))
        return x




if __name__ == "__main__":
    # calculate the model size
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', default='cfgs/uvg-12.5M.yaml', type=str)
    args = parser.parse_args()

    with open(args.opt_path, 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    cfg['model']['video_size'] = [600, 3, 1080, 1920]
    model = ModNeRV(cfg['model'])

    params = model.count_params()
    print("Enc_%.3fM_Dec_%.3fM_Total_%.3fM" % (params['enc_params'], params['dec_params'], params['model_params']))

    data = {'t': torch.tensor([4]),}
    out_img, _ = model(data)
    print(out_img.shape)