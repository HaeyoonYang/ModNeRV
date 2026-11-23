from config import *
from dataloader import dataset
from ModNeRV import ModNeRV
from utils import *
from train import evaluate



def get_args_parse():
    parser = argparse.ArgumentParser('ModNeRV compression', add_help=False)

    parser.add_argument('--compress_folder', default='experiments/~', type=str, help='name of the folder to fine-tune after prune/quant')
    # pruning
    parser.add_argument('--prune_epochs', default=1, type=int, help='training epochs for fine-tuning')
    parser.add_argument('--prune_warmup_epochs', default=0, type=int, help='epochs for lr warmup')
    parser.add_argument('--prune_lr_scale', default=0.1, type=float, help='lr scaling for fine-tuning')
    parser.add_argument('--prune_ratio', default=0.1, type=float, help='prune ratio')
    # quantization
    parser.add_argument('--quant_epochs', default=1, type=int, help='training epochs for fine-tuning')
    parser.add_argument('--quant_warmup_epochs', default=0, type=int, help='epochs for lr warmup')
    parser.add_argument('--quant_lr_scale', default=0.01, type=float, help='lr scaling for fine-tuning')
    parser.add_argument('--quant_levels', default=[8], nargs='+', type=int, help='bit-values to quantize')
    parser.add_argument('--quant_noise', default=0.9, type=float, help='quant noise')
    parser.add_argument('--quant_ste', default=False, action='store_true', help='whether to do ste(straight through estimator) or not')
    parser.add_argument('--resume_folder', default=None, type=str, help='name of the folder to resume fine-tuning')

    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get cfg yaml file
    if args.resume_folder is None:
        with open(os.path.join(args.compress_folder, 'config.yaml'), 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

            cfg['compress'] = dict()
            cfg['compress']['prune_optim'] = copy.deepcopy(cfg['train']['optim'])
            cfg['compress']['prune_optim']['epochs'] = args.prune_epochs
            cfg['compress']['prune_optim']['warmup_epochs'] = args.prune_warmup_epochs
            cfg['compress']['prune_optim']['lr'] *= args.prune_lr_scale
            cfg['compress']['prune_ratio'] = args.prune_ratio

            cfg['compress']['quant_optim'] = copy.deepcopy(cfg['train']['optim'])
            cfg['compress']['quant_optim']['epochs'] = args.quant_epochs
            cfg['compress']['quant_optim']['warmup_epochs'] = args.quant_warmup_epochs
            cfg['compress']['quant_optim']['lr'] *= args.quant_lr_scale
            cfg['compress']['quant_levels'] = args.quant_levels
            cfg['compress']['quant_noise'] = args.quant_noise
            cfg['compress']['quant_ste'] = args.quant_ste
    else:
        with open(os.path.join(args.resume_folder, 'config.yaml'), 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # fix the seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # set directory to save & set logger
    tm = time.localtime()
    tm_str = time.strftime("%Y%m%d_%H%M%S", tm)
    if args.resume_folder is not None:
        SAVE_DIR = args.resume_folder
    else:
        n_str = os.path.basename(os.path.normpath(args.compress_folder)) + f'-pq_e{args.prune_epochs}{args.quant_epochs}_pratio{args.prune_ratio}_qlevel{min(args.quant_levels)}'
        SAVE_DIR = os.path.join("experiments", cfg['exp_name'], cfg['data']['dataset'], n_str)
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(SAVE_DIR, "ckpts"), exist_ok=True)

    loggername = os.path.join(SAVE_DIR, 'logs.txt')
    f = open(loggername, 'a')
    writer = SummaryWriter(os.path.join(SAVE_DIR, 'tensorboard_%s' % tm_str))

    for k, v in sorted((args.__dict__).items()):
        f.write("\t%15s: \t%s\n" % (k, v))
        print("\t%15s: \t%s" % (k, v))
    for k, v in sorted(cfg.items()):
        f.write("\t%15s: \t%s\n" % (k, v))
        print("\t%15s: \t%s" % (k, v))

    # load dataset
    trainset = dataset(cfg['data'])
    valset = dataset(cfg['data'])

    trainloader = DataLoader(
        trainset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        drop_last=False,
    )
    valloader = DataLoader(
        valset,
        batch_size=cfg['val']['batch_size'],
        shuffle=False,
        num_workers=cfg['val']['num_workers'],
        drop_last=False,
    )

    if args.resume_folder is None:
        with open(os.path.join(SAVE_DIR, 'config.yaml'), 'w') as fp:
            yaml.safe_dump(cfg, fp)

    init_epoch = -1
    # load model & weights
    model = ModNeRV(cfg['model']).to(device)
    if args.resume_folder is None:
        weight_dir = os.path.join(args.compress_folder, 'ckpts/model_latest.pth')
        assert os.path.exists(weight_dir), "No WEIGHT FILE !!!"
        ckpt = torch.load(weight_dir)
        model.load_state_dict(ckpt['network'])
    else:
        weight_dir = os.path.join(args.resume_folder, 'ckpts/model_latest.pth')
        assert os.path.exists(weight_dir), "No WEIGHT FILE !!!"
        ckpt = torch.load(weight_dir)
        init_epoch = ckpt['epoch_idx']

    # set optimizer & scheduler
    prune_optimizer, prune_scheduler = get_optimizer_scheduler(cfg['compress']['prune_optim'], model, trainloader) if cfg['compress']['prune_optim']['epochs'] > 0 else (None, None)
    quant_optimizer, quant_scheduler = get_optimizer_scheduler(cfg['compress']['quant_optim'], model, trainloader) if cfg['compress']['quant_optim']['epochs'] > 0 else (None, None)

    # set pruning on model params
    prune_net = cfg['compress']['prune_ratio'] < 1
    if prune_net:
        param_list = []
        for k, v in model.named_parameters():
            if 'weight' in k:
                if 'stem' in k:
                    param_list.append(model.decoder.stem)
                elif 'blocks' in k:
                    block_ind = int(k.split('.')[2])
                    if 'pconv' in k:
                        param_list.append(model.decoder.blocks[block_ind].pconv)
                    elif 'conv' in k:
                        conv_ind = int(k.split('.')[4])
                        param_list.append(model.decoder.blocks[block_ind].conv[conv_ind])
                    elif 'g_linear' in k:
                        param_list.append(model.decoder.blocks[block_ind].g_linear)
                    elif 'b_linear' in k:
                        param_list.append(model.decoder.blocks[block_ind].b_linear)
                # elif 'head' in k:
                #     head_ind = int(k.split('.')[2])
                #     param_list.append(model.decoder.head[head_ind])
        param_to_prune = [(ele, 'weight') for ele in param_list]
        prune_base_ratio = cfg['compress']['prune_ratio']
        prune.global_unstructured(
            param_to_prune,
            pruning_method=prune.L1Unstructured,
            amount = prune_base_ratio,
        )

    if init_epoch < 0:
        pass
    elif init_epoch < cfg['compress']['prune_optim']['epochs']-1:
        model.load_state_dict(ckpt['network'])
    elif init_epoch < cfg['compress']['prune_optim']['epochs'] + cfg['compress']['quant_optim']['epochs'] - 1:
        init_quantization(model, min(cfg['compress']['quant_levels']), cfg['compress']['quant_noise'], cfg['compress']['quant_ste'])
        model.load_state_dict(ckpt['network'])

    # get model params
    params = model.count_params()
    param_str = f"Encoder_{round(params['enc_params'], 3)}M_Decoder_{round(params['dec_params'], 3)}M_Total_{round(params['model_params'], 3)}M"
    f.write(str(model) + '\n' + f'Model Parameters: {param_str}\n')
    print("Model Parameters: ", param_str)

    # compute sparsity
    sparsity_num = 0.
    for k, v in model.named_buffers():
        if 'weight_mask' in k:
            sparsity_num += (v == 0).sum()
    f.write(f"Number of pruned parameters: {sparsity_num}\ttotal: {params['model_params']*1e6}\n")
    f.write(f"Model sparsity: {sparsity_num / 1e6 / params['model_params']}\n")
    print(f"Number of pruned parameters: {sparsity_num}\ttotal: {params['model_params']*1e6}")
    print(f"Model sparsity: {sparsity_num / 1e6 / params['model_params']}")

    # load from resume_folder
    if args.resume_folder is not None:
        prune_optimizer.load_state_dict(ckpt['prune_optimizer'])
        prune_scheduler.load_state_dict(ckpt['prune_scheduler'])
        quant_optimizer.load_state_dict(ckpt['quant_optimizer'])
        quant_scheduler.load_state_dict(ckpt['quant_scheduler'])
        f.write("Loaded previous weights from %s. Current epoch = #%d\n" % (weight_dir, init_epoch+1))
        print("Loaded previous weights from %s. Current epoch = #%d" % (weight_dir, init_epoch+1))
    else:
        f.write("Start fine-tuning. Current epoch = #%d\n" % (init_epoch+1))
        print("Start fine-tuning. Current epoch = #%d" % (init_epoch+1))

    cur_epoch = init_epoch
    start_tm = datetime.now()
    # Fine-tuning for pruning
    if cur_epoch < cfg['compress']['prune_optim']['epochs'] - 1:
        f.write("Start fine-tuning for pruning\n")
        print("Start fine-tuning for pruning")
        for epoch_idx in range(cur_epoch+1, cfg['compress']['prune_optim']['epochs']):
            psnr_list, ssim_list = [], []
            epoch_loss = 0
            epoch_r_loss, epoch_m_loss = 0, 0
            model.train()

            epoch_sttm = datetime.now()
            for i, data in enumerate(trainloader):
                data = tensor2cuda(data, device)
                Igt = data['Igt']
                Ir, _ = model(data)

                # calculate loss & backpropagate
                r_loss = loss_fn(Ir, Igt, cfg['train']['loss'])
                if 'FFT' in cfg['train']['mask_loss']:
                    Mgt = torch.fft.fftn(Igt)
                    Mr = torch.fft.fftn(Ir)
                    m_loss = loss_fn(Mr, Mgt, cfg['train']['mask_loss']) * cfg['train']['mask_alpha']
                else:
                    m_loss = torch.tensor(0.0, device=device)

                loss = r_loss + m_loss

                prune_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, 2.0)
                prune_optimizer.step()
                prune_scheduler.step_update(num_updates=epoch_idx*len(trainloader)+i)

                # log loss
                epoch_loss += loss.item() * cfg['train']['batch_size']
                epoch_r_loss += r_loss.item() * cfg['train']['batch_size']
                epoch_m_loss += m_loss.item() * cfg['train']['batch_size']
                psnr_list.append(calculate_psnr(Ir, Igt))
                if i % cfg['train']['print_freq'] == 0 or i == (len(trainloader)-1):
                    pred_psnr = torch.cat(psnr_list).mean()
                    print_str = "[%s] Epoch[%03d/%03d], Step[%03d/%03d] - lr: %.4f, pred_PSNR: %.2f" % (
                        datetime.now().strftime("%Y/%m/%d %H:%M:%S"), epoch_idx+1, cfg['compress']['prune_optim']['epochs'], i+1, len(trainloader), prune_optimizer.param_groups[0]['lr'], pred_psnr.item()
                    )
                    f.write(print_str + '\n')
                    print(print_str)

            prune_scheduler.step(epoch=epoch_idx)

            writer.add_scalar("Train/pred_PSNR", pred_psnr, epoch_idx+1)
            writer.add_scalar("Train/lr", prune_optimizer.param_groups[0]['lr'], epoch_idx+1)
            writer.add_scalar("Train/loss", epoch_loss, epoch_idx+1)
            writer.add_scalar("Train/R_loss", epoch_r_loss, epoch_idx+1)
            writer.add_scalar("Train/M_loss", epoch_m_loss, epoch_idx+1)
            epoch_edtm = datetime.now()
            print_str = "Time/epoch: \tCurrent: %.2f \tAverage: %.2f" % (
                (epoch_edtm - epoch_sttm).total_seconds(), (epoch_edtm - start_tm).total_seconds() / (epoch_idx-init_epoch)
            )
            f.write(print_str + '\n')
            print(print_str)

            # save model
            save_ckpt = {
                'epoch_idx': epoch_idx,
                'network': model.state_dict(),
                'prune_optimizer': prune_optimizer.state_dict(),
                'prune_scheduler': prune_scheduler.state_dict(),
                'quant_optimizer': quant_optimizer.state_dict(),
                'quant_scheduler': quant_scheduler.state_dict(),
            }
            torch.save(save_ckpt, os.path.join(SAVE_DIR, 'ckpts/model_latest.pth'))
        cur_epoch = cfg['compress']['prune_optim']['epochs'] - 1

        init_quantization(model, min(cfg['compress']['quant_levels']), cfg['compress']['quant_noise'], cfg['compress']['quant_ste'])

        save_ckpt = {
            'epoch_idx': epoch_idx,
            'network': model.state_dict(),
            'prune_optimizer': prune_optimizer.state_dict(),
            'prune_scheduler': prune_scheduler.state_dict(),
            'quant_optimizer': quant_optimizer.state_dict(),
            'quant_scheduler': quant_scheduler.state_dict(),
        }
        torch.save(save_ckpt, os.path.join(SAVE_DIR, 'ckpts/model_pruned.pth'))

    # Fine-tuning for quantization
    if cfg['compress']['prune_optim']['epochs'] - 1 <= cur_epoch < cfg['compress']['prune_optim']['epochs'] + cfg['compress']['quant_optim']['epochs']:
        f.write("Start fine-tuning for quantization\n")
        print("Start fine-tuning for quantization")
        for epoch_idx in range(cur_epoch+1, cfg['compress']['prune_optim']['epochs'] + cfg['compress']['quant_optim']['epochs']):
            psnr_list, ssim_list = [], []
            epoch_loss = 0
            epoch_r_loss, epoch_m_loss = 0, 0
            model.train()

            epoch_sttm = datetime.now()
            for i, data in enumerate(trainloader):
                data = tensor2cuda(data, device)
                Igt = data['Igt']
                Ir, _ = model(data)

                # calculate loss & backpropagate
                r_loss = loss_fn(Ir, Igt, cfg['train']['loss'])
                if 'FFT' in cfg['train']['mask_loss']:
                    Mgt = torch.fft.fftn(Igt)
                    Mr = torch.fft.fftn(Ir)
                    m_loss = loss_fn(Mr, Mgt, cfg['train']['mask_loss']) * cfg['train']['mask_alpha']
                else:
                    m_loss = torch.tensor(0.0, device=device)

                loss = r_loss + m_loss

                quant_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, 2.0)
                quant_optimizer.step()
                quant_scheduler.step_update(num_updates=(epoch_idx-cfg['compress']['prune_optim']['epochs']+1)*len(trainloader)+i)

                # log loss
                epoch_loss += loss.item() * cfg['train']['batch_size']
                epoch_r_loss += r_loss.item() * cfg['train']['batch_size']
                epoch_m_loss += m_loss.item() * cfg['train']['batch_size']
                psnr_list.append(calculate_psnr(Ir, Igt))
                if i % cfg['train']['print_freq'] == 0 or i == (len(trainloader)-1):
                    pred_psnr = torch.cat(psnr_list).mean()
                    print_str = "[%s] Epoch[%03d/%03d], Step[%03d/%03d] - lr: %.4f, pred_PSNR: %.2f" % (
                        datetime.now().strftime("%Y/%m/%d %H:%M:%S"), epoch_idx-cfg['compress']['prune_optim']['epochs']+1, cfg['compress']['quant_optim']['epochs'], i+1, len(trainloader), quant_optimizer.param_groups[0]['lr'], pred_psnr.item()
                    )
                    f.write(print_str + '\n')
                    print(print_str)

            quant_scheduler.step(epoch=(epoch_idx-cfg['compress']['prune_optim']['epochs']+1))

            writer.add_scalar("Train/pred_PSNR", pred_psnr, epoch_idx+1)
            writer.add_scalar("Train/lr", quant_optimizer.param_groups[0]['lr'], epoch_idx+1)
            writer.add_scalar("Train/loss", epoch_loss, epoch_idx+1)
            writer.add_scalar("Train/R_loss", epoch_r_loss, epoch_idx+1)
            writer.add_scalar("Train/M_loss", epoch_m_loss, epoch_idx+1)
            epoch_edtm = datetime.now()
            print_str = "Time/epoch: \tCurrent: %.2f \tAverage: %.2f" % (
                (epoch_edtm - epoch_sttm).total_seconds(), (epoch_edtm - start_tm).total_seconds() / (epoch_idx-init_epoch)
            )
            f.write(print_str + '\n')
            print(print_str)

            # save model
            save_ckpt = {
                'epoch_idx': epoch_idx,
                'network': model.state_dict(),
                'prune_optimizer': prune_optimizer.state_dict(),
                'prune_scheduler': prune_scheduler.state_dict(),
                'quant_optimizer': quant_optimizer.state_dict(),
                'quant_scheduler': quant_scheduler.state_dict(),
            }
            torch.save(save_ckpt, os.path.join(SAVE_DIR, 'ckpts/model_latest.pth'))

        # remove quantization & pruning
        remove_quantization(model)
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])

        save_ckpt = {
            'epoch_idx': epoch_idx,
            'network': model.state_dict(),
        }
        torch.save(save_ckpt, os.path.join(SAVE_DIR, 'ckpts/model_fine-tuned.pth'))

    f.write("Training completed in %s\n" % (str(datetime.now() - start_tm)))
    print("Training completed in %s" % (str(datetime.now() - start_tm)))


    # compute sparsity
    sparsity_num = 0.
    for k, v in model.named_parameters():
        if 'weight' in k:
            sparsity_num += (v == 0).sum()
    f.write(f"Number of pruned parameters: {sparsity_num}\ttotal: {params['model_params']*1e6}\n")
    f.write(f"Model sparsity: {sparsity_num / 1e6 / params['model_params']}\n")
    print(f"Number of pruned parameters: {sparsity_num}\ttotal: {params['model_params']*1e6}")
    print(f"Model sparsity: {sparsity_num / 1e6 / params['model_params']}")

    # quantization & compression
    f.write("Compress model weights into bitstream\n")
    print("Compress model weights into bitstream")
    qat_state_dict = copy.deepcopy(model.state_dict())
    for qbit in cfg['compress']['quant_levels']:
        model.load_state_dict(qat_state_dict)
        # compression
        f.write("Quantization: %d bits\n" % qbit)
        print("Quantization: %d bits" % qbit)

        num_bytes = compress_and_save_model(model, qbit, SAVE_DIR)
        bits_per_pixel = num_bytes * 8 / (valset.video_size[0] * valset.video_size[-2] * valset.video_size[-1])

        f.write("Compressed model size: %.2fMB\tBits Per Pixel (BPP): %.4f\n" % (num_bytes / 10**6, bits_per_pixel))
        print("Compressed model size: %.2fMB\tBits Per Pixel (BPP): %.4f" % (num_bytes / 10**6, bits_per_pixel))

        # decompress
        decomp_state_dict = decompress_and_load_model(model, qbit, SAVE_DIR)
        model.load_state_dict(decomp_state_dict)

        # FPS, PSNR, MS-SSIM
        model_psnr, model_ssim, model_fps = evaluate(model, valloader, cfg['val'], f, device, log=False)
        print_str = "PSNR: %.2f\tMS-SSIM: %.4f\tFPS: %.1f" % (model_psnr, model_ssim, model_fps)
        f.write(print_str + '\n')
        print(print_str)




if __name__ == "__main__":
    args = get_args_parse()
    assert args.compress_folder is not None or args.resume_folder is not None, "compress_folder/resume_folder is necessary!"

    train(args)