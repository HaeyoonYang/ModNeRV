from config import *
from dataloader import dataset
from ModNeRV import ModNeRV
from utils import *
from deepspeed.profiling.flops_profiler import get_model_profile



def get_args_parse():
    parser = argparse.ArgumentParser('ModNeRV', add_help=False)

    parser.add_argument('--opt_path', default='cfgs/uvg-12.5M.yaml', type=str, help='path to cfg yaml file')
    parser.add_argument('--dataset', default='Beauty', type=str, help='trainset: UVG videos')
    parser.add_argument('--epochs', default=300, type=int, help='training epochs')
    parser.add_argument('--resume_folder', default=None, type=str, help='name of the folder to resume training')

    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get cfg from yaml file
    if args.resume_folder is not None:
        with open(os.path.join(args.resume_folder, 'config.yaml'), 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        with open(args.opt_path, 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # fix the seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    # set directory to save & set logger
    tm = time.localtime()
    tm_str = time.strftime("%Y%m%d_%H%M%S", tm)
    if args.resume_folder is not None:
        SAVE_DIR = args.resume_folder
    else:
        SAVE_DIR = os.path.join("experiments", cfg['exp_name'], args.dataset, tm_str)
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(SAVE_DIR, "ckpts"), exist_ok=True)

    loggername = os.path.join(SAVE_DIR, "logs.txt")
    f = open(loggername, 'a')
    writer = SummaryWriter(os.path.join(SAVE_DIR, "tensorboard_%s" % (tm_str)))

    for k, v in sorted((args.__dict__).items()):
        f.write("\t%15s: \t%s\n" % (k, v))
        print("\t%15s: \t %s" % (k, v))
    for k, v in sorted(cfg.items()):
        f.write("\t%15s: \t%s\n" % (k, v))
        print("\t%15s: \t %s" % (k, v))

    cfg['data']['dataset'] = args.dataset
    cfg['train']['optim']['epochs'] = args.epochs
    cfg['train']['optim']['warmup_epochs'] = int(args.epochs * cfg['train']['optim']['warmup_epoch_ratio'])

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

    cfg['model']['video_size'] = trainset.video_size
    if args.resume_folder is None:
        with open(os.path.join(SAVE_DIR, "config.yaml"), 'w') as fp:
            yaml.safe_dump(cfg, fp)

    # load model
    model = ModNeRV(cfg['model']).to(device)

    # get model params
    params = model.count_params()
    param_str = f"Encoder_{round(params['enc_params'], 3)}M_Decoder_{round(params['dec_params'], 3)}M_Total_{round(params['model_params'], 3)}M"
    f.write(str(model) + '\n' + f'Model Parameters: {param_str}\n')
    print("Model Parameters: ", param_str)

    # compute MACs
    with torch.no_grad():
        model.eval()
        _, macs, _ = get_model_profile(model=model, args=[tensor2cuda(next(iter(valloader)), device)], print_profile=False, detailed=False, warm_up=1, as_string=False)
        macs /= cfg['val']['batch_size']
    f.write(f"MACs: {macs / 10**9 :.2f}G\n")
    print(f"MACs: {macs / 10**9 :.2f}G")

    # set optimizer & scheduler
    optimizer, scheduler = get_optimizer_scheduler(cfg['train']['optim'], model, trainloader)

    # load resume ckpt
    init_epoch = -1
    if args.resume_folder is not None:
        weight_dir = os.path.join(args.resume_folder, 'ckpts/model_latest.pth')
        assert os.path.exists(weight_dir), "No file to resume training!!! Specify a different folder"

        ckpt = torch.load(weight_dir)
        model.load_state_dict(ckpt['network'])
        init_epoch = ckpt['epoch_idx']
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        f.write("Loaded previous weights from %s. Current epoch = #%d\n" % (weight_dir, init_epoch+1))
        print("Loaded previous weights from %s. Current epoch = #%d" % (weight_dir, init_epoch+1))
    else:
        f.write("Start training. Current epoch = #%d\n" % (init_epoch+1))
        print("Start training. Current epoch = #%d" % (init_epoch+1))

    # Training
    best_psnr, best_msssim = 0, 0
    start_tm = datetime.now()
    for epoch_idx in range(init_epoch+1, args.epochs):
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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, 2.0)
            optimizer.step()
            scheduler.step_update(num_updates=epoch_idx*len(trainloader)+i)

            # log loss
            epoch_loss += loss.item() * cfg['train']['batch_size']
            epoch_r_loss += r_loss.item() * cfg['train']['batch_size']
            epoch_m_loss += m_loss.item() * cfg['train']['batch_size']
            psnr_list.append(calculate_psnr(Ir, Igt))
            # ssim_list.append(calculate_msssim(Ir, Igt))
            if i % cfg['train']['print_freq'] == 0 or i == (len(trainloader)-1):
                pred_psnr = torch.cat(psnr_list).mean()
                # pred_msssim = torch.cat(ssim_list).mean()
                print_str = "[%s] Epoch[%03d/%03d], Step[%03d/%03d] - lr: %.4f, pred_PSNR: %.2f" % (
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), epoch_idx+1, args.epochs, i+1, len(trainloader), optimizer.param_groups[0]['lr'], pred_psnr.item()
                )
                f.write(print_str + '\n')
                print(print_str)

        scheduler.step(epoch=epoch_idx)

        writer.add_scalar("Train/pred_PSNR", pred_psnr, epoch_idx+1)
        writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], epoch_idx+1)
        writer.add_scalar("Train/loss", epoch_loss, epoch_idx+1)
        writer.add_scalar("Train/R_loss", epoch_r_loss, epoch_idx+1)
        writer.add_scalar("Train/M_loss", epoch_m_loss, epoch_idx+1)
        epoch_edtm = datetime.now()
        print_str = "Time/epoch: \tCurrent: %.2f \tAverage: %.2f" % (
            (epoch_edtm-epoch_sttm).total_seconds(), (epoch_edtm-start_tm).total_seconds() / (epoch_idx-init_epoch)
        )
        f.write(print_str + '\n')
        print(print_str)

        # Evaluation
        if (epoch_idx+1) % cfg['train']['save_freq'] == 0 or (args.epochs-epoch_idx) == 1:
            model_psnr, model_ssim, _ = evaluate(model, valloader, cfg['val'], f, device)
            print_str = "Eval at epoch %d: pred_PSNR: %.2f | pred_MS-SSIM: %.4f" % (
                epoch_idx+1, model_psnr, model_ssim,
            )
            f.write(print_str + '\n')
            print(print_str)

            writer.add_scalar("Val/pred_PSNR", model_psnr, epoch_idx+1)
            writer.add_scalar("Val/pred_MS-SSIM", model_ssim, epoch_idx+1)

            if pred_psnr >= best_psnr:
                best_psnr = pred_psnr
                save_ckpt = {
                    'epoch_idx': epoch_idx,
                    'network': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(save_ckpt, os.path.join(SAVE_DIR, 'ckpts/model_best_psnr.pth'))

        # save model
        save_ckpt = {
            'epoch_idx': epoch_idx,
            'network': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(save_ckpt, os.path.join(SAVE_DIR, 'ckpts/model_latest.pth'))

    f.write("Training completed in %s\n" % (str(datetime.now() - start_tm)))
    print("Training completed in %s" % (str(datetime.now() - start_tm)))


@torch.no_grad()
def evaluate(model, loader, cfg, f, device, log=True):
    model.eval()
    time_list, psnr_list, ssim_list = [], [], []
    total_frms = 0
    for i, data in enumerate(loader):
        data = tensor2cuda(data, device)
        Igt = data['Igt']
        Ir, dec_time = model(data)

        # collecting decoding fps
        time_list.append(dec_time)
        total_frms += Igt.size(0)

        # compute PSNR, MS-SSIM
        pred_psnr, pred_ssim = calculate_psnr(Ir, Igt), calculate_msssim(Ir, Igt)
        psnr_list.append(pred_psnr)
        ssim_list.append(pred_ssim)

        # log
        if log:
            if i % cfg['print_freq'] == 0 or i == (len(loader)-1):
                # avg_time = sum(time_list) / len(time_list)
                fps = total_frms / sum(time_list)
                print_str = "[%s] Eval at Step [%03d/%03d] - FPS: %.1f, pred_PSNR: %.2f, pred_MS-SSIM: %.4f" % (
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), i+1, len(loader), fps, torch.stack(psnr_list, dim=-1).mean(-1).item(), torch.stack(ssim_list, dim=-1).mean(-1).item()
                )
                f.write(print_str + '\n')
                print(print_str)

    model_psnr = torch.stack(psnr_list, dim=-1).mean(-1).item()
    model_ssim = torch.stack(ssim_list, dim=-1).mean(-1).item()
    avg_time = sum(time_list) / len(time_list)
    model_fps = total_frms / sum(time_list)              # total_frames / total_decoding_time

    return model_psnr, model_ssim, model_fps



if __name__ == "__main__":
    args = get_args_parse()
    assert args.opt_path is not None, "opt_path is necessary!"

    train(args)