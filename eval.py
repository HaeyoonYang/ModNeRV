from config import *
from dataloader import dataset
from ModNeRV import ModNeRV
from utils import *
from deepspeed.profiling.flops_profiler import get_model_profile



def get_args_parse():
    parser = argparse.ArgumentParser('ModNeRV', add_help=False)
    parser.add_argument('--test_folder', default='experiments/12.5M/Beauty', type=str, help='folder name for inference')
    parser.add_argument('--visualize', default=False, action='store_true', help='save image or not')
    return parser.parse_args()


def eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get cfg from yaml file
    if args.test_folder is not None:
        with open(os.path.join(args.test_folder, 'config.yaml'), 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        with open(args.opt_path, 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # set directory to save & set logger
    SAVE_DIR = os.path.join(args.test_folder, "test")
    os.makedirs(SAVE_DIR, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(SAVE_DIR, "viz"), exist_ok=True)
    loggername = os.path.join(SAVE_DIR, "testlogs.txt")
    f = open(loggername, 'w')

    for k, v in sorted((args.__dict__).items()):
        f.write("\t%15s: \t%s\n" % (k, v))
        print("\t%15s: \t %s" % (k, v))
    for k, v in sorted(cfg.items()):
        f.write("\t%15s: \t%s\n" % (k, v))
        print("\t%15s: \t %s" % (k, v))

    # load dataset
    valset = dataset(cfg['data'])
    valloader = DataLoader(
        valset,
        batch_size=cfg['val']['batch_size'],
        shuffle=False,
        num_workers=cfg['val']['num_workers'],
        drop_last=False,
    )

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

    # load weight
    weight_dir = os.path.join(args.test_folder, "ckpts/model_latest.pth")
    assert os.path.exists(weight_dir), "No TEST FILE !!!"
    ckpt = torch.load(weight_dir)
    model.load_state_dict(ckpt['network'])
    f.write("Loaded test weights from %s. Test epoch = #%d\n" % (weight_dir, ckpt['epoch_idx']+1))
    print("Loaded test weights from %s. Test epoch = #%d" % (weight_dir, ckpt['epoch_idx']+1))

    # Inference
    f.write("Start Inference ...\n")
    print("Start Inference ...")
    start_tm = datetime.now()
    model.eval()
    psnr_list, ssim_list, time_list = [], [], []
    total_frms = 0
    with torch.no_grad():
        for i, data in enumerate(valloader):
            data = tensor2cuda(data, device)
            Igt = data['Igt']
            Ir, dec_time = model(data)

            # calculate FPS, PSNR, MS-SSIM
            time_list.append(dec_time)
            total_frms += Igt.size(0)
            pred_psnr, pred_ssim = calculate_psnr(Ir, Igt), calculate_msssim(Ir, Igt)
            psnr_list.append(pred_psnr)
            ssim_list.append(pred_ssim)

            # print
            f.write("[%04d.png] PSNR: %.2f\tMS-SSIM: %.4f\tFPS: %.1f, Dec_time: %.4f\n" % (i+1, pred_psnr.item(), pred_ssim.item(), Igt.size(0)/dec_time, dec_time))

            # save image
            if args.visualize:
                save_img_path = os.path.join(SAVE_DIR, 'viz', '%04d_r.png' % (i+1))
                pred_img = torch2npimg(Ir[0])
                cv2.imwrite(save_img_path, pred_img)

    f.write("========================================\n")
    fps = total_frms / sum(time_list)
    psnr = torch.stack(psnr_list, dim=-1).mean(-1).item()
    msssim = torch.stack(ssim_list, dim=-1).mean(-1).item()
    print_str = "Total PSNR: %.2f\tMS-SSIM: %.4f\tFPS: %.1f" % (psnr, msssim, fps)
    f.write(print_str + "\n")
    print(print_str)

    print_str = "Evaluate completed in %s" % (str(datetime.now() - start_tm))
    f.write("\n" + print_str + "\n")
    print(print_str)

    # # generate video
    # os.system(f"ffmpeg -framerate 5 -i {SAVE_DIR}/viz/%04d_r.png -c:v copy {SAVE_DIR}/out_vid.mkv")



if __name__ == "__main__":
    args = get_args_parse()
    assert args.test_folder is not None, "test_folder is necessary!"

    eval(args)