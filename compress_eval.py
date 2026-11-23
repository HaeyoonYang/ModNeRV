from config import *
from dataloader import dataset
from ModNeRV import ModNeRV
from utils import *
from deepspeed.profiling.flops_profiler import get_model_profile



def get_args_parse():
    parser = argparse.ArgumentParser('ModNeRV compression', add_help=False)

    parser.add_argument('--test_folder', default='experiments/~', type=str, help='folder name for the model to test')
    parser.add_argument('--quant_bit', default=8, type=int, help='quantization bit')
    parser.add_argument('--visualize', action='store_true', default=False, help='whether to visualize')

    return parser.parse_args()


def eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get cfg yaml file
    with open(os.path.join(args.test_folder, 'config.yaml'), 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    # set directory to save & set logger
    SAVE_DIR = os.path.join(args.test_folder, f"Q{args.quant_bit}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(SAVE_DIR, "viz"), exist_ok=True)
    loggername = os.path.join(SAVE_DIR, 'testlogs.txt')
    f = open(loggername, 'w')

    for k, v in sorted((args.__dict__).items()):
        f.write("\t%15s: \t%s\n" % (k, v))
        print("\t%15s: \t%s" % (k, v))

    # load dataset
    valset = dataset(cfg['data'])
    valloader = DataLoader(
        valset,
        batch_size=cfg['val']['batch_size'],
        shuffle=False,
        num_workers=cfg['val']['num_workers'],
        drop_last=False,
    )

    # load model & weights
    model = ModNeRV(cfg['model']).to(device)
    # pq fine-tuned model
    if os.path.exists(os.path.join(args.test_folder, 'ckpts/model_fine-tuned.pth')):
        weight_dir = os.path.join(args.test_folder, 'ckpts/model_fine-tuned.pth')
    # not fine-tuned model
    else:
        weight_dir = os.path.join(args.test_folder, 'ckpts/model_best_psnr.pth')
    assert os.path.exists(weight_dir), "No WEIGHT FILE !!!"
    ckpt = torch.load(weight_dir)
    model.load_state_dict(ckpt['network'])
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

    # compute sparsity
    sparsity_num = 0.
    for k, v in model.named_parameters():
        sparsity_num += (v == 0).sum()
    f.write(f"Number of pruned parameters: {sparsity_num}\ttotal: {params['model_params']}\n")
    f.write(f"Model sparsity: {sparsity_num / 1e6 / params['model_params']}\n")
    print(f"Number of pruned parameters: {sparsity_num}\ttotal: {params['model_params']}")
    print(f"Model sparsity: {sparsity_num / 1e6 / params['model_params']}")

    # compress model
    if not os.path.exists(os.path.join(args.test_folder, f"Q{args.quant_bit}.zip")):
        f.write("Compress model weights into bitstream\n")
        f.write("Quantization: %d bits\n" % args.quant_bit)
        print("Compress model weights into bitstream")
        print("Quantization: %d bits" % args.quant_bit)

        num_bytes = compress_and_save_model(model, args.quant_bit, SAVE_DIR)
        bits_per_pixel = num_bytes * 8 /(valset.video_size[0] * valset.video_size[-2] * valset.video_size[-1])
        f.write("Compressed model size: %.2fMB\tBits Per Pixel (BPP): %.4f\n" % (num_bytes / 10**6, bits_per_pixel))
        print("Compressed model size: %.2fMB\tBits Per Pixel (BPP): %.4f" % (num_bytes / 10**6, bits_per_pixel))

        f.write("Decompress model weights from bitstream\n")
        print("Decompress model weights from bitstream")
        decomp_state_dict = decompress_and_load_model(model, args.quant_bit, SAVE_DIR)

    else:
        f.write("Load compress model weights from bitstream\n")
        f.write("Quantization: %d bits\n" % args.quant_bit)
        print("Load compress model weights from bitstream")
        print("Quantization: %d bits" % args.quant_bit)

        num_bytes = os.path.getsize(os.path.join(args.test_folder, f"Q{args.quant_bit}.zip"))
        bits_per_pixel = num_bytes * 8 /(valset.video_size[0] * valset.video_size[-2] * valset.video_size[-1])
        f.write("Compressed model size: %.2fMB\tBits Per Pixel (BPP): %.4f\n" % (num_bytes / 10**6, bits_per_pixel))
        print("Compressed model size: %.2fMB\tBits Per Pixel (BPP): %.4f" % (num_bytes / 10**6, bits_per_pixel))

        f.write("Decompress model weights from bitstream\n")
        print("Decompress model weights from bitstream")
        decomp_state_dict = decompress_and_load_model(model, args.quant_bit, args.test_folder)
    model.load_state_dict(decomp_state_dict)


    # Evaluation
    f.write("\n======================================= \n")
    model.eval()
    start_tm = datetime.now()
    psnr_list, ssim_list, fps_list = [], [], []
    with torch.no_grad():
        for i, data in enumerate(valloader):
            data = tensor2cuda(data, device)
            Igt = data['Igt']
            Ir, dec_time = model(data)

            # FPS, PSNR, MS-SSIM
            fps_list.append(1 / dec_time)
            pred_psnr, pred_ssim = calculate_psnr(Ir, Igt), calculate_msssim(Ir, Igt)
            psnr_list.append(pred_psnr)
            ssim_list.append(pred_ssim)

            # print
            f.write("[%04d.png] PSNR: %.2f\tMS-SSIM: %.4f\tFPS: %.1f\n" % (i+1, pred_psnr.item(), pred_ssim.item(), 1/dec_time))

            # save image
            if args.visualize:
                save_img_path = os.path.join(SAVE_DIR, "viz", "%04d_r.png" % (i+1))
                pred_img = torch2npimg(Ir[0])
                cv2.imwrite(save_img_path, pred_img)

    f.write("======================================= \n")
    fps = sum(fps_list) / len(fps_list)
    psnr = torch.stack(psnr_list, dim=-1).mean(-1).item()
    msssim = torch.stack(ssim_list, dim=-1).mean(-1).item()
    print_str = "Total PSNR: %.2f\tMS-SSIM: %.4f\tFPS: %.1f" % (psnr, msssim, fps)
    f.write(print_str + "\n")
    print(print_str)

    print_str = "Evaluation completed in %s" % (str(datetime.now() - start_tm))
    f.write("\n" + print_str + "\n")
    print(print_str)

    # # generate video
    # os.system(f"ffmpeg -framerate 5 -i {SAVE_DIR}/viz/%04d_r.png -c:v copy {SAVE_DIR}/out_vid.mkv")





if __name__ == "__main__":
    args = get_args_parse()

    eval(args)