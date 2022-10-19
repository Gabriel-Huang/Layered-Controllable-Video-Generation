import argparse, os, sys, glob, math, time
import torch
import numpy as np
from torch.serialization import save
from omegaconf import OmegaConf
from PIL import Image, ImageChops
import imageio

from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import trange
import PIL
from tqdm import tqdm
import cv2
import torchvision.transforms as T

from skimage import io,morphology
from scipy import signal


def save_mask(x, path):
    x = x.detach().cpu().squeeze(0)
    x = torch.clamp(x, 0, 1)
    # x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = np.repeat(x, 3, axis=-1)
    x = Image.fromarray(x).save(path)


def preprocess_image(image_path, size = (256,96)):
    image = Image.open(image_path)
    image = image.resize(size)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = torch.unsqueeze(T.ToTensor()(image), 0)
    image = 2*image - 1
    return image

def preprocess_mask(mask_path):
    image = Image.open(mask_path)
    # if not image.mode == "RGB":
    #     image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = T.ToTensor()(image)
    image = image/255
    return image


def meshgrid(out_shape):
    y, x = torch.meshgrid([torch.linspace(0, 1, steps=out_shape[-2]), torch.linspace(0, 1, steps=out_shape[-1])])
    x, y = x.flatten(), y.flatten()
    grid = torch.stack([x, y], dim=1)
    return grid

def spatial_transformer(image,delta_xy,shape, fill = False):
    cur_dev = image.device
    grid = meshgrid(shape).to(cur_dev)
    if fill:
        x = (grid[:,0]-0.5)*2 - delta_xy[0]
        y = grid[:,1] - delta_xy[1]
    else:
        x = (grid[:,0]-0.5)*2 - delta_xy[0]
        y = (grid[:,1]-0.5)*2 - delta_xy[1]        
    wgrid = torch.stack([x, y], dim=-1).view(*shape, 2)
    moved_image = torch.nn.functional.grid_sample(image.unsqueeze(0).unsqueeze(0), wgrid.unsqueeze(0),'bilinear','zeros')
    moved_image = moved_image.squeeze()
    return moved_image



def sample_video(vqgan, path, num_frames, path_gt=None, save_path = None, mask_path = None, bsuv = None, user_control = True, fill=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in os.listdir(path):
        if i.split('.')[-1] == 'png':
            # index_num = f'{(i):05}'
            # get current mask:
            inp_frame_path = os.path.join(path, i)
            x = preprocess_image(inp_frame_path).to(vqgan.device)
            _, _, _ = vqgan(x)
            cur_mask = vqgan.mask_gen
            cur_mask = torch.where(cur_mask > 0.5, 1, 0) * 1.0
            mask_name = i.split('.')[0] +'_mask.png'
            mask_save_path = os.path.join(save_path, mask_name)
            save_mask(cur_mask, mask_save_path)

        '''
        # get GT next frame's mask
        gt_next_frame = preprocess_image(path_gt + f'/{(i + 1):05}.png').to(vqgan.device)
        _, _, _ = vqgan(gt_next_frame)
        gt_mask = vqgan.mask_gen

        cur_mask = cur_mask.squeeze()
        # get shape
        H,W = cur_mask.shape
        # init delta x y
        delta_xy = torch.zeros((2))
        delta_xy.requires_grad = True
        # init solver
        solver = torch.optim.SGD([delta_xy], lr=1e-1)
        # top padding
        if fill:
            cur_mask[0] = torch.max(cur_mask[:2],dim=0)[0]
            cur_mask = torch.nn.ReplicationPad2d((0,0,H,0))(cur_mask.unsqueeze(0).unsqueeze(0)).squeeze().detach()
        else:
            cur_mask = cur_mask.detach()

        # optimization loop
        for p in range(50):
            moved_mask = spatial_transformer(cur_mask, delta_xy, (256,256),fill)
            moved_mask = moved_mask.unsqueeze(0).unsqueeze(0)

            loss = torch.mean(torch.square(gt_mask.detach() - moved_mask))
            solver.zero_grad()
            loss.backward()
            solver.step()

        mask_name = next_index_num +'_mask.png'
        mask_save_path = os.path.join(save_path, mask_name)
        mask = moved_mask * 255
        mask_GT = torch.where(mask > 100, 1, 0) * 1.0
        save_mask(mask_GT, mask_save_path)
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
        default=' /ubc/cs/research/shield/projects/gabrie20/projects/taming-transformers/logs/2021-08-10T12-15-18_bair_vqgan/'
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="/ubc/cs/research/shield/projects/gabrie20/projects/taming-transformers/configs/bair_vqgan.yaml",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    parser.add_argument(
        "--outdir",
        # required=True,
        type=str,
        help="Where to write outputs to.",
        default='/ubc/cs/research/shield/projects/gabrie20/data/robot_dataset/'
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Sample from among top-k predictions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default='/ubc/cs/research/shield/projects/gabrie20/data/robot_dataset/',
        help="dataroot",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="number of frames to generate in test phase",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='latest-500',
        help="checkpoint to load",
    )
    parser.add_argument(
        "--user_control",
        action="store_true",
        help="wether to use user control mode",
    )
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    return data


def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data
    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", opt.ckpt + ".ckpt")
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)

    print('ckpt:', ckpt)
    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(config))

    dsets, vqgan, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
    print(f"Global step: {global_step}")

    outdir = opt.outdir
    print("Writing samples to ", outdir)
    folder_gen = os.path.join(opt.dataroot, 'test_fixed_length')
    # for action in os.listdir(folder_gen):
    videos = os.listdir(os.path.join(folder_gen))
    for v in tqdm(videos):
        path = os.path.join(opt.dataroot, 'test_fixed_length', v)
        save_path = os.path.join(opt.dataroot, 'test_fixed_length_masks', v)
        path_gt = os.path.join(opt.dataroot, 'test_fixed_length', v)
        # for f in os.listdir(save_path):
        #     if f != '00000.png' and f != '00001.png' and f != '00000_mask.png' and f[-1] != 'l':
        #         os.remove(save_path + '/' + f)
        sample_video(vqgan, path, opt.num_frames, save_path = save_path, path_gt = path_gt, user_control = opt.user_control)
        