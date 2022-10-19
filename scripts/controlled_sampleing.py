import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageChops
import imageio

from main import instantiate_from_config, DataModuleFromConfig
from tqdm import trange
from tqdm import tqdm
import cv2
import torchvision.transforms as T

from modules.affine_transformation.affine_transform import optimize_mask

def save_image(x, path):
    x = x.detach().cpu().squeeze()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x).save(path)

def save_mask(x, path):
    x = x.detach().cpu().squeeze(0)
    x = torch.clamp(x, 0, 1)
    # x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = np.repeat(x, 3, axis=-1)
    x = Image.fromarray(x).save(path)

def preprocess_image(image_path, size = (160,208)):
    image = Image.open(image_path)
    image = image.resize(size)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = torch.unsqueeze(T.ToTensor()(image), 0)
    image = 2*image - 1
    return image

def preprocess_mask(mask_path):
    image = Image.open(mask_path).convert('L')
    # if not image.mode == "RGB":
    #     image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    image = T.ToTensor()(image)
    image = image/255
    return image

def get_GIF(folder, num_frames = 16, GT = False):

    NUM_FRAMES = num_frames
    
    images = []
    for f in range(NUM_FRAMES):
        images.append(imageio.imread(folder + f'/{(f):05}.png'))
    save_path = folder + '/output.gif'
    imageio.mimsave(save_path, images)

    if not GT:
        masks = []
        for z in range(NUM_FRAMES-1):
            masks.append(imageio.imread(folder + f'/{(z):05}_mask.png'))
        
        save_path = folder + '/mask.gif'
        imageio.mimsave(save_path, masks)

    # foregrounds = []
    # for x in range(NUM_FRAMES):
    #     foregrounds.append(imageio.imread(os.path.join(vids, str(x+1)+'_foreground.png')))
    # save_path = folder + sub + '_foreground.gif'
    # imageio.mimsave(save_path, foregrounds)

    # backgrounds = []
    # for y in range(NUM_FRAMES):
    #     backgrounds.append(imageio.imread(os.path.join(vids, str(y+1)+'_background.png')))
    # save_path = folder + sub + '_background.gif'
    # imageio.mimsave(save_path, backgrounds)

def shift_mask(image_a, x_off = 0, y_off = 5):
    img_a = Image.open(image_a).convert('L')
    width, height = img_a.size

    # img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_b = ImageChops.offset(img_a, x_off, y_off)
    img_b.paste((0), (0, 0, x_off, height))
    img_b.paste((0), (0, 0, width, y_off))

    frameDelta = np.asarray(img_b) 
    frameDelta = np.clip(frameDelta, 0, 255)
    thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
    mask_GT = (thresh).astype(np.float32)
    mask_GT = T.ToTensor()(mask_GT)/255
    return mask_GT

def construct_affine_mat(translation = [0,0], rotation = 0, scaling = [1,1], shear = [0,0]):
    '''
    trans_para: [Translation, Rotation, Scaling, Shear]

    examples:
    Translation: [0, 0]
    Rotation: Theta
    Scaling: [1,1]
    shear: [0, 0]

    '''
    translation_mat = torch.eye(3)
    translation_mat[0, 2] = torch.tensor(translation[0])
    translation_mat[1, 2] = torch.tensor(translation[1])

    rotation_mat = torch.eye(3)
    rotation_mat[0,0] = torch.cos(torch.tensor(rotation))
    rotation_mat[0,1] = -torch.sin(torch.tensor(rotation))
    rotation_mat[1,0] = torch.sin(torch.tensor(rotation))
    rotation_mat[1,1] = torch.cos(torch.tensor(rotation))

    scale_mat = torch.eye(3)
    scale_mat[0, 0] = torch.tensor(scaling[0])
    scale_mat[1, 1] = torch.tensor(scaling[1])

    shear_mat = torch.eye(3)
    shear_mat[0, 1] = torch.tensor(shear[0])
    shear_mat[1, 0] = torch.tensor(shear[1])

    affine_mat =torch.matmul(shear_mat ,torch.matmul(scale_mat, torch.matmul(rotation_mat, translation_mat)))

    return affine_mat

# @torch.no_grad()
def sample_video(vqgan, path, num_frames, path_gt=None, save_path = None, mask_path = None, bsuv = None, user_control = True, translate = (0,0)):
    # print(path)
    for i in range(num_frames):
        index_num = f'{(i):05}'
        next_index_num = f'{(i+1):05}'
        example = {}
        inp_frame_path = path+'/'+index_num+'.png'
        example['image'] = preprocess_image(inp_frame_path)
        x = example['image'].to(vqgan.device)

        # 1st forward pass for mask
        # for user control:
        if user_control:
            _, _, _ = vqgan(x)
            curr_mask = vqgan.mask_gen
            cur_dev = curr_mask.device
            affine_mat = construct_affine_mat(translation = [translate[0]/256,translate[1]/256], rotation = 0, scaling = [1,1], shear = [0,0]).to(cur_dev)
            # affine_mat = construct_affine_mat(translation = [0,0], rotation = 0, scaling = [1.2,1.2], shear = [0,0]).to(cur_dev)
            # affine_mat = construct_affine_mat(translation = [0,0], rotation = -0.3, scaling = [1,1], shear = [0,0]).to(cur_dev)
            _, moved_mask = optimize_mask(curr_mask.squeeze(), None, mode = 'affine', iter = 1000, user_input = affine_mat)
            moved_mask = torch.unsqueeze(moved_mask, 0)
            moved_mask = torch.unsqueeze(moved_mask, 0)
            mask_GT = moved_mask

            mask_name = index_num +'_mask.png'
            mask_save_path = os.path.join(save_path, mask_name)
            save_mask(curr_mask, mask_save_path)

            mask_name = index_num +'_mov_mask.png'
            mask_save_path = os.path.join(save_path, mask_name)
            save_mask(moved_mask, mask_save_path)

            # mask_a = path + '/'+index_num+'_mask.png'
            # mask_a = path + '/00000_mask.png'

            # mask_GT = move_mask(mask_a)
            # if i < 8:
            #     mask_GT = morph_mask(mask_a, step = -3, percent = 0.33)
            # else:
            #     mask_GT = morph_mask(mask_a, step = 3, percent = 0.33)
            # if i < 15:
            # mask_GT = shift_mask(mask_a, 16, 0)
            # # else:
            # #     mask_GT = shift_mask(mask_a, -6, 0)
            # mask_GT = mask_GT.to(vqgan.device)
            # mask_GT = torch.unsqueeze(mask_GT, 1)
            # mask_name = index_num +'_mask_shift.png'
            # mask_save_path = os.path.join(save_path, mask_name)
            # save_mask(mask_GT, mask_save_path)
        # for evaluation:
        else:

            gt_next_frame = preprocess_image(path_gt + f'/{(i + 1):05}.png').to(vqgan.device)
            _, _, _ = vqgan(gt_next_frame)
            mask_GT = vqgan.mask_gen

            # ######################################
            mask_GT = torch.where(mask_GT > 0.5, 1, 0) * 1.0

            _, _, _ = vqgan(x)
            mask_gen = vqgan.mask_gen
            mask_gen = torch.where(mask_gen > 0.5, 1, 0) * 1.0

            mask_gen_name = index_num +'_mask_gen.png'
            mask_gt_name = index_num +'_mask_gt.png'
            save_mask(mask_gen, os.path.join(save_path, mask_gen_name))
            save_mask(mask_GT, os.path.join(save_path, mask_gt_name))
 
            mask_gt = mask_GT.squeeze()
            mask_gen = mask_gen.squeeze()
            # print(mask_gen.shape)
            _, moved_mask = optimize_mask(mask_gen, mask_gt, mode = 'affine', iter = 1000)
            moved_mask = torch.unsqueeze(moved_mask, 0)
            moved_mask = torch.unsqueeze(moved_mask, 0)
            mask_GT = moved_mask
            ########################################

            mask_name = index_num +'_mask.png'
            mask_save_path = os.path.join(save_path, mask_name)
            save_mask(mask_GT, mask_save_path)
            # save_mask(mask_GT, mask_save_path)
            # inp_mask_path = mask_path+'/'+next_index_num+'_mask.png'
            # mask_GT = preprocess_mask(inp_mask_path)


        # 2nd forwad pass for next frame
        mask_GT = mask_GT.cuda()
        reconstruction, _, mask = vqgan(x, maskGT = mask_GT, finetune = True)

        next_index_num = f'{(i + 1):05}'
        image_name = next_index_num + '.png'
        save_p = os.path.join(save_path, image_name)
        save_image(reconstruction, save_p)

    
    # generate GIF:
    # get_GIF(save_path, num_frames = 9)
    # get_GIF(path_gt, num_frames = num_frames, GT = True)


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
        default='latest-133000',
        help="checkpoint to load",
    )
    parser.add_argument(
        "--user_control",
        action="store_true",
        help="wether to use user control mode",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='test',
        help="save path",
    )
    parser.add_argument(
        "-tx",
        type=int,
        default=0,
        help="translate X by:",
    )
    parser.add_argument(
        "-ty",
        type=int,
        default=0,
        help="translate Y by:",
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
    folder_gen = os.path.join(opt.dataroot, 'breakout')
    # for action in os.listdir(folder_gen):
    videos = os.listdir(os.path.join(folder_gen))
    for v in tqdm(videos):
        path = os.path.join(opt.dataroot, opt.save_path, v)
        save_path = os.path.join(opt.dataroot, opt.save_path, v)
        path_gt = os.path.join(opt.dataroot, 'test_GT', v)
        for f in os.listdir(save_path):
            if f != '00000.png' and f[-1] != 'l':
                os.remove(save_path + '/' + f)
        # if v == '00049' or v == '00018' or v == '00083' or v == '00096':
        sample_video(vqgan, path, opt.num_frames, save_path = save_path, path_gt = path_gt, user_control = opt.user_control, translate = (opt.tx, opt.ty))
        