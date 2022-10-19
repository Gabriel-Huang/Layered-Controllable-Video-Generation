import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import imageio

from main import instantiate_from_config
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

def preprocess_image(image_path, size = (256,256)):
    image = Image.open(image_path)
    image = image.resize(size)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = torch.unsqueeze(T.ToTensor()(image), 0)

    image = 2*image - 1
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
def sample_video(vqgan, path, frame_id, save_path = None, translate = (0,0)):
    i = frame_id

    index_num = f'{(i):05}'
    next_index_num = f'{(i+1):05}'

    inp_frame_path = path+'/'+index_num+'.png'
    x = preprocess_image(inp_frame_path).to(vqgan.device)

    # 1st forward pass for mask
    _, _, _ = vqgan(x)
    # torch.onnx.export(vqgan, x, 'bair.onnx', verbose = True, opset_version=12)
    curr_mask = vqgan.mask_gen
    cur_dev = curr_mask.device
    affine_mat = construct_affine_mat(translation = [translate[0]/256, translate[1]/256], rotation = 0, scaling = [1,1], shear = [0,0]).to(cur_dev)
    _, moved_mask = optimize_mask(curr_mask.squeeze(), None, mode = 'translation', iter = 1000, user_input = affine_mat)
    moved_mask = torch.unsqueeze(moved_mask, 0)
    moved_mask = torch.unsqueeze(moved_mask, 0)
    mask_GT = moved_mask

    mask_name = next_index_num +'_mask.png'
    mask_save_path = os.path.join(path, mask_name)
    save_mask(moved_mask, mask_save_path)

    # mask_name = index_num +'_mov_mask.png'
    # mask_save_path = os.path.join(save_path, mask_name)
    # save_mask(moved_mask, mask_save_path)

    # 2nd forwad pass for next frame
    mask_GT = mask_GT.to(vqgan.device)
    reconstruction, _, _ = vqgan(x, maskGT = mask_GT)

    next_index_num = f'{(i + 1):05}'
    image_name = next_index_num + '.png'
    save_p = os.path.join(path, image_name)
    save_image(reconstruction, save_p)

    
    # get_GIF(save_path, num_frames = num_frames)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
        default='/media/gabrie20/Work/projects/controllable_video_generation/logs/remote_checkpoints/'
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
        default="/media/gabrie20/Work/projects/controllable_video_generation/web_app/models/bair_config.yaml",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    
    parser.add_argument(
        "--num_frames",
        type=int,
        default=1,
        help="number of frames to generate in test phase",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='finetune_translation',
        help="checkpoint to load",
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


def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data

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
    return model, global_step


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

    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    vqgan, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)

    image_path = 'web_app/image'
    for f in os.listdir(image_path):
        if f != '00000.png' and f[-1] != 'l':
            os.remove(image_path + '/' + f)
    sample_video(vqgan, image_path, opt.num_frames, save_path = image_path)
    
    # torch.save(vqgan.state_dict(), 'bair_inference_model.pt')