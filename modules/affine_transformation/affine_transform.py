import os
import torch
import numpy as np
from skimage import io
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
import sys
import h5py

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/yjin/workspace/video_generation/train_masks_h5/')
    parser.add_argument('--mode', type=str, default='affine')
    parser.add_argument('--range_lower', type=int, default=1000)
    parser.add_argument('--range_upper', type=int, default=1001)
    parser.add_argument('--num_cpus', type=int, default=1)
    config = parser.parse_args()
    return config

def optimize_mask (cur_mask, tar_mask=None, mode='translation', iter=250, user_input=None):
    
    def _meshgrid(out_shape):
        y, x = torch.meshgrid([torch.linspace(0, 1, steps=out_shape[-2]), torch.linspace(0, 1, steps=out_shape[-1])])
        x, y = x.flatten(), y.flatten()
        grid = torch.stack([x, y,torch.ones(y.shape)], dim=1)
        return grid
    
    def _spatial_transformer(image,affine_mat,shape):
#     def _spatial_transformer(image,delta_xyt,shape,top_padding=False):
        cur_dev = image.device
        grid = _meshgrid(shape).to(cur_dev)
        grid[:,:2] = (grid[:,:2]-0.5)*2
        wgrid = torch.matmul(grid,torch.transpose(affine_mat, 0, 1))[:,:2]
#         wgrid = torch.stack((grid[:,0]+delta_xyt[0],grid[:,1]+delta_xyt[1]),dim=1)
        wgrid = wgrid.view(*shape, 2)
        moved_image = torch.nn.functional.grid_sample(image.unsqueeze(0).unsqueeze(0), wgrid.unsqueeze(0),'bilinear','zeros')
        moved_image = moved_image.squeeze()
        return moved_image

    def _construct_affine_mat(trans_para, mode):
        cur_dev = trans_para.device
        affine_mat = torch.zeros((3,3)).to(cur_dev)
        affine_mat[2,2] = 1
        if mode == 'affine':
            affine_mat[:2,:] = trans_para
        elif mode == 'translation':
            affine_mat[0,0] = 1
            affine_mat[1,1] = 1
            affine_mat[0,2] = trans_para[0]
            affine_mat[1,2] = trans_para[1]
        elif mode == 'translation+roatation':
            affine_mat[0,2] = trans_para[0]
            affine_mat[1,2] = trans_para[1]
            affine_mat[0,0] = torch.cos(trans_para[2])
            affine_mat[0,1] = -torch.sin(trans_para[2])
            affine_mat[1,0] = torch.sin(trans_para[2])
            affine_mat[1,1] = torch.cos(trans_para[2])            
        return affine_mat

    # move mask according to user input
    if user_input is not None:
        moved_mask = _spatial_transformer(cur_mask,user_input,cur_mask.shape)        
        return user_input, moved_mask.detach()

    # init transformation paramters
    if mode == 'affine':
        trans_para = torch.zeros(2,3)
        trans_para[0,0] = 1
        trans_para[1,1] = 1
    elif mode == 'translation':
        trans_para = torch.zeros(2)
    elif mode == 'translation+roatation':
        trans_para = torch.zeros(3)
    else:
        print('supported mode: translation, translation+roatation, affine')
        raise NotImplementedError
    cur_dev = cur_mask.device
    trans_para = trans_para.to(cur_dev)
    trans_para.requires_grad = True
    
    # init solver
    solver = torch.optim.SGD([trans_para], lr=1e-1)
    
    # optimization loop
    for i in range(iter):
        affine_mat = _construct_affine_mat(trans_para,mode)
        moved_mask = _spatial_transformer(cur_mask,affine_mat,tar_mask.shape)
        loss = torch.mean(torch.square(moved_mask-tar_mask))
        solver.zero_grad()
        loss.backward()
        solver.step()
    return trans_para.detach(), moved_mask.detach()

def process_video_seq(video_dir,mode):
    # read all frames
    # video_path_list = [os.path.join(video_dir,f) for f in os.listdir(video_dir) if f.endswith('mask.png') and len(f.split('_'))==2]
    # mask_list = {}
    # key_list = []
    # for mask_path  in video_path_list:
    #     key = os.path.basename(mask_path).split('_')[0]
    #     mask_list[key] = io.imread(mask_path)
    #     key_list.append(key) 
    # key_list.sort()
    # for key_1, key_2 in zip(key_list[:-1],key_list[1:]):
    #     cur_mask = torch.tensor(mask_list[key_1][:,:,0]/255).type(torch.FloatTensor)
    #     tar_mask = torch.tensor(mask_list[key_2][:,:,0]/255).type(torch.FloatTensor)
    #     _,opt_mask = optimize_mask(cur_mask,tar_mask,mode,1000)
    #     opt_mask = (opt_mask[:, :, None].repeat(1,1,3)*255).numpy().astype(np.uint8)
    #     io.imsave(os.path.join(video_dir,key_1+'_{}_mask.png'.format(mode)),opt_mask)
    with h5py.File(video_dir, "r") as f:
        key_list = list(f.keys())
        key_list.sort()
        mask_list = {}
        for key in key_list:
            mask_list[key] = f[key][()]
    with h5py.File(video_dir.split('.')[0]+'_opt_{}.h5'.format(mode), "w") as f:
        for key_1, key_2 in zip(key_list[:-5],key_list[5:]):
            cur_mask = torch.tensor(mask_list[key_1][:,:,0]/255).type(torch.FloatTensor)
            tar_mask = torch.tensor(mask_list[key_2][:,:,0]/255).type(torch.FloatTensor)
            _,opt_mask = optimize_mask(cur_mask,tar_mask,mode,1000)
            opt_mask = (opt_mask[:, :, None].repeat(1,1,3)*255).numpy().astype(np.uint8)
            f.create_dataset(key_1, data=opt_mask)
def main():
    config = get_config()
    # Parallel(n_jobs=config.num_cpus)(delayed(process_video_seq)(os.path.join(config.root_dir,str(i).zfill(5)),config.mode) for i in tqdm(range(config.range_lower,config.range_upper)))
    Parallel(n_jobs=config.num_cpus)(delayed(process_video_seq)(os.path.join(config.root_dir,str(i).zfill(5)+'.h5'),config.mode) for i in tqdm(range(config.range_lower,config.range_upper)))
if __name__ == '__main__':
    main()