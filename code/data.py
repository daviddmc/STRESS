from scipy.ndimage import rotate
import torch.multiprocessing as mp
import numpy as np
import nibabel as nib
import torch
import os
from trajectory import get_trajectory
import traceback
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates, affine_transform
from scipy.stats import special_ortho_group
from itertools import product
from time import time
from config import sigma as _sigma

def read_nifti(nii_filename):
    data = nib.load(nii_filename)
    return np.squeeze(data.get_data().astype(np.float32))

def down_up(*imgs, start=0):
    res = []
    num_split = (len(imgs) + 1) // 2
    for s, img in enumerate(imgs, start):
        ss = s % num_split
        X, Y, Z = np.meshgrid((np.arange(img.shape[0]) - ss) / num_split, np.arange(img.shape[1]), np.arange(img.shape[2]), indexing='ij')
        res.append(map_coordinates(img[ss::num_split], [X, Y, Z], order=3, mode='nearest'))
    return res
    
class EPIDataset(torch.utils.data.Dataset):
    def __init__(self, num_split, stage, is_denoise=False, denoiser=None):
        assert stage in ['train', 'val', 'test']
        self.data_dir = '/home/junshen/new'
        self.is_test = stage == 'test'
        self.folders = sorted(os.listdir(self.data_dir))
        if stage == 'test':
           self.folders = [folder for i, folder in enumerate(self.folders) if i % 6 == 0]
        #elif stage == 'val':
        #    self.folders = [folder for i, folder in enumerate(self.folders) if i % 6 == 0]
        #else:
        #    self.folders = [folder for i, folder in enumerate(self.folders) if folder not in data_test]
        self.proc = []
        self.res = []
        num_p = 10
        self.queue = mp.Queue(1024)
        n_new = 6
        self.queue2 = mp.Queue(n_new)
        for _ in range(n_new):
            self.queue2.put(None)
        
        if denoiser is not None:
            denoise_queue = [{'in':mp.Queue(1),'out':mp.Queue(1)} for _ in range(num_p)]
            self.denoiser = mp.Process(target=denoise_fn, args=(denoiser, denoise_queue))
            self.denoiser.daemon = True
            self.denoiser.start()
            self.denoise_queue = denoise_queue
        else:
            denoise_queue = [None] * num_p
            self.denoise_queue = denoise_queue

        # use multiple processes to fetch data
        for i in range(num_p):
            proc = mp.Process(target=prefetch_volumes_test if self.is_test else prefetch_volumes, 
                args=(self.data_dir, self.folders[i::num_p], self.queue, self.queue2, num_split, is_denoise, self.denoise_queue[i]))
            proc.daemon = True
            proc.start()
            self.proc.append(proc)
            
    def load_data(self):
        if len(self.res) == 0:
            N = 0
            while True:
                res = self.queue.get()
                if res is None:
                    N += 1
                    if N == len(self.proc):
                        break
                else:
                    self.res.append(res)
            self.res = sorted(self.res, key=lambda x:x[-1])

            #name = [x[-1] for x in self.res]
            #np.save('name', name)
            #raise Exception('test')

            if self.denoise_queue[0] is not None:
                self.denoise_queue[0]['in'].put(None)

            print("test set len: %d" % len(self.res))
        
    def __len__(self):
        if self.is_test:
            self.load_data()
            return len(self.res)
        else:
            return int(1e8)

    def __getitem__(self, idx):
        if self.is_test:
            self.load_data()
            return self.res[idx][:3]
        else:
            return self.queue.get()
    

def prefetch_volumes(data_dir, folders, queue, q2, num_split, is_denoise, denoiser):
    a = 32
    volumes = [None] * len(folders)
    files = [[]] * len(folders)
    starts = [None] * len(folders)
    start0s = [None] * len(folders)
    for i in range(len(folders)):
        files[i] = sorted(os.listdir(os.path.join(data_dir, folders[i])))
        img = read_nifti(os.path.join(data_dir, folders[i], files[i][0]))
        err0 = np.mean(((img[:, :, 20] + img[:, :, 22]) / 2 - img[:, :, 21])**2)
        err1 = np.mean(((img[:, :, 21] + img[:, :, 23]) / 2 - img[:, :, 22])**2)
        start0s[i] = 0 if err0 < err1 else 1
    
    try:
        while(True):
            for i in range(len(volumes)):

                new_vol = False
                if volumes[i] is not None:
                    try:
                        _ = q2.get_nowait()
                        new_vol = True
                    except:
                        pass

                if volumes[i] is None or new_vol:
                    fid = np.random.choice(np.arange(num_split, len(files[i])-num_split))
                    
                    angle = np.random.uniform(360)
                    hrs = []
                    
                    for dt in range(-num_split+1, num_split):
                        t = fid + dt
                            
                        img = read_nifti(os.path.join(data_dir, folders[i], files[i][t]))
                        img = (img - 70.0) / 100.0
                        img = rotate(img, angle, axes=(0, 1), reshape=False)
                        ss = (start0s[i] + t) % num_split
                        
                        if denoiser is not None:
                            d = 128-img.shape[0]
                            d1 = d//2
                            d2 = d - d1
                            if d1 >= 0:
                                frames = np.pad(img[..., ss::num_split], [(d1, d2),(d1, d2), (0, 0)], mode='constant')
                            else:
                                frames = img[-d1:d2, -d1:d2, ss::num_split]
                            frames = torch.tensor(frames[None]).permute(3, 0, 1, 2)
                            for n_slice in range(0, frames.shape[0], 16):
                                denoiser['in'].put(frames[n_slice:n_slice+16])
                                frames[n_slice:n_slice+16] = denoiser['out'].get()
                            if d1 >= 0:
                                frames = frames.squeeze().permute(1,2,0)[d1:-d2,d1:-d2].numpy()
                            else:
                                frames = np.pad(frames.squeeze().permute(1,2,0).numpy(), [(-d1, -d2),(-d1, -d2), (0, 0)], mode='constant')
                            img[..., ss::num_split] = frames
                
                        if is_denoise:
                            img = img[..., ss::num_split]
                        else:
                            X, Y, Z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), (np.arange(img.shape[2]) - ss) / num_split, indexing='ij')
                            img = map_coordinates(img[..., ss::num_split], [X, Y, Z], order=3, mode='nearest')

                        hrs.append(img)
                        
                    if is_denoise:
                        hrs = np.concatenate(hrs, -1)[None]
                        volumes[i] = (hrs, hrs)
                    else:
                        volumes[i] = (np.stack(down_up(*hrs), 0), np.stack(hrs, 0))
                    starts[i] = (start0s[i] + fid) % num_split

                    if new_vol:
                        q2.put(None)

                lr, hr = volumes[i]
                if is_denoise:
                    z = np.random.randint(lr.shape[3])
                    hr = hr[:, :, :, z]
                    d = 128-hr.shape[1]
                    d1 = d//2
                    d2 = d - d1
                    if d1 >= 0:
                        hr = np.pad(hr, [(0,0),(d1, d2),(d1, d2)], mode='constant')
                    else:
                        hr = hr[:, -d1:d2, -d1:d2]
                    lr = hr
                else:
                    y = np.random.randint(lr.shape[1] - a)
                    x = np.random.randint(lr.shape[2] - a)
                    z = np.random.randint((lr.shape[3] - starts[i]) // num_split)
                    lr = lr[:, y:y+a, x:x+a, starts[i] + z * num_split]
                    hr = hr[:, y:y+a, x:x+a, starts[i] + z * num_split]
                axis = np.random.choice([None, 1, 2])
                if axis is not None:
                    lr = np.flip(lr, axis).copy()
                    hr = np.flip(hr, axis).copy()
                lr = torch.tensor(lr, dtype=torch.float32)
                hr = torch.tensor(hr, dtype=torch.float32)
                queue.put((lr, hr))
    except:
        traceback.print_exc() 
        print("error: %s" % mp.current_process().name)
        
def prefetch_volumes_test(data_dir, folders, queue, q2, num_split, is_denoise, denoiser):
    try:
        for folder in folders:
            files = sorted(os.listdir(os.path.join(data_dir, folder)))
            img = read_nifti(os.path.join(data_dir, folder, files[0]))
            err0 = np.mean(((img[:, :, 20] + img[:, :, 22]) / 2 - img[:, :, 21])**2)
            err1 = np.mean(((img[:, :, 21] + img[:, :, 23]) / 2 - img[:, :, 22])**2)
            start = 0 if err0 < err1 else 1
            
            for fid in list(range(len(files)))[::(len(files)//7)][1:-1]:
                imgs = []
                combined = np.zeros_like(img)
                for dt in range(-num_split+1, num_split):
                    t = fid + dt
                    img = read_nifti(os.path.join(data_dir, folder, files[t]))
                    img = (img - 70.0) / 100.0
                    ss = (start + t) % num_split

                    combined[..., ss::num_split] += img[..., ss::num_split] * (num_split - np.abs(dt)) / num_split

                    if dt == 0:
                        if num_split == 4:
                            start_gt = (start+fid+2) % num_split
                            gt = 0 * img - 1000
                            gt[..., start_gt::num_split] = img[..., start_gt::num_split]
                        else:
                            gt = 0

                    if denoiser is not None:
                        d = 128-img.shape[0]
                        d1 = d//2
                        d2 = d - d1
                        if d1 >= 0:
                            frames = np.pad(img[..., ss::num_split], [(d1, d2),(d1, d2), (0, 0)], mode='constant')
                        else:
                            frames = img[-d1:d2, -d1:d2, ss::num_split]
                        frames = torch.tensor(frames[None]).permute(3, 0, 1, 2)
                        for n_slice in range(0, frames.shape[0], 16):
                            denoiser['in'].put(frames[n_slice:n_slice+16])
                            frames[n_slice:n_slice+16] = denoiser['out'].get()
                        if d1 >= 0:
                            frames = frames.squeeze().permute(1,2,0)[d1:-d2,d1:-d2].numpy()
                        else:
                            frames = np.pad(frames.squeeze().permute(1,2,0).numpy(), [(-d1, -d2),(-d1, -d2), (0, 0)], mode='constant')
                        img[..., ss::num_split] = frames

                    if is_denoise:
                        imgs.append(img[..., ss::num_split])
                    else:
                        X, Y, Z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), (np.arange(img.shape[2]) - ss) / num_split, indexing='ij')
                        imgs.append(map_coordinates(img[..., ss::num_split], [X, Y, Z], order=3, mode='nearest'))

                if is_denoise:
                    imgs = np.concatenate(imgs, -1)
                    d = 128-img.shape[0]
                    d1 = d//2
                    d2 = d - d1
                    if d1 >= 0:
                        imgs = np.pad(imgs, [(d1, d2),(d1, d2), (0, 0)], mode='constant')
                    else:
                        imgs = imgs[-d1:d2, -d1:d2]
                else:
                    imgs = np.stack(imgs, 0)
                queue.put((imgs, gt, combined, (start+fid) % num_split, os.path.join(folder, files[fid])))
    except:
        traceback.print_exc() 
        print("test error: %s" % mp.current_process().name)
    queue.put(None)
    return
    
        
class SimDataset(torch.utils.data.Dataset):
    def __init__(self, num_split, stage, is_denoise=False, denoiser=None):
        assert stage in ['test', 'train']
        self.is_test = stage == 'test'
        test_ga = ['25', '28', '30', '33', '35']
        data_dir = '/home/junshen/fetalSR/CRL_Fetal_Brain_Atlas_2017v3/'
        files = [f for f in os.listdir(data_dir) if ('STA' in f) and ('_' not in f)]
        if self.is_test:
            files = [f for f in files if any(ga in f for ga in test_ga)]
        else:
            files = [f for f in files if all(ga not in f for ga in test_ga)]
        files = [os.path.join(data_dir, f) for f in files]
        trajs = get_trajectory()
        self.proc = []
        
        num_p = 10
        self.queue = mp.Queue(1024)
        
        # use multiple processes to fetch data
        if self.is_test:
            imgs = [nib.load(f).get_fdata().astype(np.float32) / 1000.0 for f in files]
            trajs = [trajs[t][0] for t in [0, 5, 10, 15, 20, 25, 30]]
            imgs_trajs = list(product(imgs, trajs))
            self.length = len(imgs_trajs)
            self.res = []
        else:
            n_new = 6
            self.queue2 = mp.Queue(n_new)
            for _ in range(n_new):
                self.queue2.put(None)
            self.length = int(1e8)
                
        if denoiser is not None:
            denoise_queue = [{'in':mp.Queue(1),'out':mp.Queue(1)} for _ in range(num_p)]
            self.denoiser = mp.Process(target=denoise_fn, args=(denoiser, denoise_queue))
            self.denoiser.daemon = True
            self.denoiser.start()
            self.denoise_queue = denoise_queue
        else:
            denoise_queue = [None] * num_p
            self.denoise_queue = denoise_queue

        for i in range(num_p):
            if self.is_test:
                proc = mp.Process(target=prefetch_sim_volumes_test, args=(imgs_trajs[i::num_p], self.queue, num_split, is_denoise, denoise_queue[i]))
            else:
                proc = mp.Process(target=prefetch_sim_volumes, args=(files[i::num_p], self.queue, trajs, self.queue2, num_split, is_denoise, denoise_queue[i]))
            proc.daemon = True
            proc.start()
            self.proc.append(proc)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_test:
            if len(self.res) == 0:
                for i in range(self.length):
                    self.res.append(self.queue.get())
                    print(i, self.length)
                if self.denoise_queue[0] is not None:
                    self.denoise_queue[0]['in'].put(None)
            return self.res[idx]
        else:
            return self.queue.get()

def prefetch_sim_volumes(files, queue, trajs, q2, num_split, is_denoise, denoiser):
    #denoiser = denoiser.cuda()
    a = 64 # 64 (0.031)
    starts = [None] * len(files)
    volumes = [None] * len(files)
    imgs = [nib.load(f).get_fdata().astype(np.float32) / 1000.0 for f in files]
    try:    
        while True:
            for j in range(len(imgs)):
                new_vol = False
                if volumes[j] is not None:
                    try:
                        _ = q2.get_nowait()
                        new_vol = True
                    except:
                        pass
                if volumes[j] is None or new_vol:
                    
                    start = np.random.choice(num_split)
                    traj, T = trajs[np.random.choice(len(trajs))]
                    t0 = np.random.uniform(0, T)
                    hr, gt, combined, starts[j] =  sim_scan(imgs[j], num_split, traj, t0, 1.0 / imgs[j].shape[-1], start, np.eye(3,3), is_denoise, denoiser)
                    
                    lr = down_up(*hr)
                    volumes[j] = (np.stack(lr, 0), np.stack(hr, 0))
                    if new_vol:
                        q2.put(None)

                lr, hr = volumes[j]
                while True:
                    y = np.random.randint(lr.shape[1] - a)
                    x = np.random.randint(lr.shape[2] - a)
                    z = np.random.randint((lr.shape[3] - starts[j]) // num_split)
                    if is_denoise:
                        lr_ = lr[lr.shape[0]//2:lr.shape[0]//2+1, :, :, starts[j] + z * num_split]
                        hr_ = hr[lr.shape[0]//2:lr.shape[0]//2+1, :, :, starts[j] + z * num_split]
                        hr_ = np.pad(hr_, [(0,0),(28, 29),(1, 2)], mode='constant')
                    else:
                        lr_ = lr[:, y:y+a, x:x+a, starts[j] + z * num_split]
                        hr_ = hr[:, y:y+a, x:x+a, starts[j] + z * num_split]
                    if np.max(hr_) > 1:
                        lr, hr = lr_, hr_
                        break    
                axis = np.random.choice([None, 1, 2])
                if axis is not None:
                    lr = np.flip(lr, axis).copy()
                    hr = np.flip(hr, axis).copy()
                lr = torch.tensor(lr, dtype=torch.float32)
                hr = torch.tensor(hr, dtype=torch.float32)
                queue.put((lr, hr))
    except:
        traceback.print_exc() 
        print("error: %s" % mp.current_process().name)
        
def prefetch_sim_volumes_test(imgs_trajs, queue, num_split, is_denoise, denoiser):
    #denoiser = denoiser.cuda()
    try:
        for img, traj in imgs_trajs:
            t0 = 9
            inputs, gt, combined, start = sim_scan(img, num_split, traj, t0, 1.0 / img.shape[-1], 0, np.eye(3,3), is_denoise, denoiser)
            if is_denoise:
                gt = np.stack(gt, -1)
                inputs = inputs[len(inputs)//2][..., start::num_split]
                gt = np.pad(gt, [(28, 29),(1, 2),(0,0)], mode='constant')
                inputs = np.pad(inputs, [(28, 29),(1, 2),(0,0)], mode='constant')
            else:
                inputs = np.stack(inputs, 0)
            queue.put((inputs, gt, combined))
    except:
        traceback.print_exc() 
        print("test error: %s" % mp.current_process().name)
    return
    
def sim_scan(img, num_split, traj, t0, dt, start, rot0, is_denoise=False, model=None):
    #model = model.cuda
    t0 = t0 - dt * (img.shape[2] - img.shape[2] / 2 / num_split)
    idx = start
    i = 0
    gt = []
    all_frames = []
    combined = [0] * img.shape[2]
    frames = []
    sigma = img.max() * _sigma
    while True:
        if idx >= img.shape[2]:
            if model is not None:
                frames = np.pad(np.stack(frames, 0), [(0,0), (28, 29),(1, 2)], mode='constant')
                frames = torch.tensor(np.stack(frames, 0)[:, None])
                for n_slice in range(0, frames.shape[0], 16):
                    model['in'].put(frames[n_slice:n_slice+16])
                    frames[n_slice:n_slice+16] = model['out'].get()
                frames = frames.squeeze().permute(1,2,0)[28:-29,1:-2].numpy()
            else:
                frames = np.stack(frames, -1)
            X, Y, Z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), (np.arange(img.shape[2]) - start) / num_split, indexing='ij')
            all_frames.append(map_coordinates(frames, [X, Y, Z], order=3, mode='nearest'))
            if len(all_frames) == 2 * num_split - 1:
                return all_frames, gt, np.stack(combined, -1), (start + 1) % num_split 
            start = (start + 1) % num_split
            idx = start
            frames = []
            
        Rt = traj(t0 + i * dt)
        R = Rotation.from_euler('xyz', Rt[:3]).as_matrix() @ rot0
        t = Rt[3:] - [img.shape[0]/2, img.shape[1]/2, img.shape[2]/2] @ R.T + [img.shape[0]/2, img.shape[1]/2, img.shape[2]/2]
        frame = affine_transform(img, R, t, order=1)
        if (len(all_frames) == num_split - 1):
            if is_denoise:
                gt.append(frame[..., idx])
            elif ((idx - start) // num_split == img.shape[2] // num_split // 2):
                gt = frame
        frames.append(frame[..., idx])
        if sigma > 0:
            noise1 = np.random.normal(scale=sigma, size=frames[-1].shape).astype(np.float32)
            noise2 = np.random.normal(scale=sigma, size=frames[-1].shape).astype(np.float32)
            frames[-1] = np.sqrt((frames[-1] + noise1)**2 + noise2**2)
        #if num_split - 1 - num_split//2  <= len(all_frames) < 2*num_split - 1 - num_split//2:
        #    combined[idx] = frames[-1]
        combined[idx] += frames[-1] * (num_split - np.abs(len(all_frames) - num_split + 1)) / num_split
        idx += num_split
        i += 1

def denoise_fn(model, queues):
    model = model.cuda()
    #print('test')
    while True:
        for q in queues:
            try:
                inputs = q['in'].get_nowait()
                if inputs is None:
                    return
                with torch.no_grad():
                    q['out'].put(model(inputs.cuda()).cpu())
            except:
                pass
