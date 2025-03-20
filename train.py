import gc
import numpy as np
import torch.nn as nn
import yaml
import shutil
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter

from datasets.dataset import img_batch_tensor2numpy, Chunked_sample_dataset
# from models.ml_memAE_sc import ML_MemAE_SC
# from model.utils import Reconstruction3DDataLoader,Reconstruction3DFlowDataLoader
from model.CrossAttention import MotionGuide ##july
from utils.initialization_utils import weights_init_kaiming
from utils.model_utils import loader, saver, only_model_saver
from utils.vis_utils import visualize_sequences,vs
# import ml_memAE_sc_eval
from einops import rearrange
from losses.loss import Gradient_Loss, Intensity_Loss, aggregate_kl_loss

def train(config, training_chunked_samples_dir, testing_chunked_samples_file):
    paths = dict(log_dir="%s/%s_CrossAttention" % (config["log_root"], config["dataset_name"]),
                 ckpt_dir="%s/%s_CrossAttention" % (config["ckpt_root"], config["dataset_name"]))
    if not os.path.exists(paths["ckpt_dir"]):
        os.makedirs(paths["ckpt_dir"])
    if not os.path.exists(paths["log_dir"]):
        os.makedirs(paths["log_dir"])

    batch_size = config["batchsize"]
    epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    device = config["device"]

    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))
    # loss functions
    grad_loss = Gradient_Loss(config["alpha"],
                              config["model_paras"]["img_channels"] * config["model_paras"]["clip_pred"],
                              device).to(device)
    intensity_loss = Intensity_Loss(l_num=config["intensity_loss_norm"]).to(device)
    mse_loss = nn.MSELoss()

    model = MotionGuide()
    model.to(config["device"])
    # model = nn.DataParallel(model) #July

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], eps=1e-7, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.8)

    step = 0
    epoch_last = 0

    if not config["pretrained"]:
        model.apply(weights_init_kaiming)
    else:
        assert (config["pretrained"] is not None)
        model_state_dict, optimizer_state_dict, step = loader(config["pretrained"])
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        epoch_last = int(config["pretrained"].split('-')[-1])
        print('pretrained models loaded!', epoch_last)

    writer = SummaryWriter(paths["log_dir"])
    # copy config file
    shutil.copyfile("./cfgs/CrossAttention.yaml",
                    os.path.join(config["log_root"], config["dataset_name"]+"_CrossAttention", "CrossAttention.yaml"))

    # Training
    best_auc = -1
    for epoch in range(epoch_last, epochs + epoch_last):
        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file), last_flow=False)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            for idx, train_data in tqdm(enumerate(dataloader),
                                        desc="Training Epoch %d, Chunk File %d" % (epoch + 1, chunk_file_idx),
                                        total=len(dataloader)):
                model.train()

                sample_frames, sample_ofs, boxs, _, _ = train_data  #motion(flow),ML_MemAE_SC reconstruction
                # print(sample_ofs.shape) #(256,3,32,32)(B,C,H,W)
                sample_ofs = sample_ofs.to(device)
                sample_frames = sample_frames.to(device)
                # print(sample_frames[:,0:12,:,:].shape)
                # print(sample_ofs.shape)

                out,out_Flow,_,_ = model(sample_frames[:,0:12,:,:], sample_ofs)
                # print(out.shape)
                loss_frame = intensity_loss(out, sample_frames[:,12:,:,:])
                loss_grad = grad_loss(out, sample_frames[:,12:,:,:])
                # print(out_Flow.shape,sample_ofs.shape)
                loss_flow_recon = mse_loss(out_Flow, sample_ofs)

                
                loss_all = config["lam_frame"] * loss_frame + \
                           config["lam_grad"] * loss_grad + \
                            config["lam_recon"] * loss_flow_recon

                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()

                if step % config["logevery"] == config["logevery"] - 1:
                    print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss_all))

                    writer.add_scalar('loss_total/train', loss_all, global_step=step + 1)
                    # writer.add_scalar('loss_recon/train', loss_recon, global_step=step + 1)
                    # writer.add_scalar('loss_sparsity/train', config["lam_sparse"] * loss_sparsity, global_step=step + 1)

                    num_vis = 4
                step += 1
            del dataset

        scheduler.step()

        if epoch % config["saveevery"] == config["saveevery"] - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], config["model_savename"])
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=300)

            # evaluation
            # with torch.no_grad():
            #     auc = ml_memAE_sc_eval.evaluate(config, model_save_path + "-%d" % (epoch + 1),
            #                                     testing_chunked_samples_file,
            #                                     suffix=str(epoch + 1))
            #     if auc > best_auc:
            #         best_auc = auc
            #         only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))

            #     writer.add_scalar("auc", auc, global_step=epoch + 1)

    print("================ Best AUC %.4f ================" % best_auc)



def cal_training_stats(config, ckpt_path, training_chunked_samples_dir, stats_save_path):
    device = config["device"]
    model = MotionGuide().to(config["device"]).eval()

    # load weights
    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    # print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    of_training_stats = []
    frame_training_stats = []

    print("=========Forward pass for training stats ==========")
    with torch.no_grad():

        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, shuffle=False)

            for idx, data in tqdm(enumerate(dataloader),
                                  desc="Training stats calculating, Chunked File %02d" % chunk_file_idx,
                                  total=len(dataloader)):
                sample_frames, sample_ofs, _, _, _ = data
                # net_in = rearrange(sample_frames, 'b (c n) h w -> b c n h w', c=3)
                # net_in = net_in[:,:,0:4,:,:]
                sample_frames = sample_frames.to(device)
                sample_ofs = sample_ofs.to(device)

                out,out_flow,_,_ = model(sample_frames[:,0:12,:,:],sample_ofs)

                loss_frame = score_func(out, sample_frames[:,12:,:,:]).cpu().data.numpy()
                # print(loss_frame.shape)
                loss_of = score_func(out_flow, sample_ofs).cpu().data.numpy()

                of_scores = np.sum(np.sum(np.sum(loss_of, axis=3), axis=2), axis=1)
                frame_scores = np.sum(np.sum(np.sum(loss_frame, axis=3), axis=2), axis=1)

                of_training_stats.append(of_scores)
                frame_training_stats.append(frame_scores)
            del dataset
            gc.collect()

    print("=========Forward pass for training stats done!==========")
    of_training_stats = np.concatenate(of_training_stats, axis=0)
    frame_training_stats = np.concatenate(frame_training_stats, axis=0)

    training_stats = dict(of_training_stats=of_training_stats,
                          frame_training_stats=frame_training_stats)
    # save to file
    torch.save(training_stats, stats_save_path)


if __name__ == '__main__':
    config = yaml.safe_load(open("./cfgs/CrossAttention.yaml")) ##july

    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    training_chunked_samples_dir = os.path.join(dataset_base_dir, dataset_name, "training/chunked_samples")
    testing_chunked_samples_file = os.path.join(dataset_base_dir, dataset_name,
                                                "testing/chunked_samples/chunked_samples_00.pkl")

    train(config, training_chunked_samples_dir, testing_chunked_samples_file)
