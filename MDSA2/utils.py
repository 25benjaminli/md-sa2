import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import monai
import os
import time
import string
import matplotlib as mpl

import numpy as np
import logging


# import module
from importlib import import_module
from segment_anything import sam_model_registry
# cudnn
import torch.backends.cudnn as cudnn
import random
import gc
import SimpleITK as sitk
from matplotlib.widgets import Slider

from dotenv import load_dotenv
load_dotenv(override=True)

# os.getenv("DATASET_PATH") = os.path.join(os.getenv("PROJECT_PATH"), 'data', 'brats', 'BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData')
def join(*paths):
   return os.path.normpath(os.path.join(*paths))

def visualize_3D_volumes(names_to_volumes, names_to_labels, names_to_preds):
  """
  Args:
    names_to_volumes: dictionary of names to volumes. Each volume can be a 3D array, where it's C x D x H x W
    names_to_labels: dictionary of names to labels. Each label can be a 3D array, where it's C x D x H x W
    names_to_preds: dictionary of names to predictions. Each prediction can be a 3D array, where it's C x D x H x W
  """

  # make an interactive system where you can change the slice index while viewing all the volumes and labels simultaneously

  assert len(names_to_volumes) == len(names_to_labels) and len(names_to_labels) == len(names_to_preds), f"number of volumes and labels must be the same, got {len(names_to_labels)} labels and {len(names_to_volumes)} volumes and {len(names_to_preds)} preds"

  print("shape of volumes", names_to_volumes[list(names_to_volumes.keys())[0]].shape) # 3x155x224x224
  print("shape of labels", names_to_labels[list(names_to_labels.keys())[0]].shape) # 3x155x224x224
  print("shape of preds", names_to_preds[list(names_to_preds.keys())[0]].shape) # 3x155x224x224

  C, D, H, W = names_to_volumes[list(names_to_volumes.keys())[0]].shape
  def update_plots(slice_idx):
    # fig, axes, volumes, labels, predictions, C, 
    slice_idx = int(slice_idx)
    for i in range(C):
        
        axes[i, 0].imshow(real_volume[i, slice_idx, :, :], cmap='gray')
        axes[i, 0].set_title(f'Brain {vol_name}')

        axes[i, 1].imshow(volumes[i, slice_idx, :, :], cmap='gray')
        axes[i, 1].set_title(f'Pred Label {vol_name}')
        
        axes[i, 2].imshow(labels[i, slice_idx, :, :], cmap='gray')
        axes[i, 2].set_title(f'Label {label_name}')
        
        axes[i, 3].imshow(predictions[i, slice_idx, :, :], cmap='gray')
        axes[i, 3].set_title(f'Prediction {pred_name}')
    
    fig.canvas.draw_idle()

  volume_names = list(names_to_volumes.keys())
  label_names = list(names_to_labels.keys())
  pred_names = list(names_to_preds.keys())
  
  for batch_idx in range(len(volume_names)):
    fig, axes = plt.subplots(3, 4, figsize=(12, 12))  # 3x3 grid
    # increase spacing between subplots
    plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.5)  # Leave space for slider

    # Initial slice index
    initial_slice = 0
    vol_name, label_name, pred_name = volume_names[batch_idx], label_names[batch_idx], pred_names[batch_idx]

    volumes = names_to_volumes[vol_name].detach().cpu().numpy()
    labels = names_to_labels[label_name].detach().cpu().numpy()
    predictions = names_to_preds[pred_name].detach().cpu().numpy()

    # load brain_volume_{vol_name.zfill(3)}.npy
    real_volume = np.load(f'{os.getenv("UNREFINED_VOLUMES_PATH")}/brain_volume_{vol_name.zfill(3)}.npy')
    print("real volume shape", real_volume.shape)

    # update_plots(fig, axes, volumes, labels, predictions, C, initial_slice)
    update_plots(initial_slice)

    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    depth_slider = Slider(ax_slider, 'Depth', 0, D-1, valinit=initial_slice, valstep=1)

    # Update plots on slider value change
    depth_slider.on_changed(update_plots)

    # Display the interactive plot
    plt.show()


def get_volume_number(path):
  # two scenarios currently: either brats 2020 or brats africa
  # MUST be implemented in order to save volumes properly

  if "BraTS20" in path:
     return os.path.basename(os.path.dirname(path)).split('_')[-1]
  elif "BraTS-SSA" in path:
     # extremley hacky and dumb way to get it but it works kinda
     return str(int(os.path.basename(os.path.dirname(path)).split('-')[-2])).zfill(3)
  
def clear_cache():
  gc.collect()
  torch.cuda.empty_cache()

  # time.sleep(5)

def ReadImage(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def set_deterministic(seed):
  # call this before initializing the dataloaders, ALWAYS
  cudnn.benchmark = False
  cudnn.deterministic = True

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  monai.utils.set_determinism(seed)


def plot_img_test(img, label, preds, writer, iteration,epoch,image_time, log_dir, bbox_gt=None, bbox_pred=None, save=False, batch_val=0, vol_name=None, isTrain=False):
    # img: 4, 3, 224, 224
    # label: 4, 3, 224, 224
    # preds: 4, 3, 224, 224
    # print("image, label shape", img.shape, label.shape)
    mpl.rcParams['figure.dpi'] = 140
    
    # print(img.shape, label.shape, preds.shape)
    img_curr, label_curr, pred_curr = img[batch_val].cpu().numpy(), label[batch_val].cpu().numpy(), preds[batch_val].cpu().numpy()

    # maxxed_label = torch.argmax(img_curr, axis=0) # assuming 3, 224, 224
    fig, axes = plt.subplots(img_curr.shape[0], 3)
    fig.tight_layout()

    fontdict = {'fontsize': 10, 'fontweight': 'medium'}

    image_modalities = ['flair', 't1', 't1ce']
    label_tags = ['tumor core', 'whole tumor', 'enhancing tumor']

    # first row is image
    for col in range(3):
        axes[0, col].imshow(img_curr[col])
        axes[0, col].set_title(f'{image_modalities[col]}', fontdict=fontdict)
    
    # second row is ground truth
    for col in range(3):
        axes[1, col].imshow(label_curr[col])
        axes[1, col].set_title(f'GT {label_tags[col]}', fontdict=fontdict)

    # third row is predictions
    for col in range(3):
        axes[2, col].imshow(pred_curr[col])
        axes[2, col].set_title(f'PRED {label_tags[col]}', fontdict=fontdict)

    plt.close(fig)

def adjust_optimizer_lr(optimizer, args, iter_num):
    if args.use_warmup and iter_num < args.warmup_period:
          lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
          for param_group in optimizer.param_groups:
              param_group['lr'] = lr_


class ArrHelper:
  def __init__(self, arr, name, verbose=True):
    self.arr = arr
    self.name = name
    self.verbose = verbose
    self.shouldPrintName = True

    if torch.is_tensor(arr):
      self.device = arr.device

  def print_name(self):
    if self.shouldPrintName:
      print('\n')
      print(f'------- {self.name} -------')
      print('\n')

  def print(self):
    self.print_name()
    if self.verbose:
      if torch.is_tensor(self.arr):
        print('tensor: ', self.arr, self.arr.shape, self.arr.dtype, self.arr.device)
      else:
        print('np arr: ', self.arr, self.arr.shape, self.arr.dtype)
    else:
      if torch.is_tensor(self.arr):
        print('tensor: ', self.arr.shape, self.arr.dtype, self.arr.device)
      else:
        print('np arr: ', self.arr.shape, self.arr.dtype)


  def print_uniques(self):
    self.print_name()
    if torch.is_tensor(self.arr): print('tensor uniques: ', torch.unique(self.arr), 'for size', self.arr.shape)
    else: print('np arr uniques: ', np.unique(self.arr), 'for size', self.arr.shape)


  def print_min_max(self):
    self.print_name()
    print('tensor min, max: ', self.arr.min(), self.arr.max())

  def master_debug(self):
    print(f'------- {self.name} -------')
    self.shouldPrintName = False
    self.print()
    self.print_uniques()
    self.print_min_max()
    self.shouldPrintName = True

  def print_stats(self, channelWise=True):
    self.print_name()

    if not channelWise:
      print('mean', self.arr.mean(), 'std', self.arr.std())
    else:
      for channel in range(self.arr[0].shape[0]):
        print('mean', self.arr[0][channel].mean(), 'std', self.arr[0][channel].std())

    self.shouldPrintName = False
    self.print_min_max()
    self.shouldPrintName = True

  def get_numpy(self):
    return self.arr.detach().cpu().numpy()

  def get_torch(self, device=torch.device('cuda')):
    return torch.from_numpy(self.arr).to(device)

  def set_numpy(self):
    self.arr = self.get_numpy()

  def get_torch(self, device=torch.device('cuda')):
    self.arr = self.get_torch()

  def plot(self, axes, print_channel_arr=None, print_channel_plot=None, bbox=None):
    """
    @param axes: axes to draw on
    @param print_channel_arr: if None, means print all channels in the image. Otherwise, plot that specific channel.
    Assumes B C H W format. For my purposes, B is just 1.
    @param print_channel_plot: if None, it means you don't care where your image goes in the

    @param bbox: default None. Assumes typical (x1, y1, x2, y2) format.
    Assumes B 1 4 format. B is just 1

    """
    # self.arr.squeeze() # squeeze to remove first dimension

    # if tensor, convert to numpy
    # print(f'------- {self.name} -------')

    if torch.is_tensor(self.arr): self.set_numpy()  #convert to numpy
    if torch.is_tensor(bbox): bbox = bbox.detach().cpu().numpy()

    fontdict = {'fontsize': 8, 'fontweight': 'medium'}

    # process number of channels. Print each channel separately
    if print_channel_arr is None:
      if bbox is not None:
        # if bbox dims are 1,4, it is a single box, otherwise the first is ground truth, the second is prediction
        
        # x1, y1, x2, y2 = bbox.squeeze() # squeeze to remove first dimension
        # print("bounding box shape", bbox.shape)

        for channel in range(self.arr.squeeze().shape[0]):
          # plot on that channel
          axes[channel].imshow(self.arr[0][channel], cmap='gray')
          for idx in range(bbox.shape[0]):
            x1, y1, x2, y2 = bbox[idx].squeeze()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r' if idx == 0 else 'g', facecolor='none')
            axes[channel].add_patch(rect) # add bbox per graph
          axes[channel].set_title(f'ch {channel} {self.name}', fontdict=fontdict)
      else:
        for channel in range(self.arr.squeeze().shape[0]):
          # plot on that channel
          axes[channel].imshow(self.arr[0][channel], cmap='gray')
          axes[channel].set_title(f'ch {channel} {self.name}', fontdict=fontdict)


    else:
      # specific channel only
      if bbox is not None:
        x1, y1, x2, y2 = bbox.squeeze() # squeeze to remove first dimension
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        if print_channel_plot is not None:
          # means that you want it on a particular channel in the axes
          axes[print_channel_plot].imshow(self.arr[0][print_channel_arr], cmap='gray') # these are DISTINCT!!
          axes[print_channel_plot].add_patch(rect) # add bbox per graph
          axes[print_channel_plot].set_title(f'ch {print_channel_arr} {self.name}', fontdict=fontdict)
        else:
          # infer that they are the same
          axes[print_channel_arr].imshow(self.arr[0][print_channel_arr], cmap='gray') # these are DISTINCT!!
          axes[print_channel_arr].add_patch(rect) # add bbox per graph
          axes[print_channel_arr].set_title(f'ch {print_channel_arr} {self.name}', fontdict=fontdict)
      else:
        if print_channel_plot is not None:
          # means that you want it on a particular channel in the axes
          axes[print_channel_plot].imshow(self.arr[0][print_channel_arr], cmap='gray') # these are DISTINCT!!
          axes[print_channel_plot].set_title(f'ch {print_channel_arr} {self.name}', fontdict=fontdict)
        else:
          # infer that they are the same
          axes[print_channel_arr].imshow(self.arr[0][print_channel_arr], cmap='gray') # these are DISTINCT!!
          axes[print_channel_arr].set_title(f'ch {print_channel_arr} {self.name}', fontdict=fontdict)


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()
        self.cache = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, check_nan=True):
        self.val = val
        self.sum += val * n
        self.count += n
        self.cache.append(val)

        # print(self.val)
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def get_avgmeter_str(message, *args):
  st = ""
  for idx, arg in enumerate(args):
    st += f'{arg.name} {arg.avg} {arg.name} avg {sum(arg.avg)/3} \n' if idx < len(args) - 1 else f'{arg.name} {arg.avg} {arg.name} avg {sum(arg.avg)/3}'
  return message + '\n' + st

def print_avgmeters(message, *args):
  logging.info(get_avgmeter_str(message, *args))


def save_checkpoint(model, epoch, filename="model_aggregator.pt", best_acc=0, dir_add='../checkpoints'):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def register_medsam(model_config):
  sam, img_embedding_size = sam_model_registry[model_config.vit_name](checkpoint=model_config.sam_ckpt,
                                          image_size=model_config.img_size,
                                          num_classes=model_config.num_classes,
                                          pixel_mean=[0, 0, 0],
                                          pixel_std=[1, 1, 1]) # change pixel std and mean?? num classes is not including background
  
  print("image embedding size", img_embedding_size)
  # load checkpoint for sam if it exists
  
  sam = sam.to('cuda')

  return sam

def register_net(model_config):
  sam, img_embedding_size = sam_model_registry[model_config.vit_name](checkpoint=model_config.sam_ckpt,
                                          image_size=model_config.img_size,
                                          num_classes=model_config.num_classes,
                                          pixel_mean=[0, 0, 0],
                                          pixel_std=[1, 1, 1]) # change pixel std and mean?? num classes is not including background
  
  print("image embedding size", img_embedding_size)
  sam = sam.to('cuda')
  pkg = import_module(model_config.module)
  net = pkg.LoRA_Sam(sam, model_config.rank).cuda()
  if model_config.ft_ckpt is not None:
    net.load_lora_parameters(model_config.ft_ckpt)

  return net

def register_net_sam2(model_config):
  from sam2.build_sam import build_sam2
  from sam2.sam2_image_predictor import SAM2ImagePredictor

  # model cfg and checkpoint inferred from config
  if model_config.vit_name == "tiny":
    model_cfg = "sam2_hiera_t.yaml"
  elif model_config.vit_name == "small":
    model_cfg = "sam2_hiera_s.yaml"
  elif model_config.vit_name == "b+":
    model_cfg = "sam2_hiera_b+.yaml"
  elif model_config.vit_name == "large":
    model_cfg = "sam2_hiera_l.yaml"
  
  # load the cfg
  import yaml

  base_path = os.getenv("PROJECT_PATH")

  with open(f"{base_path}/MDSA2/segment_anything_2/sam2_configs/{model_cfg}", 'r') as stream:
    model_cfg_di = yaml.safe_load(stream)
    # modify the image_size parameter
    model_cfg_di['model']['image_size'] = model_config.img_size
    model_cfg_di['model']['compile_image_encoder'] = model_config.compile

  # write the modified cfg to the same file
  with open(f"{base_path}/MDSA2/segment_anything_2/sam2_configs/{model_cfg}", 'w') as stream:
    yaml.dump(model_cfg_di, stream)
  
  if model_config.vit_name == "tiny":
     sam2_checkpoint = f"{base_path}/MDSA2/checkpoints/sam2_hiera_tiny.pt" # 2.1 or 2
  elif model_config.vit_name == "b+":
     sam2_checkpoint = f"{base_path}/MDSA2/checkpoints/sam2_hiera_base_plus.pt"
  elif model_config.vit_name == "small":
     sam2_checkpoint = f"{base_path}/MDSA2/checkpoints/sam2_hiera_small.pt"

  elif model_config.vit_name == "large":
      sam2_checkpoint = f"{base_path}/MDSA2/checkpoints/sam2_hiera_large.pt"
  # elif model_config.vit_name == "none":
  #     sam2_checkpoint = None

  sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
  predictor = SAM2ImagePredictor(sam_model=sam2_model)

  # Set training parameters

  predictor.model.sam_prompt_encoder.train(False)
  predictor.model.memory_encoder.train(False)


  predictor.model.image_encoder.train(True)
  predictor.model.sam_mask_decoder.train(True)
  from models import LoRA_SAM2, SAM2_Regular
  if model_config.rank > 0:
    print("using lora rank", model_config.rank)
    net = LoRA_SAM2(predictor, model_config.rank).cuda()

    if model_config.ft_ckpt is not None:
      net.load_lora_parameters(model_config.ft_ckpt)
  else:
    # print("using regular")
    net = SAM2_Regular(predictor).cuda() # means you just use the regular one

    if model_config.ft_ckpt is not None: # not actually lora, but just the pretrained one...
      # print("loading ckpt", model_config.ft_ckpt)
      thing = torch.load(model_config.ft_ckpt, weights_only=False)
      if "state_dict" in thing.keys():
         state_dict = thing["state_dict"]
      else:
         state_dict = thing
      net.predictor.model.load_state_dict(state_dict)

  return net

def generate_rndm_path(base_path, make_downstream_folder=True, length=10):
  if not os.path.exists(base_path):
     # then just make it and return it
      os.makedirs(base_path)
      return base_path
  if make_downstream_folder:
    l = join(base_path, ''.join(random.choices(string.ascii_lowercase + string.digits, k=length)))
  else:
    l = base_path + '_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
  if os.path.exists(l):
    return generate_rndm_path(base_path, length)
  else:
     os.makedirs(l)
  return l

def initialize_logger(snapshot_path, log_file='train.log'):
  for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
  
  logging.basicConfig(filename=snapshot_path + f"/{log_file}", level=logging.INFO, format="%(asctime)s - %(message)s")
  logger = logging.getLogger(__name__)
  
  print("logging to", snapshot_path + "/train.log")
  return logger

def generate_snapshot_path(config, output_path='./runs', isVal = False):
  snapshot_path = ""
  if config.custom_name != None:
    if not os.path.exists(f"{output_path}/{config.dataset}/{config.custom_name}"):
      snapshot_path = f"{output_path}/{config.dataset}/{config.custom_name}"
      os.makedirs(snapshot_path, exist_ok=False)
    else:
      ct = 0
      for f in os.listdir(f'{output_path}/{config.dataset}'):
        # print(f, config.custom_name in f)
        if config.custom_name in f:
           ct+=1

      snapshot_path = f"{output_path}/{config.dataset}/{config.custom_name}_{ct+1}"
      os.makedirs(snapshot_path, exist_ok=False)
  else:
    ct = 0
    test_string = "test"
    for f in os.listdir(f'{output_path}'):
      if os.path.isdir(f) and test_string in f:
          ct+=1

    snapshot_path = f"{output_path}/{config.dataset}/{test_string}_{ct+1}"
    os.makedirs(snapshot_path, exist_ok=False)

  return snapshot_path

def run_fn_and_timer(fn, name, **kwargs):
  start = time.time()
  result = fn(**kwargs)
  end = time.time()
  print(f'{name} took {end - start} seconds')
  return result


def print_data(logger, acc_dict,exclude_keys=[]):
    for key in acc_dict:
        # print(f"ACCURACY FOR {key}")
        if type(acc_dict[key]) == AverageMeter:
            avg_metric = acc_dict[key].avg
            logger.info(f"{key}: {avg_metric}")
            for i, metric in enumerate(avg_metric):
                logger.info(f"{key} {i}: {metric}")
            logger.info(
                " ".join([f"{key} {i}: {metric}" for i, metric in enumerate(avg_metric)]),
                ", avg:", float(np.mean(avg_metric))
            )
        elif type(acc_dict[key]) == list or type(acc_dict[key]) == np.ndarray:
            # logger.info(f"{key}: {acc_dict[key]}")
            logger.info(
                f"{key} {acc_dict[key]} avg: {np.mean(acc_dict[key]).tolist()}"
            )
        elif key not in exclude_keys:
            logger.info(f"{key}: {acc_dict[key]}")

    """
        ", time {:.2f}s".format(time.time() - start_time),
    ", avg inference time", np.mean(inference_durations)
    """
    
def calculate_heuristic(info, weights, minus, volume_list):
  # calculate a single time to sort for curriculum learner
  #  mean_m, mean_d, mean_n = np.mean(metrics), np.mean(distance_vars), np.mean(num_pixels)
  """
  info: {
    "metrics": np array (300)
    "...": np array (300)
    "...": np array (300)
    ...
  }

  minus: {
    "metrics": np array (300),
    ...
  }

  volume_list: ['135', '074', ...]

  pseudocode: 
  vol_maps = {}
  vol_list = [] # size

  # nonzeros are the same regardless of the key
  random_key = info.keys()[0]
  nonzeros = info[random_key].nonzeros # return the indices of nonzeros


  for current_volume in nonzeros:
    # this will return the current volume number (since it's the index)
    index_of_volume = volume_list.indexof(current_volume)
    # now we can zip this index together with the stats

    aggregated_value = 0
    for key in info:
      value = (info[k][current_volume]/np.max(info[key]))
      aggregated_value += (1-value) * weights[key] if minus[key] else value * weights[key]
    
    arr.append((index_of_volume, aggregated_value))

  return arr
  
  """

  heuristic_summed = [] # size
  heuristic_arr = []

  # nonzeros are the same regardless of the key
  random_key = list(info.keys())[0]
  # print("random key", random_key)
  # print(info[random_key])
  nonzeros = np.nonzero(info[random_key])[0] # return the indices of nonzeros


  print("number of nonzeros", len(nonzeros))

  for current_volume in nonzeros:
    # this will return the current volume number (since it's the index)
    index_of_volume = volume_list.index(str(current_volume).zfill(3)) # getting index of volume for the dataset
    # now we can zip this index together with the stats

    aggregated_value = 0
    heuristic_arr.append([])
    for key in info:
      value = (info[key][current_volume]/np.max(info[key])) if key != 'losses' else info[key][current_volume]
      aggregated_value += (1-value) * weights[key] if minus[key] else value * weights[key]
      heuristic_arr[-1].append((1-value) * weights[key] if minus[key] else value * weights[key])
    
    heuristic_summed.append((index_of_volume, aggregated_value))

  return heuristic_summed, heuristic_arr