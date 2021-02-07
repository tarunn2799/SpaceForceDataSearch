import torch
import numpy as np
import PIL.Image as Image
from torch.nn import functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.optim import Adam
import pandas as pd 
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import shutil
from sklearn.metrics import classification_report
import os
from imutils import paths 
from os import path
import splitfolders
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import PIL
from PIL import Image
from pytorch_lightning.loggers import WandbLogger

from typing import List, Optional
from pytorch_lightning.metrics import Accuracy
from pathlib import Path
from argparse import ArgumentParser

from sklearn.metrics import f1_score, accuracy_score

from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

#internal imports
from transforms_dali import SimCLRTrainDataTransform
from finetuner_dali_distrib import finetuner
from encoders_dali import load_encoder
from ssl_dali_distrib import cli_main, SIMCLR
from utils import plot_metrics, plot_umap, get_embeddings, n_random_subset, prepare_dataset, class_distrib, farthest_point, min_max_diverse_embeddings, animate, get_embeddings_test
from cli_main import cli_main
from TSNE import TSNE_visualiser


def generate_classification_report(ckpt,dataset_path,size):
  
  model = finetuner.load_from_checkpoint(ckpt)
  model.eval()
  model.cuda()

  preds = []
  labels = []
  
  image_size = size
  t = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((size,size)),
    transforms.ToTensor()
  ])
  full_ds = ImageFolder(root = dataset_path)
  full_ds_loader = torch.utils.data.DataLoader(full_ds, batch_size=1,shuffle = False, drop_last=False)
  files = [full_ds.samples[i][0] for i in range(len(full_ds))]
  labels = [full_ds.samples[i][1] for i in range(len(full_ds))]

  for f in files:
    im = Image.open(f).convert('RGB')
    image = t(im)
    # im = im.resize((image_size,image_size))
    # image = np.transpose(im, (2,0,1)).copy()
    im = torch.tensor(image).unsqueeze(0).float()
    im = im.cuda()

    with torch.no_grad():
        probs = model(im)
        result = torch.argmax(probs, dim = 1)[0].detach().cpu().item()
        preds.append(result)

  labels_w_names = []
  preds_w_names = []
  for i in range(len(labels)):
    labels_w_names.append(full_ds.classes[labels[i]])
    preds_w_names.append(full_ds.classes[preds[i]])
  print('Accuracy:')
  print(sum(np.array(preds) == np.array(labels))/len(np.array(preds)))
  print(classification_report(labels_w_names, preds_w_names, full_ds.classes))
  return classification_report(labels_w_names, preds_w_names, full_ds.classes, output_dict=True)


def driver():

  parser = ArgumentParser()
  parser.add_argument("--image_size", default = 256, type=int, help="Size of the image")
  parser.add_argument("--DATA_PATH", type=str, help="path to folders with images")
  parser.add_argument("--encoder", default=None , type=str, help="encoder to initialize. Can accept SimCLR model checkpoint or just encoder name in from encoders_dali")
  parser.add_argument("--batch_size", default=128, type=int, help="batch size for SSL")
  parser.add_argument("--num_workers", default=1, type=int, help="number of workers to use to fetch data")
  parser.add_argument("--hidden_dims", default=128, type=int, help="hidden dimensions in classification layer added onto model for finetuning")
  parser.add_argument("--epochs", default=400, type=int, help="number of epochs to train model")
  parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for training model")
  parser.add_argument("--patience", default=-1, type=int, help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.")
  parser.add_argument("--val_split", default=0.2, type=float, help="percent in validation data")
  parser.add_argument("--withhold_split", default=0, type=float, help="decimal from 0-1 representing how much of the training data to withold from either training or validation. Used for experimenting with labels neeeded")
  parser.add_argument("--gpus", default=1, type=int, help="number of gpus to use for training")
  parser.add_argument("--log_name", type=str, help="name of model to log on wandb and locally")
  parser.add_argument("--online_eval", default=False, type=bool, help="Do finetuning on model if labels are provided as a sanity check")
  parser.add_argument("--num_iterations", default=1, type=int, help="Number of times pretraining occurs")
  parser.add_argument("--subset_size", default= 0.1, type = float, help= "size of the subset of dataset that goes into SimCLR training")
  parser.add_argument("--buffer_dataset_path", type= str, help = "Where the subsets are stored everytime before passing into SimCLR")
  parser.add_argument("--metric", default="count", type=str, help="Type of Metric to evaluate pretraining")
  parser.add_argument("--UMAP", default = False, type =bool, help = 'Toggles UMAP plots')
  parser.add_argument("--TSNE", default = False, type= bool, help= 'Toggles TSNE plots' )
  parser.add_argument("--sampling", default = 'da', type = str, help= 'Select between random sampling or diverse embeddings')
  args = parser.parse_args()
  size = args.image_size
  DATA_PATH = args.DATA_PATH
  batch_size = args.batch_size
  num_workers = args.num_workers
  hidden_dims = args.hidden_dims
  epochs = args.epochs
  lr = args.lr
  patience = args.patience
  val_split = args.val_split
  withhold = args.withhold_split
  gpus = args.gpus
  encoder = args.encoder
  log_name = 'SIMCLR_SSL_' + args.log_name + '.ckpt'
  online_eval = args.online_eval
  num_iters = args.num_iterations
  subset_size = args.subset_size
  buffer_dataset_path = args.buffer_dataset_path
  metric = args.metric
  umap_exec = args.UMAP
  tsne_exec = args.TSNE
  sampling = args.sampling

  train_image_paths = list(paths.list_images(DATA_PATH))
  random_points_fnames = n_random_subset(subset_size, train_image_paths)
  prepare_dataset(buffer_dataset_path , random_points_fnames)

  ims = []
  for folder in os.listdir(DATA_PATH):
    for im in os.listdir(f'{DATA_PATH}/{folder}'):
      ims.append(f'{DATA_PATH}/{folder}/{im}')

  metric_array = []
  classification_reports = []

  #creating a folder for storing graphs
  try:
    os.mkdir("./graphs/")
    if umap_exec: 
      os.mkdir("./graphs/UMAP/")
      os.mkdir("./graphs/UMAP/DiversityAlgorithm/")
      os.mkdir("./graphs/UMAP/Full_Dataset/")
    if tsne_exec:
      os.mkdir("./graphs/TSNE/")
      os.mkdir("./graphs/TSNE/DiversityAlgorithm/")
      os.mkdir("./graphs/TSNE/Full_Dataset/")
  except:
    shutil.rmtree("./graphs/")
    os.mkdir("./graphs/")
    if umap_exec: 
      os.mkdir("./graphs/UMAP/")
      os.mkdir("./graphs/UMAP/DiversityAlgorithm/")
      os.mkdir("./graphs/UMAP/Full_Dataset/")
    if tsne_exec:
      os.mkdir("./graphs/TSNE/")
      os.mkdir("./graphs/TSNE/DiversityAlgorithm/")
      os.mkdir("./graphs/TSNE/Full_Dataset/")
  
  
  for i in range(num_iters):
    print("-----------------Iteration: ",i+1,"----------------------")
    ckpt = cli_main(size, buffer_dataset_path, batch_size, num_workers, hidden_dims, epochs, lr, 
            patience, val_split, withhold, gpus, encoder, log_name + "_Cycle_"+str(i)+".ckpt", online_eval)
    print("CHECKPOINT",ckpt)
    encoder = ckpt
 
    print("Obtaining Embeddings")
    embedding = get_embeddings_test(encoder, ims, size)
    if sampling == 'da':
      print("Applying Diversity Algorithm...")
      da_files, da_embeddings, da_distances = min_max_diverse_embeddings(subset_size, ims,  embedding, i = farthest_point(embedding))
      
      print("Preparing New Dataset...")
      prepare_dataset(buffer_dataset_path, da_files)
      
      print("New Dataset abiding formats successfully created")
      metric_array.append(class_distrib(buffer_dataset_path, metric= metric))
      
      print("Running Finetuning...")
      os.system("python /content/SpaceForceDataSearch/finetuner_dali_distrib.py --DATA_PATH \""+ buffer_dataset_path +"\" --encoder \"" + encoder +"\" --num_workers 2 --log_name \"ft_resnet18_dasubset_cycle_"+str(i)+ "\" --epochs 50 --batch_size 128 --image_size 224")

      print("Generating Classification Report...")

      classification_reports.append(generate_classification_report("./models/FineTune/FineTune_ft_resnet18_dasubset_cycle_"+str(i)+".ckpt","/data/NASA_Rudy/Alphabet_Soup/GRID_A-Z/64",size))
      print("Number and Shape of Embeddings:", len(embedding),embedding[0].shape)
      if umap_exec:
        plot_umap(da_embeddings, da_files, count= i, path="./graphs/UMAP/DiversityAlgorithm/")
        plot_umap(embedding, ims, count= i, path="./graphs/UMAP/Full_Dataset/")
      ##TSNE 
      if tsne_exec:
        print("Starting TSNE")
        da_tsne = TSNE_visualiser(da_embeddings, da_files)
        full_tsne = TSNE_visualiser(embedding, ims)
        
        tsne_results = da_tsne.fit_tsne(da_tsne.feature_list)
        
        print('TSNE Full Dataset')
        full_tsne_results = full_tsne.fit_tsne(full_tsne.feature_list)
        print("TSNE Fit complete")
        
        da_tsne.scatter_plot(tsne_results, da_tsne.labels, "./graphs/TSNE/DiversityAlgorithm/", i)
        full_tsne.scatter_plot(full_tsne_results, full_tsne.labels, "./graphs/TSNE/Full_Dataset/", i)
        da_tsne.show_tsne(tsne_results[:, 0], tsne_results[:, 1], da_tsne.filenames, "./graphs/TSNE/DiversityAlgorithm", i)
        da_tsne.tsne_to_grid_plotter_manual(tsne_results[:, 0], tsne_results[:, 1], da_tsne.filenames, "./graphs/TSNE/DiversityAlgorithm", i)
        print("TSNE Graphs created and stored")
    
    if sampling == 'random':
      random_files = n_random_subset(subset_size, ims)
      prepare_dataset(buffer_dataset_path, random_files)

      print("New Dataset abiding formats successfully created")
      metric_array.append(class_distrib(buffer_dataset_path, metric= metric))
      
      print("Running Finetuning...")
      os.system("python /content/SpaceForceDataSearch/finetuner_dali_distrib.py --DATA_PATH \""+ buffer_dataset_path +"\" --encoder \"" + encoder +"\" --num_workers 2 --log_name \"ft_resnet18_randomsubset_cycle_"+str(i)+ "\" --epochs 50 --batch_size 128 --image_size 224")
      # print('Skipping Finetuning')

      print("Generating Classification Report...")
    
      classification_reports.append(generate_classification_report("./models/FineTune/FineTune_ft_resnet18_randomsubset_cycle_"+str(i)+".ckpt","/data/NASA_Rudy/Alphabet_Soup/GRID_A-Z/64",size))
      print("Number and Shape of Embeddings:", len(embedding),embedding[0].shape)
      if umap_exec:
        plot_umap(embedding, ims, count= i, path="./graphs/UMAP/Full_Dataset/")
      if tsne_exec:
        print("Starting TSNE")
        full_tsne = TSNE_visualiser(embedding, ims)
        print('TSNE Full Dataset')
        full_tsne_results = full_tsne.fit_tsne(full_tsne.feature_list)
        print("TSNE Fit complete")
        full_tsne.scatter_plot(full_tsne_results, full_tsne.labels, "./graphs/TSNE/Full_Dataset/", i)

    if umap_exec:
      animate("./graphs/UMAP/Full_Dataset",fps=1)
    if tsne_exec:
      animate("./graphs/TSNE/Full_Dataset/", fps = 1)
    if metric != "count":
      print(metric_array)
      plot_metrics(metric, metric_array)

  complete_report = pd.concat([pd.DataFrame(report).transpose()['f1-score'] for report in classification_reports],axis = 1)
  complete_report.to_csv('./performance_report_final.csv')
  metric_report = pd.DataFrame(metric_array)
  metric_report.to_csv('./class_distrib_report.csv')
  print(complete_report)


if __name__ == "__main__":
  driver()
