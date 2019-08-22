import sys, os, glob, random
import time
import parser
import torch
import torch.nn as nn
# from AdaAdam import AdaAdam
import torch.optim as OPT
import numpy as np
from copy import deepcopy
from tqdm import tqdm, trange
import logging

from torchtext import data
import DataProcessing
from DataProcessing.MLTField import MTLField
from DataProcessing.NlcDatasetSingleFile import NlcDatasetSingleFile
from CNNModel import CNNModel


logger = logging.getLogger(__name__)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO )
batch_size = 10
seed = 12345678
torch.manual_seed(seed)
Train = False


device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
n_gpu = torch.cuda.device_count()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

def load_train_test_files(listfilename, test_suffix='.test'):
    filein = open(listfilename, 'r')
    file_tuples = []
    task_classes = ['.t2', '.t4', '.t5']
    for line in filein:
        array = line.strip().split('\t')
        line = array[0]
        for t_class in task_classes:
            trainfile = line + t_class + '.train'
            devfile = line + t_class + '.dev'
            testfile = line + t_class + test_suffix
            file_tuples.append((trainfile, devfile, testfile))
    filein.close()
    return file_tuples

filelist = 'data/Amazon_few_shot/workspace.filtered.list'
targetlist = 'data/Amazon_few_shot/workspace.target.list'
workingdir = 'data/Amazon_few_shot'
emfilename = 'glove.6B.300d'
emfiledir = '..'

datasets = []
list_datasets = []


file_tuples = load_train_test_files(filelist)
print(file_tuples)

TEXT = MTLField(lower=True)
for (trainfile, devfile, testfile) in file_tuples:
    print(trainfile, devfile, testfile)
    LABEL1 = data.Field(sequential=False)
    train1, dev1, test1 = NlcDatasetSingleFile.splits(
        TEXT, LABEL1, path=workingdir, train=trainfile,
        validation=devfile, test=testfile)
    datasets.append((TEXT, LABEL1, train1, dev1, test1))
    list_datasets.append(train1)
    list_datasets.append(dev1)
    list_datasets.append(test1)

target_datasets = []
target_file = load_train_test_files(targetlist)
print(target_file)

for (trainfile, devfile, testfile) in target_file:
    print(trainfile, devfile, testfile)
    LABEL2 = data.Field(sequential=False)
    train2, dev2, test2 = NlcDatasetSingleFile.splits(TEXT, LABEL2, path=workingdir, 
    train=trainfile,validation=devfile, test=testfile)
    target_datasets.append((TEXT, LABEL2, train2, dev2, test2))

    

datasets_iters = []
for (TEXT, LABEL, train, dev, test) in datasets:
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=batch_size, device=device,shuffle=True)
    train_iter.repeat = False
    datasets_iters.append((train_iter, dev_iter, test_iter))

fsl_ds_iters = []
for (TEXT, LABEL, train, dev, test) in target_datasets:
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train,dev, test), batch_size=batch_size, device=device)
    train_iter.repeat = False
    fsl_ds_iters.append((train_iter, dev_iter, test_iter))

num_batch_total = 0
for i, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
    # print('DATASET%d'%(i+1))
    # print('train.fields', train.fields)
    # print('len(train)', len(train))
    # print('len(dev)', len(dev))
    # print('len(test)', len(test))
    # print('vars(train[0])', vars(train[0]))
    num_batch_total += len(train) / batch_size

TEXT.build_vocab(list_datasets, vectors = emfilename, vectors_cache = emfiledir)
# TEXT.build_vocab(list_dataset)

# build the vocabulary
for taskid, (TEXT, LABEL, train, dev, test) in enumerate(datasets):
    LABEL.build_vocab(train, dev, test)
    LABEL.vocab.itos = LABEL.vocab.itos[1:]

    for k, v in LABEL.vocab.stoi.items():
        LABEL.vocab.stoi[k] = v - 1

    # print vocab information
    # print('len(TEXT.vocab)', len(TEXT.vocab))
    # print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # print(LABEL.vocab.itos)
    # print(len(LABEL.vocab.itos))

    # print(len(LABEL.vocab.stoi))
fsl_num_tasks = 0
for taskid, (TEXT, LABEL, train, dev, test) in enumerate(target_datasets):
    fsl_num_tasks += 1
    LABEL.build_vocab(train, dev, test)
    LABEL.vocab.itos = LABEL.vocab.itos[1:]
    for k, v in LABEL.vocab.stoi.items():
        LABEL.vocab.stoi[k] = v - 1

nums_embed = len(TEXT.vocab)
dim_embed = 100
dim_w_hid = 200
dim_h_hid = 100
Inner_lr = 2e-6
Outer_lr = 1e-5

n_labels = []
for (TEXT, LABEL, train, dev, test) in datasets:
   n_labels.append(len(LABEL.vocab))
print(n_labels)
num_tasks = len(n_labels)
print("num_tasks", num_tasks)
winsize = 3
num_labels = len(LABEL.vocab.itos)
model = CNNModel(nums_embed, num_labels, dim_embed, dim_w_hid, dim_h_hid, winsize, batch_size)

print("GPU Device: ", device)
model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
opt = OPT.Adam(model.parameters(), lr=Inner_lr)
Inner_epochs = 4
epochs = 2

N_task = 5

task_list = np.arange(num_tasks)
print("Total Batch: ", num_batch_total)
output_model_file = '/tmp/CNN_MAML_output'
if Train:
    for t in trange(int(num_batch_total*epochs/Inner_epochs), desc="Iterations"):
        selected_task = np.random.choice(task_list, N_task,replace=False)
        weight_before = deepcopy(model.state_dict())
        update_vars = []
        fomaml_vars = []
        for task_id in selected_task:
            # print(task_id)
            (train_iter, dev_iter, test_iter) = datasets_iters[task_id]
            train_iter.init_epoch()
            model.train()
            n_correct = 0
            n_step = 0
            for inner_iter in range(Inner_epochs):
                batch = next(iter(train_iter))

                # print(batch.text)
                # print(batch.label)
                logits = model(batch.text)
                loss = criterion(logits.view(-1, num_labels), batch.label.data.view(-1))
                

                n_correct = (torch.max(logits, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
                n_step = batch.batch_size
                loss.backward()
                opt.step()
                opt.zero_grad()
            task_acc = 100.*n_correct/n_step
            if t%10 == 0:
                logger.info("Iter: %d, task id: %d, train acc: %f", t, task_id, task_acc)
            weight_after = deepcopy(model.state_dict())
            update_vars.append(weight_after)
            model.load_state_dict(weight_before)

        new_weight_dict = {}
        for name in weight_before:
            weight_list = [tmp_weight_dict[name] for tmp_weight_dict in update_vars]
            weight_shape = list(weight_list[0].size())
            stack_shape = [len(weight_list)] + weight_shape
            stack_weight = torch.empty(stack_shape)
            for i in range(len(weight_list)):
                stack_weight[i,:] = weight_list[i] 
            new_weight_dict[name] = torch.mean(stack_weight, dim=0).cuda()
            new_weight_dict[name] = weight_before[name]+(new_weight_dict[name]-weight_before[name])/Inner_lr*Outer_lr
        model.load_state_dict(new_weight_dict)


    torch.save(model.state_dict(), output_model_file)

model.load_state_dict(torch.load(output_model_file))
logger.info("***** Running evaluation *****")
fsl_task_list = np.arange(fsl_num_tasks)
weight_before = deepcopy(model.state_dict())
fsl_epochs = 3
Total_acc = 0
opt = OPT.Adam(model.parameters(), lr=3e-4)

for task_id in fsl_task_list:
    model.train()
    (train_iter, dev_iter, test_iter) = fsl_ds_iters[task_id]
    train_iter.init_epoch()
    batch = next(iter(train_iter))
    for i in range(fsl_epochs):
        logits = model(batch.text)
        loss = criterion(logits.view(-1, num_labels), batch.label.data.view(-1))
        n_correct = (torch.max(logits, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_size = batch.batch_size
        train_acc = 100. * n_correct / n_size
        loss = criterion(logits.view(-1, num_labels), batch.label.data.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        logger.info("  Task id: %d, fsl epoch: %d, Acc: %f, loss: %f", task_id, i, train_acc, loss)

    model.eval()
    test_iter.init_epoch()
    n_correct = 0
    n_size = 0
    for test_batch_idx, test_batch in enumerate(test_iter):
        with torch.no_grad():
            logits = model(test_batch.text)
        loss = criterion(logits.view(-1, num_labels), test_batch.label.data.view(-1))
        n_correct += (torch.max(logits, 1)[1].view(test_batch.label.size()).data == test_batch.label.data).sum()
        n_size += test_batch.batch_size
    test_acc = 100.* n_correct/n_size
    logger.info("FSL test Number: %d, Accuracy: %f",n_size, test_acc)
    Total_acc += test_acc
    model.load_state_dict(weight_before)

print("Mean Accuracy is : ", float(Total_acc)/fsl_num_tasks)

        
