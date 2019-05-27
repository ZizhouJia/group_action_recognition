import model_utils
import model_utils.solver as solver
import model_utils.runner as runner
import model_utils.optimizer as optimizer
import model
from torchvision import transforms
import video_transforms 
import dataset
import torch.utils.data as Data
import h5py
import numpy as np
import copy
import model_utils.dataset as dl
import multiprocessing
import torch

video_t=transforms.Compose([
                video_transforms.center_crop(),
                video_transforms.resize(299),
                video_transforms.to_tensor(),
                video_transforms.normalize()
            ])

def generate_frames_dataset(batch_size=1):
    ks=dataset.ks_dataset.KsDataset(transform=video_t)
    dataloader=dl.BufferDataLoader(ks,batch_size=batch_size,shuffle=False,num_workers=32,buffer_size=100)
   # dataloader=Data.DataLoader(ks,batch_size=1,shuffle=False)
    return dataloader

def generate_feature_dataset(batch_size,path):
    f=h5py.File(path,"r")
    features=f["feature"]
    id=f["image_id"]
    label=f["label"]
    indexs=np.arrange(features.shape[0])
    np.random.seed(666)
    np.random.shuffle(indexs)
    features[:,:,:,:]=features[index]
    id[:]=features[index]
    label[:]=label[index]
    train_set=int(features.shape[0]*0.8)
    val_set=int(features.shape[0]*0.1)
    test_set=features.shape[0]-train_set-val_set
    train_loader=Data.DataLoader(dataset.feature_dataset.feature_dataset(features[:train_set],label[:train_set]),batch_size=batch_size,shuffle=True,num_workers=8)
    test_loader=Data.DataLoader(dataset.feature_dataset.feature_dataset(features[val_set:],label[val_set:]),batch_size=1,shuffle=False,num_workers=8)
    val_loader=Data.DataLoader(dataset.features_dataset.feature_dataset(features[train_set:val_set],label[train_set:val_set]),batch_size=batch_size,shuffle=False,num_workers=8)
    return train_loader,val_loader,test_loader

def generate_yt8m_dataset(batch_size):
    train_dataset=dataset.yt8m_dataset.yt8mDataset(data_type="train")
    valid_dataset=dataset.yt8m_dataset.yt8mDataset(data_type="valid")
    test_dataset=dataset.yt8m_dataset.yt8mDataset(data_type="test")
    train_loader=Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    valid_loader=Data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    test_loader=Data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    return train_loader,valid_loader,test_loader

def generate_kuaishou_dataset(batch_size):
    train_dataset=dataset.feature_dataset.feature_dataset(data_type="train")
    valid_dataset=dataset.feature_dataset.feature_dataset(data_type="valid")
    test_dataset=dataset.feature_dataset.feature_dataset(data_type="test")
    train_loader=Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    valid_loader=Data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    test_loader=Data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    return train_loader,valid_loader,test_loader

    
def generate_optimizer(models,base_lr,fc_lr,weight_decay):
    fc_params=list(map(id,models[0].module.c_layer.parameters()))
    base_params=filter(lambda p:id(p) not in fc_params,models[0].module.parameters())
    optimizer=torch.optim.Adam([
            {"params":base_params,"lr":base_lr},
            {"params":models[0].module.c_layer.parameters(),"lr":fc_lr}
            ],lr=base_lr,weight_decay=weight_decay)
    return [optimizer]

r=runner.runner()

tasks=[]

config=model_utils.config.common_solver_config()

config["epochs"]=200
config["dataset_function"]=generate_kuaishou_dataset
config["learning_rate_decay_iteration"]=4000000
config["dataset_function_params"]={"batch_size":32}
config["model_class"]=[model.DbofModel.DbofModel]
config["model_params"]=[{"vocab_size":3,"pretrain":True}]
config["optimizer_function"]=generate_optimizer
config["optimizer_params"]={"base_lr":0.000000,"fc_lr":0.0002,"weight_decay":0.0}
config["task_name"]="NetVLADModel_baseline"
config["mem_use"]=[10000,10000,10000,10000]
config["grad_plenty"]=1.0
config["distilling_mode"]=False
test_task={
"solver":{"class":solver.vedio_classify_solver,"params":{}},
"config":config,
}

config=model_utils.config.config()
config["task_name"]="extract_2048_features"
config["model_class"]=[model.inception_v3.inception_v3]
config["model_params"]=[{"pca_dir":None}]
config["mem_use"]=[10000,10000,10000,10000,10000,10000,10000,10000]
config["summary_writer_open"]=False
config["dataset_function"]=generate_frames_dataset
config["dataset_function_params"]={"batch_size":1}
config["save_path_2048"]="kuaishou_feature2048.h5"
config["save_path_1024"]="kuaishou_feature.h5"
config["pca_save_path"]="./pca_matrix/kuaishou"
config["model_path"]=None
config["split_times"]=1

extract_2048_features={
"solver":{"class":solver.feature_extractor_solver,"params":{}},
"config":config
}

config=copy.deepcopy(config)

config["task_name"]="extract_1024_features"
config["model_params"]=[{"pca_dir":"./pca_matrix/kuaishou"}]
config["save_path"]="kuaishou_feature.h5"
config["pca_save_path"]=None

extract_1024_features={
"solver":{"class":solver.feature_extractor_solver,"params":{}},
"config":config
}


#tasks.append(extract_2048_features)
#tasks.append(extract_1024_features)
tasks.append(test_task)
r.generate_tasks(tasks)
r.main_loop()
