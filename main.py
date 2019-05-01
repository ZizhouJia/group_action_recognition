import model_utils.sovler as solver
import model_utils.runner as runner
import model_utils.optimizer as optimizer
import model

def generate_frames_dataset(batch_size,path):
    return None

def generate_feature_dataset(batch_size,path):
    return None,None,None


r=runner.runner()

tasks=[]

config=model_utils.config.config()

config["epochs"]=5
config["dataset_function"]=generate_dataset
config["dataset_function_params"]={"batch_size":160}
config["learning_rate_decay_iteration"]=4000000
config["dataset_function"]=generate_feature_dataset
config["dataset_function_params"]={"batch_size":160,"path":""}
config["model_class"]=[model.DbofModel.DbofModel]
config["model_params"]=[{},{}]
config["optimizer_function"]=optimizer.generate_optimizers
config["optimizer_params"]={"lrs":[,0.0002],"optimizer_type":"adam","weight_decay":1.0}}
config["task_name"]="DbofModel_baseline"
config["mem_use"]=[10000,10000,10000,10000]

test_task={
"solver":{"class":solver.vedio_classify_solver,"param":{}},
"config":config,
"mem_use":[10000,10000,10000,10000]
}

config=model_utils.config.config()
config["model_class"]=[model.inception_v3.inception_v3]
config["model_params"]=[{"pca_dir":None}]
config["summary_writer_open"]=False
config["dataset_function"]=generate_frames_dataset
config["dataset_function_params"]={"batch_size":1,"path":""}
config["save_path"]=None
config["pca_save_path"]="./pca_matrix/kuaishou"

extract_2048_features={
"solver":{"class":solver.feature_extractor_solver,"param":{}},
"config":config
mem_use:[10000]
}

config["model_params"]=[{"pca_dir":"./pca_matrix/kuaishou"}]
config["save_path"]="."
config["pca_save_path"]=None

extract_1024_features={
"solver":{"class":solver.feature_extractor_solver,"param":{}},
"config":config
mem_use:[10000]
}


tasks.append(extract_2048_features)
tasks.append(extract_1024_features)
tasks.append(test_task)
r.generate_tasks(tasks)
r.main_loop()
