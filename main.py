import model_utils.sovler as solver
import model_utils.runner as runner
import model_utils.optimizer as optimizer
import

def generate_dataset(batch_size):
    return None,None,None


r=runner.runner()

tasks=[]

config=model_utils.config.config()

config["epochs"]=5
config["dataset_function"]=generate_dataset
config["dataset_function_params"]={"batch_size":160}
config["learning_rate_decay_iteration"]=4000000
config["dataset_function"]=generate_dataset
config["dataset_function_params"]={"batch_size":160}

test_task={
"task_name":"test_baseline",
"solver":{"class":solver.vedio_classify_solver,"param":{}},
"models":{"class":[],"param":[]},
"optimizers":{"function":optimizer.generate_optimizers,
"params":{"lrs":[0.0002],"optimizer_type":"sgd","weight_decay":1.0}},
"config":config
"mem_use":[10000,10000,10000,10000]
}

tasks.append(test_task)

r.generate_tasks(tasks)
r.main_loop()
