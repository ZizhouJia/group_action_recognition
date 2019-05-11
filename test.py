import model_utils.dataset as Data
import time
import numpy as np
class debugger_dataset(Data.Dataset):
  def __getitem__(self,index):
      return np.zeros((100,100))

  def __len__(self):
      return 100


dataloader=Data.BufferDataLoader(debugger_dataset(),batch_size=1,shuffle=True,num_workers=128,buffer_size=300)

last_start=time.time()
for i in range(0,100):
    for step,data in enumerate(dataloader):
        start_time=time.time()
        print(str(i)+","+str(step))
        print(str(start_time-last_start))
        last_start=start_time

