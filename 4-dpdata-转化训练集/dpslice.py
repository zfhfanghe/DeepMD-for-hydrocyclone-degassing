import dpdata
import numpy as np

# read energy force from aimd cp2k.out
data=dpdata.LabeledSystem('.',cp2k_output_name="cp2k.out",fmt='cp2kdata/md')
print('# the data contains %d frames' % len(data))

index_validation = np.random.choice(100,size=10,replace=False)
# random choose 10 index for validation_data

index_training = list(set(range(100))-set(index_validation))
# other indexes are training_data

data_training = data.sub_system(index_training)
data_validation = data.sub_system(index_validation)

data_training.to_deepmd_npy('training_data')
# all training data put into directory:"training_data"
data_validation.to_deepmd_npy('validation_data')
# all validation data put into directory:"validation_data"
print(len(data_training))
print(len(data_validation))

