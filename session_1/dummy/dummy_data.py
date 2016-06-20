import h5py
import numpy as np

dummy = h5py.File('dummy_data.h5')
x_1 = np.ones(5)
dummy.create_dataset(name = 'x_1', dtype = x_1.dtype, shape = x_1.shape, data = x_1)

x_2 = np.arange(5)
dummy.create_dataset(name = 'x_2', dtype = x_2.dtype, shape = x_2.shape, data = x_2)

dummy.close()
