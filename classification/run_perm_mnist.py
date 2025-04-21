import numpy as np

# import cls_utils as cu
import reg_utils as cu
from copy import deepcopy
import torch
import gzip
import pickle
import sys

sys.path.extend(["alg/"])


class PermutedMnistGenerator:
    def __init__(self, max_iter=10):  # max_iter default 10 since 10 digits (10 tasks)
        path = "data/mnist.pkl.gz"
        f = gzip.open(path, "rb")
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        train_set, valid_set, test_set = u.load()
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))  # shape [60k, 784]
        self.Y_train = np.hstack((train_set[1], valid_set[1]))  # shape [60k]
        self.X_test = test_set[0]  # shape [10k, 784]
        self.Y_test = test_set[1]  # shape [10k]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            np.random.seed(self.cur_iter)
            perm_inds = np.arange(self.X_train.shape[1])
            np.random.shuffle(perm_inds)

            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[
                :, perm_inds
            ]  # use a single random permutation on pixels over entire trainset
            next_y_train = np.eye(10)[self.Y_train]

            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:, perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


hidden_size = [100, 100]
batch_size = 256
no_epochs = 10
single_head = True
num_tasks = 10

torch.manual_seed(1)
np.random.seed(1)

coreset_size = 0
data_gen = PermutedMnistGenerator(num_tasks)
vcl_result = cu.run_vcl(
    hidden_size,
    no_epochs,
    data_gen,
    cu.rand_from_batch,
    coreset_size,
    batch_size,
    single_head,
)
print(vcl_result)

torch.manual_seed(1)
np.random.seed(1)

coreset_size = 200


# uncertainty guided
data_gen = PermutedMnistGenerator(num_tasks)
unc_vcl_result = cu.run_vcl(
    hidden_size,
    no_epochs,
    data_gen,
    cu.k_center,
    coreset_size,
    batch_size,
    single_head,
    unc_coreset=True,
)
print(unc_vcl_result)

torch.manual_seed(1)
np.random.seed(1)

data_gen = PermutedMnistGenerator(num_tasks)
rand_vcl_result = cu.run_vcl(
    hidden_size,
    no_epochs,
    data_gen,
    cu.rand_from_batch,
    coreset_size,
    batch_size,
    single_head,
)
print(rand_vcl_result)


torch.manual_seed(1)
np.random.seed(1)

data_gen = PermutedMnistGenerator(num_tasks)
kcen_vcl_result = cu.run_vcl(
    hidden_size,
    no_epochs,
    data_gen,
    cu.k_center,
    coreset_size,
    batch_size,
    single_head,
)
print(kcen_vcl_result)


filename = "classification/results/permuted_reg_full_e10_t10.jpg"
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
unc_vcl_avg = np.nanmean(unc_vcl_result, 1)
cu.plot(filename, vcl_avg, rand_vcl_avg, kcen_vcl_avg, unc_vcl_avg)
