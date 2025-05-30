import numpy as np
import reg_utils as cu

# import cls_utils as cu
import torch
import gzip
import pickle
import sys

sys.path.extend(["alg/"])


class SplitMnistGenerator:
    def __init__(self):
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

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            train_0_id = np.where(self.Y_train == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.Y_train == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack(
                (self.X_train[train_0_id], self.X_train[train_1_id])
            )

            next_y_train = np.vstack(
                (np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1)))
            )
            next_y_train = np.hstack((next_y_train, 1 - next_y_train))

            test_0_id = np.where(self.Y_test == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.Y_test == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack(
                (np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1)))
            )
            next_y_test = np.hstack((next_y_test, 1 - next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


hidden_size = [256, 256]
batch_size = None
no_epochs = 120
single_head = False


torch.manual_seed(2)
np.random.seed(2)

coreset_size = 0
data_gen = SplitMnistGenerator()
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

coreset_size = 40

# uncertainty guided
data_gen = SplitMnistGenerator()
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

data_gen = SplitMnistGenerator()
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

data_gen = SplitMnistGenerator()
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

filename = "classification/results/split_reg_final_e120.jpg"
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
unc_vcl_avg = np.nanmean(unc_vcl_result, 1)
cu.plot(filename, vcl_avg, rand_vcl_avg, kcen_vcl_avg, unc_vcl_avg)
