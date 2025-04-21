import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from reg import Vanilla_NN, MFVI_NN

matplotlib.use("agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def rand_from_batch(x_coreset, y_coreset, x_train, y_train, coreset_size):
    idx = np.random.choice(x_train.shape[0], coreset_size, replace=False)
    x_coreset.append(x_train[idx])
    y_coreset.append(y_train[idx])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    return x_coreset, y_coreset, x_train, y_train


def update_distance(dists, x_train, current_id):
    current_dist = np.linalg.norm(x_train - x_train[current_id], axis=1)
    dists = np.minimum(current_dist, dists)
    return dists


def k_center(x_coreset, y_coreset, x_train, y_train, coreset_size):
    dists = np.full(x_train.shape[0], np.inf)
    current_id = 0
    dists = update_distance(dists, x_train, current_id)
    idx = [current_id]
    for i in range(1, coreset_size):
        current_id = np.argmax(dists)
        dists = update_distance(dists, x_train, current_id)
        idx.append(current_id)
    x_coreset.append(x_train[idx])
    y_coreset.append(y_train[idx])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    return x_coreset, y_coreset, x_train, y_train


def uncertainty_coreset(
    x_coreset, y_coreset, x_train, y_train, coreset_size, model, task_id
):
    if isinstance(model, Vanilla_NN):
        return rand_from_batch(x_coreset, y_coreset, x_train, y_train, coreset_size)
    with torch.no_grad():
        preds = model(
            torch.FloatTensor(x_train).to(device),
            task_id - 1,
            model.no_train_samples,
        )
    pred_means = preds.mean(dim=0)
    pred_entropy = torch.distributions.Categorical(logits=pred_means).entropy()

    multiplier = len(pred_entropy) // coreset_size
    _, idx = torch.topk(pred_entropy, coreset_size * multiplier)
    idx = idx.cpu().numpy()
    idx = idx[::multiplier]
    np.random.shuffle(idx)

    x_coreset.append(x_train[idx])
    y_coreset.append(y_train[idx])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    return x_coreset, y_coreset, x_train, y_train


def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    return merged_x, merged_y


def get_scores(
    model,
    x_testsets,
    y_testsets,
    x_coresets,
    y_coresets,
    hidden_size,
    no_epochs,
    single_head,
    batch_size=None,
):
    mf_weights, mf_variances = model.get_weights()
    rmse = []

    if single_head:
        if len(x_coresets) > 0:
            x_train, y_train = merge_coresets(x_coresets, y_coresets)
            x_train = torch.from_numpy(x_train).to(device)
            y_train = torch.from_numpy(y_train).to(device)
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            final_model = MFVI_NN(
                x_train.shape[1],
                hidden_size,
                y_train.shape[1],
                x_train.shape[0],
                prev_means=mf_weights,
                prev_log_vars=mf_variances,
            )
            final_model.train_model(x_train, y_train, 0, no_epochs, bsize)
        else:
            final_model = model

    for i in range(len(x_testsets)):
        if not single_head:
            if len(x_coresets) > 0:
                x_train, y_train = x_coresets[i], y_coresets[i]
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = MFVI_NN(
                    x_train.shape[1],
                    hidden_size,
                    y_train.shape[1],
                    x_train.shape[0],
                    prev_means=mf_weights,
                    prev_log_vars=mf_variances,
                )
                x_train = torch.from_numpy(x_train).to(device)
                y_train = torch.from_numpy(y_train).to(device)
                final_model.train_model(x_train, y_train, i, no_epochs, bsize)
            else:
                final_model = model

        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]

        pred = final_model.prediction_prob(x_test, head)
        pred = np.mean(pred, axis=0)
        curr_rmse = np.sqrt(np.mean((pred - y_test) ** 2))
        rmse.append(curr_rmse)

    return np.array(rmse)


def concatenate_results(score, all_score):
    if all_score.size == 0:
        return np.array([score])
    return np.vstack((all_score, np.array(score)))


def plot(filename, vcl, rand_vcl, kcen_vcl, unc_vcl):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig = plt.figure(figsize=(7, 3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl)) + 1, vcl, label="VCL", marker="o")
    plt.plot(
        np.arange(len(rand_vcl)) + 1, rand_vcl, label="VCL + Random Coreset", marker="o"
    )

    plt.plot(
        np.arange(len(kcen_vcl)) + 1,
        kcen_vcl,
        label="VCL + K-center Coreset",
        marker="o",
    )

    plt.plot(
        np.arange(len(unc_vcl)) + 1,
        unc_vcl,
        label="VCL + Entropy-guided Coreset",
        marker="o",
    )

    ax.set_xticks(range(1, len(vcl) + 1))
    ax.set_ylabel("RMSE")
    ax.set_xlabel(r"\# tasks")
    ax.legend()

    fig.savefig(filename, bbox_inches="tight")
    plt.close()


def run_vcl(
    hidden_size,
    no_epochs,
    data_gen,
    coreset_method,
    coreset_size=0,
    batch_size=None,
    single_head=True,
    unc_coreset=False,
):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])
    mf_weights = None
    mf_variances = None

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        if task_id == 0:
            ml_model = Vanilla_NN(
                in_dim,
                hidden_size,
                out_dim,
                x_train.shape[0],
                prev_weights=None,
            ).to(device)
            ml_model.train_model(
                x_train,
                y_train,
                task_id,
                num_epochs=no_epochs,
                display_epoch=max(1, no_epochs // 10),
                batch_size=bsize,
            )
            mf_weights = ml_model.get_weights()
            if unc_coreset and coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = uncertainty_coreset(
                    x_coresets,
                    y_coresets,
                    x_train.cpu().numpy(),
                    y_train.cpu().numpy(),
                    coreset_size,
                    ml_model,
                    head,
                )

        if coreset_size > 0:
            if unc_coreset:
                if task_id != 0:
                    x_coresets, y_coresets, x_train, y_train = uncertainty_coreset(
                        x_coresets,
                        y_coresets,
                        x_train.cpu().numpy(),
                        y_train.cpu().numpy(),
                        coreset_size,
                        mf_model,
                        head,
                    )
            else:
                x_coresets, y_coresets, x_train, y_train = coreset_method(
                    x_coresets,
                    y_coresets,
                    x_train.cpu().numpy(),
                    y_train.cpu().numpy(),
                    coreset_size,
                )
            x_train = torch.FloatTensor(x_train).to(device)
            y_train = torch.FloatTensor(y_train).to(device)

        mf_model = MFVI_NN(
            in_dim,
            hidden_size,
            out_dim,
            x_train.shape[0],
            prev_means=mf_weights,
            prev_log_vars=mf_variances,
        ).to(device)

        mf_model.train_model(
            x_train,
            y_train,
            head,
            num_epochs=no_epochs,
            display_epoch=max(1, no_epochs // 10),
            batch_size=bsize,
        )

        mf_weights, mf_variances = mf_model.get_weights()

        acc = get_scores(
            mf_model,
            x_testsets,
            y_testsets,
            x_coresets,
            y_coresets,
            hidden_size,
            no_epochs * 2,
            single_head,
            batch_size,
        )

        acc_padded = np.full(data_gen.max_iter, np.nan)
        acc_padded[: len(acc)] = acc
        all_acc = concatenate_results(acc_padded, all_acc)

    return all_acc
