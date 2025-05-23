import torch
from torch.utils.data import DataLoader
from utils.generic import freeze_model
from copy import deepcopy
from utils.viz import plt_test_confmat, plt_confmats, plt_test_confmat_task
from parse import args
import pickle as pkl
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.transforms import expansion_transforms, to_tensor_and_normalize, expansion_transforms_tiny, \
    expansion_transforms_synbols, expansion_transforms_core50
import os
from Explainability import lime_tool, visualize_meaningfull_layers, occlusion_sensitivity, maximal_response


def get_stat_exp(y, y_hats, exp_idx, task_id, task_predictions):
    """ Compute accuracy and task accuracy for each experience."""
    conf_mat = torch.zeros((exp_idx + 1, exp_idx + 1))

    for i in range(exp_idx + 1):
        ybuff = y[task_id == i]
        y_hats_buff = y_hats[task_id == i]
        acc = (ybuff == y_hats_buff).sum() / y_hats_buff.shape[0]

        for j in range(exp_idx + 1):
            conf_mat[i, j] = ((task_id == i) & (task_predictions == j)).sum() / (task_id == i).sum()

        print(
            f"EXP:{i}, acc:{acc:.3f}, task:{conf_mat[i, i]:.3f}, distrib:{[round(conf_mat[i, j].item(), 3) for j in range(exp_idx + 1)]}")


def entropy(vec):
    return -torch.sum(vec * torch.log(vec + 1e-7), dim=1)


def test_onering(strategy, test_set, plot=False):
    strategy.model.eval()

    # test_set=torch.shuffle
    dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

    confusion_mat = torch.zeros((args.n_classes, args.n_classes))
    confusion_mat_taw = torch.zeros((args.n_classes, args.n_classes))

    y_hats = []
    y_taw = []
    ys = []
    task_predictions = []
    task_ids = []

    # per diversi dataset quello che è usare diversi task id del dataset e vedere le performance della rete con l'inserimento di nuovi task id diversi
    for i, (x, y, task_id) in enumerate(dataloader):

        frag_preds = torch.zeros((y.shape[0], args.n_classes + args.n_experiences))  # [n_batch, classes+unk_classes]

        entropy_frag = torch.zeros((y.shape[0], args.n_experiences))  # [n_batch , n_experiences]

        # iterate through the experiences
        for j in range(strategy.experience_idx + 1):
            # create a temporary model copy
            model = freeze_model(deepcopy(strategy.model))

            strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
            model.load_bn_params(j)
            model.exp_idx = j

            pred = model(x.to(args.device))
            # print(pred.shape)
            # input("stop")

            if not args.dataset == 'CORE50':  # if NOT

                # take the class UNK for that task
                last = pred[:, args.n_classes + j]
                last = torch.unsqueeze(last, dim=1)
                pred = pred[:, j * args.classes_per_exp:(j + 1) * args.classes_per_exp]
                pred = torch.cat([pred, last], dim=1)

            softmax = torch.softmax(pred, dim=1)

            # insert the respective probability
            frag_preds[:, j * args.classes_per_exp:(j + 1) * args.classes_per_exp] = softmax[:, :-1]
            frag_preds[:, args.n_classes + j] = softmax[:, -1]

            # calculate the entropies
            entropia = entropy(softmax)
            entropy_frag[:, j] = entropia

        # nello stesso for, quindi questi sono elementi dello stesso batch

        frag_preds_max = torch.argsort(frag_preds[:, :args.n_classes], dim=1, descending=True)[:, 0]
        task_predicted = torch.floor(frag_preds_max / 10)

        # for the task awareness
        task_id_int = task_id.to(torch.int32)  # Convert to Python int
        n_class_min = task_id_int * args.classes_per_exp
        n_class_max = (task_id_int + 1) * args.classes_per_exp

        batch_size = frag_preds.shape[0]
        frag_preds_task = []

        for i in range(batch_size):
            frag_preds_task.append(frag_preds[i, n_class_min[i]:n_class_max[i]])

        frag_preds_task = torch.stack(frag_preds_task)
        frag_preds_taskmax = frag_preds_task.argmax(dim=1)

        y_hats.append(frag_preds_max)
        y_taw.append(frag_preds_taskmax)

        task_predictions.append(task_predicted)

        task_ids.append(task_id)  # real task id for each sample
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')
    y_taw = torch.cat(y_taw, dim=0).to('cpu')
    task_predictions = torch.cat(task_predictions, dim=0).to('cpu')
    task_ids = torch.cat(task_ids, dim=0).to('cpu')

    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1
        confusion_mat_taw[y[i], y_taw[i]] += 1

    # task confusion matrix and forgetting mat
    for j in range(strategy.experience_idx + 1):
        i = strategy.experience_idx
        acc_conf_mat_task = confusion_mat[j * args.classes_per_exp:(j + 1) * args.classes_per_exp,
                            j * args.classes_per_exp:(j + 1) * args.classes_per_exp].diag().sum() / confusion_mat[
                                                                                                    i * args.classes_per_exp:(
                                                                                                                                     i + 1) * args.classes_per_exp,
                                                                                                    :].sum()
        strategy.confusion_mat_task[i][j] = acc_conf_mat_task
        strategy.forgetting_mat[i][j] = strategy.confusion_mat_task[:, j].max() - acc_conf_mat_task

    # compute accuracy
    accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
    accuracy_taw = confusion_mat_taw.diag().sum() / confusion_mat_taw.sum()

    task_accuracy = (task_predictions == task_ids).sum() / y_hats.shape[0]

    print(f"Test Accuracy: {accuracy:.4f},Test Accuracy taw: {accuracy_taw:.4f}, Task accuracy: {task_accuracy:.4f}")

    get_stat_exp(y, y_hats, strategy.experience_idx, task_ids, task_predictions)

    if plot:
        plt_test_confmat(args.run_name, confusion_mat, strategy.experience_idx)
        if strategy.experience_idx == args.n_experiences - 1:
            plt_test_confmat_task(args.run_name, strategy.confusion_mat_task)
            torch.save(strategy.forgetting_mat, f'./logs/{args.run_name}/forgetting_mat.pt')
            torch.save(strategy.confusion_mat_task, f'./logs/{args.run_name}/confusion_mat_task.pt')

    if strategy.experience_idx == args.n_experiences - 1:
        print(" SAVE THE LOGS")
        res = {}
        res['y'] = y.cpu().numpy()
        res['y_hats'] = y_hats.cpu().numpy()
        res['frag_preds'] = frag_preds.cpu().numpy()
        res['entropy_frag'] = entropy_frag.cpu().numpy()
        res['y_taw'] = y_taw.cpu().numpy()

        # write to file
        with open(f'./logs/{args.run_name}/res.pkl', 'wb') as f:
            pkl.dump(res, f)

    return accuracy, task_accuracy, accuracy_taw


def apply_transforms_and_permute(batch, num_permutations):
    """
    Apply default transforms and generate permutations of each image in a batch.

    Args:
        batch (torch.Tensor): Tensor of shape (batch_size, channels, height, width)
        num_permutations (int): Number of permutations per image

    Returns:
        torch.Tensor: Tensor of shape (batch_size * num_permutations, channels, height, width)
    """
    batch_size, channels, height, width = batch.shape
    permuted_images = []

    if args.dataset == "TinyImageNet":
        trans = transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        )
    elif args.dataset == "CIFAR100":
        trans = transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        )
    elif args.dataset == "Synbols":
        trans = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        )

    elif args.dataset == "CORE50_CI":
        trans = transforms.Normalize(
            (0.5998523831367493, 0.5575963854789734, 0.5395311713218689),
            (0.20457075536251068, 0.2166813313961029, 0.22945666313171387)
        )

    for img in batch:
        # keep the original image
        permuted_images.append(trans(img))

        for _ in range(num_permutations - 1):
            if args.dataset == "TinyImageNet":
                transformed_img = expansion_transforms_tiny(img)
            elif args.dataset == "CIFAR100":
                transformed_img = expansion_transforms(img)  # Apply transformations to tensor
            elif args.dataset == "Synbols":
                transformed_img = expansion_transforms_synbols(img)
            elif args.dataset == "CORE50_CI":
                transformed_img = expansion_transforms_core50(img)
            permuted_images.append(transformed_img)

    return torch.stack(permuted_images, dim=0)


def process_permuted_images(batch, model, num_permutations, exp_idx):
    """
    Computes network output for all permuted images and aggregates per original image.

    Args:
        batch (torch.Tensor): Original batch of images (batch_size, channels, height, width)
        model (nn.Module): A neural network model
        num_permutations (int): Number of permutations per image

    Returns:
        torch.Tensor: Aggregated output per original image (batch_size, output_dim)
    """
    batch_size = batch.shape[0]

    # Generate permuted images
    permuted_batch = apply_transforms_and_permute(batch, num_permutations)

    # Compute outputs for permuted images
    permuted_outputs = model(permuted_batch.to(args.device))  # Shape: (batch_size * num_permutations, output_dim)

    # Aggregate results per original image (e.g., averaging across permutations)

    return permuted_outputs


def position(probs, entropies, temperature, th=0.20):
    """
    This function creates an histogram that
    keeps  track of the task that have a lower entropy thant the th value
    for each image"""
    probs_th = probs.clone()

    # print(probs.shape)#[ n_frag,n_batch ,n_class] [task, immagine, classi del task]
    # print(entropies.shape)#[n_frag, batch]
    # input("observe the shapes")

    masked_entropies = entropies < th

    dict_entropy = {}
    for index in range(entropies.shape[1]):  # fix the image
        counter = 0
        # probs_im=probs[:,index,:]
        for task in range(entropies.shape[0]):

            if masked_entropies[:, index].sum() == 0:
                # print(masked_entropies[:, index])
                continue  # skip if the image has all the values strictly major than the th
            else:
                if masked_entropies[task, index]:  # True if below the threshold
                    counter += 1
                    # continue #keep the values
                else:  # remove the probs if above the threshold
                    probs_th[task, index, :] = 0

        if counter not in dict_entropy.keys():
            dict_entropy[counter] = 1
        else:
            dict_entropy[counter] += 1

    # save the histogram

    plt.figure(figsize=(8, 5))
    plt.bar(list(dict_entropy.keys()), list(dict_entropy.values()), color='blue', alpha=0.7)
    plt.xlabel("Number of Tasks Below Threshold")
    plt.ylabel("Frequency")
    plt.title("Histogram of Task Counts Below Entropy Threshold")
    plt.xticks(sorted(dict_entropy.keys()))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Define the folder path

    save_dir = f"/davinci-1/home/micherusso/PycharmProjects/MIND_real/plots/run_{args.seed}"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plt.savefig(f"{save_dir}/entropy_histogram_{temperature}.png")
    # plt.show()
    # Save plot)
    # print(dict_entropy)
    return probs_th


def show_channel_images(channel_image, cols=8, title="Maximally Activating Patterns", save=False,
                        save_path="activations.png"):
    num_images = len(channel_image)
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(2 * cols, 2 * rows))

    for i, img in enumerate(channel_image):
        plt.subplot(rows, cols, i + 1)
        if isinstance(img, torch.Tensor):
            img = img.squeeze(dim=0).detach().cpu().permute(1, 2, 0).numpy()
        img -= img.min()
        img /= (img.max() + 1e-8)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Filter {i}', fontsize=8)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def test(strategy, test_set, temperature, n_perturb, plot=True):
    strategy.model.eval()
    dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

    confusion_mat = torch.zeros((args.n_classes, args.n_classes))
    confusion_mat_th = torch.zeros((args.n_classes, args.n_classes))
    confusion_mat_e = torch.zeros((args.n_classes, args.n_classes))
    confusion_mat_taw = torch.zeros((args.n_classes, args.n_classes))

    y_hats = []
    y_hats_e = []
    y_taw = []
    ys = []
    task_predictions = []
    task_ids = []
    y_hats_th = []
    check_entropy_th = False
    explainability = args.explainability

    for i, (x, y, task_id) in enumerate(dataloader):

        frag_preds = []
        entropy_frag = []
        batch_size = y.shape[0]
        for j in range(strategy.experience_idx + 1):
            # create a temporary model copy
            model = freeze_model(deepcopy(strategy.model))

            strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)

            model.load_bn_params(j)
            model.exp_idx = j

            if explainability and strategy.experience_idx == 9:  # save the results only for the last iteration

                # lime_tool(x,model)

                occlusion_sensitivity(model, x, y, task_ids=strategy.experience_idx)
                channel_image = []

                for layer_idx in range(2):
                    for channel in range(16):
                        print(f"channel used {channel} and layer used {layer_idx}")

                        channel_image.append(maximal_response(model, channel, layer_idx))

                show_channel_images(channel_image, save=True)

            # new Permutations
            if n_perturb != 0:

                pred = process_permuted_images(x, model, n_perturb, j)
            else:

                pred = model(x.to(args.device))

            if not args.dataset == 'CORE50':
                last = pred[:, args.n_classes + j].unsqueeze(1)

                pred = pred[:, j * args.classes_per_exp:(j + 1) * args.classes_per_exp]

                pred = torch.cat([pred, last], dim=1)

            # mute the unk_classes
            probs = torch.softmax(pred / temperature, dim=1)
            # print(probs.shape)

            if n_perturb != 0:
                probs = probs.view(batch_size, n_perturb, args.classes_per_exp + 1)  # modificare l'ultima

                probs = probs.mean(dim=1)  # Shape: (batch_size, output_dim)

            # print(probs.size)
            # input("check")
            frag_preds.append(probs[:, :-1])
            entropy_frag.append(entropy(torch.softmax(pred / temperature, dim=1)))

        # on_shell_confidence, elsewhere_confidence = confidence(frag_preds, task_id)
        # print(f"on_shell confidence:{[round(c.item(), 2) for c in on_shell_confidence]}\nelsewhere confidence:{[round(c.item(), 2) for c in elsewhere_confidence]}")

        frag_preds = torch.stack(frag_preds)  # [n_frag, bsize, n_classes]
        entropy_frag = torch.stack(entropy_frag)  # [bsize, n_frag]

        batch_size = frag_preds.shape[1]

        ### select across the top 2 of likelihood the head  with the lowest entropy
        # buff -> batch_size  x 2, 0-99 val
        # frag_preds_th=torch.zeros(frag_preds.shape).to("cpu")

        if check_entropy_th:
            print(" TH USED FOR THE ENTROPIES")
            frag_preds_th = position(frag_preds, entropy_frag, temperature)

        # max among the classes,           sort the max classes and take the two most likely
        buff = frag_preds.max(dim=-1)[0].argsort(dim=0)[-2:]  # [2, bsize]
        # buff_th = frag_preds_th.max(dim=-1)[0].argsort(dim=0)[-2:].to("cpu")  # [2, bsize]

        # buff_entropy ->  2 x batch_size, entropy values
        indices = torch.arange(batch_size)
        if buff.shape[0] == 1:
            buff_entropy = entropy_frag[buff[0, :], indices].unsqueeze(0)
        else:
            a = entropy_frag[buff[0, :], indices]
            b = entropy_frag[buff[1, :], indices]
            buff_entropy = torch.stack((a, b))  # [2, bsize]

        # index of min entropy ->  1 x batch_size, val 0-1
        index_min = buff_entropy.argsort(dim=0)[0, :]

        index_class = buff[index_min, indices]

        task_predictions.append(buff[-1])
        if args.dataset == 'CORE50':
            y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1))
            # y_hats_th.append(frag_preds[buff_th[-1], indices].argmax(dim=1))
            y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1))
            y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1))

        else:
            y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1) + args.classes_per_exp * index_class)
            y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1) + args.classes_per_exp * buff[-1])
            # y_hats_th.append(frag_preds_th[buff[-1], indices].argmax(dim=1) + args.classes_per_exp * buff_th[-1])
            y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1) + (
                    args.classes_per_exp * task_id.to(args.cuda)).to(torch.int32))

        task_ids.append(task_id)
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    # y_hats_th = torch.cat(y_hats_th, dim=0).to('cpu')
    y_hats_e = torch.cat(y_hats_e, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')
    y_taw = torch.cat(y_taw, dim=0).to('cpu')
    task_predictions = torch.cat(task_predictions, dim=0).to('cpu')
    task_ids = torch.cat(task_ids, dim=0).to('cpu')

    # assign +1 to the confusion matrix for each prediction that matches the label
    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1
        # confusion_mat_th[y[i], y_hats_th[i]] += 1
        confusion_mat_e[y[i], y_hats_e[i]] += 1
        confusion_mat_taw[y[i], y_taw[i]] += 1

    # task confusion matrix and forgetting mat
    for j in range(strategy.experience_idx + 1):
        i = strategy.experience_idx
        acc_conf_mat_task = confusion_mat[j * args.classes_per_exp:(j + 1) * args.classes_per_exp,
                            j * args.classes_per_exp:(j + 1) * args.classes_per_exp].diag().sum() / confusion_mat[
                                                                                                    i * args.classes_per_exp:(
                                                                                                                                     i + 1) * args.classes_per_exp,
                                                                                                    :].sum()
        strategy.confusion_mat_task[i][j] = acc_conf_mat_task
        strategy.forgetting_mat[i][j] = strategy.confusion_mat_task[:, j].max() - acc_conf_mat_task

    # compute accuracy
    accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
    accuracy_th = confusion_mat_th.diag().sum() / confusion_mat_th.sum()
    accuracy_e = confusion_mat_e.diag().sum() / confusion_mat_e.sum()
    accuracy_taw = confusion_mat_taw.diag().sum() / confusion_mat_taw.sum()

    task_accuracy = (task_predictions == task_ids).sum() / y_hats.shape[0]
    print(
        f"Test Accuracy: {accuracy:.4f},Test Accuracy with entropy: {accuracy_e:.4f},Test Accuracy taw: {accuracy_taw:.4f}, Task accuracy: {task_accuracy:.4f}")
    if check_entropy_th:
        get_stat_exp(y, y_hats_th, strategy.experience_idx, task_ids, task_predictions)

    get_stat_exp(y, y_hats, strategy.experience_idx, task_ids, task_predictions)

    if plot:
        plt_test_confmat(args.run_name, confusion_mat, strategy.experience_idx)
        if strategy.experience_idx == args.n_experiences - 1:
            plt_test_confmat_task(args.run_name, strategy.confusion_mat_task)
            torch.save(strategy.forgetting_mat, f'./logs/{args.run_name}/forgetting_mat.pt')
            torch.save(strategy.confusion_mat_task, f'./logs/{args.run_name}/confusion_mat_task.pt')

    if strategy.experience_idx == args.n_experiences - 1:
        res = {}
        res['y'] = y.cpu().numpy()
        res['y_hats'] = y_hats.cpu().numpy()
        res['y_hats_e'] = y_hats_e.cpu().numpy()
        res['frag_preds'] = frag_preds.cpu().numpy()
        res['entropy_frag'] = entropy_frag.cpu().numpy()
        res['y_taw'] = y_taw.cpu().numpy()

        # write to file
        with open(f'./logs/{args.run_name}/res.pkl', 'wb') as f:
            pkl.dump(res, f)

    return accuracy, task_accuracy, accuracy_e, accuracy_taw


#################### MIND TESTS ####################
def test_single_exp(pruner, tested_model, loader, exp_idx, distillation):
    confusion_mat = torch.zeros((args.n_classes, args.n_classes))
    y_hats = []
    ys = []

    for i, (x, y, _) in enumerate(loader):
        model = freeze_model(deepcopy(tested_model))
        preds = torch.softmax(model(x.to(args.device)), dim=1)
        preds = preds[:, :args.n_classes]  # remove extra classes

        y_hats.append(preds.argmax(dim=1))
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')

    # assign +1 to the confusion matrix for each prediction that matches the label
    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1

    return confusion_mat


def test_single_exp_onering(pruner, tested_model, loader, exp_idx, distillation):
    confusion_mat = torch.zeros((args.n_classes + 10, args.n_classes + 10))
    y_hats = []
    ys = []
    for i, (x, y, _) in enumerate(loader):
        model = freeze_model(deepcopy(tested_model))
        # remove the real class
        preds = torch.softmax(model(x.to(args.device)), dim=1)
        y = torch.full(y.shape, args.n_classes + exp_idx)
        # create second peak

        y_hats.append(preds.argsort(dim=1, descending=True)[:, 1])
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')

    # assign +1 to the confusion matrix for each prediction that matches the label
    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1

    return confusion_mat


def test_robustness_OOD(strategy, loader, exp_idx, temperature, distillation=True, sanity_check=False, ):
    dataloader = DataLoader(loader, batch_size=100, shuffle=False, num_workers=8)
    confusion_mat = torch.zeros((args.classes_per_exp + 1, args.classes_per_exp + 1), dtype=torch.int32)
    y_hats = []
    ys = []
    entropy_vector = []
    entropy_vector_unk = []
    entropy_total = []

    for i, (x, y, task_id) in enumerate(dataloader):

        # print(task_id)
        # load the model label
        model = freeze_model(deepcopy(strategy.model))  ####
        if not distillation:
            model = freeze_model(deepcopy(strategy.fresh_model))  ####
        if distillation:
            strategy.pruner.set_gating_masks(model, exp_idx, weight_sharing=args.weight_sharing, distillation=True)
            model.load_bn_params(exp_idx)
            model.exp_idx = exp_idx

        preds = model(x.to(args.device))

        last = preds[:, args.n_classes + exp_idx]
        if distillation and sanity_check:
            print(last)

        last = torch.unsqueeze(last, dim=1)
        preds = preds[:, exp_idx * args.classes_per_exp:(exp_idx + 1) * args.classes_per_exp]
        preds = torch.cat([preds, last], dim=1)

        preds = torch.softmax(preds / temperature, dim=1)

        entropia = entropy(preds)

        classes_indexes_start, classes_indexes_end = exp_idx * args.classes_per_exp, (
                    exp_idx + 1) * args.classes_per_exp

        for index, label in enumerate(y):
            if label < classes_indexes_start or label >= classes_indexes_end:
                # assign to UNK class
                y[index] = 10
                entropy_vector_unk.append(entropia[index].cpu().numpy())
            else:
                # classi del task
                y[index] = y[index] - classes_indexes_start
                entropy_vector.append(entropia[index].cpu().numpy())

                # input("guarda bene se è giusto")

        # print(preds.shape)

        # input("vedere cosa printi in output")

        y_hats.append(preds.argmax(dim=1))
        # print(preds.argmax(dim=1))
        # input("guarda qua")
        ys.append(y)
        entropy_total.append(entropia)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')

    y = torch.cat(ys, dim=0).to('cpu')

    entropy_total = torch.cat(entropy_total, dim=0).to("cpu")

    # assign +1 to the confusion matrix for each prediction that matches the label
    confusion_mat_m = torch.zeros((args.classes_per_exp + 1, args.classes_per_exp + 1), dtype=torch.int32)
    confusion_mat_m_unk = torch.zeros((args.classes_per_exp + 1, args.classes_per_exp + 1), dtype=torch.int32)
    confusion_mat_min = torch.zeros((args.classes_per_exp + 1, args.classes_per_exp + 1), dtype=torch.int32)
    confusion_mat_min_unk = torch.zeros((args.classes_per_exp + 1, args.classes_per_exp + 1), dtype=torch.int32)

    task_accuracy = 0.0

    for i in range(y.shape[0]):

        # y[i], y_hats[i]

        if entropy_total[i] < 0.25:  # classificati come task elements

            if y[i] == 10:
                confusion_mat_min_unk[y[i], y_hats[i]] += 1
            else:
                confusion_mat_min[y[i], y_hats[i]] += 1
                task_accuracy += 1


        else:  # classificcati come unk

            if y[i] == 10:

                confusion_mat_m_unk[y[i], y_hats[i]] += 1
                task_accuracy += 1
            else:

                confusion_mat_m[y[i], y_hats[i]] += 1

        confusion_mat[y[i], y_hats[i]] += 1  # Normal case: assign confusion matrix

    print(f"task accuracy with entropy th 0.25 {task_accuracy / y.shape[0]}")
    sanity_check = True

    if sanity_check:
        task_accuracy = compute_task_accuracy(confusion_mat_min)
        # print(task_accuracy)
        # print(confusion_mat_min)
        # print(f"number of elements: {confusion_mat_min.sum()}")
        # input("<0.1 task label")
        task_accuracy = compute_task_accuracy(confusion_mat_min_unk)
        # print(task_accuracy)
        # print(f"number of elements: {confusion_mat_min_unk.sum()}")
        # input("<0.1 unk label")
        # print(confusion_mat_min_unk)
        task_accuracy = compute_task_accuracy(confusion_mat_m)
        # print(task_accuracy)
        # print(confusion_mat_m)
        # print(f"number of elements: {confusion_mat_m.sum()}")
        # input(">=0.1 task label")
        task_accuracy = compute_task_accuracy(confusion_mat_m_unk)
        # print(task_accuracy)
        # print(confusion_mat_m_unk)
        # print(f"number of elements: {confusion_mat_m_unk.sum()}")
        # input(">=0.1 unk label")

    save_dir = f"plots/run_{args.seed}"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot histograms
    plt.figure(figsize=(10, 6))
    entropy_vector = np.array(entropy_vector, dtype=float)
    entropy_vector_unk = np.array(entropy_vector_unk, dtype=float)
    sns.histplot(entropy_vector, bins=30, color='blue', label="Entropy 1", alpha=0.6)
    sns.histplot(entropy_vector_unk, bins=30, color='red', label="Entropy 2", alpha=0.6)

    # Labels and legend
    plt.xlabel("Entropy Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Two Entropy Distributions")
    plt.legend()

    # plt.show()
    # Save the plot instead of showing it
    plt.savefig(f"plots/run_{args.seed}/entropy_histogram_{strategy.experience_idx}_{temperature}.png", dpi=300,
                bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    # input("STOP")
    # print(confusion_mat)
    # Convert to NumPy and save to a text file
    np.savetxt(f"plots/run_{args.seed}/confusion_matrix_{exp_idx}_{temperature}.txt", confusion_mat.numpy(), fmt='%d')
    # input("funzione di confsione rotta")
    task_accuracy = compute_task_accuracy(confusion_mat)

    # print(f"Accuracy for Task {exp_idx}: {task_accuracy:.4f}")
    # input("STOP")
    return confusion_mat


def compute_task_accuracy(confusion_mat):
    accuracy = confusion_mat.diag().sum() / confusion_mat.sum()

    return accuracy


def test_during_training(pruner, train_dloader, test_dloader, model, fresh_model, scheduler, epoch, exp_idx,
                         distillation, plot=True):
    if distillation:
        model = model.eval()
    else:
        model = fresh_model.eval()

    with torch.no_grad():

        train_conf_mat = test_single_exp(pruner, model, train_dloader, exp_idx, distillation)
        test_conf_mat = test_single_exp(pruner, model, test_dloader, exp_idx, distillation)

        # check if the second class is the unk class for that task
        if epoch >= args.sweep and distillation:
            train_conf_mat_onering = test_single_exp_onering(pruner, model, train_dloader, exp_idx, distillation)
            test_conf_mat_onering = test_single_exp_onering(pruner, model, test_dloader, exp_idx, distillation)

        # compute accuracy
        train_acc = train_conf_mat.diag().sum() / train_conf_mat.sum()
        test_acc = test_conf_mat.diag().sum() / test_conf_mat.sum()

        if epoch >= args.sweep and distillation:
            train_acc_onering = train_conf_mat_onering.diag().sum() / train_conf_mat_onering.sum()
            test_acc_onering = test_conf_mat_onering.diag().sum() / test_conf_mat_onering.sum()

        if epoch >= args.sweep and distillation:
            print(
                f"    e:{epoch:03}, tr_acc:{train_acc:.4f}, te_acc:{test_acc:.4f} tr_acc_onering:{train_acc_onering:.4f}, te_acc_onering:{test_acc_onering:.4f} lr:{scheduler.get_last_lr()[0]:.5f}")
        else:
            print(
                f"    e:{epoch:03}, tr_acc:{train_acc:.4f}, te_acc:{test_acc:.4f}, lr:{scheduler.get_last_lr()[0]:.5f}")
        if plot:
            plt_confmats(args.run_name, train_conf_mat, test_conf_mat, distillation, exp_idx)

    model.train()

    return train_acc, test_acc


def confidence(frag_preds, task_id):
    on_shell_probs = []
    elsewhere_probs = []
    for i, frag in enumerate(frag_preds):
        on_shell_probs.append(torch.softmax(frag[task_id == i], dim=-1))
        elsewhere_probs.append(torch.softmax(frag[task_id != i], dim=-1))

    max_on_shell_probs = torch.max(torch.stack(on_shell_probs), dim=-1)[0]
    on_shell_confidence = [(1. / (1. - p + 1e-6).mean()) for p in max_on_shell_probs]

    max_elsewhere_probs = torch.max(torch.stack(elsewhere_probs), dim=-1)[0]
    elsewhere_confidence = [(1. / (1. - p + 1e-6).mean()) for p in max_elsewhere_probs]

    return on_shell_confidence, elsewhere_confidence
