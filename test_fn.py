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

def get_stat_exp(y,y_hats,exp_idx, task_id,task_predictions):
    """ Compute accuracy and task accuracy for each experience."""
    conf_mat = torch.zeros((exp_idx+1, exp_idx+1))

    for i in range(exp_idx+1):
        ybuff= y[task_id==i]
        y_hats_buff=y_hats[task_id==i]
        acc = (ybuff==y_hats_buff).sum()/y_hats_buff.shape[0]

        for j in range(exp_idx+1):
            conf_mat[i,j] = ((task_id==i)&(task_predictions==j)).sum()/(task_id==i).sum()

        print(f"EXP:{i}, acc:{acc:.3f}, task:{conf_mat[i,i]:.3f}, distrib:{[round(conf_mat[i,j].item(), 3) for j in range(exp_idx+1)]}")


def entropy(vec):
    return -torch.sum(vec * torch.log(vec + 1e-7), dim=1)

#create the augmentation of the image and assign a good label using a score ???
def query_expansion(original, num_perturbations):
    cutout_transform = transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    augmented_images= []
    for _ in range(num_perturbations):
        augmented_images.append(cutout_transform(original))  # Apply Cutout

    #insert an expansion function
    return torch.stack(augmented_images)

def scoring_function(pred):
    #create a scoring function
    avg_probs = pred.mean(dim=0)
    return avg_probs



def predict_with_perturbations(strategy,test_set, batch):
    """
    For each image in the batch:
    - Generate multiple perturbations
    - Get predictions for each perturbed version
    - Combine predictions and return the final class for each original image

    Args:
        batch (torch.Tensor): Input batch of images, shape (batch_size, C, H, W)

    Returns:
        torch.Tensor: Final predicted class for each image in the batch (batch_size,)
    """
    strategy.model.eval()
    dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

    confusion_mat = torch.zeros((args.n_classes, args.n_classes))
    confusion_mat_e = torch.zeros((args.n_classes, args.n_classes))
    confusion_mat_taw = torch.zeros((args.n_classes, args.n_classes))

    y_hats = []
    y_hats_e = []
    y_taw = []
    ys = []
    task_predictions = []
    task_ids = []
    augmentations=args.num_perturbations

    for i, (x, y, task_id) in enumerate(dataloader):
        frag_preds = []

        entropy_frag = []
        for j in range(strategy.experience_idx+1):
            # create a temporary model copy
            model = freeze_model(deepcopy(strategy.model))

            strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
            model.load_bn_params(j)
            model.exp_idx = j
            pred=[]

            #create a perturbation for each image into the batch
            for index,image in enumerate (x):
                augmented_x=query_expansion(x)
                pred_aug = model(x.to(args.device))
                pred_aug= torch.softmax(pred_aug , dim=1)
                final_pred=scoring_function(pred_aug)
                pred.append(final_pred)

            pred=torch.stack(pred,dim=1)

            if not args.dataset == 'CORE50':
                #take the last class UNK

                last=pred[:, -1:]
                pred = pred[:, j*args.classes_per_exp:(j+1)*args.classes_per_exp]
                pred=torch.cat([pred, last], dim=1 )

            softmax=torch.softmax(pred / args.temperature, dim=1)
            #check the capture in the UNK task
            check=True
            if check:
                out_task=(task_id!=j)
                out_indexes=torch.nonzero(out_task)
                print(out_indexes.shape)
                print(out_indexes)
                print(softmax[out_indexes, -1:])
                input("check")


            softmax = torch.softmax(pred / args.temperature, dim=1)[:, :-1]
            entropia = entropy(softmax)

        #continuare a sistemare
    return


def test(strategy, test_set, plot=False):

    strategy.model.eval()

    #test_set=torch.shuffle
    dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

    confusion_mat = torch.zeros((args.n_classes,args.n_classes))
    confusion_mat_th = torch.zeros((args.n_classes,args.n_classes))
    confusion_mat_e = torch.zeros((args.n_classes,args.n_classes))
    confusion_mat_taw = torch.zeros((args.n_classes,args.n_classes))

    y_hats = []
    y_hats_e = []
    y_hats_th = []
    y_taw = []
    ys = []
    task_predictions = []
    task_ids = []
    dict_entropy={}

    # per diversi dataset quello che è usare diversi task id del dataset e vedere le performance della rete con l'inserimento di nuovi task id diversi
    for i, (x, y, task_id) in enumerate(dataloader):
        frag_preds = []

        entropy_frag = []
        for j in range(strategy.experience_idx+1):
            # create a temporary model copy
            model = freeze_model(deepcopy(strategy.model))

            strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
            model.load_bn_params(j)
            model.exp_idx = j

            pred = model(x.to(args.device))
            #print(pred.shape)
            #input("stop")
            if not args.dataset == 'CORE50':#if NOT
                #take the last class UNK

                last=pred[:, -1:]
                pred = pred[:, j*args.classes_per_exp:(j+1)*args.classes_per_exp]
                pred=torch.cat([pred, last], dim=1 )





            softmax=torch.softmax(pred , dim=1)

            entropia = entropy(softmax)

            #check the capture in the UNK task
            check=False
            if check:
                out_task = (task_id != j)
                out_indexes = torch.nonzero(out_task)

                with open("output_log.txt", "a") as log_file:  # Open file in append mode
                    if len(out_indexes) > 1:
                        log_file.write(f"Softmax (first 5 indices) task id: {task_id}, subnetwork {j}:\n{softmax[out_indexes[:5], :]}\n\n")
                        print(softmax[out_indexes[:5], :])
                        #input("check the distribution")  # Keeps original behavior

                    log_file.write(f"Softmax (last column) task id: {task_id}, subnetwork {j} :\n{softmax[out_indexes, -1:]}\n\n")
                    #input("check")  # Keeps original behavior

            softmax = torch.softmax(pred , dim=1)[:, :-1] #remove the last class for the accuracy



            frag_preds.append(softmax)

            entropy_frag.append(entropia)


        #nello stesso for, quindi questi sono elementi dello stesso batch

        frag_preds = torch.stack(frag_preds)  # [n_frag, bsize, n_classes]

        entropy_frag = torch.stack(entropy_frag)  # [bsize, n_frag] contains the entropy

        batch_size = frag_preds.shape[1]

        #divide per entropy
        entropy_th=0.20

        entropy_mask= entropy_frag<entropy_th #[nfrag, bsize]
        frag_preds_th=frag_preds.clone()
        entropy_good = torch.sum(entropy_mask, dim=0)

        print(entropy_good.shape)
        print(frag_preds_th.shape)
        print(entropy_mask.shape)

        for index in range(frag_preds.shape[1]):#scandisco le immagini

            for value in range (entropy_mask.shape[0]):#scandisco i task

                muted_tasks=entropy_mask[value, index]

                #flag to zero, don't interfere

                all_muted_flag= entropy_good[index]==0

                if all_muted_flag:
                    continue #don't do enything

                else:

                    if not muted_tasks:
                        #mute the classes related to the task
                        frag_preds_th[value, index,:]=0.0



        #print(frag_preds_th)
        #print(frag_preds)
        #input("guarda un attimo")
        #create a similar torch vector to check what happens by thesholding the entropy

        for el in entropy_good:

                if int(el.to("cpu")) not in dict_entropy.keys():

                    dict_entropy[int(el.to("cpu"))]=1
                else:
                    dict_entropy[int(el.to("cpu"))]+=1

        ### select across the top 2 of likelihood the head  with the lowest entropy
        # buff -> batch_size  x 2, 0-99 val

        buff = frag_preds.max(dim=-1)[0].argsort(dim=0)[-2:] # [2, bsize]
        buff_th = frag_preds_th.max(dim=-1)[0].argsort(dim=0)[-2:] # [2, bsize]

        # buff_entropy ->  2 x batch_size, entropy values
        indices = torch.arange(batch_size)

        if buff.shape[0] == 1:
            buff_entropy = entropy_frag[buff[0, :], indices].unsqueeze(0)

        else:
            a = entropy_frag[buff[0, :], indices]
            b = entropy_frag[buff[1, :], indices]
            buff_entropy = torch.stack((a, b)) # [2, bsize]


        # index of min entropy ->  1 x batch_size, val 0-1
        index_min = buff_entropy.argsort(dim=0)[0,:]

        index_class = buff[index_min, indices]

        task_predictions.append(buff[-1])

        if args.dataset == 'CORE50':
            y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1))
            y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1))
            y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1))

        else:
            y_hats_th.append(frag_preds[buff_th[-1], indices].argmax(dim=1) + args.classes_per_exp*buff_th[-1])
            y_hats_e.append(frag_preds[index_class, indices].argmax(dim=1)+ args.classes_per_exp*index_class)
            y_hats.append(frag_preds[buff[-1], indices].argmax(dim=1) + args.classes_per_exp*buff[-1])
            y_taw.append(frag_preds[task_id.to(torch.int32), indices].argmax(dim=-1) + (args.classes_per_exp*task_id.to(args.cuda)).to(torch.int32))

        task_ids.append(task_id) #real task id for each sample
        ys.append(y)

    #print(dict_entropy)
    total_number=sum(dict_entropy.values())

    bins = list(dict_entropy.keys())
    frequencies = list(dict_entropy.values())
    #to cpu
    # Plot the bar chart
    plt.bar(bins, frequencies, width=0.5, color='blue', edgecolor='black', alpha=0.7)

    # Labels and title
    plt.xlabel('number of elements below min th')
    plt.ylabel('Frequency')
    plt.title('Frequency of tasks under the minimum entropy per image')
    plt.xticks(bins)
    plt.savefig(f"distribution_{strategy.experience_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    y_hats_th = torch.cat(y_hats_th, dim=0).to('cpu')
    y_hats_e = torch.cat(y_hats_e, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')
    y_taw = torch.cat(y_taw, dim=0).to('cpu')
    task_predictions = torch.cat(task_predictions, dim=0).to('cpu')

    task_ids = torch.cat(task_ids, dim=0).to('cpu')
    print(f"number of samples below the threshold th 0.20 : {total_number}, total number of samples: {y.shape[0]}")


    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1
        confusion_mat_th[y[i], y_hats_th[i]] += 1
        confusion_mat_e[y[i], y_hats_e[i]] += 1
        confusion_mat_taw[y[i], y_taw[i]] += 1

        #find this entropy




    
    #task confusion matrix and forgetting mat
    for j in range(strategy.experience_idx+1):
        i = strategy.experience_idx
        acc_conf_mat_task = confusion_mat[j*args.classes_per_exp:(j+1)*args.classes_per_exp, j*args.classes_per_exp:(j+1)*args.classes_per_exp].diag().sum()/confusion_mat[i*args.classes_per_exp:(i+1)*args.classes_per_exp,:].sum()
        strategy.confusion_mat_task[i][j] = acc_conf_mat_task
        strategy.forgetting_mat[i][j] = strategy.confusion_mat_task[:, j].max()-acc_conf_mat_task

    # You should divide the accuracy with 0.2 for the choosen task , and see whether the accuracy is good or not

    # compute accuracy
    accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
    accuracy_th = confusion_mat_th.diag().sum() / confusion_mat_th.sum()
    accuracy_e = confusion_mat_e.diag().sum() / confusion_mat_e.sum()
    accuracy_taw = confusion_mat_taw.diag().sum() / confusion_mat_taw.sum()



    task_accuracy = (task_predictions==task_ids).sum()/y_hats.shape[0] #modificare questa cosa

    print(f"Test Accuracy: {accuracy:.4f},Test Accuracy with entropy: {accuracy_e:.4f},Test Accuracy taw: {accuracy_taw:.4f}, Task accuracy: {task_accuracy:.4f}, accuracy_th: {accuracy_th}")
    get_stat_exp(y, y_hats, strategy.experience_idx, task_ids,task_predictions)
    print("using accuracy_th")
    get_stat_exp(y,y_hats_th, strategy.experience_idx, task_ids,task_predictions)

    if plot:
        plt_test_confmat(args.run_name, confusion_mat, strategy.experience_idx)
        if strategy.experience_idx == args.n_experiences-1:
            plt_test_confmat_task(args.run_name, strategy.confusion_mat_task)
            torch.save(strategy.forgetting_mat, f'./logs/{args.run_name}/forgetting_mat.pt')
            torch.save(strategy.confusion_mat_task, f'./logs/{args.run_name}/confusion_mat_task.pt')


    if strategy.experience_idx == args.n_experiences-1:
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
        preds= preds[:, :-1]
        #frag preds size = (hid,bsize,100) and I want to reshape (bsize, hid, 100)
        y_hats.append(preds.argmax(dim=1))
        ys.append(y)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')
    y = torch.cat(ys, dim=0).to('cpu')

    # assign +1 to the confusion matrix for each prediction that matches the label
    for i in range(y.shape[0]):
        confusion_mat[y[i], y_hats[i]] += 1

    return confusion_mat

def test_single_exp_one_ring(strategy,  loader, exp_idx, distillation=True):

    dataloader = DataLoader(loader, batch_size=100, shuffle=False, num_workers=8)
    confusion_mat = torch.zeros((args.classes_per_exp+1, args.classes_per_exp+1),  dtype=torch.int32)
    y_hats = []
    ys = []
    entropy_vector=[]
    entropy_vector_unk=[]
    entropy_total=[]


    for i, (x, y, task_id) in enumerate(dataloader):


        #print(task_id)
        #load the model label
        model = freeze_model(deepcopy(strategy.model))  ####
        if  not distillation:
            model = freeze_model(deepcopy(strategy.fresh_model)) ####
        if distillation:
            strategy.pruner.set_gating_masks(model, exp_idx, weight_sharing=args.weight_sharing, distillation=True)
            model.load_bn_params(exp_idx)
            model.exp_idx = exp_idx

        preds = model(x.to(args.device))

        last = preds[:, -1:]
        preds = preds[:, exp_idx * args.classes_per_exp:(exp_idx + 1) * args.classes_per_exp]
        preds = torch.cat([preds, last], dim=1)

        preds=torch.softmax(preds, dim=1)

        entropia=entropy(preds)

        classes_indexes_start, classes_indexes_end =exp_idx * args.classes_per_exp,(exp_idx + 1) * args.classes_per_exp


        for index, label in enumerate(y):
            if label< classes_indexes_start or label>=classes_indexes_end:
                #assign to UNK class
                y[index]=10
                entropy_vector_unk.append(entropia[index].cpu().numpy())
            else :
                #classi del task
                y[index]=y[index]-classes_indexes_start
                entropy_vector.append(entropia[index].cpu().numpy())

                #input("guarda bene se è giusto")

        #print(preds.shape)

        #input("vedere cosa printi in output")

        y_hats.append(preds.argmax(dim=1))
        #print(preds.argmax(dim=1))
        #input("guarda qua")
        ys.append(y)
        entropy_total.append(entropia)

    # concat labels and preds
    y_hats = torch.cat(y_hats, dim=0).to('cpu')

    y = torch.cat(ys, dim=0).to('cpu')

    entropy_total=torch.cat( entropy_total, dim=0).to("cpu")

    # assign +1 to the confusion matrix for each prediction that matches the label
    confusion_mat_m=torch.zeros((args.classes_per_exp+1, args.classes_per_exp+1),  dtype=torch.int32)
    confusion_mat_m_unk=torch.zeros((args.classes_per_exp+1, args.classes_per_exp+1),  dtype=torch.int32)
    confusion_mat_min=torch.zeros((args.classes_per_exp+1, args.classes_per_exp+1),  dtype=torch.int32)
    confusion_mat_min_unk=torch.zeros((args.classes_per_exp+1, args.classes_per_exp+1),  dtype=torch.int32)

    task_accuracy=0.0

    for i in range(y.shape[0]):

        #y[i], y_hats[i]

        if entropy_total[i]<0.25: #classificati come task elements


            if y[i]==10:
                confusion_mat_min_unk[y[i], y_hats[i]] += 1
            else:
                confusion_mat_min[y[i], y_hats[i]] += 1
                task_accuracy+=1


        else: #classificcati come unk

            if y[i]==10:

                confusion_mat_m_unk[y[i], y_hats[i]] += 1
                task_accuracy += 1
            else:

                confusion_mat_m[y[i], y_hats[i]] += 1

        confusion_mat[y[i], y_hats[i]] += 1  # Normal case: assign confusion matrix

    print(f"task accuracy with entropy th 0.25 {task_accuracy/y.shape[0]}")
    sanity_check=True

    if sanity_check:
        task_accuracy = compute_task_accuracy(confusion_mat_min)
        print(task_accuracy)
        print(confusion_mat_min)
        print(f"number of elements: {confusion_mat_min.sum()}")
        #input("<0.1 task label")
        task_accuracy = compute_task_accuracy(confusion_mat_min_unk)
        print(task_accuracy)
        print(f"number of elements: {confusion_mat_min_unk.sum()}")
        #input("<0.1 unk label")
        print(confusion_mat_min_unk)
        task_accuracy = compute_task_accuracy(confusion_mat_m)
        print(task_accuracy)
        print(confusion_mat_m)
        print(f"number of elements: {confusion_mat_m.sum()}")
       # input(">=0.1 task label")
        task_accuracy = compute_task_accuracy(confusion_mat_m_unk)
        print(task_accuracy)
        print(confusion_mat_m_unk)
        print(f"number of elements: {confusion_mat_m_unk.sum()}")
        #input(">=0.1 unk label")

    # Plot histograms
    plt.figure(figsize=(10, 6))
    entropy_vector = np.array(entropy_vector, dtype=float)
    entropy_vector_unk = np.array(entropy_vector_unk, dtype=float)
    sns.histplot(entropy_vector, bins=30, color='blue', label="Entropy 1", alpha=0.6)
    sns.histplot(entropy_vector_unk, bins=30,  color='red', label="Entropy 2", alpha=0.6)

    # Labels and legend
    plt.xlabel("Entropy Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Two Entropy Distributions")
    plt.legend()

    #plt.show()
    # Save the plot instead of showing it
    plt.savefig(f"entropy_histogram_{strategy.experience_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    #input("STOP")
    #print(confusion_mat)
    # Convert to NumPy and save to a text file
    np.savetxt(f"confusion_matrix_{exp_idx}.txt", confusion_mat.numpy(), fmt='%d')
    #input("funzione di confsione rotta")
    task_accuracy = compute_task_accuracy(confusion_mat)

    print(f"Accuracy for Task {exp_idx}: {task_accuracy:.4f}")
    #input("STOP")
    return confusion_mat

def compute_task_accuracy(confusion_mat):

    accuracy = confusion_mat.diag().sum() / confusion_mat.sum()


    return accuracy

def test_during_training(pruner, train_dloader, test_dloader, model, fresh_model, scheduler, epoch, exp_idx, distillation, plot=True):
    if distillation:
        model = model.eval()
    else:
        model = fresh_model.eval()

    with torch.no_grad():
        train_conf_mat = test_single_exp(pruner, model, train_dloader, exp_idx, distillation)
        test_conf_mat = test_single_exp(pruner, model, test_dloader, exp_idx, distillation)
        # compute accuracy
        train_acc = train_conf_mat.diag().sum() / train_conf_mat.sum()
        test_acc = test_conf_mat.diag().sum() / test_conf_mat.sum()
        print(f"    e:{epoch:03}, tr_acc:{train_acc:.4f}, te_acc:{test_acc:.4f} lr:{scheduler.get_last_lr()[0]:.5f}")

        if plot:
            plt_confmats(args.run_name, train_conf_mat, test_conf_mat, distillation, exp_idx)

    model.train()

    return train_acc, test_acc


def confidence(frag_preds, task_id):
    on_shell_probs = []
    elsewhere_probs = []
    for i, frag in enumerate(frag_preds):
        on_shell_probs.append(torch.softmax(frag[task_id==i], dim = -1))
        elsewhere_probs.append(torch.softmax(frag[task_id!=i], dim = -1))


    max_on_shell_probs = torch.max(torch.stack(on_shell_probs), dim = -1)[0]
    on_shell_confidence = [(1./(1.-p + 1e-6).mean()) for p in max_on_shell_probs]

    max_elsewhere_probs = torch.max(torch.stack(elsewhere_probs), dim = -1)[0]
    elsewhere_confidence = [(1./(1.-p + 1e-6).mean()) for p in max_elsewhere_probs]

    return on_shell_confidence, elsewhere_confidence
