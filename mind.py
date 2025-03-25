import torch
from copy import deepcopy
from pruner import Pruner
from utils.generic import freeze_model, unfreeze_model
from test_fn import test_during_training
from parse import args
from torch.nn import CrossEntropyLoss
from utils.viz import plt_masks_grad_weight
import torch.nn.functional as F

class MIND():

    def __init__(self, model):

        # default values        
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.experience_idx = -1
        self.optimizer = None
        self.train_epochs = None

        # model + loss
        self.model = model
        self.criterion = CrossEntropyLoss(label_smoothing=0.)

        # params pruner
        self.pruner = Pruner(self.model, train_bias=False, train_bn=False)
        self.distillation = False

        #param distill
        self.distill_model = None

        # logging stuff
        self.log_every = args.log_every
        self.plot_gradients = args.plot_gradients
        self.plot_gradients_of_layer = args.plot_gradients_of_layer

        self.confusion_mat_task = torch.zeros((args.n_experiences, args.n_experiences))
        self.forgetting_mat = torch.zeros((args.n_experiences, args.n_experiences))

    def get_ce_loss2(self):
        out = self.mb_output[:, torch.nonzero(self.model.output_mask[self.experience_idx])].squeeze(-1)
        return self.criterion(out, self.mb_y.to(args.device)-(self.experience_idx*10))


    def get_ce_loss(self):
        """ Cross entropy loss. """

        return self.criterion(self.mb_output.to(args.device), self.mb_y.to(args.device))

    def get_one_ring_loss(self):
        "one ring loss definition"


        #eliminare i pattern della classe reale e rimpiazzarli con la classe UNK

        new_removed_y=torch.full(self.mb_y.shape, self.train_scenario.nb_classes+self.experience_idx-1)#-1 because one element is removed later
        #one_hot=torch.nn.functional.one_hot(new_removed_y, num_classes=self.train_scenario.nb_classes+10)

        #eliminate one element for the one ring
        p = [(i, j.item()) for i, j in enumerate(self.mb_y)]

        n_exp=args.n_classes/args.classes_per_exp

        outs_ = torch.cat(
            [torch.cat((self.mb_output[i][0:j], self.mb_output[i][j + 1:])) for i, j in p]
        ).view(self.mb_output.shape[0], args.n_classes+n_exp-1)#automatizzare qua


        #print(self.mb_output[:,100])


        loss_unk = self.criterion(outs_.to(args.device), new_removed_y.to(args.device))


        #plot_UNK_position(self.mb_output, self.experience_idx)
        return loss_unk

    def get_distill_loss_JS(self):
        """ Distillation loss. (jensen-shannon) """
        with torch.no_grad():
            old_y = self.distill_model.forward(self.mb_x)
        #logits for the distilled network

        new_y = self.mb_output
        soft_log_old = torch.nn.functional.log_softmax(old_y+10e-5, dim=1)
        soft_log_new = torch.nn.functional.log_softmax(new_y+10e-5, dim=1)
        soft_old = torch.nn.functional.softmax(old_y+10e-5, dim=1)
        soft_new = torch.nn.functional.softmax(new_y+10e-5, dim=1)

        dist1 = torch.nn.functional.kl_div(soft_log_new+10e-5, soft_old+10e-5, reduction='batchmean')
        dist2 = torch.nn.functional.kl_div(soft_log_old+10e-5, soft_new+10e-5, reduction='batchmean')

        dist = ((dist1+dist2)/2).mean()

        return dist

    def get_distill_loss_KL(self):
        """ KL divergence loss. """
        with torch.no_grad():
            old_y = self.distill_model.forward(self.mb_x)

        new_y = self.mb_output
        soft_log_old = torch.nn.functional.log_softmax(old_y+10e-5, dim=1)
        soft_log_new = torch.nn.functional.log_softmax(new_y+10e-5, dim=1)

        kl_div = torch.nn.functional.kl_div(soft_log_new+10e-5, soft_log_old+10e-5, reduction='batchmean', log_target=True)

        return kl_div
    def get_distill_loss_Cosine(self):
        """ Cosine distance loss. """
        with torch.no_grad():
            old_y = self.distill_model.forward(self.mb_x)
        new_y = self.mb_output
        cosine_sim = F.cosine_similarity(new_y, old_y, dim=1)

        return 1 - cosine_sim.mean()

    def get_distill_loss_L2(self):
    
        """ L2 distance loss. """
        with torch.no_grad():
            old_y = self.distill_model.forward(self.mb_x)

        new_y = self.mb_output
        l2_distance = F.pairwise_distance(new_y, old_y)

        return l2_distance.mean()
    

    def train(self):

        for epoch in range(self.train_epochs):

            self.epoch = epoch

            if epoch ==40 and not self.distillation:
                print(" START ONE RING TRAINING")

            if not self.distillation:
                loss_ce, loss_distill, loss_ce_onering = self.training_epoch_fresh_model()
            else:
                loss_ce, loss_distill, loss_ce_onering = self.training_epoch()


            if self.scheduler:
                self.scheduler.step()

            if (epoch) % self.log_every == 0 or epoch+1 == self.train_epochs:
                print(f"    epoch: {self.epoch}, cross entropy: {loss_ce:.4f}, one ring loss: {loss_ce_onering:.4f}, loss distillation: {loss_distill/10:.4f}")
                with open(f"logs/{args.run_name}/results/loss.csv", "a") as f:
                    f.write(f"{self.experience_idx},{self.distillation},{epoch},{loss_ce:.4f},{loss_distill:.4f}\n")
                with open(f"logs/{args.run_name}/results/acc.csv", "a") as f:
                    acc_train, acc_test = test_during_training(self.pruner,
                                                               self.train_dataloader,
                                                               self.val_dataloader,
                                                               self.model,
                                                               self.fresh_model,
                                                               self.scheduler,
                                                               epoch,
                                                               self.experience_idx,
                                                               self.distillation,
                                                               plot=True)

                    f.write(f"{self.experience_idx},{self.distillation},{epoch},{acc_train:.4f},{acc_test:.4f}\n")

        if self.distillation:
            self.model.save_bn_params(self.experience_idx)


    def training_epoch_fresh_model(self):
        base_model = deepcopy(self.fresh_model)
        # freeze the model
        freeze_model(base_model)
        self.fresh_model.train()

        if args.packnet_original:
            if self.experience_idx != 0:
                for m in self.fresh_model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()

        for i, (self.mb_x, self.mb_y, self.mb_t) in enumerate(self.train_dataloader):

            self.mb_x = self.mb_x.to(args.device)
            self.loss = torch.tensor(0.).to(args.device)
            self.loss_ce = torch.tensor(0.).to(args.device)
            self.loss_distill = torch.tensor(0.).to(args.device)
            self.loss_ce_onering= torch.tensor(0.).to(args.device)
            self.mb_output = self.fresh_model.forward(self.mb_x.to(args.device))


            if self.epoch>100:




               self.loss_ce_onering=self.get_one_ring_loss()

            self.loss_ce = self.get_ce_loss()

            #loss distill is always zero
            self.loss += self.loss_ce + self.loss_distill + self.loss_ce_onering
            self.loss.backward()

            if self.plot_gradients and (len(self.train_dataloader)-2)==i and self.epoch == self.train_epochs-1:
                plt_masks_grad_weight(args.run_name, self.fresh_model, self.pruner, self.experience_idx, distillation=False, layer_idx=self.plot_gradients_of_layer)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            if args.self_distillation:
                freeze_model(self.fresh_model)
                self.pruner.ripristinate_weights(self.fresh_model, base_model, self.experience_idx, self.distillation)
                unfreeze_model(self.fresh_model)

        return self.loss_ce, self.loss_distill , self.loss_ce_onering


    def training_epoch(self):

        # to ripristinate the model
        base_model = deepcopy(self.model)
        # freeze the model
        freeze_model(base_model)

        #modalità training
        self.model.train()

        if args.packnet_original or args.no_bn:
            if self.experience_idx != 0:
                for m in self.model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()

        for i, (self.mb_x, self.mb_y, self.mb_t) in enumerate(self.train_dataloader):
            self.mb_x = self.mb_x.to(args.device)
            self.loss = torch.tensor(0.).to(args.device)
            self.loss_ce = torch.tensor(0.).to(args.device)
            self.loss_distill = torch.tensor(0.).to(args.device)
            self.loss_ce_onering = torch.tensor(0.).to(args.device)

            self.mb_output = self.model.forward(self.mb_x.to(args.device))
            #plot_UNK_position(self.mb_output, self.experience_idx)



            if args.distill_beta > 0:
                if args.distill_loss == 'JS':
                    self.loss_distill = args.distill_beta*self.get_distill_loss_JS()
                elif args.distill_loss == 'KL':
                    self.loss_distill = args.distill_beta*self.get_distill_loss_KL()
                elif args.distill_loss == 'Cosine':
                    self.loss_distill = args.distill_beta*self.get_distill_loss_Cosine()*10.
                elif args.distill_loss == 'L2':
                    self.loss_distill = args.distill_beta*self.get_distill_loss_L2()
                
            if self.epoch>=args.sweep:

                self.loss_ce_onering = self.get_one_ring_loss()
                self.loss_distill=0.0


            self.loss_ce = self.get_ce_loss()

            self.loss += self.loss_ce + self.loss_distill +self.loss_ce_onering
            self.loss.backward()
            
            if self.plot_gradients and (len(self.train_dataloader)-2)==i and self.epoch == self.train_epochs-1:
                plt_masks_grad_weight(args.run_name, self.model, self.pruner, self.experience_idx, distillation=True, layer_idx=self.plot_gradients_of_layer)
            
            self.optimizer.step()
            self.optimizer.zero_grad()



            freeze_model(self.model)

            #muta i pesi che erano stati mutati in precedenza
            self.pruner.ripristinate_weights(self.model, base_model, self.experience_idx, self.distillation)
            unfreeze_model(self.model)

        return self.loss_ce, self.loss_distill, self.loss_ce_onering

def plot_UNK_position(probabilities, n_task):

    starter_pos, ending_pos=n_task*args.classes_per_exp, (n_task+1)*args.classes_per_exp

    prob_last=probabilities[:, args.n_classes+n_task]
    prob_last = prob_last.unsqueeze(1)

    probs= torch.cat([probabilities[:, starter_pos:ending_pos], prob_last], dim=1)
    probs= torch.softmax(probs, dim=1)
    #print(probs.shape)
    sorted_probs= probs.argsort(dim=1, descending=True)
    #print(sorted_probs)

    sorted_probs=sorted_probs.tolist()
    unk_pos = {}
    for sorted_prob in sorted_probs:
        #print(sorted_prob)
        index=sorted_prob.index(10)
        if index not in unk_pos.keys():
            unk_pos[index] =  1
        else:
            unk_pos[index]= unk_pos[index]+1

    print(f"position of UNK  {unk_pos}")

    #input("STOP")




