# basic package
import torch
import numpy as np
from tqdm import tqdm
# local package
from model.model import Net


def onehot_embedding(labels, num_classes=10, device=None):
    labels = labels.squeeze(1)
    y = torch.eye(num_classes)
    y = y.to(device)
    return y[labels]


class ModelInterface():
    def __init__(self, args):
        super(ModelInterface, self).__init__()
        self.model = Net(args=args)
        self.model.to(args.device)
        self.classes = args.num_class

    def KL(self, alpha, c):
        beta = torch.ones((1, c)).cuda()
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def mse_loss(self, y, alpha, classes, global_step, lambda_epochs):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        m = alpha / S
        label = y
        # label = F.one_hot(y, num_classes=classes)
        A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        annealing_coef = min(1, global_step / lambda_epochs)
        alp = E * (1 - label) + 1
        C = annealing_coef * self.KL(alp, classes)
        return (A + B) + C

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag  # 矩阵白色部分的和

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))  # 1-c 是棕色部分
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def train(self, train_data, val_data, criterion, optimizer, checkpoint_filepath, writer, args):
        best_score = 0.0 if args.monitor_acc else np.inf
        num_steps_wo_improvement = 0
        threshold = args.threshold
        save_times = 0
        epochs = args.epochs
        print('Start training...')
        for epoch in range(epochs):
            losses = 0.0
            self.model.train()

            dis_str = 'Epoch {}/{}, Saves:{}'.format(epoch + 1, epochs,
                                                     save_times) if not args.early_stop else 'Steps {}/{} in epoch:{}, Saves:{}'.format(
                num_steps_wo_improvement + 1, threshold, epoch + 1, save_times)
            for i, batch in enumerate(tqdm(train_data, desc=dis_str)):
                loss = 0.0
                bbox = batch[0][:, :, -args.time_scale:].to(args.device)
                vel = batch[1][:, :, -args.time_scale:].to(args.device)
                label = batch[-1].reshape(-1, 1).to(args.device).long()
                if args.time_crop and np.random.randint(10) >= 5:
                    bbox = bbox[:, :, -args.time_crop_scale:]
                    vel = vel[:, :, -args.time_crop_scale:]
                optimizer.zero_grad()

                y = onehot_embedding(label, args.num_class, args.device)
                evidence = dict()
                evidence[0], evidence[1], evidence[2] = self.model(bbox, vel)

                alpha = dict()
                for n in range(len(evidence)):
                    alpha[n] = evidence[n] + 1
                    loss += self.mse_loss(y, alpha[n], args.num_class, epochs, 10)

                alpha_a = self.DS_Combin(alpha)
                loss += self.mse_loss(y, alpha_a, args.num_class, global_step=epochs, lambda_epochs=10)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                losses += loss.item()

            writer.add_scalar('train_loss', losses / (i + 1), epoch)
            print(f"Epoch {epoch + 1}: Train loss: {losses / (i + 1)}")

            valid_loss = self.evaluate(val_data, args)
            writer.add_scalar('val_loss', valid_loss, epoch)

            if best_score > valid_loss and not args.monitor_acc:
                best_score = valid_loss
                num_steps_wo_improvement = 0
                torch.save(self.model.state_dict(), checkpoint_filepath)
                save_times += 1
                print("Save model {} times.\n".format(save_times))
            else:
                num_steps_wo_improvement += 1
                print("-> No improvement, step {} of {}".format(num_steps_wo_improvement, threshold))
            if args.early_stop and num_steps_wo_improvement >= threshold:
                break
        out_str = "Training finished, best acc: {}, save times: {}".format(best_score,
                                                                           save_times) if args.monitor_acc else "Training finished, best loss: {}, save times: {}".format(
            best_score, save_times)
        print(out_str)

    def evaluate(self, data, args):
        losses = 0.0
        epochs = args.epochs
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data, desc='Validation')):
                loss = 0.0
                bbox = batch[0][:, :, -args.time_scale:].to(args.device)
                vel = batch[1][:, :, -args.time_scale:].to(args.device)
                label = batch[-1].reshape(-1, 1).to(args.device).long()

                y = onehot_embedding(label, args.num_class, args.device)
                evidence = dict()
                evidence[0], evidence[1], evidence[2] = self.model(bbox, vel)

                alpha = dict()
                for n in range(len(evidence)):
                    alpha[n] = evidence[n] + 1
                    loss += self.mse_loss(y, alpha[n], args.num_class, epochs, 10)

                alpha_a = self.DS_Combin(alpha)
                loss += self.mse_loss(y, alpha_a, args.num_class, global_step=epochs, lambda_epochs=10)
                loss = torch.mean(loss)
                losses += loss.item()

        print(f'Validation loss: {losses / (i + 1)}')
        return losses / (i + 1)

    def test(self, test_data, check_file, args):
        check_point = torch.load(check_file)
        self.model.load_state_dict(check_point)
        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(tqdm(test_data, desc='Testing...')):
                bbox = batch[0][:, :, -args.time_scale:].to(args.device)
                vel = batch[1][:, :, -args.time_scale:].to(args.device)
                label = batch[-1].to(args.device).long()

                evidence = dict()
                e1, e2, e3 = self.model(bbox, vel)
                evidence[0], evidence[1], evidence[2] = e1.reshape(1, -1), e2.reshape(1, -1), e3.reshape(1, -1)
                alpha = dict()
                for n in range(len(evidence)):
                    alpha[n] = evidence[n]+1
                alpha_a = self.DS_Combin(alpha)
                evidence_a = alpha_a-1
                _,predicted = torch.max(evidence_a.data,1)
                if i == 0:
                    preds = predicted
                    labels = label
                else:
                    preds = torch.cat((preds, predicted), 0)
                    labels = torch.cat((labels, label), 0)
        return preds, labels


