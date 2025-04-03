from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from LLMRepresentation import LLMRepresentation
# from MutualInformation import MutualInformationLoss
from WeightingNet import WeightingNet

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    # def train(self, setting):
    #     train_data, train_loader = self._get_data(flag='TRAIN')
    #     vali_data, vali_loader = self._get_data(flag='TEST')
    #     test_data, test_loader = self._get_data(flag='TEST')

    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     time_now = time.time()

    #     train_steps = len(train_loader)
    #     early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    #     model_optim = self._select_optimizer()
    #     criterion = self._select_criterion()

    #     for epoch in range(self.args.train_epochs):
    #         iter_count = 0
    #         train_loss = []

    #         self.model.train()
    #         epoch_time = time.time()

    #         for i, (batch_x, label, padding_mask) in enumerate(train_loader):
    #             iter_count += 1
    #             model_optim.zero_grad()

    #             batch_x = batch_x.float().to(self.device)
    #             padding_mask = padding_mask.float().to(self.device)
    #             label = label.to(self.device)

    #             outputs = self.model(batch_x, padding_mask, None, None)
    #             loss = criterion(outputs, label.long().squeeze(-1))
    #             train_loss.append(loss.item())

    #             if (i + 1) % 100 == 0:
    #                 print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                 speed = (time.time() - time_now) / iter_count
    #                 left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #                 print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                 iter_count = 0
    #                 time_now = time.time()

    #             loss.backward()
    #             nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
    #             model_optim.step()

    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         train_loss = np.average(train_loss)
    #         vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
    #         test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

    #         print(
    #             "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
    #             .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
    #         early_stopping(-val_accuracy, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path))

    #     return self.model
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 模块初始化
        llm_rep = LLMRepresentation(output_dim=self.args.d_model).to(self.device)  # 模块保留
        weighting_net = WeightingNet(input_dim=self.args.num_class).to(self.device)

        optimizer_llm = torch.optim.Adam(llm_rep.parameters(), lr=self.args.learning_rate)
        optimizer_weight = torch.optim.Adam(weighting_net.parameters(), lr=self.args.learning_rate)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            llm_rep.train()
            weighting_net.train()
            train_loss = []

            for batch_x, labels, padding_mask in train_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                labels = labels.to(self.device)

                # 时间表示（用于结构完整）
                time_rep = self.model(batch_x, padding_mask, None, None)
                pooled_time = time_rep.mean(dim=1)

                # 文本表示（图(a)模块结构，但不参与loss）
                descriptions = ["This is a time series classification sample."] * batch_x.size(0)
                _ = llm_rep(descriptions)

                # 分类预测 + loss
                pred = self.model(batch_x, padding_mask, None, None)
                cls_loss = criterion(pred, labels.long().squeeze(-1))

                # 重加权模块（图(b)结构）
                ω_O, _ = weighting_net(pred.detach())  # 使用 detach 防止梯度泄露
                total_loss = ω_O.mean() * cls_loss

                model_optim.zero_grad()
                optimizer_llm.zero_grad()
                optimizer_weight.zero_grad()
                total_loss.backward()
                model_optim.step()
                optimizer_llm.step()

                # Bi-level：通过验证集更新权重模块
                with torch.no_grad():
                    vali_x, vali_y, vali_mask = next(iter(vali_loader))
                    vali_x = vali_x.float().to(self.device)
                    vali_y = vali_y.to(self.device)
                    vali_mask = vali_mask.float().to(self.device)
                    val_pred = self.model(vali_x, vali_mask, None, None)
                    val_loss = criterion(val_pred, vali_y.long().squeeze(-1))

                val_loss.requires_grad_(True)
                optimizer_weight.zero_grad()
                val_loss.backward()
                optimizer_weight.step()

                train_loss.append(total_loss.item())

            train_avg = np.mean(train_loss)
            vali_loss, val_acc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc = self.vali(test_data, test_loader, criterion)

            print(f"Epoch {epoch+1} | Train Loss: {train_avg:.4f} | Vali Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            early_stopping(-val_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return