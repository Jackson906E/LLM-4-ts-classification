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
from models.LLMRepresentation import LLMRepresentation
from models.WeightingNet import WeightingNet
# from MutualInformation import MutualInformationLoss
import math
import torch.nn.functional as F

warnings.filterwarnings('ignore')


def get_text_description(batch_x, args):
    descriptions = []

    for x in batch_x:
        description = "This is a time series classification sample. The content is: {}."
        description += "Input statistics: min value: {}, max value: {}, median value: {}."

        round_x = torch.round(x)

        description = description.format(str(round_x.tolist()), str(round_x.min().item()), str(round_x.max().item()), str(round_x.median().item()))
        
        descriptions.append(description)

    return descriptions

def contrastive_loss(time_rep, text_rep):
    # 计算时间序列和文本描述的余弦相似度
    # 将输入展平
    b = text_rep.shape[0]
    d = text_rep.shape[-1]
    time_rep = time_rep.view(b, -1, d)
    text_rep = text_rep.view(b, -1, d)

    tau = 0.07
    z_time = time_rep.mean(dim=1)   # [b, d]
    z_text = text_rep.mean(dim=1)   # [b, d]


    z_time = F.normalize(z_time, p=2, dim=1)   # (b, d)
    z_text = F.normalize(z_text, p=2, dim=1)   # (b, d)

    
    # 计算余弦相似度矩阵
    S = torch.matmul(z_time, z_text.t())  # (b, T, L)

    # S: (b, T, L) 已经是归一化后的点积
    logits = S / tau                   # (b, T, L)
    # 正样本假设都在对角线上：对每个 batch i，label[i] = [0,1,2,...,T-1]
    targets = torch.arange(b, device=logits.device)
    # F.cross_entropy 支持 3D 输入，自动对最后两维做 softmax + NLL
    loss_nce = F.cross_entropy(logits, targets)
    
    return loss_nce


class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.fusion_layer = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, time_rep, text_rep):
        # 1. 计算attention scores (Q·K^T)
        attention_scores = torch.matmul(time_rep, text_rep.transpose(-2, -1)) / math.sqrt(self.d_model)
        # 2. 应用softmax得到attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)     
        # 3. 计算加权和得到context vector (weights·V)
        context_vector = torch.matmul(attention_weights, text_rep)
        # 4. 通过线性层降维回原始维度
        fused_rep = self.fusion_layer(context_vector)
        # 5. Mean Pooling
        fused_rep = fused_rep.mean(dim=1)
        # 6. 将fused_rep展平并通过MLP得到分类logits
        fused_rep = self.mlp(fused_rep)

        return fused_rep

class Exp_Classification_LLM(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification_LLM, self).__init__(args)
        self.llm_rep = LLMRepresentation(output_dim=self.args.d_model).to(self.device)
        self.cross_attn = CrossAttention(self.args.d_model).to(self.device)


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

                descriptions = get_text_description(batch_x, self.args)
                text_rep = self.llm_rep(descriptions)
                time_rep = self.model(batch_x, padding_mask, None, None, out_proj=False)

                outputs = self.cross_attn(time_rep, text_rep)

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
        #llm_rep = LLMRepresentation(output_dim=self.args.d_model).to(self.device)
        # weighting_net = WeightingNet(input_dim=self.args.num_class).to(self.device)
        #cross_attn = CrossAttention(self.args.d_model).to(self.device)

        optimizer_llm = torch.optim.Adam(self.llm_rep.parameters(), lr=self.args.learning_rate)
        optimizer_cross_attn = torch.optim.Adam(self.cross_attn.parameters(), lr=self.args.learning_rate)
        # optimizer_weight = torch.optim.Adam(weighting_net.parameters(), lr=self.args.learning_rate)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            self.llm_rep.train()
            self.cross_attn.train()
            #weighting_net.train()
            train_loss = []

            for batch_x, labels, padding_mask in train_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                labels = labels.to(self.device)

                # print('batch_x shape:', batch_x.shape)
                # print('labels shape:', labels.shape)
                # print('padding_mask shape:', padding_mask.shape)
                
                # 时间表示（用于结构完整）
                time_rep = self.model(batch_x, padding_mask, None, None, out_proj=False)

                # 文本表示（图(a)模块结构，但不参与loss）
                descriptions = get_text_description(batch_x, self.args)
                text_rep = self.llm_rep(descriptions)

                # print('time_rep:', time_rep.shape)
                # print('text_rep:', text_rep.shape)

                # 计算cross attention，text_rep作为kv，time_rep作为q
                # # 1. 计算attention scores (Q·K^T)
                # attention_scores = torch.matmul(time_rep, text_rep.transpose(-2, -1)) / math.sqrt(self.args.d_model)
                # # 2. 应用softmax得到attention weights
                # attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)     
                # # 3. 计算加权和得到context vector (weights·V)
                # context_vector = torch.matmul(attention_weights, text_rep)
                # # 4. 通过线性层降维回原始维度
                # fusion_layer = nn.Linear(context_vector.size(-1), self.args.d_model).to(self.device)

                # fused_rep = fusion_layer(context_vector)

                # # Mean Pooling
                # fused_rep = fused_rep.mean(dim=1)

                # # 将fused_rep展平并通过MLP得到分类logits
                # fused_rep = fused_rep.reshape(fused_rep.size(0), -1)  # 展平
                # mlp = nn.Sequential(
                #     nn.Linear(fused_rep.size(-1), fused_rep.size(-1)*4),
                #     nn.ReLU(),
                #     nn.Dropout(0.1),
                #     nn.Linear(fused_rep.size(-1)*4, self.model.configs.num_class)
                # ).to(self.device)

                # pred = mlp(fused_rep)

                # pred = time_rep

                pred = self.cross_attn(time_rep, text_rep)

                cls_loss = criterion(pred, labels.long().squeeze(-1))

                ctr_loss = contrastive_loss(time_rep, text_rep)
                # ctr_loss = 0

                total_loss = cls_loss + 0.5 * ctr_loss

                # print(f'cls_loss: {cls_loss:.4f}, contrastive_loss: {ctr_loss:.4f}')

                model_optim.zero_grad()
                optimizer_cross_attn.zero_grad()
                optimizer_llm.zero_grad()

                total_loss.backward()

                model_optim.step()
                optimizer_cross_attn.step()
                optimizer_llm.step()

                # # 分类预测 + loss
                # pred = self.model(batch_x, padding_mask, None, None)
                # cls_loss = criterion(pred, labels.long().squeeze(-1))

                # # 重加权模块（图(b)结构）
                # ω_O, _ = weighting_net(pred.detach())  # 使用 detach 防止梯度泄露
                # total_loss = ω_O.mean() * cls_loss

                # model_optim.zero_grad()
                # optimizer_llm.zero_grad()
                # optimizer_weight.zero_grad()
                # total_loss.backward()
                # model_optim.step()
                # optimizer_llm.step()

                # # Bi-level：通过验证集更新权重模块
                # with torch.no_grad():
                #     vali_x, vali_y, vali_mask = next(iter(vali_loader))
                #     vali_x = vali_x.float().to(self.device)
                #     vali_y = vali_y.to(self.device)
                #     vali_mask = vali_mask.float().to(self.device)
                #     val_pred = self.model(vali_x, vali_mask, None, None)
                #     val_loss = criterion(val_pred, vali_y.long().squeeze(-1))

                # val_loss.requires_grad_(True)
                # optimizer_weight.zero_grad()
                # val_loss.backward()
                # optimizer_weight.step()

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

                descriptions = get_text_description(batch_x, self.args)
                text_rep = self.llm_rep(descriptions)
                time_rep = self.model(batch_x, padding_mask, None, None, out_proj=False)

                outputs = self.cross_attn(time_rep, text_rep)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

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
        file_name='result_classification_llm.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return