import torch
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics

class New_Trainer(object):
    def __init__(self, model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, scaler_phy, loss_phy, lr_scheduler=None):
        super(New_Trainer, self).__init__()
        self.model = model
        # self.model_stu = model_stu
        self.args = args
        self.loss = loss
        self.loss_phy = loss_phy
        self.loss_kl = loss_kl
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.batch_seen = 0
        self.scaler_phy = scaler_phy
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, self.args.save_pretrain_path)
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=False)
        print('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        total_val1 = 0
        total_val2 = 0
        # val_pred = []
        # val_true = []

        with torch.no_grad():
            for batch_idx, (data, data_mat, target, phy_target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_base_dim]
                label = target[..., :]
                output, phy_output = self.model(data, label=None, batch_seen=None, data_mat=data_mat)
                # output = self.model(data, label=None, batch_seen=None, data_mat=data_mat)
                # if self.args.real_value:
                #     label = self.scaler.inverse_transform(label[..., :self.args.output_dim])
                loss1, _ = self.loss(output, label[..., :])
                loss2, _ = self.loss_phy(phy_output, phy_target)
                loss = 0.7*loss1 + 0.3*loss2
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    total_val1 += loss1.item()
                    total_val2 += loss2.item()
                    
        val_loss = total_val_loss / len(val_dataloader)
        val_loss1 = total_val1 / len(val_dataloader)
        val_loss2 = total_val2 / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}, loss1: {:.6f},loss2:{:.6f}'.format(epoch, val_loss,val_loss1,val_loss2))

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        total_flow_loss = 0
        total_s_loss = 0
        for batch_idx, (data, data_mat ,target, phy_target) in enumerate(self.train_loader):
            self.batch_seen += 1
            data = data[..., :self.args.input_base_dim]
            label = target[..., :]
            self.optimizer.zero_grad()

            out, phy_out = self.model(data, label=None, batch_seen=None, data_mat=data_mat)
            # out = self.model(data, label=None, batch_seen=None, data_mat=data_mat)
            # print(out.shape, label.shape)
            loss1, _ = self.loss(out, label[..., :])
            loss2, _ = self.loss_phy(phy_out, phy_target)
            loss = 0.7*loss1+0.3*loss2
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            # calculate total loss
            #log information
            if batch_idx ==self.train_per_epoch:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        train_epoch_loss1 = total_loss1/self.train_per_epoch
        # train_epoch_loss2 = total_loss2/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, loss1: {:.6f},loss2:{:.6f}'.format(epoch, train_epoch_loss, train_epoch_loss1,train_epoch_loss2))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        # train_epoch_flow_loss for params selecting

        return train_epoch_loss

    def train(self):
        best_model = None
        best_model_test = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        up_epoch = [int(i) for i in list(self.args.up_epoch.split(','))]
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # print(time.time()-epoch_time)
            if epoch in up_epoch:
                best_loss = float('inf')

            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            val_loss_list.append(val_epoch_loss)
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                best_model_test = self.model
                model_save_path = f"./{self.args.model}/{self.args.dataset}/{self.args.dataset}_1.pth"
                torch.save(best_model_test.state_dict(), model_save_path)

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        # if not self.args.debug:
        if self.args.debug:
            # torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        # self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        model_save_path = f"./{self.args.model}/{self.args.dataset}/{self.args.dataset}_1.pth"
        torch.save(best_model_test.state_dict(), model_save_path)
        self.test(best_model_test, self.args, self.test_loader, self.scaler, self.logger)


    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, data_mat, target, phy_target) in enumerate(data_loader):
                data = data[..., :args.input_base_dim]
                # label = target[..., :args.input_base_dim + args.input_extra_dim]
                output, _ = model(data, label=None, batch_seen=None, data_mat=data_mat)
                # output = model(data, label=None, batch_seen=None, data_mat=data_mat)
                label = target[..., :args.output_dim]
                y_true.append(label)
                y_pred.append(output)
        # print(y_true.shape)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        
        # if args.real_value:
        #     y_pred = torch.cat(y_pred, dim=0)
        # else:
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        # np.save('./{}_true.npy'.format(args.dataset+'_'+args.model+'_'+args.mode), y_true.cpu().numpy())
        # np.save('./{}_pred.npy'.format(args.dataset+'_'+args.model+'_'+args.mode), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, corr = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}, CORR:{:.4f}%".format(
                t + 1, mae, rmse, mape*100, corr))
        mae, rmse, mape, _, corr = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
                    mae, rmse, mape*100, corr))