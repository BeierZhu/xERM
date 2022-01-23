import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, load_state_dict, rename_parallel_state_dict, autocast, use_fp16, MiscMeter, shot_acc
import model.model as module_arch
import torch.nn.functional as F

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config

        self.distill = config._config.get('distill', False)
        
        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = config._config.get('add_extra_info', False)

        if self.distill:
            print("** Distill is on, please double check distill_checkpoint in config **")
            self.teacher_model = config.init_obj('distill_arch', module_arch)
            teacher_checkpoint = torch.load(config['distill_checkpoint'], map_location="cpu")

            self.teacher_model = self.teacher_model.to(self.device)

            teacher_state_dict = teacher_checkpoint["state_dict"]

            rename_parallel_state_dict(teacher_state_dict)
            
            if len(self.device_ids) > 1:
                print("Using multiple GPUs for teacher model")
                self.teacher_model = torch.nn.DataParallel(self.teacher_model, device_ids=self.device_ids)
                load_state_dict(self.teacher_model, {"module." + k: v for k, v in teacher_state_dict.items()}, no_ignore=True)
            else:
                load_state_dict(self.teacher_model, teacher_state_dict, no_ignore=True)

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        if use_fp16:
            self.logger.warn("FP16 is enabled. This option should be used with caution unless you make sure it's working and we do not provide guarantee.")
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.balanced_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.imbalanced_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.xERM = False
        if 'balanced_model_path' in config._config:
            print('=> load balanced model')
            balanced_model = config.init_obj('arch', module_arch)
            balanced_checkpoint = torch.load(config['balanced_model_path'], map_location="cpu")
            balanced_state_dict = balanced_checkpoint["state_dict"]
            load_state_dict(balanced_model, balanced_state_dict, no_ignore=True)
            self.balanced_model = balanced_model
            self.balanced_model.to(self.device)
            for param in self.balanced_model.parameters():
                param.require_grad = False
            self.balanced_model.eval()
            self.xERM = True
            print('=> load student as well')
            load_state_dict(self.model, balanced_state_dict, no_ignore=True)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.real_model._hook_before_iter()
        self.train_metrics.reset()

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        for batch_idx, data in enumerate(self.data_loader):
            if self.distill and len(data) == 4:
                data, target, idx, contrast_idx = data
                idx, contrast_idx = idx.to(self.device), contrast_idx.to(self.device)
            else:
                data, target = data
                idx, contrast_idx = None, None
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                if self.real_model.requires_target:
                    output = self.model(data, target=target)
                    output, loss = output
                else:
                    extra_info = {}
                    output = self.model(data)
                    if self.xERM:
                        with torch.no_grad():
                            _ = self.balanced_model(data)
                            imbalanced_logits = self.balanced_model.backbone.logits
                            balanced_logits = [imbalanced_logit -  2*torch.log(self.data_loader.prior).cuda() + torch.log(self.valid_data_loader.prior).cuda() for imbalanced_logit in imbalanced_logits]
                    if self.distill:
                        with torch.no_grad():
                            teacher = self.teacher_model(data)
                            if idx is not None: # Contrast
                                extra_info.update({
                                    "idx": idx,
                                    "contrast_idx": contrast_idx
                                })
                            if isinstance(output, dict): # New return that does support DataParallel
                                feat_students = output["feat"]
                                extra_info.update({
                                    "feat_students": feat_students,
                                })
                                if isinstance(teacher, dict):
                                    feat_teachers = teacher["feat"]
                                    extra_info.update({
                                        "feat_teachers": feat_teachers,
                                    })
                            else: # Old return that does not support DataParallel
                                extra_info.update({
                                    "feat_students": self.real_model.backbone.feat,
                                    "feat_teachers": self.teacher_model.backbone.feat,
                                    "feat_students_before_GAP": self.real_model.backbone.feat_before_GAP,
                                    "feat_teachers_before_GAP": self.teacher_model.backbone.feat_before_GAP,
                                })
                        if isinstance(teacher, dict):
                            teacher = teacher["output"]

                    if self.add_extra_info:
                        if isinstance(output, dict):
                            logits = output["logits"]
                            extra_info.update({
                                "logits": logits.transpose(0, 1)
                            })
                        else:
                            extra_info.update({
                                "logits": self.real_model.backbone.logits
                            })

                    if isinstance(output, dict):
                        output = output["output"]

                    if self.distill:
                        loss = self.criterion(student=output, target=target, teacher=teacher, extra_info=extra_info)
                    elif self.xERM:
                        loss = self.criterion(output_logits=output, balanced_logits=balanced_logits, imbalanced_logits=imbalanced_logits, target=target, extra_info=extra_info)
                    elif self.add_extra_info:
                        loss = self.criterion(output_logits=output, target=target, extra_info=extra_info)
                    else:
                        loss = self.criterion(output_logits=output, target=target)

            if not use_fp16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, return_length=True))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} max group LR: {:.6f} min group LR: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    max([param_group['lr'] for param_group in self.optimizer.param_groups]),
                    min([param_group['lr'] for param_group in self.optimizer.param_groups])))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def check_ensemble(self):
        balanced_metrics = MiscMeter()
        imbalanced_metrics = MiscMeter()
        ensemble_metrics = MiscMeter()
        upper_bound_metrics = MiscMeter()

        balanced_total_preds = torch.empty(0, dtype=torch.long).cuda()
        imbalanced_total_preds = torch.empty(0, dtype=torch.long).cuda()
        ensemble_total_preds = torch.empty(0, dtype=torch.long).cuda()
        upper_total_preds = torch.empty(0, dtype=torch.long).cuda()
        total_target = torch.empty(0, dtype=torch.long).cuda()

        alpha = 0.5

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                balanced_output = self.balanced_model(data)
                imbalanced_output = self.imbalanced_model(data)
                ensemble_output = (1- alpha) * balanced_output + alpha * imbalanced_output

                balanced_prob = F.softmax(balanced_output, -1)
                imbalanced_prob = F.softmax(imbalanced_output, -1)

                _, balanced_preds = torch.max(balanced_prob, 1)
                _, imbalanced_preds = torch.max(imbalanced_prob, 1)
                # _, ensemble_preds = torch.max(balanced_prob+imbalanced_prob, 1)
                _, ensemble_preds = torch.max(ensemble_output, 1)

                balanced_right = balanced_preds == target
                imbalanced_right = imbalanced_preds == target

                preds_bound = balanced_preds.clone()
                preds_bound[balanced_right] = balanced_preds[balanced_right]
                preds_bound[imbalanced_right] = imbalanced_preds[imbalanced_right]

                balanced_metrics.update((balanced_preds==target).sum().item()/data.size(0), n=data.size(0))
                imbalanced_metrics.update((imbalanced_preds==target).sum().item()/data.size(0), n=data.size(0))
                ensemble_metrics.update((ensemble_preds==target).sum().item()/data.size(0), n=data.size(0))
                upper_bound_metrics.update((preds_bound==target).sum().item()/data.size(0), n=data.size(0))

                total_target = torch.cat((total_target, target))
                balanced_total_preds = torch.cat((balanced_total_preds, balanced_preds))
                imbalanced_total_preds = torch.cat((imbalanced_total_preds, imbalanced_preds))
                ensemble_total_preds = torch.cat((ensemble_total_preds, ensemble_preds))
                upper_total_preds = torch.cat((upper_total_preds, preds_bound))

        print(f'balanced accuracy:  {balanced_metrics.avg:.3f} '
              f'imblanced accuracy: {imbalanced_metrics.avg:.3f} '
              f'ensemble accuracy: {ensemble_metrics.avg:.3f} '
              f'upper bound accuracy: {upper_bound_metrics.avg:.3f} ')

        many, med, few = shot_acc(balanced_total_preds, total_target, self.data_loader)
        print(f'balanced many: {many:.3f} med: {med:.3f} few: {few:.3f} ')

        many, med, few = shot_acc(imbalanced_total_preds, total_target, self.data_loader)
        print(f'imbalanced many: {many:.3f} med: {med:.3f} few: {few:.3f} ')        

        many, med, few = shot_acc(ensemble_total_preds, total_target, self.data_loader)
        print(f'ensemble many: {many:.3f} med: {med:.3f} few: {few:.3f} ')

        many, med, few = shot_acc(upper_total_preds, total_target, self.data_loader)
        print(f'upper bound many: {many:.3f} med: {med:.3f} few: {few:.3f} ')


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        if self.xERM:
            self.balanced_model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if hasattr(self.model, "confidence_model") and self.model.confidence_model:
                cumulative_sample_num_experts = torch.zeros((self.model.backbone.num_experts, ), device=self.device)
                num_samples = 0
                confidence_model = True
            else:
                confidence_model = False
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                if confidence_model:
                    output, sample_num_experts = self.model(data)
                    num, count = torch.unique(sample_num_experts, return_counts=True)
                    cumulative_sample_num_experts[num - 1] += count
                    num_samples += data.size(0)
                else:
                    output = self.model(data)

                    if self.xERM:
                        imbalanced_logits = self.balanced_model(data)
                        balanced_logits = imbalanced_logits -  2*torch.log(self.data_loader.prior).cuda() + torch.log(self.valid_data_loader.prior).cuda()

                if isinstance(output, dict):
                    output = output["output"]

                # output = output - torch.log(self.data_loader.prior).cuda() + torch.log(self.valid_data_loader.prior).cuda()
                # loss = self.criterion(output, target)
                loss = F.cross_entropy(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                    if self.xERM:
                        self.balanced_metrics.update(met.__name__, met(balanced_logits, target, return_length=True))
                        self.imbalanced_metrics.update(met.__name__, met(imbalanced_logits, target, return_length=True))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if confidence_model:
                print("Samples with num_experts:", *[('%.2f'%item) for item in (cumulative_sample_num_experts * 100 / num_samples).tolist()])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        if self.xERM:
            self.imbalanced_metrics.result()
            self.balanced_metrics.result()
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
