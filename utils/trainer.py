import os
import shutil
import time

import wandb
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

from utils.data_utils import load_dataloader
from utils.loss import DualCLLoss, CELoss, SupConLoss
from utils.models import EncoderModel


class Trainer(object):

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.model = self._get_model()
        self.tokenizer = self._get_tokenizer()

        start_time = time.time()
        dataloader = load_dataloader(args, self.tokenizer)
        self.train_dataloader = dataloader['train']
        self.valid_dataloader = dataloader['valid']
        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        logger.info(f'Time Spent on Making DataLoader : {elapsed_mins} minutes {elapsed_secs} seconds')

        self.step_per_epoch = len(self.train_dataloader)

        self.loss_fn = self._get_loss_fn()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()

        self.best_score = 0
        self.best_model_folder = None

        self.early_stopping_counter = 0
        self.is_early_stopping = False

    def train_epoch(self, epoch):
        self.model.train()

        self.optimizer.zero_grad()

        train_iterator = tqdm(self.train_dataloader, total=self.step_per_epoch, desc='Train Iteration')
        for step, (batch, label) in enumerate(train_iterator):
            total_step = epoch * self.step_per_epoch + step

            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            targets = label.to(self.args.device)

            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                outputs = self.model(batch)

            loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()

            preds = torch.argmax(outputs['predicts'], dim=-1)
            acc = accuracy_score(targets.cpu(), preds.cpu())

            if (total_step + 1) % self.args.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            self.scheduler.step()

            self.train_loss.update(loss.item(), self.args.train_batch_size)
            self.train_acc.update(acc, self.args.train_batch_size)

            if total_step != 0 and total_step % (self.args.eval_steps * self.args.accumulation_steps) == 0:
                valid_acc, valid_loss = self.validate()

                self.model.train()
                current_lr = self.scheduler.optimizer.param_groups[0]['lr']
                if self.args.wandb:
                    wandb.log({
                        'train/loss': self.train_loss.avg,
                        'train/acc': self.train_acc.avg,
                        'train/learning_rate': current_lr,
                        'eval/loss': valid_loss,
                        'eval/acc': valid_acc,
                        'eval/best_score': valid_acc if valid_acc > self.best_score else self.best_score,
                    })

                self.logger.info(
                    f'STEP {total_step} | train loss: {self.train_loss.avg:.4f} | train acc: {self.train_acc.avg:.4f} | lr: {current_lr}'
                )
                self.logger.info(
                    f'STEP {total_step} | eval loss: {valid_loss:.4f} | eval acc: {valid_acc:.4f}'
                )

                if valid_acc > self.best_score:
                    self.logger.info(f'BEST_BEFORE : {self.best_score:.4f}, NOW : {valid_acc:.4f}')
                    self.logger.info('Saving Model...')
                    self.best_score = valid_acc
                    self.early_stopping_counter = 0
                    self.save_model(total_step)
                else:
                    self.early_stopping_counter += 1
                    if self.args.patience < self.early_stopping_counter:
                        self.is_early_stopping = True
                        return

    def validate(self):
        self.model.eval()

        valid_acc = AverageMeter()
        valid_loss = AverageMeter()

        valid_iterator = tqdm(self.valid_dataloader, desc="Valid Iteration")

        with torch.no_grad():
            for step, (batch, label) in enumerate(valid_iterator):
                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                targets = label.to(self.args.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch)

                loss = self.loss_fn(outputs, targets)

                preds = torch.argmax(outputs['predicts'], dim=-1)
                acc = accuracy_score(targets.cpu(), preds.cpu())

                valid_loss.update(loss.item(), self.args.valid_batch_size)
                valid_acc.update(acc, self.args.valid_batch_size)

        return valid_acc.avg, valid_loss.avg

    def _get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)

        return optimizer

    def _get_scheduler(self):
        t_total = self.step_per_epoch * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer, round(self.args.warmup_proportion * t_total), t_total)

        return scheduler

    def _get_loss_fn(self):
        if self.args.method == 'ce':
            loss_fn = CELoss()
        elif self.args.method == 'scl':
            loss_fn = SupConLoss(alpha=self.args.alpha, temp=self.args.temp)
        elif self.args.method == 'dualcl':
            loss_fn = DualCLLoss(alpha=self.args.alpha, temp=self.args.temp)
        else:
            raise ValueError('Unknown loss (%s)' % self.args.loss_fn)

        return loss_fn

    def _get_model(self):
        base_model = RobertaModel.from_pretrained(self.args.pretrained_model_path)
        model = EncoderModel(self.args, base_model)

        if self.args.saved_model_state_path is not None:
            print('Loading Model')
            load_state = torch.load(self.args.saved_model_state_path, map_location=torch.device('cpu'))
            model.load_state_dict(load_state['model_state_dict'], strict=True)

        model.to(self.args.device)

        if 'cuda' in self.args.device.type:
            self.logger.info(
                '> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.args.device.index)))

        return model

    def _get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_tokenizer_path,
                                                  do_lower_case=False,
                                                  unk_token='<unk>',
                                                  sep_token='</s>',
                                                  pad_token='<pad>',
                                                  cls_token='<s>',
                                                  mask_token='<mask>',
                                                  model_max_length=self.args.max_len)
        # if self.args.method == 'dualcl':
            # special_tokens_dict = {'additional_special_tokens': ['[NEG]', '[POS]']}
            # num_added_toks = tokenizer.add_special_tokens((special_tokens_dict))
            # self.logger.info(f'num_added_tokens: {num_added_toks}')
            # self.model.base_model.resize_token_embeddings(len(tokenizer))

        return tokenizer

    def save_model(self, step):
        if self.best_model_folder:
            shutil.rmtree(self.best_model_folder)

        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path, exist_ok=True)

        file_name = f'STEP_{step}_{self.args.method}_ACC{self.best_score:.4f}'
        output_path = os.path.join(self.args.output_path, file_name)

        os.makedirs(output_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(output_path, 'model_state_dict.pt'))

        self.logger.info(f'Model Saved at {output_path}')
        self.best_model_folder = output_path


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
