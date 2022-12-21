import time

import wandb as wandb

from utils.setting import Setting
from utils.trainer import Trainer


def train():
    args, logger = Setting().run()

    trainer = Trainer(args, logger)

    if args.wandb:
        name = f'{args.output_path.split("/")[-1]}'
        wandb.init(project='NSMC Dual Contrastive Learning',
                   name=name,
                   config=vars(args))

    for epoch in range(args.epochs):
        logger.info(f'Start Training Epoch {epoch + 1}')

        start_time = time.time()  # 시작 시간 기록

        trainer.train_epoch(epoch)

        logger.info(f'Finish Training Epoch {epoch + 1}')

        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        logger.info(f'Time Spent on Training (per epoch) : {elapsed_mins} minutes {elapsed_secs} seconds')

        if trainer.is_early_stopping:
            trainer.logger.info(f'Early Stopping, best_valid_accuracy: {trainer.best_score:.4f}')
            return

    logger.info('Training Finished')


if __name__ == '__main__':
    train()
