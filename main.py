import time
from src import utils
import torch
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, 'train')
    num_steps_per_epoch = len(train_loader)

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(opt, optimizer, epoch)  # update the learning rate

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            outputs = model(inputs, labels)
            outputs['loss'].backward()

            optimizer.step()

            train_results = utils.log_results(train_results, outputs, num_steps_per_epoch)
        
        utils.print_results('train', time.time() - start_time, train_results, epoch)
        start_time = time.time()

        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, 'val', epoch=epoch)
    
    return model


def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)
            outputs = model.forward_downstream_classification_model(inputs, labels)
            test_results = utils.log_results(test_results, outputs, num_steps_per_epoch)
    
    utils.print_results(partition, time.time() - test_time, test_results, epoch)
    model.train()


def main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    validate_or_test(opt, model, 'val')

    if opt.training.final_test:
        validate_or_test(opt, model, 'test')


if __name__ == '__main__':
    config_path = 'config.yaml'
    opt = OmegaConf.load(config_path)
    main(opt)