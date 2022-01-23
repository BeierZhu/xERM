import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    train_loader = config.init_obj('data_loader', module_data)
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        training=False,
        num_workers=2
    )
    train_loader = config.init_obj('data_loader', module_data)

    # build model architecture
    if 'returns_feat' in config['arch']['args']:
        model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=False)
    else:
        model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = config.init_obj('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    num_classes = config._config["arch"]["args"]["num_classes"]
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()

    if hasattr(model, "confidence_model") and model.confidence_model:
        cumulative_sample_num_experts = torch.zeros((model.backbone.num_experts, ), dtype=torch.float, device=device)
        cumulative_sample_num_experts_each_shot = [torch.zeros((model.backbone.num_experts, ), dtype=torch.float, device=device) for _ in range(3)]
        num_samples = 0
        confidence_model = True
    else:
        cumulative_sample_num_experts = None
        cumulative_sample_num_experts_each_shot = None
        confidence_model = False

    get_class_acc = True
    if get_class_acc:
        train_data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=256,
            training=True
        )
        test_cls_num_list = np.array(data_loader.cls_num_list)
        train_cls_num_list = np.array(train_data_loader.cls_num_list)
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        few_shot = train_cls_num_list < 20

    total_preds = torch.empty(0, dtype=torch.long).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            if confidence_model:
                output, sample_num_experts = model(data)
                num, count = torch.unique(sample_num_experts, return_counts=True)
                cumulative_sample_num_experts[num - 1] += count.type(torch.float)
                num_samples += data.size(0)

                many_shot_tensor = torch.tensor(many_shot, device=device)
                medium_shot_tensor = torch.tensor(medium_shot, device=device)
                few_shot_tensor = torch.tensor(few_shot, device=device)

                for i, mask_shot in enumerate([many_shot_tensor, medium_shot_tensor, few_shot_tensor]):
                    num, count = torch.unique(sample_num_experts[mask_shot[target]], return_counts=True)
                    (cumulative_sample_num_experts_each_shot[i])[num - 1] += count.float()
            else:
                output = model(data)

            if config['PC_eval']:
                output = output - 2*torch.log(train_loader.prior).cuda() + torch.log(data_loader.prior).cuda()

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set

            # loss = loss_fn(output, target)
            batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size

            _, preds = torch.max(output, 1)
            total_preds = torch.cat((total_preds, preds))
            total_labels = torch.cat((total_labels, target))
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

            for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    cls_accs = shot_acc(total_preds, total_labels, train_loader, acc_per_cls=False)
    print(cls_accs)
    cls_accs = shot_precision(total_preds, total_labels, train_loader, acc_per_cls=False)
    print(cls_accs)

    if confidence_model:
        print("Samples with num_experts:", *[('%.2f'%item) for item in (cumulative_sample_num_experts * 100 / num_samples).tolist()])
        print({"many_hp_num": (cumulative_sample_num_experts_each_shot[0]/cumulative_sample_num_experts_each_shot[0].sum()).cpu().tolist(),
            "medium_hp_num": (cumulative_sample_num_experts_each_shot[1]/cumulative_sample_num_experts_each_shot[1].sum()).cpu().tolist(),
            "few_hp_num": (cumulative_sample_num_experts_each_shot[2]/cumulative_sample_num_experts_each_shot[2].sum()).cpu().tolist()})

    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
    
    acc = acc_per_class.cpu().numpy()
    print('Acc for each class: \n', acc)

    print('Acc with class mean:', acc_per_class.mean().item())

    # np.save("test_acc.npy", acc)

    if get_class_acc:
        # Here we assume each class has same number of instances
        assert np.all(test_cls_num_list == test_cls_num_list[0])

        many_shot_acc = acc[many_shot].mean()
        medium_shot_acc = acc[medium_shot].mean()
        few_shot_acc = acc[few_shot].mean()
        print("{}, {}, {}".format(np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2)))

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })

    if get_class_acc:
        log.update({
            "many_class_num": many_shot.sum(),
            "medium_class_num": medium_shot.sum(),
            "few_class_num": few_shot.sum(),
            "many_shot_acc": many_shot_acc,
            "medium_shot_acc": medium_shot_acc,
            "few_shot_acc": few_shot_acc,
        })
    logger.info(log)

def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.targets).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    overall_acc = (preds==labels).sum()/len(preds)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return overall_acc, np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return overall_acc, np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def shot_precision(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.targets).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        num_pred = (preds == l).sum()
        test_class_count.append(num_pred)
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            if test_class_count[i] != 0: 
                many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            if test_class_count[i] != 0: 
                low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            if test_class_count[i] != 0: 
                median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
