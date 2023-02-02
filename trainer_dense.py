import argparse

import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from utils import *

#Mine
from omegaconf import OmegaConf
from tqdm import tqdm
import warnings
import wandb
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--network', default='split', type=str, help='split, mtan')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert, autol')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--with_noise', action='store_true', help='with noise prediction task')
parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
parser.add_argument('--autol_lr', default=1e-4, type=float, help='learning rate for auto-lambda')
parser.add_argument('--task', default='all', type=str, help='primary tasks, use all for MTL setting')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')

parser.add_argument('--pretrained', action='store_true', help='If pretrained')
parser.add_argument('--checkpoint_path', default='', type=str, help='where the checkpoint is located')


parser.add_argument('--wandbtlogger', default=True, type=bool,help ='use wandb or not')
parser.add_argument('--wandbprojectname', default='taskonomy-autolambda', type=str, help='c')
parser.add_argument('--wandbentity', default='wandbdimar', type=str, help='c')

opt = parser.parse_args()
option_dict = vars(opt)
print(option_dict)
#Initialize weights and biases logger
if opt.task == 'all' or opt.task == 'seg':
    prev_best_test_metrc = -np.inf
else:
    prev_best_test_metrc = np.inf
    
    #prev_best_test_metrc = test_metrc

if opt.wandbtlogger:
    print("Started logging in wandb")
    #wandb_config = OmegaConf.to_container(option_dict, resolve=True, throw_on_missing=True)
    wandb_config = option_dict
    wandb.init(project=opt.wandbprojectname,entity=opt.wandbentity,
        name='{}_{}_{}_{}'.format(opt.dataset,opt.task,opt.weight,opt.network),
        config = wandb_config)

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
if opt.with_noise:
    train_tasks = create_task_flags('all', opt.dataset, with_noise=True)
else:
    train_tasks = create_task_flags('all', opt.dataset, with_noise=False)

pri_tasks = create_task_flags(opt.task, opt.dataset, with_noise=False)

train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
      .format(opt.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.upper()))
print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
      .format(opt.weight.title(), opt.grad_method.upper()))

if opt.network == 'split':
    model = MTLDeepLabv3(train_tasks).to(device)
elif opt.network == 'mtan':
    model = MTANDeepLabv3(train_tasks).to(device)
    
if opt.pretrained == True:
    model.load_state_dict(torch.load(opt.checkpoint_path))

total_epoch = 200

# choose task weighting here
if opt.weight == 'uncert':
    logsigma = torch.tensor([-0.7] * len(train_tasks), requires_grad=True, device=device)
    params = list(model.parameters()) + [logsigma]
    logsigma_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)

if opt.weight in ['dwa', 'equal']:
    T = 2.0  # temperature used in dwa
    lambda_weight = np.ones([total_epoch, len(train_tasks)])
    params = model.parameters()

if opt.weight == 'autol':
    params = model.parameters()
    autol = AutoLambda(model, device, train_tasks, pri_tasks, opt.autol_init)
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = optim.Adam([autol.meta_weights], lr=opt.autol_lr)

optimizer = optim.SGD(params, lr=0.1, weight_decay=1e-4, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

# define dataset
if opt.dataset == 'nyuv2':
    dataset_path = 'dataset/nyuv2'
    train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 4

elif opt.dataset == 'sim_warehouse':
    dataset_path = 'dataset/sim_warehouse'
    train_set = SimWarehouse(root=dataset_path, train=True, augmentation=True)
    test_set = SimWarehouse(root=dataset_path, train=False)
    batch_size = 16
    
elif opt.dataset == 'taskonomy':
    dataset_path = 'dataset/taskonomy'
    train_set = Taskonomy(root=dataset_path, train=True, augmentation=False)
    test_set = Taskonomy(root=dataset_path, train=False)
    batch_size = 8
    
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

# a copy of train_loader with different data order, used for Auto-Lambda meta-update
if opt.weight == 'autol':
    val_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)

# apply gradient methods
if opt.grad_method != 'none':
    rng = np.random.default_rng()
    grad_dims = []
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), len(train_tasks)).to(device)


# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)
train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)
test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset, include_mtl=True)
for index in range(total_epoch):

    # apply Dynamic Weight Average
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[index, :] = 1.0
        else:
            w = []
            for i, t in enumerate(train_tasks):
                w += [train_metric.metric[t][index - 1, 0] / train_metric.metric[t][index - 2, 0]]
            w = torch.softmax(torch.tensor(w) / T, dim=0)
            lambda_weight[index] = len(train_tasks) * w.numpy()

    # iteration for all batches
    model.train()
    train_dataset = iter(train_loader)
    if opt.weight == 'autol':
        val_dataset = iter(val_loader)

    for k in tqdm(range(train_batch)):
        train_data, train_target = next(iter(train_dataset))
        train_data = train_data.to(device)
        train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}

        # update meta-weights with Auto-Lambda
        if opt.weight == 'autol':
            val_data, val_target = next(iter(val_dataset))
            val_data = val_data.to(device)
            val_target = {task_id: val_target[task_id].to(device) for task_id in train_tasks.keys()}

            meta_optimizer.zero_grad()
            autol.unrolled_backward(train_data, train_target, val_data, val_target,
                                    scheduler.get_last_lr()[0], optimizer)
            meta_optimizer.step()

        # update multi-task network parameters with task weights
        optimizer.zero_grad()
        train_pred = model(train_data)
        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        train_loss_tmp = [0] * len(train_tasks)

        if opt.weight in ['equal', 'dwa']:
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]

        if opt.weight == 'uncert':
            train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]

        if opt.weight == 'autol':
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(autol.meta_weights)]

        loss = sum(train_loss_tmp)

        if opt.grad_method == 'none':
            loss.backward()
            optimizer.step()

        # gradient-based methods applied here:
        elif opt.grad_method == "graddrop":
            for i in range(len(train_tasks)):
                train_loss_tmp[i].backward(retain_graph=True)
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()
            g = graddrop(grads)
            overwrite_grad(model, g, grad_dims, len(train_tasks))
            optimizer.step()

        elif opt.grad_method == "pcgrad":
            for i in range(len(train_tasks)):
                train_loss_tmp[i].backward(retain_graph=True)
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()
            g = pcgrad(grads, rng, len(train_tasks))
            overwrite_grad(model, g, grad_dims, len(train_tasks))
            optimizer.step()

        elif opt.grad_method == "cagrad":
            for i in range(len(train_tasks)):
                train_loss_tmp[i].backward(retain_graph=True)
                grad2vec(model, grads, grad_dims, i)
                model.zero_grad_shared_modules()
            g = cagrad(grads, len(train_tasks), 0.4, rescale=1)
            overwrite_grad(model, g, grad_dims, len(train_tasks))
            optimizer.step()

        train_metric.update_metric(train_pred, train_target, train_loss)

    train_str,train_metrc = train_metric.compute_metric()
    #print(f'Train metrc {loss}')
    train_metric.reset()

    # evaluating test data
    model.eval()
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_target = next(iter(test_dataset))
            test_data = test_data.to(device)
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in
                         enumerate(train_tasks)]

            test_metric.update_metric(test_pred, test_target, test_loss)

    test_str,test_metrc = test_metric.compute_metric()
    #print(metrc)
    #print(test_metric.get_best_performance(opt.task))

        
    if test_metrc >= prev_best_test_metrc:
        print(test_metrc,prev_best_test_metrc)
        prev_best_test_metrc = test_metrc
        torch.save(model.state_dict(),'models/{}_{}_{}_{}.pth'.format(opt.dataset,opt.task,opt.weight,opt.grad_method))
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))
    wandb.log({'train_loss': loss, 'test_metrc':test_metrc, 'best_all': test_metric.get_best_performance(opt.task)})

    #print(type(test_metric.get_best_performance(opt.task)),test_metric.get_best_performance(opt.task))

    if opt.weight == 'autol':
        meta_weight_ls[index] = autol.meta_weights.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': meta_weight_ls}

        print(get_weight_str(meta_weight_ls[index], train_tasks))

    if opt.weight in ['dwa', 'equal']:
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': lambda_weight}

        print(get_weight_str(lambda_weight[index], train_tasks))

    if opt.weight == 'uncert':
        logsigma_ls[index] = logsigma.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': logsigma_ls}

        print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), train_tasks))

    np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.npy'
            .format(opt.network, opt.dataset, opt.task, opt.weight, opt.grad_method, opt.seed), dict)
