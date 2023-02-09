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
import yaml
warnings.filterwarnings("ignore")

def train_sweep():
    with wandb.init():
        wb_config = wandb.config
        for k, v in wb_config.items():
            ks = k.split(".")
            opt[ks[0]][ks[1]] = v
        #print(opt)
        training(opt)
      
        
def training(opt):
    #Initialize weights and biases logger
    if not opt.sweep.sweep_t and opt.wandb.t_logger:
        print("Started logging in wandb")
        wandb_config = OmegaConf.to_container(opt, resolve=True, throw_on_missing=True)
        wandb.init(project=opt.wandb.project_name,entity=opt.wandb.entity,
            name='{}_{}_{}_{}_{}'.format(opt.data.dataset,opt.network.task,opt.network.weight,opt.network.archit,opt.network.grad_method),
            config = wandb_config)

    elif opt.sweep.sweep_t:
        print("Started logging in wandb")
        wandb_config = OmegaConf.to_container(opt, resolve=True, throw_on_missing=True)
        wandb.init(config = wandb_config)
    
    if opt.network.task == 'all' or opt.network.task == 'seg':
        prev_best_test_metrc = -np.inf
    else:
        prev_best_test_metrc = np.inf
        

    torch.manual_seed(opt.reprod.o_seed)
    np.random.seed(opt.reprod.o_seed)
    random.seed(opt.reprod.o_seed)

    # create logging folder to store training weights and losses
    if not os.path.exists('logging'):
        os.makedirs('logging')

    # define model, optimiser and scheduler
    device = torch.device("cuda:{}".format(opt.training.gpu) if torch.cuda.is_available() else "cpu")
    if opt.data.with_noise:
        train_tasks = create_task_flags('all', opt.data.dataset, with_noise=True)
    else:
        train_tasks = create_task_flags('all', opt.data.dataset, with_noise=False)

    pri_tasks = create_task_flags(opt.network.task, opt.data.dataset, with_noise=False)

    train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
    pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
    print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
        .format(opt.data.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.archit.upper()))
    print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
        .format(opt.network.weight.title(), opt.network.grad_method.upper()))

    if opt.network.archit == 'split':
        model = MTLDeepLabv3(train_tasks).to(device)
    elif opt.network.archit == 'mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
        
    
    if opt.training.pretrained == True:
        model.load_state_dict(torch.load(opt.training.checkpoint_path))

    total_epoch = opt.training.epochs
    if opt.wandb.t_logger:
        wandb.watch(model, log_freq=100, log_graph=True)
        
    # choose task weighting here
    if opt.network.weight == 'uncert':
        logsigma = torch.tensor([-0.5] * len(train_tasks), requires_grad=True, device=device)
        print(logsigma)
        params = list(model.parameters()) + [logsigma]
        logsigma_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
        print(logsigma_ls[0:3])

    if opt.network.weight in ['dwa', 'equal']:
        T = 2.0  # temperature used in dwa
        lambda_weight = np.ones([total_epoch, len(train_tasks)])
        params = model.parameters()

    if opt.network.weight == 'autol':
        params = model.parameters()
        autol = AutoLambda(model, device, train_tasks, pri_tasks, opt.network.autol_init)
        meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
        meta_optimizer = optim.Adam([autol.meta_weights], lr=opt.network.autol_lr)
    
    learning_rate = opt.training.optim_lr
    
    if opt.training.optim == 'sgd':
        optimizer = optim.SGD(params, lr=learning_rate, weight_decay=opt.training.weight_decay, momentum=0.9)
    elif opt.training.optim == 'adam':
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=opt.training.weight_decay)
    
    #optimizer = optim.SGD(params, lr=0.1, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

    # define dataset
    if opt.data.dataset == 'nyuv2':
        dataset_path = 'dataset/nyuv2'
        train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
        test_set = NYUv2(root=dataset_path, train=False)

    elif opt.data.dataset == 'sim_warehouse':
        dataset_path = 'dataset/sim_warehouse'
        train_set = SimWarehouse(root=dataset_path, train=True, augmentation=True)
        test_set = SimWarehouse(root=dataset_path, train=False)
        
    elif opt.data.dataset == 'taskonomy':
        dataset_path = 'dataset/taskonomy'
        train_set = Taskonomy(root=dataset_path, train=True, augmentation=False)
        test_set = Taskonomy(root=dataset_path, train=False)
        
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=opt.training.batch_size,
        shuffle=True,
        num_workers=4
    )

    # a copy of train_loader with different data order, used for Auto-Lambda meta-update
    if opt.network.weight == 'autol':
        val_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=opt.training.batch_size,
            shuffle=True,
            num_workers=4
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=opt.training.batch_size,
        shuffle=False
    )

    # apply gradient methods
    if opt.network.grad_method != 'none':
        rng = np.random.default_rng()
        grad_dims = []
        for mm in model.shared_modules():
            for param in mm.parameters():
                grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), len(train_tasks)).to(device)


    # Train and evaluate multi-task network
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    train_metric = TaskMetric(train_tasks, pri_tasks, opt.training.batch_size, total_epoch, opt.data.dataset)
    test_metric = TaskMetric(train_tasks, pri_tasks, opt.training.batch_size, total_epoch, opt.data.dataset, include_mtl=True)
    for index in range(total_epoch):
        # apply Dynamic Weight Average
        if opt.network.weight == 'dwa':
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
        if opt.network.weight == 'autol':
            val_dataset = iter(val_loader)

        for k in tqdm(range(train_batch)):
   
            train_data, train_target = next(iter(train_dataset))
            train_data = train_data.to(device)
            train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}

            # update meta-weights with Auto-Lambda
            if opt.network.weight == 'autol':
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

            #train_loss_tmp = [0] * len(train_tasks)
            wandb_loss,wandb_weights,weight_list,train_loss_tmp =  {},{},[],[]

            #print(f'dict of wandb_loss_prev {wandb_loss}')    
            if opt.network.weight in ['equal', 'dwa']:
                for i, w in enumerate(lambda_weight[index]):
                    if i == 20:
                        train_loss_tmp.append(w * train_loss[i] / 100) 
                    else:
                        train_loss_tmp.append(w * train_loss[i]) 

                    weight_list.append(w)

            if opt.network.weight == 'uncert':
                for i, w in enumerate(logsigma):
                    if i == 20:
                        train_loss_tmp.append((1 / (2 * torch.exp(w)) * train_loss[i] + w / 2) /  100)
                    else:
                        train_loss_tmp.append(1 / (2 * torch.exp(w)) * train_loss[i] + w / 2) 
                    weight_list.append(w)

            if opt.network.weight == 'autol':
                for i, w in enumerate(autol.meta_weights):
                    if i == 20:
                        train_loss_tmp.append(w * train_loss[i] / 100 )

                    else:
                        train_loss_tmp.append(w * train_loss[i])
                    weight_list.append(w)

            wandb_loss = {}
            for i, task_id in enumerate(train_tasks):
                wandb_loss[task_id] = {'bef': train_loss[i].item() ,'aft':train_loss_tmp[i].item()}

                wandb_weights[task_id] = weight_list[i].item()
                                
            loss = sum(train_loss_tmp)

            if opt.network.grad_method == 'none':
                loss.backward()
                optimizer.step()

            # gradient-based methods applied here:
            elif opt.network.grad_method == "graddrop":
                for i in range(len(train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = graddrop(grads)
                overwrite_grad(model, g, grad_dims, len(train_tasks))
                optimizer.step()

            elif opt.network.grad_method == "pcgrad":
                for i in range(len(train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = pcgrad(grads, rng, len(train_tasks))
                overwrite_grad(model, g, grad_dims, len(train_tasks))
                optimizer.step()

            elif opt.network.grad_method == "cagrad":
                for i in range(len(train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
                g = cagrad(grads, len(train_tasks), 0.4, rescale=1)
                overwrite_grad(model, g, grad_dims, len(train_tasks))
                optimizer.step()

            train_metric.update_metric(train_pred, train_target, train_loss)
            
        train_str,train_metrc = train_metric.compute_metric()
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
            torch.save(model.state_dict(),'models/{}_{}_{}_{}.pth'.format(opt.data.dataset,opt.network.task,opt.network.weight,opt.network.grad_method))
        test_metric.reset()

        scheduler.step()

        print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
            .format(index, train_str, test_str, opt.network.task.title(), test_metric.get_best_performance(opt.network.task)))
        
        if opt.wandb.t_logger:
            print(wandb_loss)
            wandb.log({'train_loss': loss, 'test_metrc':test_metrc, 'best_all': test_metric.get_best_performance(opt.network.task),
                'weights' : wandb_weights, 'task_loss': wandb_loss}, step=index)#, 'weighted_loss' : wandb_loss_after})
            
        #print(type(test_metric.get_best_performance(opt.task)),test_metric.get_best_performance(opt.task))

        if opt.network.weight == 'autol':
            meta_weight_ls[index] = autol.meta_weights.detach().cpu()
            dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                    'weight': meta_weight_ls}

            print(get_weight_str(meta_weight_ls[index], train_tasks))

        if opt.network.weight in ['dwa', 'equal']:
            dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                    'weight': lambda_weight}

            print(get_weight_str(lambda_weight[index], train_tasks))

        if opt.network.weight == 'uncert':
            logsigma_ls[index] = logsigma.detach().cpu()
            dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                    'weight': logsigma_ls}

            print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), train_tasks))

        np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.npy'
                .format(opt.network.archit, opt.data.dataset, opt.network.task, opt.network.weight, opt.network.grad_method, opt.reprod.o_seed), dict)


if __name__ == "__main__":

    opt = OmegaConf.load('configs/trainer.yaml')

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    
    if opt.reprod.t_seed:
        seed = opt.reprod.o_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    if opt.sweep.sweep_t:
        sweep_config = yaml.load(
            open(opt.sweep.sweep_o, "r"), Loader=yaml.FullLoader
        )
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project="MTL-warehouse-Sweep",
            entity=opt.wandb.entity,
        )
        wandb.agent(sweep_id, function=train_sweep, count=24)
        
    else:
        training(opt)