import argparse
import torch.optim as optim
import torch.utils.data.sampler as sampler
from create_network import *
from create_dataset import *

from utils import *

#Mine
from networks.ddrnet import DualResNetMTL,BasicBlock
from networks.guidedepth import GuideDepth
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb
import warnings
import yaml
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Single-task Learning: Dense Prediction Tasks')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--network', default='ddrnetsingle', type=str, help='split, mtan, ddrnetsingle,guidedepth')
parser.add_argument('--dataset', default='sim_warehouse', type=str, help='nyuv2, cityscapes')
parser.add_argument('--task', default='normal', type=str, help='choose task for single task learning')
parser.add_argument('--seed', default=29, type=int, help='gpu ID')

parser.add_argument('--pretrained', action='store_true', help='If pretrained')
parser.add_argument('--checkpoint_path', default='', type=str, help='where the checkpoint is located')

parser.add_argument('--wandbtlogger', default=True, type=bool,help ='use wandb or not')
parser.add_argument('--wandbprojectname', default='mtl-WarehouseSIM', type=str, help='c')
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

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
train_tasks = create_task_flags(opt.task, opt.dataset)

print('Training Task: {} - {} in Single Task Learning Mode with {}'
      .format(opt.dataset.title(), opt.task.title(), opt.network.upper()))

total_epoch = 200

if opt.network == 'split':
    model = MTLDeepLabv3(train_tasks).to(device)
    network = opt.network

    #optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4, momentum=0.9)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
elif opt.network == 'mtan':
    network = opt.network

    model = MTANDeepLabv3(train_tasks).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4, momentum=0.9)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
elif opt.network == 'ddrnetsingle':
    variant = '23s' #23s,23,39
    if variant == '23s':
        model = DualResNetMTL(BasicBlock, [2, 2, 2, 2], train_tasks, opt.dataset, planes=32, spp_planes=128, head_planes=64).to(device)
    elif variant == '23':
        model = DualResNetMTL(BasicBlock, [2, 2, 2, 2], train_tasks, opt.dataset, planes=64, spp_planes=128, head_planes=128).to(device)

    elif variant == '39':
        model = DualResNetMTL(BasicBlock, [3, 4, 6, 3], train_tasks, opt.dataset, planes=64, spp_planes=128, head_planes=256, augment=False).to(device)
    network = 'ddrnetsingle_{}'.format(variant)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60,80],gamma=0.1)
elif opt.network == 'guidedepth':
    model = GuideDepth(train_tasks).to(device) 
    network = opt.network
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
    


if opt.wandbtlogger:
    print("Started logging in wandb")
    #wandb_config = OmegaConf.to_container(opt, resolve=True, throw_on_missing=True)
    wandb_config = option_dict
    wandb.init(project=opt.wandbprojectname,entity=opt.wandbentity,
        name='{}_{}_{}'.format(opt.dataset,opt.task,network),
        config = wandb_config)




#optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # Just winging this one, 
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
    batch_size = 8
    
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

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=1,
    shuffle=False,
    num_workers=4
)


# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)
train_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, opt.dataset, opt.network)
test_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, opt.dataset,opt.network)

for index in range(total_epoch):

    # evaluating train data
    model.train()
    train_dataset = iter(train_loader)

    for k in tqdm(range(train_batch)):
        train_data, train_target = next(iter(train_dataset))
        train_data = train_data.to(device)
        train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}
        train_pred = model(train_data)
        optimizer.zero_grad()

        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        train_loss[0].backward()
        optimizer.step()

        train_metric.update_metric(train_pred, train_target, train_loss)

    train_str,train_metrc = train_metric.compute_metric()
    train_metric.reset()


    selection_list = [50,59,68,77,150,250] #keep 50,150,250
    test_pred_list =[]
    test_data_list = []
    test_target_list = []

    # evaluating test data
    model.eval()
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_target = next(iter(test_dataset))
            test_data = test_data.to(device)
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(test_data)
                
            if k in selection_list:
                test_pred_list.append(test_pred)
                test_data_list.append(test_data)
                test_target_list.append(test_target)
                
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

            test_metric.update_metric(test_pred, test_target, test_loss)

    test_str,test_metrc = test_metric.compute_metric()
    test_metrc = test_metrc[opt.task]
    
    
    if index == 0: 
        from datetime import datetime
        print(test_data.shape)
        import visualizer
        folder_name = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
        
        
        os.mkdir(f'results/{folder_name}')
        os.mkdir(f'results/{folder_name}/models')
        folder = f'results/{folder_name}/models'

        for i in range(len(test_pred_list)):
            test_data = test_data_list[i]  
            test_target = test_target_list[i]    
            im_rgb = Image.fromarray(visualizer.rgb_visualizer(test_data.detach().cpu().squeeze().numpy()))
            im_rgb.save(f'results/{folder_name}/rgb_im{selection_list[i]}.png')
            if opt.task == 'seg':
                im_s = visualizer.semantic_colorize(test_target['seg'].detach().cpu().numpy())
                im_s.save(f'results/{folder_name}/semantic_im{selection_list[i]}.png')         
            elif opt.task == 'depth':
                im_d = Image.fromarray(visualizer.depth_colorize(test_target['depth'].detach().cpu().squeeze().numpy()))
                im_d.save(f'results/{folder_name}/depth_im{selection_list[i]}.png')
            elif opt.task == 'normal':
                im_n = Image.fromarray(visualizer.normal_colorize(test_target['normal'].detach().cpu().squeeze().numpy()))
                im_n.save(f'results/{folder_name}/normal_im{selection_list[i]}.png')

    

    if opt.task == 'seg':
        if test_metrc >= prev_best_test_metrc:
            for i in range(len(test_pred_list)):
                test_pred = test_pred_list[i]      
                
                im_s = visualizer.semantic_colorize(test_pred[0].detach().cpu().squeeze().numpy())
                im_s.save(f'results/{folder_name}/semantic_im{selection_list[i]}_e{index}.png')            
            
            prev_best_test_metrc = test_metrc
            torch.save(model.state_dict(),'{}/{}_{}_{}.pth'.format(folder, opt.dataset,opt.task, index))
        
    else: 
        if test_metrc <= prev_best_test_metrc:
            #print(test_metrc,prev_best_test_metrc)
            for i in range(len(test_pred_list)):
                test_pred = test_pred_list[i]      
                
                if opt.task == 'depth':
                    im_d = Image.fromarray(visualizer.depth_colorize(test_pred[0].detach().cpu().squeeze().numpy()))
                    im_d.save(f'results/{folder_name}/depth_im{selection_list[i]}_e{index}.png')
                elif opt.task == 'normal':
                    im_n = Image.fromarray(visualizer.normal_colorize(test_pred[0].detach().cpu().squeeze().numpy()))
                    im_n.save(f'results/{folder_name}/normals_im{selection_list[i]}_e{index}.png')

            prev_best_test_metrc = test_metrc
            torch.save(model.state_dict(),'{}/{}_{}_{}.pth'.format(str(folder), opt.dataset,opt.task, index))
            
                
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))
    
    wandb.log({'train_loss': train_loss[0], 'test_metrc':test_metrc, 'best_all': test_metric.get_best_performance(opt.task)})

    task_dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric}
    np.save('logging/stl_{}_{}_{}_{}.npy'.format(opt.network, opt.dataset, opt.task, opt.seed), task_dict)





