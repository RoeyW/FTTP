import torch
from utils import args_parser
from update import LocalTestTune,GlobalUpdate
from data_loader import DataLoaderfromfiles,test4test_generator
import copy
import numpy as np
import pandas as pd
import os
import datetime
from model import HAR_lstm

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

np.random.seed(1)

# set the parameters
dataset_name = 'pamap'
args = args_parser(dataset_name)
# load the global model and test on the new user

setting_config  = current_time+ dataset_name+'-'+args.model_name+'-memo-'+'lr'+str(args.lr)+'beta'+str(args.beta) +'th'+str(args.th) +'wholeopdate' + str(args.whole_update)
from torch.utils.tensorboard import SummaryWriter
logger = SummaryWriter(comment=setting_config)

PATH = 'savemodels/'+args.model_name+'/'+dataset_name+'/global.pt' #global model path
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

global_model_parameter = torch.load(PATH,map_location=device)
global_model = HAR_lstm(args.input_dim, args).to(device)
global_model.load_state_dict(global_model_parameter)
global_update = GlobalUpdate(args)

test_loader_list = []

for id in range(args.test_users_num):
    test_loader_list.append(DataLoaderfromfiles(train=False,args=args,id=id))


if dataset_name == 'pamap':
    comm_round = 50
else:
    comm_round = 150

upload_times = 0.
feedback_times=0.
for test_r in range(1,comm_round):
    
    print('communication round', test_r)
    # one user uploads
    id = np.random.randint(low=0,high= args.test_users_num)
    # id=18
    

    # ==============htta communication cost===============
    uploadornot = False
    
    low_confidence_list_x = []
    low_confidence_list_y = []
    # load test data
    # test_loader = DataLoaderfromfiles(train=False,args=args,id=id)
    
    generator = test_loader_list[id].test_generator()

    # find the local model for each user
    # if id < args.num_users:
    #     local_path = 'savemodels/'+args.model_name+'/'+dataset_name+'/local'+str(id)+'.pt'
    #     local_model_parameter = torch.load(local_path,map_location=device)
    #     local_model = HAR_lstm(args.input_dim, args)
    #     local_model.load_state_dict(local_model_parameter)
    #     local_model.to(device)
    # else:
    #     local_model = copy.deepcopy(global_model)

    try:
        if dataset_name =='harthsp':
            local_path = 'savemodels/'+args.model_name+'/'+dataset_name+'/local'+str(id+15)+'.pt'
        else:
            local_path = 'savemodels/'+args.model_name+'/'+dataset_name+'/local'+str(id)+'.pt'
        print('CLIENT ID--',id)
        local_model_parameter = torch.load(local_path,map_location=device)
        local_model = HAR_lstm(args.input_dim, args)
        local_model.load_state_dict(local_model_parameter)
        local_model.to(device)
    except FileNotFoundError as e:
        # new user
        local_model = copy.deepcopy(global_model)


    new_local_model = copy.deepcopy(local_model)

    while True:
        try:
            x, y = next(generator)
            x = torch.Tensor(x).to(device)
            y = torch.tensor(y,dtype=torch.int64).to(device)
            predict_y = local_model(x)
            predict_y = torch.softmax(predict_y,dim=-1)
            prob, pseudo_label = predict_y.max(1)
            # predict_y_new = predict_y-prob.repeat(1,predict_y.shape[1])
            # _, candidate_label = predict_y_new.max(1)
            print("truthy:",y,"predict_y",pseudo_label)
            # if predict is low confidence
            if prob<args.e:
                low_confidence_list_x.append(x)
                low_confidence_list_y.append(pseudo_label)
                print('logits:',predict_y)
            
            # if the number of low confidence sample exceeds th, get a human feedback
            if len(low_confidence_list_x) > args.th:
                print('===human feedback success====')
                print(low_confidence_list_y)
                feedback_times+=1
                # h_x, h_y = next(generator)
                maxTimes_label = max(set(low_confidence_list_y),key=low_confidence_list_y.count)
                for i in low_confidence_list_y:
                    if i !=maxTimes_label: 
                        alter_label = i
                        break
                # if methods do not use human feedback
                # h_x, h_y = test_loader_list[id].feedback_data([maxTimes_label,alter_label])
                # h_x = torch.Tensor(h_x).to(device)
                # h_y = torch.tensor(h_y,dtype=torch.int64).to(device)
                # low_confidence_list_x.append(h_x)
                # low_confidence_list_y.append(h_y)
                local_update = LocalTestTune(args)

                
                

                if args.model_name =='fedavg' or 'fedprox' or 'ditto':
                    # new_local_model, loss,uploadornot = local_update.update_weights(model=local_model,data=low_confidence_list_x,label=low_confidence_list_y,uploadornot=uploadornot)
                    new_local_model, loss, uploadornot = local_update.update_weights_select_samples(model=new_local_model,data=low_confidence_list_x,label=low_confidence_list_y,uploadornot=uploadornot)
                # elif args.model_name == 'fedprox':
                #     new_local_model, loss,uploadornot = local_update.update_weights_prox(global_model=global_model,model=new_local_model,data=low_confidence_list_x,label=low_confidence_list_y,uploadornot=uploadornot)
                # # update the local model file
                if dataset_name =='harthsp':
                    LOCAL_PATH = 'savemodels/'+args.model_name+'/'+args.dataset+'/local'+str(id+15)+'.pt'
                else:
                    LOCAL_PATH = 'savemodels/'+args.model_name+'/'+args.dataset+'/local'+str(id)+'.pt'
                torch.save(new_local_model.state_dict(), LOCAL_PATH)
                low_confidence_list_x =[]
                low_confidence_list_y =[]

                # uploadornot = False
                if uploadornot:
                    upload_times+=1
                    print('=======',str(id),'=====upload sucess====')
                    if args.whole_update:
                        global_weights = global_update.smoothaggregate(new_local_model.state_dict(),global_model.state_dict(),id)
                    else:
                        w = new_local_model.embedding.state_dict().update(new_local_model.lstm_layer.state_dict()) #embedding layers
                        global_weights = global_update.smoothaggregate(w,global_model.state_dict(),id)
                    
                    # # update global model file
                    GLOBAL_PATH = 'savemodels/'+ args.model_name+'/'+dataset_name+'/global.pt' 
                    torch.save(global_weights, GLOBAL_PATH)

                    # # # update the local model file
                    # PATH = 'savemodels/'+args.model_name+'/'+args.dataset+'/local'+str(id)+'.pt'
                    # torch.save(global_weights, PATH)
                    
                    global_model.load_state_dict(global_weights)
                    
                break
            

        except StopIteration as e:
            break
    
    

    # local validation after htta   local validation stride = window size
    vali_generator = test4test_generator(args,test_loader_list[id].file_id)
    criterion =torch.nn.CrossEntropyLoss()
    correct = 0
    local_loss = 0.
    local_sample_num = 0.
    while True:
        try:
            x, y = next(vali_generator)
            test_x = torch.Tensor(x).to(device)
            test_y = torch.tensor(y,dtype=torch.int64).to(device)
            predict_y = new_local_model(test_x)
            local_loss += criterion(predict_y,test_y)

            _,predict_class = predict_y.max(1)
            correct += predict_class.eq(test_y).sum().item()
            local_sample_num += 1
        except StopIteration as e:
            break
    local_acc = correct / local_sample_num
    local_loss /= local_sample_num

    # global validation after htta 
    global_vali_generator = test4test_generator(args,'global.csv')
    correct =0.
    global_loss = 0.
    global_sample_num = 0.
    while True:
        try:
            x,y = next(global_vali_generator)
            test_x = torch.Tensor(x).to(device)
            test_y = torch.tensor(y,dtype=torch.int64).to(device)
            global_predict_logit = global_model(test_x)
            loss = criterion(global_predict_logit,test_y)
            global_loss += loss

            _,global_predict_class = global_predict_logit.max(1)
            correct += global_predict_class.eq(test_y).sum().item()
            global_sample_num +=1
        except StopIteration as e:
            break
    global_acc = correct/global_sample_num
    global_loss /=global_sample_num

    
    
    # record
    if dataset_name == 'harthsp':
        logger.add_scalar(tag=str(id+15),scalar_value=local_acc,global_step=test_r)
        tag_name = str(id+15)+'_loss'
    else:
        logger.add_scalar(tag=str(id),scalar_value=local_acc,global_step=test_r)
        tag_name = str(id)+'_loss'
    logger.add_scalar(tag=tag_name,scalar_value=local_loss,global_step=test_r)
    logger.add_scalar(tag='global',scalar_value=global_acc,global_step=test_r)
    logger.add_scalar(tag='global_loss',scalar_value=global_loss,global_step=test_r)

print('--feedbacktimes',feedback_times)
print('--uploadtimes',upload_times)
    


    









