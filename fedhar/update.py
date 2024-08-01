import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from data_loader import DataLoaderfromfiles
from scipy.interpolate import CubicSpline


class LocalUpdate(object):
    def __init__(self, args, idxs):
      self.args = args
      self.device = torch.device(args.device)
      if args.lossfn == 'NLLLoss':
        self.criterion = nn.NLLLoss()
      if args.lossfn == 'CrossEntropyLoss':
        self.criterion = nn.CrossEntropyLoss()
      self.dataloader = DataLoaderfromfiles(id=idxs,window_size=args.window_size,args=args)
      self.id = idxs

    def update_weights(self, model, global_round):
        global_model = model
        model.train()
        epoch_loss = []
      
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.wd)
            
        # dataset generator
        generator = self.dataloader.train_generator()
        for epoch in range(1, self.args.local_ep+1):
            train_loss = 0.0
            correct_train = 0.0
            total=0
            batch_loss = []
            total,correct,train_loss = 0,0,0
            for batch_num in range(self.args.batchsum):
                x,y = next(generator)
                x_g = torch.Tensor(x).to(self.device)
                y_g = torch.tensor(y,dtype=torch.int64).to(self.device)

                optimizer.zero_grad()
                logits = model(x_g) # predict y
                loss = self.criterion(logits,y_g)
                if self.args.model_name == 'fedprox':
                  proximal_term =0.
                  for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                  loss += (0.01 / 2) * proximal_term

                train_loss += loss
                loss.backward()
                optimizer.step()

                _, predicted = logits.max(1)
                total += y_g.size(0)
                correct += predicted.eq(y_g).sum().item()

            correct_train = correct/total
            train_loss /= self.args.batchsum

            epoch_loss.append(train_loss)
            print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f} | Accuracy: {:.2f}%'.format(
                    global_round, epoch, train_loss, correct_train*100))
                
      
        print('| Global Round : {} |\tLoss avg: {:.6f} '.format(
                    global_round, sum(epoch_loss) / len(epoch_loss)))
        
        # record each local model
        if global_round ==self.args.epochs:
            PATH = 'savemodels/'+self.args.model_name+'/'+self.args.dataset+'/local'+str(self.id)+'.pt'
            torch.save(model.state_dict(), PATH)
            
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_ditto(self, global_model, model, global_round):
        global_model.train()
        model.train()
        epoch_loss = []
      
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.wd)
            
        # dataset generator
        generator = self.dataloader.train_generator()
        for epoch in range(1, self.args.local_ep+1):
            train_loss = 0.0
            correct_train = 0.0
            total=0
            batch_loss = []
            total,correct,train_loss = 0,0,0
            for batch_num in range(self.args.batchsum):
                x,y = next(generator)
                x_g = torch.Tensor(x).to(self.device)
                y_g = torch.tensor(y,dtype=torch.int64).to(self.device)

                # local model optimize
                optimizer.zero_grad()
                local_logits = model(x_g)
                local_loss = self.criterion(local_logits,y_g)
                train_loss += local_loss
                proximal_term=0.
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                local_loss += (0.01 / 2) * proximal_term
                local_loss.backward()
                optimizer.step()
                
                # global model optimize
                optimizer.zero_grad()
                logits = global_model(x_g) # predict y
                loss = self.criterion(logits,y_g)
                
                loss.backward()
                optimizer.step()

                

                _, predicted = local_logits.max(1)
                total += y_g.size(0)
                correct += predicted.eq(y_g).sum().item()

            correct_train = correct/total
            train_loss /= self.args.batchsum

            epoch_loss.append(train_loss)
            print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f} | Accuracy: {:.2f}%'.format(
                    global_round, epoch, train_loss, correct_train*100))
                
      
        print('| Global Round : {} |\tLoss avg: {:.6f} '.format(
                    global_round, sum(epoch_loss) / len(epoch_loss)))
        
        # record each local model
        if global_round ==self.args.epochs:
            PATH = 'savemodels/'+self.args.model_name+'/'+self.args.dataset+'/local'+str(self.id)+'.pt'
            torch.save(model.state_dict(), PATH)
            
        return model.state_dict(),global_model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self,model):
        model.eval()
        test_x, test_y = self.dataloader.get_test_data()
        test_x = torch.tensor(test_x,dtype=torch.float32).to(self.device)
        test_y = torch.tensor(test_y,dtype=torch.int64).to(self.device)
        
        predict_y = model(test_x)
        loss = self.criterion(predict_y,test_y)

        _,predict_class = predict_y.max(1)
        correct = predict_class.eq(test_y).sum().item()
        acc = correct / len(test_y)
        return acc, loss
        

# local test-tuning 
class LocalTestTune(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.lossfn == 'NLLLoss':
          self.criterion = nn.NLLLoss()
        if args.lossfn == 'CrossEntropyLoss':
          self.criterion = nn.CrossEntropyLoss()
        
    def GenerateRandomCurves(X, sigma=0.2, knot=4):
      #时间序列数据增强
      xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
      yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
      x_range = np.arange(X.shape[0])
      cs_x = CubicSpline(xx[:,0], yy[:,0])
      cs_y = CubicSpline(xx[:,1], yy[:,1])
      cs_z = CubicSpline(xx[:,2], yy[:,2])
      return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()    
    
    def contrastive_loss(self, z,y, temperature=0.07, base_temperature=0.07):
      # contrastive loss, z is a list of hidden feature ; y is a list of labels
      batch_size = z.shape[0]
      # contrast_count = 1
      # anchor_count = contrast_count
      y = torch.unsqueeze(y, -1)

      # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
      #     has the same class as sample i. Can be asymmetric.
      mask = torch.eq(y, torch.t(y)).float()
      anchor_dot_contrast = torch.matmul(z, torch.t(z))/ temperature
    
      # # for numerical stability
      logits_max = torch.diag_embed(torch.diag(anchor_dot_contrast))  # get i-i th
      logits = anchor_dot_contrast - logits_max # remove i-i
      # # tile mask
      logits_mask = torch.ones_like(mask) - torch.eye(batch_size,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
      mask = mask * logits_mask
      # compute log_prob
      exp_logits = torch.exp(logits) * logits_mask
      log_prob = logits - \
          torch.log(torch.sum(exp_logits, axis=1, keepdims=True))

      # compute mean of log-likelihood over positive
      # this may introduce NaNs due to zero division,
      # when a class only has one example in the batch
      mask_sum = torch.sum(mask, axis=1)
      mean_log_prob_pos = torch.sum(
          mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]

      # loss
      loss = -(temperature / base_temperature) * mean_log_prob_pos
      # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
      loss = torch.mean(loss)
      return loss


    def single_constrast_loss(self, anchor_feat, anchor_y, feat,y):
      temperature=0.07
      anchor_feat = anchor_feat.unsqueeze(0)

      # obtain the mean feature of each class
      unique_label = torch.unique(y)
      mean_feat = []
      mean_feat_label = []
      for l in unique_label:
        mask = torch.nonzero(y==l)
        mean_feat_label.append(l)
        if torch.numel(mask) ==1:
           mean_feat.append(feat[mask[0]])
           continue
        mask = mask.squeeze(1)
        mean_feat.append( torch.mean(torch.index_select(feat,0,mask),keepdim=False,dim=0).unsqueeze(0))
      mean_feat = torch.cat(mean_feat)
      mean_feat_label = torch.Tensor(mean_feat_label).to(self.device)
         

      dot_product_similarity = torch.mm(mean_feat,anchor_feat.T)/temperature
      exp_dot_sim = torch.exp(dot_product_similarity)
      mask_simliar_class = torch.nonzero(mean_feat_label==anchor_y) 
      log_prob = -torch.log(exp_dot_sim/torch.sum(exp_dot_sim,dim=0,keepdim=True))
      per_loss = log_prob[mask_simliar_class]
      mean_log_constr_loss = torch.mean(per_loss)
      return mean_log_constr_loss
    
    def distance_contrast_loss(self, anchor_feat, anchor_y, feat,y):
      
      anchor_feat = anchor_feat.unsqueeze(0)

      # obtain the mean feature of each class
      unique_label = torch.unique(y)
      mean_feat = []
      mean_feat_label = []
      for l in unique_label:
        mask = torch.nonzero(y==l)
        mean_feat_label.append(l)
        if torch.numel(mask) ==1:
           mean_feat.append(feat[mask[0]])
           continue
        mask = mask.squeeze(1)
        mean_feat.append( torch.mean(torch.index_select(feat,0,mask),keepdim=False,dim=0).unsqueeze(0))
      mean_feat = torch.cat(mean_feat)
      mean_feat_label = torch.Tensor(mean_feat_label).to(self.device)
      
      # calculate distance between mean feature of different class and human feedback
      pdist = nn.PairwiseDistance(p=2)
      dist = pdist(mean_feat, anchor_feat.repeat(mean_feat.shape[0],1))
      similar_sample = torch.nonzero(mean_feat_label==anchor_y).squeeze()
      # mean_dist_loss = dist[similar_sample]/torch.sum(dist,dim=0)

      pos_dist = torch.nn.functional.relu(dist[similar_sample])
      neg_dist =0.
      for i in range(mean_feat_label.shape[0]):
          if torch.numel(similar_sample)==0:
            neg_dist += torch.nn.functional.relu(0.5-dist[i])
          elif i != similar_sample:
            neg_dist += torch.nn.functional.relu(0.5-dist[i])
      
      if torch.numel(similar_sample)==0:
         loss = neg_dist/mean_feat_label.shape[0]
      else:
         #loss = (pos_dist+neg_dist)/mean_feat_label.shape[0]
         loss = pos_dist
      
      return loss

    def variance_inerclass(self, feat,label):
      #  Features of samples in the sample class should be closed.
      unique_label = torch.unique(label)
      l_var = 0.
      # variance in each class
      for l in unique_label:
        mask = torch.nonzero(label==l).squeeze()
        if torch.numel(mask) ==1:
           l_var +=0.
           continue
        var_feat = torch.mean(torch.var(torch.index_select(feat,0,mask),correction=1,keepdim=False,dim=0))
        l_var+= var_feat
      # mean variance of different class
      l_var/= torch.numel(unique_label)
      return l_var
        

    def update_weights_select_samples(self,model,data,label,uploadornot):
      label = torch.cat(label).to(self.device)
      # print(label.shape)
      
      # init a optimizer
      if self.args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
      elif self.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.wd)
      
      

      #  data (feature,pseudo label) and (x_h,y_h)
      #  test_tune local model with prediction loss and contrastive loss
      
      # predict human feedback data
      predict_human_logit = torch.softmax(model(data[-1]),-1)
      _,predict_human_label = predict_human_logit.max(1)
      human_prediction_loss = self.criterion(predict_human_logit,label[-1].unsqueeze(0))
      print('human y',label[-1])
      print('human logit',predict_human_logit)

      #  predict pseudo labels for non-label data
      # no_label_data = torch.cat(data[:-1],axis=0)
      hidden_embedding = model.encoder(torch.cat(data))
      pseudo_logits =torch.softmax(model.output(hidden_embedding[:-1]),-1)
      pseudo_prediction_loss = self.criterion(pseudo_logits,label[:-1])


      # find possible candidate label
      max_mask = -(pseudo_logits==pseudo_logits.max(1,keepdim=True)[0]).to(dtype=torch.int32)+1
      max_logits = torch.mul(max_mask,pseudo_logits)
      candidate_label = max_logits.max(1)[1]
      select_c_loss = self.criterion(pseudo_logits,candidate_label)
      
      
      # gradient with different labels
      optimizer.zero_grad()
      human_prediction_loss.backward(retain_graph=True)
      true_grad={}
      # true_para_name = []
      for name,t_para in model.output.named_parameters():
         if t_para.requires_grad and t_para.grad is not None:
            true_grad[name]=t_para.grad.view(-1).clone()
            # true_para_name.append(name)
      

      # different between candidate gradient and true gradient
      candidate_diff= {}
      # optimizer.zero_grad()
      select_c_loss.backward(retain_graph=True)
      
      for name,para in model.output.named_parameters():
         if para.requires_grad and para.grad is not None:
            candidate_diff[name] = torch.cosine_similarity(true_grad[name],para.grad.view(-1).clone(),dim=0)
      
      # difference between pseudo gradient and true gradient
      pseudo_diff = {}
      # optimizer.zero_grad()
      pseudo_prediction_loss.backward(retain_graph=True)
      for name,para in model.output.named_parameters():
         if para.requires_grad and para.grad is not None:
            pseudo_diff[name] =torch.cosine_similarity(true_grad[name],para.grad.view(-1).clone(),dim=0)
      
      
      if sum(pseudo_diff.values())>=sum(candidate_diff.values()):
        #  说明第一顺位预测大部分错误
        pseudo_prediction_loss = self.criterion(pseudo_logits,candidate_label)
        psudo_contrast_loss = self.contrastive_loss(hidden_embedding[:-1],candidate_label)
        contrast_loss = self.distance_contrast_loss(hidden_embedding[-1],label[-1],hidden_embedding[:-1],candidate_label)
        var_loss = self.variance_inerclass(hidden_embedding[:-1],candidate_label)
        uploadornot=True
      else:
        # contrastive loss: enchor = human feedback, pos sample: mean feature with same class, negative sample: other features
        # contrast_loss = self.single_constrast_loss(hidden_embedding[-1],label[-1],hidden_embedding[:-1],label[:-1])
        psudo_contrast_loss = self.contrastive_loss(hidden_embedding[:-1],label[:-1])
        contrast_loss = self.distance_contrast_loss(hidden_embedding[-1],label[-1],hidden_embedding[:-1],label[:-1])
        var_loss = self.variance_inerclass(hidden_embedding[:-1],label[:-1])
        
      

      if label[-1] not in label[:-1] and label[-1] not in candidate_label:
         pseudo_prediction_loss = self.criterion(pseudo_logits,label[-1].repeat(pseudo_logits.shape[0]))
         psudo_contrast_loss = self.contrastive_loss(hidden_embedding[:-1],label[-1].repeat(pseudo_logits.shape[0]))
         loss = 1*human_prediction_loss + 0*pseudo_prediction_loss + 0*psudo_contrast_loss 
      else:
         loss = 1*human_prediction_loss + 0*pseudo_prediction_loss + 0*var_loss + 0*contrast_loss +0*psudo_contrast_loss 
      # loss = pseudo_prediction_loss
      # optimize the model
      print(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      # compare the performance before and after tuning func1: compare the model before and after adaptation
      if predict_human_label != label[-1]:
         uploadornot = True
      elif self.args.whole_change:
        after_predict = model(torch.cat(data[:-1],dim=0))
        _,after_p_label = after_predict.max(1)
        corr = after_p_label.eq(label[:-1]).sum().item()
        if corr!=len(after_p_label):
          uploadornot = True
      else:
        exg_predict = model.output(hidden_embedding)
        _,after_p_label = exg_predict.max(1)
        corr = after_p_label.eq(label[:-1]).sum().item()
        if corr!=len(after_p_label):
          uploadornot = True
      
      # if self.args.whole_update:
      #    return model.embedding.state_dict().update(model.lstm_layer.state_dict()), loss,uploadornot
      # else:
      #    return model.state_dict(),loss,uploadornot
      return model, loss, uploadornot
    
    def update_weights_tent(self, model, global_round):
      model.train()
      epoch_loss = []
    
      if self.args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)
      elif self.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                      weight_decay=self.args.wd)

      for epoch in range(1, self.args.local_ep+1):
        train_loss = 0.0
        correct_train = 0.0
        total=0
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.test4train_loader):
          images = images.to(self.device)
          labels = labels.to(self.device)

          optimizer.zero_grad() 
          log_probs = model(images)  
          _, predicted_outputs = log_probs.max(1)
          loss =  (-(log_probs.softmax(1) * log_probs.log_softmax(1)).sum(1)).mean(0)
          loss.backward() 
          optimizer.step()  

          train_loss += loss.data.item() 
          
          total += labels.size(0)
          correct_train += predicted_outputs.eq(labels).sum().item()


        train_loss = train_loss / (batch_idx+1)
        train_acc = correct_train / total

        epoch_loss.append(train_loss)
        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f} | Accuracy: {:.2f}%'.format(
                global_round, epoch, train_loss, train_acc*100))
              
    
      print('| Global Round : {} |\tLoss avg: {:.6f} '.format(
                  global_round, sum(epoch_loss) / len(epoch_loss)))
      
      # if global_round ==self.args.epochs:
      #   PATH = 'fedimage/savemodels/'+'fedavg'+'/'+self.args.dataset+'/local'+str(self.user_id)+'.pt'
      #   torch.save(model.state_dict(), PATH)
          
      return model, sum(epoch_loss) / len(epoch_loss)


    def gasuss_noise(self, image, mean=0,var=0.001):
        #  add gasuss_noise for image
        # image = np.array(image/255,dtype=float)
        noise = np.random.normal(mean, var**0.5,image.shape)
        out = image +noise
        if out.min() <0:
          low_clip=-1.
        else:
          low_clip=0.
        out = np.clip(out,low_clip,1.0)
        out = torch.from_numpy(np.uint8(out*255.)).to(torch.float32)
        return out


    def update_weights_memo(self, model, global_round):
          model.train()
          epoch_loss = []
        
          if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
          elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                          weight_decay=self.args.wd)

          for epoch in range(1, self.args.local_ep+1):
            train_loss = 0.0
            correct_train = 0.0
            total=0
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.test4train_loader):
              noise_imagev1 = self.gasuss_noise(images).to(self.device)
              noise_imagev2 = self.gasuss_noise(images,var=0.002).to(self.device)

              images = images.to(self.device)
              labels = labels.to(self.device)

              optimizer.zero_grad() 
              log_probs = model(images).softmax(1)
              _, predicted_outputs = log_probs.max(1)
              # prediction for noise image
              noise_log_probsv1 = model(noise_imagev1).softmax(1)
              noise_log_probsv2 = model(noise_imagev2).softmax(1)
              
              aug_log_prob = (noise_log_probsv1+noise_log_probsv2+log_probs)/3
                          
              
              loss =  (-(aug_log_prob.softmax(1) * aug_log_prob.log_softmax(1)).sum(1)).mean(0)
              loss.backward() 
              optimizer.step()  

              train_loss += loss.data.item() 
              
              total += labels.size(0)
              correct_train += predicted_outputs.eq(labels).sum().item()


            train_loss = train_loss / (batch_idx+1)
            train_acc = correct_train / total

            epoch_loss.append(train_loss)
            print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f} | Accuracy: {:.2f}%'.format(
                    global_round, epoch, train_loss, train_acc*100))
                  
        
          print('| Global Round : {} |\tLoss avg: {:.6f} '.format(
                      global_round, sum(epoch_loss) / len(epoch_loss)))
          
          # if global_round ==self.args.epochs:
          #   PATH = 'fedimage/savemodels/'+'fedavg'+'/'+self.args.dataset+'/local'+str(self.user_id)+'.pt'
          #   torch.save(model.state_dict(), PATH)
              
          return model, sum(epoch_loss) / len(epoch_loss)
    
    def marginal_entropy(self,outputs):
      logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
      avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
      min_real = torch.finfo(avg_logits.dtype).min
      avg_logits = torch.clamp(avg_logits, min=min_real)
      return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
    
    
    
    def aug_constrastive_loss(self, test_data, aug_data):
        #  postive samples are augment sample
        temperature = 0.07
        positive_res = torch.diag(torch.matmul(test_data,torch.t(aug_data) ))/temperature

        all_res = torch.matmul(test_data,torch.t(test_data))/temperature
        diag_m = torch.diag_embed(all_res)
        other_dist = all_res-diag_m
        dist = - (torch.log(positive_res) - torch.log(torch.sum(other_dist,dim=1,keepdim=True)))
        loss = torch.mean(dist)
        return loss

    
       
    def hybrid_update_weights(self,model,global_m,data,label,uploadornot):
      label = torch.cat(label).to(self.device)
      # print(label.shape)
      
      # init a optimizer
      if self.args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
      elif self.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.wd)
      
      
      # predict human feedback data
      predict_human_logit = torch.softmax(model(data[-1]),-1)
      _,predict_human_label = predict_human_logit.max(1)
      human_prediction_loss = self.criterion(predict_human_logit,label[-1].unsqueeze(0))

      #  obtain conterfactual_label
      counterfactual_label = predict_human_label.detach()
      p_hum_log = predict_human_logit.clone()
      if counterfactual_label == label[-1]:
         p_hum_log[0,predict_human_label] = -1
         counterfactual_label = p_hum_log.max(1)[1]
      counterfactual_loss = self.criterion(predict_human_logit,counterfactual_label)
      

      #  predict pseudo labels for non-label data
      # no_label_data = torch.cat(data[:-1],axis=0)
      hidden_embedding = model.encoder(torch.cat(data))
      pseudo_logits =torch.softmax(model.output(hidden_embedding[:-1]),-1)
      pseudo_prediction_loss = self.criterion(pseudo_logits,label[:-1])

      # gradient with human feedback
      optimizer.zero_grad()
      human_prediction_loss.backward(retain_graph=True)
      true_grad={}
      # true_para_name = []
      for name,t_para in model.output.named_parameters():
         if t_para.requires_grad and t_para.grad is not None:
            true_grad[name]=t_para.grad.view(-1).clone()
            # true_para_name.append(name)
      
      # gradient with counterfactual label
      # optimizer.zero_grad()
      counterfactual_loss.backward(retain_graph = True)
      counterfactual_grad={}
      for name,t_para in model.output.named_parameters():
         if t_para.requires_grad and t_para.grad is not None:
            counterfactual_grad[name]=t_para.grad.view(-1).clone()

      # difference between pseudo gradient and true gradient
      pseudo_diff = 0.
      # optimizer.zero_grad()
      persample_criterion = nn.CrossEntropyLoss(reduce=False)
      per_sample_loss = persample_criterion(pseudo_logits,label[:-1])
      candidate_set = []
      for i in range(per_sample_loss.size(dim=0)):
        per_sample_loss[i].backward(retain_graph=True)
        for name,para in model.output.named_parameters():
          if para.requires_grad and para.grad is not None:
              pseudo_diff  +=torch.cosine_similarity(true_grad[name],para.grad.view(-1).clone(),dim=0) - torch.cosine_similarity(counterfactual_grad[name],para.grad.view(-1).clone(),dim=0)
        if pseudo_diff<0:
           candidate_set.append(0)
        else:
           candidate_set.append(1)
        pseudo_diff = 0.
      

      # global model predict
      global_embedding = global_m.encoder(torch.cat(data[:-1]))
      global_norm = global_embedding.norm(p=2,dim=1,keepdim=True)
      global_embedding_norm = global_embedding.div(global_norm)
      local_norm = hidden_embedding[:-1].norm(p=2,dim=1,keepdim=True)
      local_embedding_norm = hidden_embedding[:-1].div(local_norm)
      global_local_loss = (global_embedding_norm-local_embedding_norm).norm(2).pow(2)
      contrast_loss = self.distance_contrast_loss(hidden_embedding[-1],label[-1],hidden_embedding[:-1],label[:-1])
      loss = 1*human_prediction_loss + 1*pseudo_prediction_loss  + 10*contrast_loss +100*global_local_loss
      print(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      # compare the performance before and after tuning func1: compare the model before and after adaptation
      if predict_human_label != label[-1]:
         uploadornot = True
      elif self.args.whole_change:
        after_predict = model(torch.cat(data[:-1],dim=0))
        _,after_p_label = after_predict.max(1)
        corr = after_p_label.eq(label[:-1]).sum().item()
        if corr!=len(after_p_label):
          uploadornot = True
      else:
        exg_predict = model.output(hidden_embedding)
        _,after_p_label = exg_predict.max(1)
        corr = after_p_label.eq(label[:-1]).sum().item()
        if corr!=len(after_p_label):
          uploadornot = True
      
      # if self.args.whole_update:
      #    return model.embedding.state_dict().update(model.lstm_layer.state_dict()), loss,uploadornot
      # else:
      #    return model.state_dict(),loss,uploadornot
      return model, loss, uploadornot

    def update_weights(self, model,data,label, uploadornot):
      label = torch.cat(label).to(self.device)
      # print(label.shape)
      
      # init a optimizer
      if self.args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
      elif self.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.wd)
      
      optimizer.zero_grad()

      #  data (feature,pseudo label) and (x_h,y_h)
      #  test_tune local model with prediction loss and contrastive loss
      
      # predict human feedback data
      predict_human_logit = torch.softmax(model(data[-1]),-1)
      _,predict_human_label = predict_human_logit.max(1)
      human_prediction_loss = self.criterion(predict_human_logit,label[-1].unsqueeze(0))
      print('human y',label[-1])
      print('human logit',predict_human_logit)

      #  predict pseudo labels for non-label data
      # no_label_data = torch.cat(data[:-1],axis=0)
      hidden_embedding = model.encoder(torch.cat(data))
      pseudo_logits =torch.softmax(model.output(hidden_embedding[:-1]),-1)

      pseudo_prediction_loss = self.criterion(pseudo_logits,label[:-1])
      # logits = torch.cat([pseudo_logits,predict_human_logit],dim=0)
      # prediction_loss = self.criterion(logits,torch.cat(label))

      # contrastive loss: enchor = human feedback, pos sample: mean feature with same class, negative sample: other features
      # contrast_loss = self.single_constrast_loss(hidden_embedding[-1],label[-1],hidden_embedding[:-1],label[:-1])
      psudo_contrast_loss = self.contrastive_loss(hidden_embedding[:-1],label[:-1])
      contrast_loss = self.distance_contrast_loss(hidden_embedding[-1],label[-1],hidden_embedding[:-1],label[:-1])
      var_loss = self.variance_inerclass(hidden_embedding[:-1],label[:-1])
      loss = 0*human_prediction_loss + 1*pseudo_prediction_loss + 0*var_loss + 0*contrast_loss +10*psudo_contrast_loss
      # loss = pseudo_prediction_loss

      # optimize the model
      print(loss)
      loss.backward()
      optimizer.step()


      # compare the performance before and after tuning func1: compare the model before and after adaptation
      if predict_human_label != label[-1]:
         uploadornot = True
      elif self.args.whole_change:
        after_predict = model(torch.cat(data[:-1],dim=0))
        _,after_p_label = after_predict.max(1)
        corr = after_p_label.eq(label[:-1]).sum().item()
        if corr!=len(after_p_label):
          uploadornot = True
      else:
        exg_predict = model.output(hidden_embedding)
        _,after_p_label = exg_predict.max(1)
        corr = after_p_label.eq(label[:-1]).sum().item()
        if corr!=len(after_p_label):
          uploadornot = True
      
      # if self.args.whole_update:
      #    return model.embedding.state_dict().update(model.lstm_layer.state_dict()), loss,uploadornot
      # else:
      #    return model.state_dict(),loss,uploadornot
      return model, loss, uploadornot

    def update_weights_ditto(self, global_model,model,data,label,uploadornot):
          
          uploadornot = False

          # init a optimizer
          if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                            momentum=self.args.momentum)
          elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                            weight_decay=self.args.wd)
          
          
          # =====================local update
          optimizer.zero_grad()

          #  data (feature,pseudo label) and (x_h,y_h)
          #  test_tune local model with prediction loss and contrastive loss
          
          # predict human feaback data
          predict_human_logit = model(data[-1])
          _,predict_human_label = predict_human_logit.max(1)
          human_prediction_loss = self.criterion(predict_human_logit,label[-1])

          #  predict pseudo labels for non-label data
          hidden_embedding = model.encoder(torch.cat(data))
          pseudo_logits = model.output(hidden_embedding[:-1])

          pseudo_prediction_loss = self.criterion(pseudo_logits,torch.cat(label[:-1]))
          # logits = torch.cat([pseudo_logits,predict_human_logit],dim=0)
          # prediction_loss = self.criterion(logits,torch.cat(label))

          contrast_loss = self.contrastive_loss(hidden_embedding[:-1],torch.cat(label[:-1],dim=0))

          # ############################ local loss ###############
          loss = human_prediction_loss+ pseudo_prediction_loss + self.args.beta * contrast_loss
          # loss = human_prediction_loss+ pseudo_prediction_loss 
          # loss = pseudo_prediction_loss

          # prox loss 
          proximal_term =0.
          for w, w_t in zip(model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
          loss += proximal_term

          # optimize the model
          loss.backward()
          optimizer.step()

          # =======================global update
          # init a optimizer
          if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.lr,
                                            momentum=self.args.momentum)
          elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(global_model.parameters(), lr=self.args.lr,
                                            weight_decay=self.args.wd)
          optimizer.zero_grad()
          loss = 0.
          
          predict_human_logit = global_model(data[-1])
          _,predict_human_label = predict_human_logit.max(1)
          human_prediction_loss = self.criterion(predict_human_logit,label[-1])

          #  predict pseudo labels for non-label data
          hidden_embedding = global_model.encoder(torch.cat(data))
          pseudo_logits = global_model.output(hidden_embedding[:-1])

          pseudo_prediction_loss = self.criterion(pseudo_logits,torch.cat(label[:-1]))
          # logits = torch.cat([pseudo_logits,predict_human_logit],dim=0)
          # prediction_loss = self.criterion(logits,torch.cat(label))

          contrast_loss = self.contrastive_loss(hidden_embedding[:-1],torch.cat(label[:-1],dim=0))

          ################################### gloabl loss ####################
          loss = human_prediction_loss+ pseudo_prediction_loss + self.args.beta * contrast_loss
          # loss = human_prediction_loss+ pseudo_prediction_loss 
          # loss = pseudo_prediction_loss

          loss.backward()
          optimizer.step()


          # compare the performance before and after tuning func1: compare the model before and after adaptation
          if predict_human_label != label[-1]:
            uploadornot = True
          elif self.args.whole_change:
            after_predict = global_model(torch.cat(data[:-1],dim=0))
            _,after_p_label = after_predict.max(1)
            corr = after_p_label.eq(torch.cat(label[:-1])).sum().item()
            if corr!=len(after_p_label):
              uploadornot = True
          else:
            exg_predict = global_model.output(hidden_embedding)
            _,after_p_label = exg_predict.max(1)
            corr = after_p_label.eq(label[:-1]).sum().item()
            if corr!=len(after_p_label):
              uploadornot = True
          
          # if self.args.whole_update:
          #    return model.embedding.state_dict().update(model.lstm_layer.state_dict()), loss,uploadornot
          # else:
          #    return model.state_dict(),loss,uploadornot
          return global_model, model, loss, uploadornot

    def update_weights_prox(self,global_model,model,data,label,uploadornot):
          
          # init a optimizer
          if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                            momentum=self.args.momentum)
          elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                            weight_decay=self.args.wd)
          
          optimizer.zero_grad()

          #  data (feature,pseudo label) and (x_h,y_h)
          #  test_tune local model with prediction loss and contrastive loss
          
          # predict human feaback data
          predict_human_logit = model(data[-1])
          _,predict_human_label = predict_human_logit.max(1)
          human_prediction_loss = self.criterion(predict_human_logit,label[-1])

          #  predict pseudo labels for non-label data
          no_label_data = torch.cat(data[:-1],axis=0)
          hidden_embedding = model.encoder(torch.cat(data))
          pseudo_logits = model.output(hidden_embedding[:-1])

          pseudo_prediction_loss = self.criterion(pseudo_logits,torch.cat(label[:-1]))
          # logits = torch.cat([pseudo_logits,predict_human_logit],dim=0)
          # prediction_loss = self.criterion(logits,torch.cat(label))

          contrast_loss = self.contrastive_loss(hidden_embedding[:-1],torch.cat(label[:-1],dim=0))
          # loss = human_prediction_loss + pseudo_prediction_loss + self.args.beta * contrast_loss
          loss = human_prediction_loss + pseudo_prediction_loss   
          # loss = pseudo_prediction_loss

          # prox loss 
          proximal_term =0.
          for w, w_t in zip(model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
          
          loss +=  (0.01 / 2) * proximal_term
          # optimize the model
          loss.backward()
          optimizer.step()


          # compare the performance before and after tuning func1: compare the model before and after adaptation
          if predict_human_label != label[-1]:
            uploadornot = True
          elif self.args.whole_change:
            after_predict = model(torch.cat(data[:-1],dim=0))
            _,after_p_label = after_predict.max(1)
            corr = after_p_label.eq(torch.cat(label[:-1])).sum().item()
            if corr!=len(after_p_label):
              uploadornot = True
          else:
            exg_predict = model.output(hidden_embedding)
            _,after_p_label = exg_predict.max(1)
            corr = after_p_label.eq(label[:-1]).sum().item()
            if corr!=len(after_p_label):
              uploadornot = True
          
          # if self.args.whole_update:
          #    return model.embedding.state_dict().update(model.lstm_layer.state_dict()), loss,uploadornot
          # else:
          #    return model.state_dict(),loss,uploadornot
          return model, loss, uploadornot
    
    def update_weights_tent(self, model, data, label, uploadornot, global_round):
      model.train()
      epoch_loss = []
    
      if self.args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)
      elif self.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                      weight_decay=self.args.wd)

      for epoch in range(1, self.args.local_ep+1):
        train_loss = 0.0
        correct_train = 0.0
        total=0
        batch_loss = []
        for x,labels in zip(data,label):
          x = x.to(self.device)
          labels = labels.to(self.device)

          optimizer.zero_grad() 
          log_probs = model(x)  
          _, predicted_outputs = log_probs.max(1)
          loss =  (-(log_probs.softmax(1) * log_probs.log_softmax(1)).sum(1)).mean(0)
          loss.backward() 
          optimizer.step()  

          train_loss += loss.data.item() 
          
          total += labels.size(0)
          correct_train += predicted_outputs.eq(labels).sum().item()


        train_loss = train_loss / len(data)
        train_acc = correct_train / total

        epoch_loss.append(train_loss)
        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f} | Accuracy: {:.2f}%'.format(
                global_round, epoch, train_loss, train_acc*100))
              
    
      print('| Global Round : {} |\tLoss avg: {:.6f} '.format(
                  global_round, sum(epoch_loss) / len(epoch_loss)))
      
      # if global_round ==self.args.epochs:
      #   PATH = 'fedimage/savemodels/'+'fedavg'+'/'+self.args.dataset+'/local'+str(self.user_id)+'.pt'
      #   torch.save(model.state_dict(), PATH)
          
      return model, sum(epoch_loss) / len(epoch_loss) ,uploadornot


    def gasuss_noise(self, image, mean=0,var=0.001):
        #  add gasuss_noise for image
        # image = np.array(image/255,dtype=float)
        noise = np.random.normal(mean, var**0.5,image.shape)
        out = image + torch.from_numpy(noise)
        # if out.min() <0:
        #   low_clip=-1.
        # else:
        #   low_clip=0.
        # out = np.clip(out,low_clip,1.0)
        # out = torch.from_numpy(np.uint8(out*255.)).to(torch.float32)
        return out


    def update_weights_memo(self, model, data, label, global_round):
          model.train()
          epoch_loss = []
        
          if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
          elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                          weight_decay=self.args.wd)

          for epoch in range(1, self.args.local_ep+1):
            train_loss = 0.0
            correct_train = 0.0
            total=0
            batch_loss = []
            for images,labels in zip(data,label):
              noise_imagev1 = self.gasuss_noise(images).to(self.device)
              noise_imagev2 = self.gasuss_noise(images,var=0.002).to(self.device)

              images = images.to(self.device)
              labels = labels.to(self.device)

              optimizer.zero_grad() 
              log_probs = model(images).softmax(1)
              _, predicted_outputs = log_probs.max(1)
              # prediction for noise image
              noise_log_probsv1 = model(noise_imagev1).softmax(1)
              noise_log_probsv2 = model(noise_imagev2).softmax(1)
              
              aug_log_prob = (noise_log_probsv1+noise_log_probsv2+log_probs)/3
                          
              
              loss =  (-(aug_log_prob.softmax(1) * aug_log_prob.log_softmax(1)).sum(1)).mean(0)
              loss.backward() 
              optimizer.step()  

              train_loss += loss.data.item() 
              
              total += labels.size(0)
              correct_train += predicted_outputs.eq(labels).sum().item()


            train_loss = train_loss / len(data)
            train_acc = correct_train / total

            epoch_loss.append(train_loss)
            print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f} | Accuracy: {:.2f}%'.format(
                    global_round, epoch, train_loss, train_acc*100))
                  
        
          print('| Global Round : {} |\tLoss avg: {:.6f} '.format(
                      global_round, sum(epoch_loss) / len(epoch_loss)))
          
          # if global_round ==self.args.epochs:
          #   PATH = 'fedimage/savemodels/'+'fedavg'+'/'+self.args.dataset+'/local'+str(self.user_id)+'.pt'
          #   torch.save(model.state_dict(), PATH)
              
          return model, sum(epoch_loss) / len(epoch_loss)
    
    def marginal_entropy(self,outputs):
      logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
      avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
      min_real = torch.finfo(avg_logits.dtype).min
      avg_logits = torch.clamp(avg_logits, min=min_real)
      return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
    
    
    def aug_constrastive_loss(self, test_data, aug_data):
        #  postive samples are augment sample
        temperature = 0.07
        positive_res = torch.diag(torch.matmul(test_data,torch.t(aug_data) ))/temperature

        all_res = torch.matmul(test_data,torch.t(test_data))/temperature
        diag_m = torch.diag_embed(all_res)
        other_dist = all_res-diag_m
        dist = - (torch.log(positive_res) - torch.log(torch.sum(other_dist,dim=1,keepdim=True)))
        loss = torch.mean(dist)
        return loss

   
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class GlobalUpdate(object):
  def __init__(self, args):
    self.weight_list=[0.1 for i in range(args.test_users_num)]
    self.frequency_list = [0 for i in range(args.test_users_num)]
    self.decay_gamma = 0.9
    self.args = args

  
  def smoothaggregate(self, l_w,g_w,user_id):
    """
    Returns the global weights after smooth aggreagation.
    """
    if user_id<self.args.num_users:
       alpha = 0.3
    else:
       alpha = 0.3
    self.weight_list[user_id] = alpha*np.exp(-self.decay_gamma*self.frequency_list[user_id])
    for key in l_w.keys():
            g_w[key]= (1-self.weight_list[user_id])*g_w[key] + self.weight_list[user_id]* l_w[key]
    self.frequency_list[user_id]+=1
    return g_w
