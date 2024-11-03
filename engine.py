import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os

from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.cluster import KMeans
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self.server_model_param = {}
        self.client_model_params = {}
        self.cluster_model_param = {}
        self.identity = {}
        self.num_client_clusters = self.config['user_cluster']
        self.num_item_clusters = self.config['item_cluster']
        self.item_clusters = []
        self.temperature = self.config['cl_t']
        self.base_temperature = self.config['base_t']
        self.reg = self.config['reg']
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.supcon_loss = 0
        self.mae = torch.nn.L1Loss()
        self.labels = torch.tensor([1,2])
        self.count = {}
        for i in range(self.config["num_users"]):
            self.count[i] = 0

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers, labels):
        """train a batch and return an updated model."""
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            labels = labels.cuda()

        optimizer, optimizer_i = optimizers
        # update score function.
        optimizer.zero_grad()
        ratings_pred,supcon = model_client(items,labels)
       
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss = loss + self.reg * supcon
        loss.backward()
        optimizer.step()
       
        # update item embedding.
        optimizer_i.zero_grad()
        ratings_pred, supcon= model_client(items,labels)
    
        
        loss_i = self.crit(ratings_pred.view(-1), ratings)
        loss_i = loss_i + self.reg * supcon
        loss_i.backward()
        optimizer_i.step()
    
        return model_client, loss_i.item()
    
    def aggregate_clients_params(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():
            user_params = round_user_params[user]
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)

    def compute_cluster(self,  neighborhood_users,round_user_params):
        len_cluster = 0
        center_cluster = copy.deepcopy(round_user_params[neighborhood_users[0]]['embedding_item.weight'])
        for user_id in neighborhood_users:
            if len_cluster == 0:
                center_cluster.data = round_user_params[user_id]['embedding_item.weight'].data
                len_cluster += 1
            else:
                center_cluster.data += round_user_params[user_id]['embedding_item.weight'].data
                len_cluster += 1
        center_cluster.data = center_cluster.data/len_cluster
        return center_cluster
    
    def find_elbow(self, data,round_id):
        """Find the elbow point in a dataset."""
        # Compute coordinates of the line connecting the first and last point
        n_points = len(data)
        all_coords = np.vstack((range(n_points), data)).T
        first_point = all_coords[0]
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = all_coords - first_point
        scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
        vec_line = np.outer(scalar_prod, line_vec_norm)
        vec_to_line = vec_from_first - vec_line
        
        # Compute distance to line (L2 norm)
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        elbow_index = np.argmax(dist_to_line)
        return elbow_index

    def cab_cluster(self, round_user_params,participants,item_clusters,round_id):
        selected_user = random.choice(participants)
        selected_item_cluster = random.choice(range(self.num_item_clusters))
        indices = [i for i, x in enumerate(item_clusters) if x == selected_item_cluster]
        self.cluster_model_param = {}
        user_similarity = {}
      
        selected_user_item = copy.deepcopy(round_user_params[selected_user]['embedding_item.weight'].data)
        for u in round_user_params.keys():
            other_user = copy.deepcopy(round_user_params[u]['embedding_item.weight'].data)
            similarity = F.cosine_similarity(selected_user_item[indices], other_user[indices],dim=1)
            user_similarity[u] = torch.sum(similarity).item()
        sorted_similarities = dict(sorted(user_similarity.items(), key=lambda item: item[1], reverse=True))
        # Find the elbow point based on the sorted values
        elbow_index = self.find_elbow(list(sorted_similarities.values()),round_id)
        big_values = list(sorted_similarities.keys())[:elbow_index+1]
        
        if len(big_values) > 0:
            center_cluster1 = self.compute_cluster(big_values,round_user_params)
            for user_id in big_values: 
                self.count[user_id] += 1 
                self.client_model_params[user_id]['embedding_item.weight'].data= copy.deepcopy(center_cluster1.data).cuda()

    def fed_train_a_round(self, all_train_data, round_id):
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = random.sample(range(self.config['num_users']), num_participants)
        else:
            participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])

        # store users' model parameters of current round.
        round_participant_params = {}
        # store all the users' train loss and mae.
        all_loss = {}
        for user in participants:
            loss = 0
            model_client = copy.deepcopy(self.model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated item embedding and score function from server.
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                model_client.load_state_dict(user_param_dict)
            # optimizer is responsible for updating score function.
            optimizer = torch.optim.SGD(model_client.affine_output.parameters(),
                                        lr=self.config['lr'], weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                          weight_decay=self.config['l2_regularization'])  # Item optimizer
            optimizers = [optimizer, optimizer_i]
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            sample_num = 0
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    
                    model_client, loss_u = self.fed_train_single_batch(model_client, batch, optimizers, self.labels)
                    loss += loss_u * len(batch[0])
                    sample_num += len(batch[0])
                all_loss[user] = loss / sample_num
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            # store client models' local parameters for global update.
            round_participant_params[user] = copy.deepcopy(self.client_model_params[user])
            del round_participant_params[user]['affine_output.weight']

        self.aggregate_clients_params(round_participant_params)
        kmeans = KMeans(n_clusters=self.num_item_clusters)
        self.item_clusters = kmeans.fit_predict(self.server_model_param['embedding_item.weight'].data)
        self.labels = copy.deepcopy(torch.tensor(self.item_clusters, dtype=torch.long))

        self.cab_cluster(round_participant_params,participants,self.item_clusters,round_id)
        if round_id == 99:
            logging.info('Participe round is {}'.format(self.count))
        return all_loss


    def fed_evaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        temp = [0] * 100
        temp[0] = 1
        ratings = torch.FloatTensor(temp)
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()
        test_scores = None
        negative_scores = None
        all_loss = {}
        for user in range(self.config['num_users']):
            user_model = copy.deepcopy(self.model)
            if user in self.client_model_params.keys():
                user_param_dict = copy.deepcopy(self.client_model_params[user])
                for key in user_param_dict.keys():
                    user_param_dict[key] = user_param_dict[key].data.cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                test_user = test_users[user: user + 1]
                test_item = test_items[user: user + 1]
                negative_user = negative_users[user*99: (user+1)*99]
                negative_item = negative_items[user*99: (user+1)*99]
                test_score= user_model.forward_test(test_item )
                negative_score = user_model.forward_test(negative_item)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[user] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        return hit_ratio, ndcg, all_loss


    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)