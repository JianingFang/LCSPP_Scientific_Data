import os
import xarray as xr
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import logging
import copy
from datetime import datetime, date
from scipy.stats import linregress
from matplotlib import rcParams
import matplotlib.pyplot as plt



ML_PATH = "AVHRR/data/ML_CORRECT_v3.1/"



IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
if IDX > 23:
    ref_var = "red"
    bid = IDX - 24
else:
    ref_var = "nir"
    bid = IDX
    
# Calibrate a model for each of the biweekly period 
biweekly_identifiers = []
for month in np.arange(1, 13):
    for half_month_index in ["a", "b"]:
        biweekly_identifiers.append('{:0>2}'.format(month) + half_month_index)
bi = biweekly_identifiers[bid]

X_train = np.load(os.path.join(ML_PATH, "snow_X_train_{}_{}.npy".format(ref_var, bi))).astype(np.float32).T
Y_train = np.load(os.path.join(ML_PATH, "snow_Y_train_{}_{}.npy".format(ref_var, bi))).astype(np.float32)
train_valid = (np.sum(np.isnan(X_train), axis=1)==0) & np.invert(np.isnan(Y_train))
X_train = X_train[train_valid, :]
Y_train = Y_train[train_valid]


#rand_idx = np.random.permutation(X_train.shape[0])
#np.save(os.path.join(ML_PATH, "{}_train_idx_v3.1.npy".format(ref_var)), rand_idx)
rand_idx = np.load(os.path.join(ML_PATH, "{}_train_idx_v3.1.npy".format(ref_var)))
X_train = X_train[rand_idx, :]
Y_train = Y_train[rand_idx]



scaler = StandardScaler()
scaler.fit(X_train)
scaler_mean = scaler.mean_
scaler_var = scaler.var_
np.save(os.path.join(ML_PATH, "snow_{}_{}_train_scaler_mean_v3.1.npy".format(ref_var, bi)), np.array(scaler_mean))
np.save(os.path.join(ML_PATH, "snow_{}_{}_train_scaler_var_v3.1.npy".format(ref_var, bi)), np.array(scaler_var))
X_train=scaler.transform(X_train)
train_end_idx = np.int64(X_train.shape[0] * 0.8)


class MLDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float().to(device = try_gpu())  
        self.y = torch.tensor(y).float().to(device = try_gpu())  
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]


def try_gpu(i=0): 
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# feedforward model construct function
def construct_model(input_dim, hidden_dim, n_hidden_layers, drop_out=None):
    layers=[]
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    if drop_out:
        layers.append(nn.Dropout(p=0.2))
    for i in range(n_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim,hidden_dim))
        layers.append(nn.ReLU())
        if drop_out:
            layers.append(nn.Dropout(p=drop_out))
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers).to(device=try_gpu())

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.001)


# In[35]:


# set up logging. Training logs will be stores in "./logs/training.log"
logging.basicConfig(level=logging.INFO,
                    filename="../notebooks/logs/ML_{}_{}_correction_training.log".format(ref_var, bi),
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def evaluate_loss(net, data_iter, loss):  
    """Evaluate the loss of a model on the given dataset."""
    loss_sum=0
    sample_sum=0
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        loss_sum += l.sum()
        sample_sum += l.numel()
    return loss_sum / sample_sum


def train(layers, input_dim, hidden_dims, lrs, bs, num_epochs, train_dataset, val_dataset=None, PATIENCE=10):
    logging.info("layers: "+ str(layers))
    logging.info("hidden_dims: "+ str(hidden_dims))
    logging.info("lrs: "+ str(lrs))
    logging.info("bs: "+ str(bs))

    best_overall=9999999999999
    best_model=None
    if val_dataset == None:
        for lr in lrs:
            for n_hidden_layers in layers:
                for hidden_dim in hidden_dims:
                    net= construct_model(input_dim, hidden_dim, n_hidden_layers)
                    for batch_size in bs:
                        net.apply(init_weights);
                        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        loss = nn.MSELoss()
                        trainer = torch.optim.Adam(net.parameters(), lr=lr)
                        best_loss = 9999999999999
                        finished=False
                        no_improvement = 0
                        best_model_param = None
                        for epoch in range(num_epochs):
                            for X, y in train_iter:
                                l = loss(net(X).squeeze(), y)
                                trainer.zero_grad()
                                l.backward()
                                trainer.step()
                            if epoch == 0 or (epoch + 1) % 1 == 0:
                                with torch.no_grad():
                                    train_loss = evaluate_loss(net, train_iter, loss)
                                    logging.info(" epoch: " + str(epoch + 1) + " train: " + str(train_loss))
                        best_model_param = copy.deepcopy(net.state_dict())
                        best_model = net
                        torch.save(best_model_param, "./models/" + ref_var + "_" + bi +  "_layer_" + str(n_hidden_layers) + "_neuron_" + str(hidden_dim) + "_lr" + str(lr) +"_batchsize" + str(batch_size))
                        logging.info("./models/"+ ref_var + "_" + bi +  "_layer_" + str(n_hidden_layers) + "_neuron_" + str(hidden_dim) + "_lr" + str(lr) +"_batchsize" + str(batch_size) + " completed. "+ " epoch: " + str(epoch + 1) + " train: " + str(train_loss))
    else:
        for lr in lrs:
            for n_hidden_layers in layers:
                for hidden_dim in hidden_dims:
                    net= construct_model(input_dim, hidden_dim, n_hidden_layers)
                    for batch_size in bs:
                        net.apply(init_weights);
                        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                        loss = nn.MSELoss()
                        trainer = torch.optim.Adam(net.parameters(), lr=lr)
                        best_loss = 9999999999999
                        finished=False
                        no_improvement = 0
                        best_model_param = None
                        for epoch in range(num_epochs):
                            for X, y in train_iter:
                                l = loss(net(X).squeeze(), y)
                                trainer.zero_grad()
                                l.backward()
                                trainer.step()
                            if epoch == 0 or (epoch + 1) % 1 == 0:
                                with torch.no_grad():
                                    train_loss = evaluate_loss(net, train_iter, loss)
                                    val_loss = evaluate_loss(net, val_iter, loss)
                                    logging.info(" epoch: " + str(epoch + 1) + " train: " + str(train_loss) + " val: " + str(val_loss))
                                    if val_loss < best_loss:
                                        no_improvement = 0
                                        best_loss = val_loss
                                        best_model_param = copy.deepcopy(net.state_dict())
                                        if val_loss < best_overall:
                                            best_overall=val_loss
                                            if best_model != net:
                                                best_model=net
                                                logging.info("NEW BEST MODEL " + str(net))
                                    if val_loss > best_loss:
                                        no_improvement = no_improvement + 1
                                        if no_improvement > PATIENCE:
                                            timestring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                                            torch.save(best_model_param, "./models/"+ ref_var + "_" + bi +  "_layer_" + str(n_hidden_layers) + "_neuron_" + str(hidden_dim) + "_lr" + str(lr) +"_batchsize" + str(batch_size))
                                            logging.info("./models/" + ref_var + "_" + bi + "_layer_" + str(n_hidden_layers) + "_neuron_" + str(hidden_dim) + "_lr" + str(lr) +"_batchsize" + str(batch_size) + " completed. "+  " epoch: " + str(epoch + 1) + " train: " + str(train_loss) + " best val: " + str(best_loss))
                                            logging.info("finish time: "+ timestring)
                                            finished = True
                                            break
                        if not finished:
                            torch.save(best_model_param, "./models/"+ ref_var + "_" + bi + "_layer_" + str(n_hidden_layers) + "_neuron_" + str(hidden_dim) + "_lr" + str(lr) +"_batchsize" + str(batch_size))
                            logging.info("./models/"+ ref_var + "_" + bi +  "_layer_" + str(n_hidden_layers) + "_neuron_" + str(hidden_dim) + "_lr" + str(lr) +"_batchsize" + str(batch_size) + " completed. "+ " epoch: " + str(epoch + 1) + " train: " + str(train_loss) + " best val: " + str(best_loss))
        logging.info(best_overall)
    return best_model

n_train = np.int64(3e6)
n_val = np.int64(1e6)
lrs = [1e-3,]
bs = [1024]
layers = [3,]
hidden_dims = [64,]
num_epochs = 10
input_dim = 7
train_ds = MLDataset(X_train[0:n_train, :], Y_train[0:n_train])
val_ds = MLDataset(X_train[n_train:n_train+n_val, :], Y_train[n_train:n_train+n_val])
train(layers, input_dim, hidden_dims, lrs, bs, num_epochs, train_ds, val_ds, PATIENCE=10)


n_train = np.int64(20e6)
n_val = np.int64(3e6)

net = construct_model(7, 64, 3);
model_name = ref_var + "_" + bi +  "_layer_" + str(3) + "_neuron_" + str(64) + "_lr" + str(0.001) +"_batchsize" + str(1024)
model_dir="../notebooks/models"
net.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location=torch.device('cpu')))
net.eval();
net=net.to(device="cpu")


train_predict = net(torch.tensor(X_train)[0:n_train, :]).detach().numpy()[:, 0]


X_test = np.load(os.path.join(ML_PATH, "snow_X_test_{}_{}.npy".format(ref_var, bi))).astype(np.float32).T
Y_test = np.load(os.path.join(ML_PATH, "snow_Y_test_{}_{}.npy".format(ref_var, bi))).astype(np.float32)
test_valid = (np.sum(np.isnan(X_test), axis=1)==0) & np.invert(np.isnan(Y_test))
X_test = X_test[test_valid, :]
Y_test = Y_test[test_valid]


test_shuffle_idx = np.random.permutation(X_test.shape[0])
np.save(os.path.join(ML_PATH, "{}_test_idx.npy".format(ref_var)), test_shuffle_idx)
test_shuffle_idx = np.load(os.path.join(ML_PATH, "{}_test_idx.npy".format(ref_var)))
X_test_sample = X_test[test_shuffle_idx, :][0:500000, :]
Y_test_sample = Y_test[test_shuffle_idx][0:500000]
X_test_sample = scaler.transform(X_test_sample)

Y_predict_sample = net(torch.tensor(X_test_sample)).cpu().detach().numpy()[:, 0]


def evaluate_performance(true, predicted, ax, IGBP_name=None):
    res = linregress(predicted, true)
    #histogram definition
    bins = [100, 100] # number of bins

    # histogram the data
    hh, locx, locy = np.histogram2d(predicted, true, bins=bins)

    # Sort the points by density, so that the densest points are plotted last
    z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(predicted,true)])
    idx = z.argsort()
    x2, y2, z2 = predicted[idx], true[idx], z[idx]

    s = ax.scatter(x2, y2, c=z2, cmap='jet', marker='.')
    axis_min=min(predicted.min(), true.min())*1.1
    axis_max=max(predicted.max(), true.max())*1.1    
    ax.plot(np.arange(-10, 10), np.arange(-10, 10), "k--")
    ax.plot(predicted, res.intercept + res.slope*predicted, 'red', label='fitted line')
    ax.set_xticks(np.arange(np.floor(axis_min)+1, np.ceil(axis_max), 1))
    ax.set_yticks(np.arange(np.floor(axis_min)+1, np.ceil(axis_max), 1))
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.set_box_aspect(1)
    ax.text(0.15, 0.85, IGBP_name,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=14)
    ax.tick_params(direction="in")
    metric_dict=dict()
    metric_dict["r2"]=r2_score(true, predicted)
    metric_dict["mse"]=mean_squared_error(true, predicted, squared=False)
    metric_dict["mae"]=mean_absolute_error(true, predicted)
    
    ax.text(0.95, 0.19, 'N={0:.{1}f}'.format(true.shape[0], 0),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.12, '$R^2$: {0:.{1}f}'.format(metric_dict["r2"], 2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.05, 'RMSE: {0:.{1}f}'.format(metric_dict["mse"], 2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=12)

rcParams['font.family'] = 'Inter'
fig_dir = "../notebooks/figs/"
fig, axs=plt.subplots(1,2, figsize=(10,5), dpi=300)
ax=axs.flatten()
evaluate_performance(Y_train[0:n_train], train_predict,  ax[0], "Train")
evaluate_performance(Y_test_sample, Y_predict_sample, ax[1], "Test")

fig.text(0.5, 0.04, 'Predicted {} error'.format(ref_var), ha='center', va='center', fontsize=14)
fig.text(0.08, 0.5, 'Predicted {} error'.format(ref_var), ha='center', va='center', rotation='vertical', fontsize=14)

plt.savefig(os.path.join(fig_dir, model_name + "snow_{}_{}_train_test_together_v3.1_snow.png".format(ref_var, bi)), dpi=300)

