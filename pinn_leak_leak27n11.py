
# Import Packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
np.random.seed(12324)
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from torchsummary import summary


class PINNDataset(Dataset):
    def __init__(self, t_collo,t_obs,h_obs,):
        self.t_collo = t_collo.astype(np.float32)
        self.t_obs = t_obs.astype(np.float32)
        self.h_obs = h_obs.astype(np.float32)
   

    def __len__(self):
        return len(self.t_obs)

    def __getitem__(self, idx):
        return {
            't_collo': torch.tensor(self.t_collo[idx], requires_grad=True),
            't_obs': torch.tensor(self.t_obs[idx], requires_grad=True),
            'h_obs': torch.tensor(self.h_obs[idx], requires_grad=True),
        }


class PINNs(nn.Module):
    def __init__(self,  lb, ub, layers):
        super(PINNs, self).__init__()
        self.lb = lb
        self.ub = ub
        self.max_q=0.01
        self.max_h=50
        self.wf=10    # weight of loss pde
        self.layers = layers
        self.a1=0.05
        self.a2=500
        self.wd=0.000001
        # ensure m1, m2 are tensors
        self.m1 = torch.tensor(m1, dtype=torch.float32, device=device)
        self.m2 = torch.tensor(m2,  dtype=torch.float32,  device=device)
        self.m4 = torch.tensor(m4,  dtype=torch.float32,  device=device).view(1, -1)
        self.A1 = torch.tensor(A1,  dtype=torch.float32,  device=device)
        self.a3 = torch.tensor(a3.T,  dtype=torch.float32,  device=device)
        self.DI=nn.Parameter(torch.tensor(np.zeros(M_h), dtype=torch.float32,  device=device))
        # self.DI=torch.tensor(DI,  dtype=torch.float32,  device=device).view(1, -1)
        self.t_q=torch.tensor(t_q, dtype=torch.float32, device=device).view(-1,1)
        self.t0=torch.tensor(0, dtype=torch.float32, device=device).view(-1,1)
        # self.h0=torch.tensor(h0, dtype=torch.float32, device=device).view(1,-1)
        # self.q0=torch.tensor(q0, dtype=torch.float32, device=device).view(1,-1)
        # Initialize NN
        self.layers=nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i],layers[i+1]))

    def forward(self,t):
        out = 2.0 * (t  - self.lb[0] ) / (self.ub[0]  - self.lb[0]) - 1.0
        for layer in self.layers[:-1]:
            out= torch.tanh(layer(out))
        u= self.layers[-1](out)
        u[:,0:M_h],u[:,M_h:]=u[:,0:M_h]*self.max_h+100 ,u[:,M_h:]*self.max_q
        return u
    
    ## PDE as loss function. Thus would use the network which we call as u_theta
    def f(self,t):
        u = self.forward(t) # the dependent variable u is given by the network based on independent variables x,t
        h,q=u[:,0:M_h],u[:,M_h:]
        # dq/dt and dh/dt by automatic defferentiation
        h_t = torch.stack([torch.autograd.grad(h[:, j], t, grad_outputs=torch.ones_like(h[:, j]), create_graph=True)[0].squeeze(-1)
        for j in range(h.shape[1])], dim=1)  # shape: [2000, 19]
        q_t = torch.stack([torch.autograd.grad(q[:, j], t, grad_outputs=torch.ones_like(q[:, j]), create_graph=True)[0].squeeze(-1)
        for j in range(q.shape[1])], dim=1)  # shape: [2000, 19]
     
        
        # dq/dt and dh/dt by EEM,
        q_abs = q * torch.abs(q)  # element-wise q * |q|
        term1 = torch.matmul(q_abs, self.m1.T)  # [2000, 21]
        term2 = torch.matmul(h, self.m2.T)      # [2000, 21]

        qt_m = term1 + term2 + self.a3 
        #qt_m = qt_m[:,:-1] # exclude downstream reservoir
        # ht_m = (A1 @ q-self.DI@torch.sqrt(h))/m4
        term1 = torch.matmul(q, self.A1.T)                       # [2000, 21]
        term2 = torch.sqrt(torch.clamp(h, min=0.01)) * torch.nn.functional.softplus(self.DI, beta=10)*1e-4     # [2000, 21]
        ht_m = (-term1 - term2) / self.m4                         # [2000, 21]

        # residuals ff and gf are expected to be zero
        ff = (q_t - qt_m) /self.a1 # 0.01 is max(dq/dt) - min(dq/dt)
        #ff = torch.cat((ff[:, :15], ff[:, 16:]), dim=1)
        ff = ff[:,:-1] # exclude downstream reservoir. I can also use ff = ff[:, :37]     # keep pipes 0..36, exclude pipe 37 
        gf = (h_t  - ht_m) /self.a2  # 50 is max(dh/dt) - min(dh/dt)
        return h,q,ff,gf


    def trainNN(self, nIter):                     
        start_time = time.time()
        mse_cost_function = torch.nn.MSELoss() # Mean squared error
        self.directory=directory="models/"+uname+"/"
        # Create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # === Phase 1: Train branch network ===
        optimizer = torch.optim.Adam(self.parameters())
        for epoch in range(nIter):
            
            
            for batch in train_loader:
                t_obs = batch['t_obs'].to(device).reshape(-1,1)
                h_obs = batch['h_obs'].to(device)
                optimizer.zero_grad() # to make the gradients zero
                #LOSS
                # Loss based on observations h
                h,_,ff,gf= self.f( t_obs) # output of u(x,t)
                mse_uh = mse_cost_function(h[:,observations], h_obs)/self.max_h/self.max_h
                # Loss based on initial conditions
                u= self.forward(self.t0 ) # output of u(x,t)
                # mse_uh0 = mse_cost_function(u[:,:M_h], self.h0)/self.max_h/self.max_h
                # mse_uq0 = mse_cost_function(u[:,M_h:], self.q0)/self.max_q/self.max_q

                # Loss based on observations q
                q= self.forward(self.t_q)[:,M_h:] # output of u(x,t)
                # all_zeros_q= torch.zeros_like(q[:,15])
                # mse_uq = mse_cost_function(q[:,15], all_zeros_q)/self.max_q/self.max_q
                all_zeros_q= torch.zeros_like(q[:,37])
                mse_uq = mse_cost_function(q[:,37], all_zeros_q)/self.max_q/self.max_q
                #Loss based on ODEs
                all_zeros_f = torch.zeros_like(ff)
                all_zeros_g = torch.zeros_like(gf)
                mse_f = mse_cost_function(ff, all_zeros_f)+mse_cost_function(gf, all_zeros_g)
            
            loss =  mse_uh+mse_uq  +self.wf*mse_f+self.wd *torch.sum(torch.nn.functional.softplus(self.DI, beta=10))

            loss.backward() # This is for computing gradients using backward propagation
            optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                start_time = time.time()
                print(f"Epoch {epoch}, Training Loss: {loss.item()},Data Loss: {mse_uh.item()},ODE Loss: {mse_f.item()}, Time: {elapsed}")

            if epoch % 2000 == 0:
                # Save Model
                print(f"Epoch {epoch},DI {torch.nn.functional.softplus(self.DI, beta=10)}")
                torch.save(self.state_dict(), directory+uname+str(epoch)+".pt")
                self.validation(val_loader=val_loader)
            if  epoch>100000 and epoch% 2000 == 0:
                # Find indices of the top 2 values
                _, top2_indices = torch.topk(self.DI.data, 2)

                # Create a mask for the top 2 indices
                mask = torch.zeros_like(self.DI.data)
                mask[top2_indices] = 1.0

                # In-place multiplication to zero out all but top 2 values
                self.DI.data.mul_(mask)
    def validation(self,val_loader):
        model.eval()  # Switch to evaluation mode
        with torch.no_grad():  # Ensures no gradients are computed
            # Perform inference (predictions)
            for batch in val_loader:
                t_obs = batch['t_obs'].to(device).reshape(-1,1)
                h_obs = batch['h_obs'].to(device)
                # Reshape x_val, t_val to (batch_size*200*26, 1) to pass through the model

                pred= model.forward(t_obs)

                for i in range(12):
                    plt.figure(i+1)
                    plt.plot(dataseth[i+1],'k',label="True")
                    plt.plot(pred[:, i].cpu().numpy(),'--',label="PINN")
                    plt.legend()
                    # plt.xlim([0,10])
                    plt.xlabel("Time (s)")
                    plt.ylabel("Head (m)")
                    plt.title(f"Sample {i+1}")
                    plt.savefig(self.directory+uname+f"node{i}.png",dpi=300)
                    plt.close()

                # internal_nodes = list(range(1, 13))  # 1 to 12
                

                # for i, node in enumerate(internal_nodes):
                #     plt.figure(i + 1)

                #     plt.plot(dataseth[node], 'k', label="True")
                #     plt.plot(pred[:, i].cpu().numpy(), '--', label="PINN")

                #     plt.legend()
                #     plt.xlabel("Time (s)")
                #     plt.ylabel("Head (m)")
                #     plt.title(f"Node {node}")

                #     plt.savefig(self.directory + uname + f"_node{node}.png", dpi=300)
                #     plt.close()

        model.train()


#### parameter setting ######
#name = 'seven_pipe_4_reach'
name = 'nineteen_pipe_2_reach'
M_h = 12+1*19  # Number of nodes (12 = number of node, 1 = number of internal moc nodes per pipe and 12 = number of physical nodes)
M_p = 19*2  # Number of pipes
tt=10
tlen = 2000
#observations = [2, 5] # observation points  I will try [2, 5] later if 3, 6 work 
#observations = [0, 4, 9] # observation points 
observations = [2, 8, 11] # observation points  
len_obs=len(observations) # number of observation points
"""load data"""
nodeData=np.load(f"data19_2R_27L11_NEW/pinn_leak_19pipe_0.2id0_nodes_data.npz")
#nodeData=np.load(f"data/pinn_leak_7pipe_results.npz") 
#nodeData=np.load(f"data/pinn_leak_nodes_pressure_only.npz") # leak in node 7 original
#nodeData=np.load(f"data/pinn_leak_71_nodes_pressure_OK.npz")  # leak in between 1 - 6      
#nodeData=np.load(f"data/pinn_leak_0_7_nodes_pressure_only.npz")  # leak in between 0 - 7 
#dataseth=np.zeros((10,tlen))
dataseth=np.zeros((14,tlen)) # 14 nodes includes reservoir node 0 and 8, which are not included in the previous 10 nodes. also leak 
sensorData=np.zeros((tlen,3)) # 3 observation points
for i,j in enumerate(observations):
    array_name = f'node{j+1}'
    loaded_array = nodeData[array_name][:]
    sensorData[:,i]=loaded_array
for i in range(14):
    array_name = f'node{i}'
    loaded_array = nodeData[array_name][:]
    dataseth[i,:]=loaded_array


#print("dataseth[5, :] =")
#print(dataseth[5, :])
#print("shape =", dataseth[5, :].shape)
#plt.plot(dataseth[5, :])
#plt.title("dataseth[5, :] / node5 head")
#plt.show()


#### load EEM model ######
L2 = np.load("data19_2R_27L11_NEW/"+name + 'L2.npy')
C2 = np.load("data19_2R_27L11_NEW/"+name + 'C2.npy')
R2 = np.load("data19_2R_27L11_NEW/"+name + 'R2.npy')
A1 = np.load("data19_2R_27L11_NEW/"+name + 'A1.npy')
A2 = np.load("data19_2R_27L11_NEW/"+name + 'A2.npy')
B1 = np.load("data19_2R_27L11_NEW/"+name + 'B1.npy')
h2 = np.load("data19_2R_27L11_NEW/"+name + 'h2.npy')
C1 = np.load("data19_2R_27L11_NEW/"+name + 'C1.npy')
# h0=np.load('data/7_pipe_3_reach_h0.npy')
# q0=np.load('data/7_pipe_3_reach_q0.npy')
h2[1]=dataseth[13,:]
# DI=np.zeros(M_h)
# DI[-2]=0.0001*np.sqrt(2*9.81)
# intermediate constans
Linv = np.linalg.inv(L2) 
m1 = -Linv @ R2  
m2 = Linv @ A1.T  
m3 = Linv @ A2.T
a3 = m3 @ h2
m4 = ((0.5 * B1 @ (C1)))
###############
t_collo = np.linspace(0, tt, tlen) 
t_q = t_collo[40:]

batchSize=2000
train_dataset = PINNDataset( t_collo,t_collo,sensorData)
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_dataset = PINNDataset( t_collo,t_collo,sensorData)
val_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False)
lb = np.array([0.]) # low boundary of inputs (t)
ub = np.array([t_collo[-1]])  # up boundary of inputs (t)
# plt.plot(dataseth[6,:])
# plt.show()
# Input: t dim(2000,1) output: (2000,12)
layers = [1, 70, 70, 70, 70, 70, 70, 70,70, 70,  M_h + M_p]  # layers
model = PINNs( lb, ub, layers).to(device) # PINN model
uname = name+f"layers{len(layers)}_wd_{model.wd}"
print(uname)

model.trainNN(500000) # training
#model.trainNN(200000) # training


