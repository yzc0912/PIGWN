import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import math
import torch.nn.init as init
from GWN.GWN import GWNET

class TimeAttention(nn.Module):
    def __init__(self, input_dim,time_step):
        super(TimeAttention, self).__init__()
        self.agg_linear = nn.Linear(input_dim, 1)
        self.time_linear = nn.Linear(time_step, time_step)  # 时间维度的注意力线性层
        self.relu = nn.ReLU()  # 添加激活函数

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.agg_linear.weight, gain=gain)
        nn.init.xavier_normal_(self.time_linear.weight, gain=gain)

    def forward(self, inputs):
        # 输入数据形状：(batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, input_dim = inputs.shape

        # 应用全连接层，得到聚合后的特征, (batch_size, seq_len, num_nodes)
        aggregated_features = self.agg_linear(inputs).squeeze(-1)
        aggregated_features = self.relu(aggregated_features)
        # (batch_size, num_nodes, seq_len)
        aggregated_features = aggregated_features.permute(0, 2, 1)

        # 对聚合后的特征应用时间维度的注意力机制，可以尝试一下先应用时间再聚合
        time_attention_weights = self.time_linear(aggregated_features)  # (batch_size, num_nodes, seq_len)
        time_attention_weights = F.softmax(time_attention_weights, dim=-1)  # 注意力权重
        # 将相邻两个时间刻的权值进行加和
        taw_sum = torch.stack([time_attention_weights[:, :, 2*i] + time_attention_weights[:, :, 2*i+1] 
                                                  for i in range(seq_len // 2)], dim=2)
        # 将注意力权重重塑为与输入相同的形状
        time_attention_weights = taw_sum.permute(0, 2, 1)  # (batch_size, seq_len//2, num_nodes)

        return time_attention_weights


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, device):
        super(SpatioConvLayer, self).__init__()
        self.ks = ks
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        # self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, graph_input):
        """
        Lk: (batch_size, Ks, num_nodes, num_nodes)
        x:  (batch_size, c_in, input_length, num_nodes)
        graph_input: (batch_size, num_nodes, num_nodes)
        x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        theta: (c_in, c_out, Ks)
        x_gc: (batch_size, c_out, input_length, num_nodes)
        return: (batch_size, c_out, input_length, num_nodes)
        """
        L = self.scaled_laplacian(graph_input)
        Lk = self.cheb_poly_approx(L, self.ks)
        x_c = torch.einsum("bknm,bitm->bitkn", Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        # x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        x_gc = F.dropout(x_gc, p=0.3, training=self.training)
        return torch.relu(x_gc)  # TODO: residual connection
    
    def scaled_laplacian(self, W):
        '''
        Normalized graph Laplacian function.
        :param W: torch.Tensor, [batch_size, n_route, n_route], weighted adjacency matrix of G.
        :return: torch.Tensor, [batch_size, n_route, n_route].
        '''
        batch_size, n = W.size(0), W.size(1)

        # Calculate diagonal degree matrix
        d = torch.sum(W, dim=2)

        # Construct graph Laplacian matrix
        L = -W.clone()
        L.view(batch_size, -1)[:, ::n + 1] = d  # Setting diagonal elements

        # Normalize Laplacian matrix
        d_sqrt = torch.sqrt(d.unsqueeze(2) * d.unsqueeze(1))
        L = L / d_sqrt
        # Calculate largest eigenvalue of L
        eigvals = torch.linalg.eigvals(L).real
        lambda_max = torch.max(eigvals, dim=1)[0]
        #print(lambda_max.device)
        #print("lambda_max:", lambda_max.shape)

        # Compute scaled Laplacian
        output = 2 * L / lambda_max.view(batch_size, 1, 1) - torch.eye(n).to(W.device)

        return output
    
    def cheb_poly_approx(self, L, Ks):
        '''
        Chebyshev polynomials approximation function.
        :param L: torch.Tensor, [batch_size, n_route, n_route], graph Laplacian.
        :param Ks: int, kernel size of spatial convolution.
        :param n: int, number of routes / size of graph.
        :return: torch.Tensor, [batch_size, n_route, Ks*n_route].
        '''
        # 获取 batch_size
        batch_size = L.size(0)
        n = L.size(1)
        # 创建零阶和一阶Laplacian矩阵
        L0 = torch.eye(n).unsqueeze(0).repeat(batch_size, 1, 1).to(L.device)  # 零阶矩阵
        L1 = L.clone()  # 一阶矩阵
        
        # 存储各阶Laplacian矩阵的列表
        L_list = [L0, L1]
        
        # 循环计算高阶Laplacian矩阵
        for i in range(Ks - 2):
            Ln = 2 * torch.bmm(L, L1) - L0
            L_list.append(Ln)
            L0, L1 = L1.clone(), Ln.clone()
        
        # 将所有Laplacian矩阵堆叠为一个张量
        L_stack = torch.stack(L_list, dim=1)  # [batch_size, Ks, n, n]
        
        return L_stack

class PhyCell(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        # TODO: 这里需要改一下args文件
        self.TimeAtt = TimeAttention(args.original_dim, args.per_step)
        self.time_step = args.per_step
        self.num_nodes = args.num_nodes
        self.input_dim = args.original_dim
        self._device = device
        self.gcn_ks = args.gcn_ks  
        self.c_in = args.gcn_c_in   
        self.c_out = args.gcn_c_out 
        self.gru_h_dim = args.gru_h_dim
        
        # 初始化GCN层
        self.gcn = SpatioConvLayer(self.gcn_ks, self.c_in, self.c_out, self._device)
        
        # 初始化GRU单元
        self.grucell = nn.GRUCell(self.time_step * self.c_out, (self.time_step//2) * self.gru_h_dim)  # 输入维度为GCN输出的两倍
        self.arr_linear = nn.Linear(self.gru_h_dim, 2)
    
    def forward(self, inputs_1, inputs_2, h_input, graph_all):
        """
        前向传播函数
        Args:
            inputs_1: 输入数据，shape为 (batch_size, seq_len, num_nodes, input_dim)
            inputs_2: 输入数据，shape为 (batch_size, seq_len/2, num_nodes, num_nodes)

        Returns:
            隐状态，shape为 (batch_size, seq_len//2, num_nodes, 2 * self.num_units)
        """
        batch_size, seq_len, num_nodes, input_dim = inputs_1.shape
        
        # GCN处理input2，得到图数据
        # attention_weights: (B, seq_len/2, N)
        attention_weights = self.TimeAtt(inputs_1).unsqueeze(-1)
        graph_inputs = torch.sum(inputs_2 * attention_weights, dim=1) # graph_input:(B, N, N),这里可能有点问题，可能需要归一
        graph_inputs = F.normalize(graph_inputs, p=1, dim=1)
        I = torch.eye(num_nodes, device=graph_inputs.device).unsqueeze(0)
        graph_all = graph_all.unsqueeze(0)
        graph_inputs = graph_inputs + I + graph_all

        # 图卷积
        inputs_1 = inputs_1.permute(0, 3, 1, 2) # (batch_size, input_dim, seq_len, num_nodes)
        gcn_outputs = self.gcn.forward(inputs_1, graph_inputs)  # gcn_out: (batch_size, c_out, input_length, num_nodes)

         # 调整h_input的形状，使其与gcn_outputs兼容
        h_input = torch.transpose(h_input, 1, 2) #(batch_size, num_nodes, seq_len//2, 2 * self.num_units)
        h_input = torch.reshape(h_input, (batch_size*num_nodes, -1))
        # TODO: 这里需要测试一下
        gcn_outputs = gcn_outputs.permute(0, 3, 2, 1) # (batch_size, num_nodes, input_length, c_out)
        gcn_outputs = torch.reshape(gcn_outputs, (batch_size*num_nodes, -1))
        # Pass gru_inputs through the GRU cell
        h1 = self.grucell(gcn_outputs, h_input)

        # Reshape the output hidden state to match the batch size and sequence length
        # gru_h_dim = 2
        h1 = h1.view(batch_size, num_nodes, seq_len//2, self.gru_h_dim)
        h1 = h1.permute(0, 2, 1, 3) # (batch_size, seq_len//2, num_nodes, 2 * self.num_units)
        # Apply linear transformation
        arr = torch.sigmoid(self.arr_linear(h1))
        # (batch_size, seq_len//2, num_nodes, 2)
        arr1 = arr[..., 0].unsqueeze(-1)  # 第一个通道
        arr2 = arr[..., 1].unsqueeze(-1)  # 第二个通道
        decay_matrix = torch.matmul(arr1, arr2.transpose(-2, -1))
        # (0, 1, 2) -> 4
        dep_input = inputs_2[:, :, :, :]
        temp = torch.sum(torch.matmul(dep_input, decay_matrix), dim=-2)
        phy_result = torch.sum(temp, dim=-2)        
        # Return the computed hidden state
        return h1, phy_result


    
class PIGWN(nn.Module):
    def __init__(self, args, device, dim_in, dim_out):
        super().__init__()
        self.device = device
        self.num_nodes = args.num_nodes
        self.args = args
        self.support = torch.tensor(args.adj_mx).to(self.device)

        self.time_step = args.per_step
        self.gru_h_dim = args.gru_h_dim
        self.input_window = args.input_window

        self.Phy = nn.ModuleList([PhyCell(self.args, self.device) for _ in range(0, self.input_window-self.time_step+1, 2)])
        self.gwn = GWNET(args, self.device, dim_in + 2, dim_out)
        self.final_linear = nn.Linear(self.time_step//2, self.input_window)
    
    def forward(self, input1, input2):
        """
        前向传播函数
        Args:
            input1: 输入1，shape为 (batch_size, seq_len, num_nodes, input_dim)
            input2: 输入2，shape为 (batch_size, seq_len//2, 2, num_nodes, num_nodes)

        """
        batch_size, seq_len, num_nodes, input_dim = input1.shape
        # Loop through time steps
        # 修改h1的形状为(batch_size, 3, num_nodes, hidden_size)
        h1 = torch.zeros(batch_size, 3, num_nodes, self.gru_h_dim, device=input1.device)
        phy_list = []
        for t1 in range(0, seq_len-5, 2):
            if t1 + 6 > seq_len:
                break
            # Get input slices
            input1_slice = input1[:, t1:t1+6]
            input2_slice = input2[:, t1//2:t1//2+3]

            # Forward pass through physical cell
            # (batch_size, seq_len//2, num_nodes, 2)
            h1, phy = self.Phy[t1//2](input1_slice, input2_slice, h1, self.support)
            phy_list.append(phy)
        stacked_phy = torch.stack(phy_list, dim=1) # (B, T, N)
        h1 = self.final_linear(h1.permute(0, 2, 3, 1)) # (batch_size, num_nodes, 2, seq_len//2)
        h1 = h1.permute(0, 3, 1, 2)
        stacked_input = torch.cat((input1, h1), dim=-1) # (batch_size, seq_len, num_nodes, input_dim+2)
        dl_result = self.gwn(stacked_input)

        return dl_result, stacked_phy
        
