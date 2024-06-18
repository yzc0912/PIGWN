import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, args_predictor):
        super(Model, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.model = args.model

        dim_in = self.input_base_dim
        dim_out = self.output_dim

        if self.model == 'MTGNN':
            from MTGNN.MTGNN import MTGNN
            self.predictor = MTGNN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STGCN':
            from STGCN.stgcn import STGCN
            self.predictor = STGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STSGCN':
            from STSGCN.STSGCN import STSGCN
            self.predictor = STSGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'ASTGCN':
            from ASTGCN.ASTGCN import ASTGCN
            self.predictor = ASTGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'GWN':
            from GWN.GWN import GWNET
            self.predictor = GWNET(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'TGCN':
            from TGCN.TGCN import TGCN
            self.predictor = TGCN(args_predictor, args.device, dim_in)
        elif self.model == 'STFGNN':
            from STFGNN.STFGNN import STFGNN
            self.predictor = STFGNN(args_predictor, dim_in)
        elif self.model == 'STGODE':
            from STGODE.STGODE import ODEGCN
            self.predictor = ODEGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'ST_WA':
            from ST_WA.ST_WA import STWA
            self.predictor = STWA(args_predictor, args.device, dim_in)
        elif self.model == 'MSDR':
            from MSDR.gmsdr_model import GMSDRModel
            args_predictor.input_dim = dim_in
            self.predictor = GMSDRModel(args_predictor, args.device)
        elif self.model == 'DMVSTNET':
            from DMVSTNET_demand.DMVSTNET import DMVSTNet
            self.predictor = DMVSTNet(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'CCRNN':
            from CCRNN_demand.CCRNN import EvoNN2
            self.predictor = EvoNN2(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STMGCN':
            from STMGCN_demand.STMGCN import ST_MGCN
            self.predictor = ST_MGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'new':
            print("正在使用模型new")
            from new.new import new
            self.predictor = new(args_predictor, args.device, dim_in, dim_out, departure_matrices_dim=self.num_node, arrival_matrices_dim=self.num_node)
        elif self.model == 'PIGWN':
            print("正在使用模型PIGWN")
            from PIGWN.PIGWN import PIGWN
            self.predictor = PIGWN(args_predictor, args.device, dim_in, dim_out)
        else:
            raise ValueError

    def forward(self, data, label=None, batch_seen=None, data_mat = None):
        if self.model == 'CCRNN':
            if label is None:
                x_predic = self.predictor(data[:, :, :, 0:self.input_base_dim], None, None)
            else:
                x_predic = self.predictor(data[:, :, :, 0:self.input_base_dim], label[:, :, :, 0:self.input_base_dim],
                                          None)
        elif self.model == 'new':
            x_predic = self.predictor(data[:, :, :, 0:self.input_base_dim], data_mat)
        elif self.model == 'PIGWN':
            x_predic, phy_predic = self.predictor(data[:, :, :, 0:self.input_base_dim], data_mat)
            # x_predic = self.predictor(data[:, :, :, 0:self.input_base_dim], data_mat)
            return x_predic, phy_predic
        else:
            x_predic = self.predictor(data[:, :, :, 0:self.input_base_dim])
        return x_predic, x_predic, x_predic, x_predic, x_predic
