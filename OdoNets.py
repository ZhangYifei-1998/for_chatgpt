import torch
from torch import nn
from torch.nn import functional as F
import math
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

# 卷积轮速网络

class CovOdoNet(nn.Module):

    def __init__(self, channel1=10, channel2=64, channel3=32, fcn1=128, fcn2=64, out_dim=3):
        super(CovOdoNet, self).__init__()

        self.conv1 = nn.Conv2d(channel1, channel2, 1)
        self.conv2 = nn.Conv2d(channel2, channel3, 1)

        self.fc1 = nn.Linear(channel3*4*1, fcn1)
        self.fc2 = nn.Linear(fcn1, fcn2)
        self.fc3 = nn.Linear(fcn2, out_dim)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FcnOdoNet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FcnOdoNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 全连接轮速网络
class FcnOdoNetV2(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(FcnOdoNetV2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class FcnOdoNetV4(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(FcnOdoNetV4, self).__init__()
        #Relu = True 代表会在原变量的基础上修改，可以节省内存或者显存，还可以省去反复申请和释放内存的时间，对输出不会有影响，只是会覆盖输入，没事儿
        #这个相当于构建了6层，每一层激活函数是Relu，然后都是全连接层
        #使用sigmoid作为激活函数（指数函数），运算量大，反向传播求误差梯度时，求导设计出发，计算量特别大，relu没有这个问题
        #sigmoid函数反向传播时，很容易就出现梯度消失的情况（在sigmoid函数接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失），从而无法完成深层网络的训练
        #Relu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_5), nn.ReLU(True))
        self.bn = nn.BatchNorm1d(n_hidden_5)
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))

        self.init_weights()


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

class FcnOdoNetV3(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(FcnOdoNetV3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))

        self.init_weights()


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 2)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  
        out = self.out(r_out)
        return out


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
                    

class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.reg = nn.Linear(hidden_size, output_size) # 回归

    #def init_state(self, batch_size):
     #   return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
      #          torch.zeros(self.num_layers, batch_size, self.hidden_size))


   
 


    def forward(self, x):
      #  hs=self.init_state(batch_size)
        x, _ , = self.rnn(x,64) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()



#class LSTM(nn.Module):
#    def __init__(self, input_size, hidden_layer_size=200, output_size=2):
 #       super().__init__()
 #       self.hidden_layer_size = hidden_layer_size
#
#        self.lstm = nn.LSTM(input_size, hidden_layer_size)
#
 #       self.linear = nn.Linear(hidden_layer_size, output_size)

 #       self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(device),
 #                           torch.zeros(1,1,self.hidden_layer_size).cuda())

 #   def forward(self, input_seq):
  #      lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)

 #       predictions = self.linear(lstm_out.view(len(input_seq), -1))
 #       predictions[-1] = predictions[-1].squeeze(-1)

   #     return predictions[-1]




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
           )                  
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)


    def forward(self, input_seq,h_state):
        lstm_out, h_state = self.lstm(input_seq )[0]
       
        out=[]
        for time in range(lstm_out.size(1)):
            every_time_out = lstm_out[:, time, :]   
            out.append(self.output_layer(every_time_out))
        return torch.stack(out, dim=1),h_state




 
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=9144):
        
        super(PositionalEncoding, self).__init__()      
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)
 
    def forward(self, x):
      #  print("PositionalEncoding",x.size())
        
        
        return x + self.pe[:x.size(0), :]
          
 
class TransAm(nn.Module):
    def __init__(self,feature_size=600,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(600)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,6)
        self.init_weights()
        
        self.feature_size = feature_size
        self.num_layers   = num_layers
        self.dropout      = dropout
        
    def feature(self):
        return{"feature_size":self.feature_size,"num_layers":self.num_layers,"dropout":self.dropout}
        
 
    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
 
    def forward(self,src):
#         print("0",src.shape)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            print(src.shape[0])
            mask = self._generate_square_subsequent_mask(600).to(device)
            self.src_mask = mask
            print("a")
            # print(self.src_mask)
#         print("1",src.shape)
#         print("src:{}".format(src))
        src = self.pos_encoder(src)
        # src = src.permute(2, 1, 0)
        print(src.shape)
        print(self.src_mask.shape)
     #   print("2",src.shape)
        output = self.transformer_encoder(src,self.src_mask.unsqueeze(1))#, self.src_mask)
        # print("output:{}".format(output))
        output = output.view(output.shape[0], -1)
     #   print("3",output.shape)
        output = self.decoder(output)
    #    print('4',output.shape)

        return output
 
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask













   
