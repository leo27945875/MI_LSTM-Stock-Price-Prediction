import torch
import torch.nn as nn
import torch.nn.functional as F


# Mean Absolute Percentage Loss:
def MAPELoss(output, target):
    output, target = output.detach().cpu(), target.detach().cpu()
    return torch.mean(torch.abs((target - output) / target)).item()


# Accuracy:
def Accuracy(output, target):
    output = torch.argmax(output, axis=1)
    return ((output == target).sum() / target.size(0)).item()


# Mish Activation Function:
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# Transformer模型中的encoder:
class TransformerEncoder(nn.Module):
    def __init__(self, embedSize, window, numHeads=4):
        super(TransformerEncoder, self).__init__()
        self.embedSize = embedSize
        self.attention = nn.MultiheadAttention(embedSize, numHeads)
        self.mlp = nn.Sequential(
                        nn.Dropout(0.1),
                        nn.Linear(embedSize, embedSize * 2),
                        Mish(),
                        nn.Linear(embedSize * 2, embedSize)
                   )
        self.layerNorm0 = nn.LayerNorm(embedSize)
        self.layerNorm1 = nn.LayerNorm(embedSize)
        
    
    def forward(self, x):
        xt = x.transpose(0, 1)
        h1 = self.attention(xt, xt, xt)[0]
        h2 = self.layerNorm0((h1 + xt).transpose(0, 1))
        h3 = self.mlp(h2)
        h4 = self.layerNorm1(h2 + h3)
        return h4 + x
                        

# 模型最前面的LSTM部分:
class Encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize, window, numLayers, dropout, 
                 bidirectional, isAttention, attentionHeads):
        super(Encoder, self).__init__()
        self.inputSize  = inputSize
        self.hiddenSize = hiddenSize
        self.bidirectional = bidirectional
        self.isAttention = isAttention
        self.lstm = nn.LSTM(
            inputSize, 
            hiddenSize, 
            numLayers, 
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        if isAttention:
            self.attention = TransformerEncoder(hiddenSize * 2 if bidirectional else hiddenSize, 
                                                window, 
                                                attentionHeads)
    
    
    def forward(self, y, index, pos, neg):
        device = y.device
        ouputSize = self.hiddenSize * 2 if self.bidirectional else self.hiddenSize
        
        # Mainstream:
        factorY, _ = self.lstm(y)
        
        # Index:
        factorI, _ = self.lstm(index)
        
        # Positive:
        factorP = torch.zeros([pos.size(0), pos.size(1), ouputSize]).to(device)
        for i in range(0, self.inputSize, pos.shape[-1]):
            factorP_I, _ = self.lstm(pos[..., i: i + self.inputSize])
            factorP += factorP_I
        
        factorP /= pos.shape[-1]
        
        # Negative:
        factorN = torch.zeros([neg.size(0), neg.size(1), ouputSize]).to(device)
        for i in range(0, self.inputSize, neg.shape[-1]):
            factorN_I, _ = self.lstm(neg[..., i: i + self.inputSize])
            factorN += factorN_I
        
        factorN /= neg.shape[-1]
        
        # Attention:
        if self.isAttention:
            factorY = self.attention(factorY)
            factorI = self.attention(factorI)
            factorP = self.attention(factorP)
            factorN = self.attention(factorN)
        
        # Output result:
        return factorY, factorI, factorP, factorN


# 模型中的Multi-Input LSTM:
class MI_LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(MI_LSTM, self).__init__()
        concatSize = inputSize + hiddenSize
        self.inputSize  = inputSize
        self.hiddenSize = hiddenSize
        self.lF  = nn.Linear(concatSize, hiddenSize)
        self.lO  = nn.Linear(concatSize, hiddenSize)
        self.lCY = nn.Linear(concatSize, hiddenSize)
        self.lCI = nn.Linear(concatSize, hiddenSize)
        self.lCP = nn.Linear(concatSize, hiddenSize)
        self.lCN = nn.Linear(concatSize, hiddenSize)
        self.lIY = nn.Linear(concatSize, hiddenSize)
        self.lII = nn.Linear(concatSize, hiddenSize)
        self.lIP = nn.Linear(concatSize, hiddenSize)
        self.lIN = nn.Linear(concatSize, hiddenSize)
        
        wA  = torch.nn.init.kaiming_normal_(torch.zeros([hiddenSize, hiddenSize]))
        bAY = torch.nn.init.uniform_(torch.zeros([hiddenSize]))
        bAI = torch.nn.init.uniform_(torch.zeros([hiddenSize]))
        bAP = torch.nn.init.uniform_(torch.zeros([hiddenSize]))
        bAN = torch.nn.init.uniform_(torch.zeros([hiddenSize]))
        self.wA  = nn.Parameter(wA)
        self.bAY = nn.Parameter(bAY)
        self.bAI = nn.Parameter(bAI)
        self.bAP = nn.Parameter(bAP)
        self.bAN = nn.Parameter(bAN)
        
        
    def forward(self, factorY, factorI, factorP, factorN):
        batch, window, device = factorY.size(0), factorY.size(1), factorY.device
        h  = torch.zeros([batch, window, self.hiddenSize]).to(device)
        cT = torch.zeros([batch, self.hiddenSize]).to(device)
        hT = torch.zeros([batch, self.hiddenSize]).to(device)
        for t in range(window):
            cT, hT = self.Step(factorY[:, t, :],
                               factorI[:, t, :],
                               factorP[:, t, :],
                               factorN[:, t, :],
                               cT, hT)
            h[:, t, :] = hT
        
        return h
    
    
    def Step(self, yT, iT, pT, nT, cT=None, hT=None):
        device, batch = yT.device, yT.size(0)
        if cT is None:
            cT = torch.zeros([batch, self.hiddenSize]).to(device)
        
        if hT is None:
            hT = torch.zeros([batch, self.hiddenSize]).to(device)
        
        hTyT = torch.cat([hT, yT], dim=-1)
        hTiT = torch.cat([hT, iT], dim=-1)
        hTpT = torch.cat([hT, pT], dim=-1)
        hTnT = torch.cat([hT, nT], dim=-1)
        
        f  = torch.sigmoid(self.lF (hTyT))
        o  = torch.sigmoid(self.lO (hTyT))
        iY = torch.sigmoid(self.lIY(hTyT))
        iI = torch.sigmoid(self.lII(hTyT))
        iP = torch.sigmoid(self.lIP(hTyT))
        iN = torch.sigmoid(self.lIN(hTyT))
        
        lY = torch.tanh(self.lCY(hTyT)) * iY
        lI = torch.tanh(self.lCI(hTiT)) * iI
        lP = torch.tanh(self.lCP(hTpT)) * iP
        lN = torch.tanh(self.lCN(hTnT)) * iN
        lT = self.GetAttention(lY, lI, lP, lN, cT)
        
        cNext = cT * f + lT
        hNext = torch.tanh(cNext) * o
        
        return cNext, hNext
        
        
    def GetAttention(self, lY, lI, lP, lN, cT):
        cTwA = cT @ self.wA
        attention = [
            torch.tanh((lY * cTwA).sum(dim=-1, keepdim=True) + self.bAY),
            torch.tanh((lI * cTwA).sum(dim=-1, keepdim=True) + self.bAI),
            torch.tanh((lP * cTwA).sum(dim=-1, keepdim=True) + self.bAP),
            torch.tanh((lN * cTwA).sum(dim=-1, keepdim=True) + self.bAN)
        ]
        attention = torch.cat(attention, dim=-1)
        attention = torch.softmax(attention, dim=-1)
        lT = (attention[:, 0: 1] * lY + 
              attention[:, 1: 2] * lI + 
              attention[:, 2: 3] * lP + 
              attention[:, 3: 4] * lN )
        return lT
        

# 模型中的Attention Layer:
class Attention(nn.Module):
    def __init__(self, inputSize, embedSize, window, attentionLayers, attentionHeads):
        super(Attention, self).__init__()
        self.inputSize = inputSize
        self.embedSize = embedSize
        self.attention = nn.Sequential(*[TransformerEncoder(inputSize, 
                                                            window, 
                                                            attentionHeads) 
                                         for _ in range(attentionLayers)])
        self.linear = nn.Linear(inputSize, embedSize)
        self.v = nn.Parameter(torch.nn.init.uniform_(torch.zeros([embedSize])))
        
    
    def forward(self, x):
        h = self.attention(x)
        h = torch.tanh(self.linear(h))
        b = torch.softmax(h @ self.v, dim=1).unsqueeze(-1)
        return (h * b).sum(dim=1)
        

# 模型最後面的FC層(regressor):
class Regressor(nn.Module):
    def __init__(self, inputSize, layers):
        super(Regressor, self).__init__()
        if layers > 1:
            blocks = ([nn.BatchNorm1d(inputSize)] + 
                      [self.MakeBlock(inputSize, inputSize * 2 ** (layers - 2))] + 
                      [self.MakeBlock(inputSize * 2 ** i, inputSize * 2 ** (i - 1))
                       for i in range(layers - 2, 0, -1)])
        elif layers == 1:
            blocks = []
        
        else:
            raise ValueError("Layers must greater than 1 .")
        
        self.extractor = nn.Sequential(*blocks)
        self.regressor = nn.Linear(inputSize, 1)
    
    
    def MakeBlock(self, inputSize, outputSize):
        return nn.Sequential(
            nn.Linear(inputSize, outputSize),
            Mish(),
            nn.BatchNorm1d(outputSize)
        )
    
    
    def forward(self, x):
        h = self.extractor(x)
        h = self.regressor(h)
        return h


# 模型最後面的FC層(classifier):
class Classifier(nn.Module):
    def __init__(self, inputSize, layers):
        super(Classifier, self).__init__()
        if layers > 1:
            blocks = ([nn.BatchNorm1d(inputSize)] + 
                      [self.MakeBlock(inputSize, inputSize * 2 ** (layers - 2))] + 
                      [self.MakeBlock(inputSize * 2 ** i, inputSize * 2 ** (i - 1))
                       for i in range(layers - 2, 0, -1)])
        elif layers == 1:
            blocks = []
        
        else:
            raise ValueError("Layers must greater than 1 .")
        
        self.extractor  = nn.Sequential(*blocks)
        self.classifier = nn.Linear(inputSize, 3)
    
    
    def MakeBlock(self, inputSize, outputSize):
        return nn.Sequential(
            nn.Linear(inputSize, outputSize),
            Mish(),
            nn.BatchNorm1d(outputSize)
        )
    
    
    def forward(self, x):
        h = self.extractor(x)
        h = self.classifier(h)
        return h
        

# 完整模型:
class Model(nn.Module):
    def __init__(self, 
                 window=30,
                 inputSize=1,
                 embedSize0=64, 
                 embedSize1=128, 
                 embedSize2=128,
                 encoderLayers=2,
                 encoderDropout=0.1,
                 encoderBidirectional=True,
                 encoderAttention=True,
                 attentionHeads=4,
                 attentionLayers=1,
                 regressorLayers=4):
        
        super(Model, self).__init__()
        self.window = window
        self.encoder = Encoder(inputSize, 
                               embedSize0, 
                               window,
                               encoderLayers, 
                               encoderDropout, 
                               encoderBidirectional,
                               encoderAttention,
                               attentionHeads)
        self.lstm = MI_LSTM(embedSize0 * 2 if encoderBidirectional else embedSize0, 
                            embedSize1)
        self.attention = Attention(embedSize1,
                                   embedSize2,
                                   window,
                                   attentionLayers,
                                   attentionHeads)
        self.regressor = Regressor(embedSize2,
                                   regressorLayers)
        
    
    def forward(self, y, index, pos, neg):
        h = self.encoder(y, index, pos, neg)
        h = self.lstm(*h)
        h = self.attention(h)
        h = self.regressor(h)
        return h
        

# 分類版本完整模型:
class ClassifierModel(nn.Module):
    def __init__(self, 
                 window=30,
                 inputSize=1,
                 embedSize0=64, 
                 embedSize1=128, 
                 embedSize2=128,
                 encoderLayers=2,
                 encoderDropout=0.1,
                 encoderBidirectional=True,
                 encoderAttention=True,
                 attentionHeads=4,
                 attentionLayers=1,
                 classificationLayers=4):
        
        super(ClassifierModel, self).__init__()
        self.window = window
        self.encoder = Encoder(inputSize, 
                               embedSize0, 
                               window,
                               encoderLayers, 
                               encoderDropout, 
                               encoderBidirectional,
                               encoderAttention,
                               attentionHeads)
        self.lstm = MI_LSTM(embedSize0 * 2 if encoderBidirectional else embedSize0, 
                            embedSize1)
        self.attention = Attention(embedSize1,
                                   embedSize2,
                                   window,
                                   attentionLayers,
                                   attentionHeads)
        self.classifier = Classifier(embedSize2,
                                     classificationLayers)
        
    
    def forward(self, y, index, pos, neg):
        h = self.encoder(y, index, pos, neg)
        h = self.lstm(*h)
        h = self.attention(h)
        h = self.classifier(h)
        return h
        
        
        