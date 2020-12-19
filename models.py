import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        modules=list(resnet50.children())[:-1]
        resnet50=nn.Sequential(*modules)
        for p in resnet50.parameters():
            p.requires_grad = True
        
        self.img_emb = resnet50
        self.linear = nn.Linear(2048, 1024)

    def forward(self, img):
        output = self.img_emb(img)
        img_emb = self.linear(output.view(output.size()[0], -1))
        return img_emb

class AttrEncoder(nn.Module):

    def __init__(self, input_size):
        super(AttrEncoder, self).__init__()

        self.linear1 = nn.Linear(input_size, 2048)
        self.linear2 = nn.Linear(2048, 1024)

    def forward(self, attr_vect):
        inter_emb = torch.tanh(F.dropout(self.linear1(attr_vect)))
        final_emb = self.linear2(inter_emb)
        return final_emb


class AttrEmbeddingNetwork(nn.Module):

    def __init__(self, vocab_size):
        super(AttrEmbeddingNetwork, self).__init__()
        
        self.embedding_dim = 1024
        self.embeddings = nn.Embedding(vocab_size, 2048)
        self.linear = nn.Linear(2048, self.embedding_dim)

    def forward(self, attr):
        attr_emb = self.linear(F.tanh(self.embeddings(attr).squeeze(dim=1)))
        return attr_emb