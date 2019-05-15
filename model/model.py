import torch
import torch.nn as nn
from collections import OrderedDict


# base net: Conv_1D
class Conv_1D(nn.Module):
    def __init__(self, num_layers=4, num_in_c=3, num_out_c=4096, use_bn=False):
        super(Conv_1D, self).__init__()
        num_out_1 = 64
        factor = pow(num_out_1 / num_out_c, 1 / (num_layers - 1))
        sizes = [num_in_c] + [round(num_out_1 / pow(factor, i)) for i in range(num_layers)]
        layers = []
        for i in range(num_layers):
            layers.append(('Conv%d'%(i+1), nn.Conv1d(sizes[i], sizes[i+1], kernel_size=3, padding=1)))
            if use_bn:
                layers.append(('BN%d'%(i+1), nn.BatchNorm1d(sizes[i+1])))
            # add PReLU from 1 to n-1
            if i<num_layers-1:
                layers.append(('PReLU%d'%(i+1), nn.PReLU()))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # shape: bs*4096*n
        return self.model(x)


# base net: Conv_2D
class Conv_2D(nn.Module):
    def __init__(self, num_layers=8, num_in_c=2, num_out_c=512, use_bn=False):
        super(Conv_2D, self).__init__()
        num_out_1 = 8
        factor = pow(num_out_1 / num_out_c, 1 / (num_layers - 1))
        sizes = [num_in_c] + [round(num_out_1 / pow(factor, i)) for i in range(num_layers)]
        layers = []
        for i in range(num_layers):
            layers.append(('Conv%d'%(i+1), nn.Conv2d(sizes[i], sizes[i+1], kernel_size=3, padding=1)))
            if use_bn:
                layers.append(('BN%d'%(i+1), nn.BatchNorm2d(sizes[i+1])))
            # add PReLU from 1 to n-1
            if i<num_layers-1:
                layers.append(('PReLU%d'%(i+1), nn.PReLU()))
            layers.append(('MaxPool%d'%(i+1), nn.MaxPool2d(kernel_size=2)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # shape: bs*512*4*4
        return self.model(x)


# base net: FC layers
class FullyConnected(nn.Module):
    def __init__(self, num_layers=2, dim_input=128, dim_output=3, use_bn=False, use_sm=False, dropout_p=0.5):
        super(FullyConnected, self).__init__()
        factor = pow(dim_input / dim_output, 1 / num_layers)
        sizes = [round(dim_input / pow(factor, i)) for i in range(num_layers + 1)]
        layers = []
        for i in range(num_layers):
            # add fc layer
            layers.append(('Linear%d'%(i+1), nn.Linear(sizes[i], sizes[i+1])))
            # add dropout layer if this layer has too much weights
            if sizes[i]*sizes[i+1]>1e+5:
                layers.append(('Dropout%d'%(i+1), nn.Dropout(p=dropout_p)))
            # add bn if use bn
            if use_bn:
                layers.append(('BN%d'%(i+1), nn.BatchNorm1d(sizes[i+1])))
            # add PReLU from 1 to n-1
            if i<num_layers-1:
                layers.append(('PReLU%d'%(i+1), nn.PReLU()))
            # add softmax to last layer if use softmax (for classification)
            if use_sm and i==num_layers-1:
                layers.append(('Softmax', nn.Softmax(dim=1)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


# Encoder: branch 1: Multi-view CNN
class MVCNN(nn.Module):
    def __init__(self):
        super(MVCNN, self).__init__()
        self.single_conv = Conv_2D(num_layers=8, num_in_c=2, num_out_c=512)
        self.fc = FullyConnected(num_layers=2, dim_input=512*4*4, dim_output=512)

    def forward(self, x):
        batch_size, num_views, num_channels, h, w = x.shape
        view_pool = []
        for i in range(num_views):
            # shape: num_views, batch_size, num_channels, h, w
            view_pool.append(self.single_conv(x[:,i]))
        view_pooled = torch.max(torch.stack(view_pool), dim=0)[0]
        view_pooled = view_pooled.view(batch_size, -1)
        feature = self.fc(view_pooled)
        return feature


# Encoder: branch 2: PointNet
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv = Conv_1D(num_layers=4, num_in_c=3, num_out_c=4096)
        self.fc = FullyConnected(num_layers=2, dim_input=4096, dim_output=512)

    def forward(self, x):
        batch_size, num_channels, num_points = x.shape
        x = self.conv(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


# Encoder: full Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mvcnn = MVCNN()
        self.point_net = PointNet()
        self.fuse = FullyConnected(num_layers=2, dim_input=1024, dim_output=128)

    def forward(self, img, pts):
        feature1 = self.mvcnn(img)
        feature2 = self.point_net(pts)
        feature = torch.cat((feature1, feature2), dim=1)
        feature = self.fuse(feature)
        return feature


# Decoder: one element of the recursive decoder
class Element(nn.Module):
    def __init__(self):
        super(Element, self).__init__()
        self.split_node = FullyConnected(num_layers=2, dim_input=128, dim_output=128+128)
        self.similar_node = FullyConnected(num_layers=2, dim_input=128, dim_output=128+8)
        self.shape_node = FullyConnected(num_layers=2, dim_input=128, dim_output=8)

    def forward(self, x, node_class=None):
        """node_class: 0--split_node; 1--similar_node; 2--shape_node"""
        if node_class==0:
            x = self.split_node(x)
            return x[:, 0:128], x[:, 128:]
        elif node_class==1:
            x = self.similar_node(x)
            return x[:, 0:128], x[:, 128:]
        elif node_class==2:
            x = self.shape_node(x)
            return x
        else:
            return False


# Decoder: full Decoder
class Decoder(nn.Module):
    """node_class: 0--split_node; 1--similar_node; 2--shape_node"""
    def __init__(self):
        super(Decoder, self).__init__()
        self.element = Element()
        self.node_classifier = FullyConnected(num_layers=3, dim_input=128, dim_output=3, use_sm=True)
        self.mse = nn.MSELoss()

    def forward(self, x, tree=None):
        # NOTE: these should be written in forward, so that they will be reset for each sample
        sims_pred = []
        shapes_pred = []
        classes_pred = []

        # NOTE: this should be written in forward, otherwise the loss will not be updated
        def recursion(x, node_index=0):
            # current node
            node = tree[node_index]
            children_index = self.get_children(node_index)

            # node classification
            node_class_label = node.node_class
            # node_class_pred = torch.argmax(self.node_classifier(x), dim=1).float().requires_grad_()
            node_class_pred = self.node_classifier(x).float().requires_grad_()
            classes_pred.append((node_index, node_class_pred))

            # decide training or test?
            if tree is not None:
                node_class = node_class_label
            else:
                node_class = node_class_pred

            # recursion
            if node_class == 0:
                left, right = self.element(x, node_class=node_class_label)
                recursion(left, node_index=children_index[0])
                recursion(right, node_index=children_index[1])
            elif node_class == 1:
                # prediction
                left, sim_pred = self.element(x, node_class=node_class_label)
                sims_pred.append((node_index, sim_pred))
                # recursion
                recursion(left, node_index=children_index[0])
            elif node_class == 2:
                # prediction
                shape_pred = self.element(x, node_class=node_class_label)
                shapes_pred.append((node_index, shape_pred))
            else:
                return False

        recursion(x, node_index=0)
        return classes_pred, sims_pred, shapes_pred

    @ staticmethod
    def get_children(index):
        """Given an index in {0,...,126}, return its left and right children in {0,...,126}"""

        def index2level(index):
            """Given an index in {0,...,126}, return its level and index in current level, from *ZERO*"""
            import numpy as np
            level = int(np.floor(np.log2(index + 1)))
            index_in_level = index - pow(2, level) + 1
            return level, index_in_level

        def level2index(level, index_in_level):
            """Given a level and index in current level, from *ZERO*, return an index in {0,...,126}"""
            index = pow(2, level) - 1 + index_in_level
            return index

        level, index_in_level = index2level(index)
        child_level = level + 1
        child_index_in_level = index_in_level * 2
        child_index = level2index(child_level, child_index_in_level)
        return child_index, child_index + 1


# FULL MODEL
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, imgs, pts, tree=None):
        feature = self.encoder(imgs, pts)
        classes_pred, sims_pred, shapes_pred = self.decoder(feature, tree)
        return classes_pred, sims_pred, shapes_pred


if __name__ == '__main__':

    import time
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt

    def train_encoder():
        """"debug the encoder"""
        USE_CUDA = torch.cuda.is_available()
        # USE_CUDA = False
        print('######## ENCODER ########')
        encoder = Encoder()
        print(encoder)
        print('\n')
        n_params = sum(param.numel() for param in encoder.parameters())
        print('Number of parameters: ', n_params)

        num_samples = 128
        batch_size = 1

        num_views, num_channels, h, w = 4, 2, 1024, 1024
        train_data_img = torch.rand(num_samples, batch_size, num_views, num_channels, h, w)

        num_channels, num_pts = 3, 4096
        train_data_pts = torch.rand(num_samples, batch_size, num_channels, num_pts)

        train_label = torch.rand(1, 128)

        mse_loss = nn.MSELoss()

        # cuda
        if USE_CUDA:
            encoder = encoder.cuda()
            train_label = train_label.cuda()

        time_start = time.time()

        loss_all = []
        for i in range(num_samples):
            # train_input_img = train_data_img[i]
            # train_input_pts = train_data_pts[i]
            # # we just train the first sample
            train_input_img = train_data_img[0]
            train_input_pts = train_data_pts[0]
            if USE_CUDA:
                train_input_img = train_input_img.cuda()
                train_input_pts = train_input_pts.cuda()
            train_pred = encoder(train_input_img, train_input_pts)
            loss = mse_loss(train_pred, train_label)

            print('Loss: %.4f' %loss.item())

            lr = 0.001 * pow(1e-4, i / 128)
            # optimizer = optim.SGD(decoder.parameters(), lr=lr)
            optimizer = optim.Adam(encoder.parameters(), lr=lr)

            # zero grade
            optimizer.zero_grad()

            # backward + optimization
            loss_all.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()

        import matplotlib.pyplot as plt
        plt.plot(loss_all)
        plt.grid(True)
        plt.show()

        print('Size of train_input: ', train_input_img.shape, train_input_pts.shape)
        print('Size of train_pred: ', train_pred.shape)
        print('Time: ', time.time() - time_start)
        print('######## END ENCODER ########\n\n')

    def train_decoder():
        """"debug the decoder"""

        # USE_CUDA = torch.cuda.is_available()
        USE_CUDA = False

        # Training data
        class Node(object):
            """Each node includes a index in {0,...,126}, a node_class in {0,1,2},
            and node data (only similar node and shape node have data)"""
            def __init__(self, index=None, node_class=None, sim=None, shape=None):
                self.index = int(index)
                self.parent_index = self.get_parent(self.index)
                self.children_index = self.get_children(self.index)
                # node_class: 0--split_node; 1--similar_node; 2--shape_node
                self.node_class = node_class
                self.sim = sim      # similar parameters data
                self.shape = shape  # shape data

            @staticmethod
            def index2level(index):
                """Given an index in {0,...,126}, return its level and index in current level, from *ZERO*"""
                level = int(np.floor(np.log2(index+1)))
                index_in_level = index - pow(2, level) + 1
                return level, index_in_level

            @staticmethod
            def level2index(level, index_in_level):
                """Given a level and index in current level, from *ZERO*, return an index in {0,...,126}"""
                index = pow(2, level) - 1 + index_in_level
                return index

            def get_parent(self, index):
                """Given an index in {0,...,126}, return its parent in {0,...,126}"""
                level, index_in_level = self.index2level(index)
                parent_level = level - 1
                parent_index_in_level = int(np.floor(index_in_level / 2))
                parent_index = self.level2index(parent_level, parent_index_in_level)
                return parent_index

            def get_children(self, index):
                """Given an index in {0,...,126}, return its left and right children in {0,...,126}"""
                level, index_in_level = self.index2level(index)
                child_level = level + 1
                child_index_in_level = index_in_level*2
                child_index = self.level2index(child_level, child_index_in_level)
                return child_index, child_index+1


        tree = []
        tree.append(Node(index=0, node_class=0, sim=None, shape=None))
        tree.append(Node(index=1, node_class=0, sim=None, shape=None))
        tree.append(Node(index=2, node_class=1, sim=np.random.randn(8), shape=None))
        tree.append(Node(index=3, node_class=1, sim=np.random.randn(8), shape=None))
        tree.append(Node(index=4, node_class=1, sim=np.random.randn(8), shape=None))
        tree.append(Node(index=5, node_class=2, sim=None, shape=np.random.randn(8)))
        tree.append(Node(index=6, node_class=None, sim=None, shape=None))
        tree.append(Node(index=7, node_class=2, sim=None, shape=np.random.randn(8)))
        tree.append(Node(index=8, node_class=None, sim=None, shape=None))
        tree.append(Node(index=9, node_class=2, sim=None, shape=np.random.randn(8)))
        tree.append(Node(index=10, node_class=None, sim=None, shape=None))
        tree.append(Node(index=11, node_class=None, sim=None, shape=None))
        tree.append(Node(index=12, node_class=None, sim=None, shape=None))
        tree.append(Node(index=13, node_class=None, sim=None, shape=None))
        tree.append(Node(index=14, node_class=None, sim=None, shape=None))

        sims_label = torch.FloatTensor([node.sim for node in tree if node.node_class==1])
        shapes_label = torch.FloatTensor([node.shape for node in tree if node.node_class==2])
        classes_label = torch.FloatTensor([node.node_class for node in tree if node.node_class is not None])

        # train the decoder

        feature = torch.rand(1, 128)
        decoder = Decoder()
        print('######## DECODER ########')
        print(decoder)
        print('\n')
        n_params = sum(param.numel() for param in decoder.parameters())
        print('Number of parameters: ', n_params)
        print('\n')

        # CUDA
        if USE_CUDA:
            decoder = decoder.cuda()
            feature = feature.cuda()
            sims_label = sims_label.cuda()
            shapes_label = shapes_label.cuda()
            classes_label = classes_label.cuda()

        mse_loss = nn.MSELoss()
        cross_entropy = nn.CrossEntropyLoss()

        loss_all = []
        for i in range(3000):

            lr = 0.01 * pow(1e-4, i / 3000)
            # lr = 0.001

            # optimizer = optim.SGD(decoder.parameters(), lr=lr)
            optimizer = optim.Adam(decoder.parameters(), lr=lr)

            # zero grade
            optimizer.zero_grad()
            # decoder.zero_grad()

            # forword
            classes_pred, sims_pred, shapes_pred = decoder(feature, tree=tree)

            # loss
            classes_pred.sort()
            classes_pred = [i[1] for i in classes_pred]
            # print(classes_pred[0])
            # print(classes_label[0])
            sims_pred.sort()
            sims_pred = [i[1] for i in sims_pred]
            shapes_pred.sort()
            shapes_pred = [i[1] for i in shapes_pred]
            loss = 0
            for j in range(len(sims_pred)):
                loss += torch.sum(torch.abs(sims_pred[j] - sims_label[j]))
                loss += torch.sum(torch.abs(shapes_pred[j] - shapes_label[j]))
            for j in range(len(classes_pred)):
                loss += cross_entropy(classes_pred[j], classes_label[j].unsqueeze(0).long())
            # loss = cross_entropy(classes_pred[0], classes_label[0].unsqueeze(0).long())
            # print(classes_pred[0])

            if i%25==0:
                print('Loss: %.4f' %loss.item())

            # # DEBUG: see weights
            # param = list(decoder.parameters())[0][0][0]
            # # print(param.shape)
            # print(param)  # see if weight was updated

            # backward + optimization
            loss_all.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()

            # # DEBUG: see grad
            # print(decoder.element.split_node.model.Linear1.bias.grad[0])

        plt.plot(loss_all)
        plt.grid(True)
        plt.show()

        # print(sims_pred[0])
        # print(sims_label[0])

        # # DEBUG: see grad functions
        # def recursion(x, loss):
        #     if x>0:
        #         loss = loss.next_functions[0][0]
        #         print(loss)
        #         recursion(x-1, loss)
        #     else:
        #         pass
        # recursion(10, loss.grad_fn)

        print('######## END DECODER ########')

    train_encoder()
    train_decoder()
