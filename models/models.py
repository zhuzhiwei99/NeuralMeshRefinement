import sys

sys.path.append('../')
from include import *


class MLP(torch.nn.Module):
    def __init__(self, Din, Dhid, Dout, activation='relu'):

        '''
        Din: input dimension
        Dhid: a list of hidden layer size
        Dout: output dimension
        '''
        super(MLP, self).__init__()

        self.layerIn = torch.nn.Linear(Din, Dhid[0])
        self.hidden = torch.nn.ModuleList()
        for ii in range(len(Dhid) - 1):
            self.hidden.append(torch.nn.Linear(Dhid[ii], Dhid[ii + 1]))
        self.layerOut = torch.nn.Linear(Dhid[-1], Dout)
        if activation == 'relu':
            self.relu = torch.nn.ReLU()
        elif activation == 'leakyrelu':
            self.relu = torch.nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'sigmoid':
            self.relu = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layerIn(x)
        x = self.relu(x)
        for ii in range(len(self.hidden)):
            x = self.hidden[ii](x)
            x = self.relu(x)
        x = self.layerOut(x)
        return x


class NMRNet(torch.nn.Module):
    # Neural Mesh Refinement network
    def __init__(self, Din, Dout, numSubd, scale=True):
        super(NMRNet, self).__init__()
        self.Din = Din
        self.Dout = Dout
        self.numSubd = numSubd  # number of subdivisions
        self.scale = scale  # if normalize the scale
        # three MLPs
        self.edge_feature_embedding = MLP(Din * 4 - 3, [Dout, Dout], Dout)
        self.graph_attention_aggregation = MLP(3 + Dout, [Dout, Dout], Dout, activation='leakyrelu')
        self.vertex_reposition = MLP(Dout, [32, 16], 3)

        self.pool = torch.nn.AvgPool2d((2, 1))  # half-edge pool

        # print network architecture
        print('NMRNet Parameters:')
        print('numSubd:', self.numSubd)
        print('scale normalization:', self.scale)
        print('edge_feature_embedding: ', self.edge_feature_embedding)
        print('graph_attention_aggregation: ', self.graph_attention_aggregation)
        print('vertex_reposition: ', self.vertex_reposition)

    def getMidpointSixHalfflap(self, poolmat, even_num, vertex_num):
        half_edge = poolmat._indices().transpose(0, 1)  # 2nE x 2
        half_edge_num = half_edge.shape[0]
        midpoint_6 = (vertex_num - even_num) * 6
        get_num = half_edge_num - midpoint_6
        _, sort_indices = torch.sort(half_edge[:, 0], dim=0)
        sort_half_edge = half_edge[sort_indices]
        midpoint_sort_half_edge = sort_half_edge[get_num:, 1]
        midpoint_6_hf = midpoint_sort_half_edge.reshape(-1, 6)

        return midpoint_6_hf

    def calHalfEdgeLength(self, fv, hfIdx):
        fv = fv[:, :3]
        half_edge = hfIdx[:, 0:2]
        half_edge_pos0 = fv[half_edge[:, 0], :]
        half_edge_pos1 = fv[half_edge[:, 1], :]
        half_edge_length = torch.norm(half_edge_pos0 - half_edge_pos1, dim=1)
        return half_edge_length

    def sixVertexNormalization(self, sort_midpoint_6hfIdx_fv, normalizeFeature=False, normalizeScale=True):

        """
        normalize the features of a one-ring-six-edge graph so that it is orientation and translation invariant
        sort_midpoint_6hfIdx_fv : Nv x 6 x 4 x Din
        """
        V_hf = sort_midpoint_6hfIdx_fv[:, 0, :, :3]

        F = torch.tensor([[0, 1, 2], [1, 0, 3]])  # half flap face list
        # 1st frame: edge vector
        b1 = (V_hf[:, 1, :] - V_hf[:, 0, :]) / torch.norm(V_hf[:, 1, :] - V_hf[:, 0, :], dim=1).unsqueeze(1)
        # 3rd frame: edge normal (avg of face normals)
        vec1 = V_hf[:, F[:, 1], :] - V_hf[:, F[:, 0], :]
        vec2 = V_hf[:, F[:, 2], :] - V_hf[:, F[:, 0], :]
        FN = torch.cross(vec1, vec2)  # nF x 2 x 3
        FNnorm = torch.norm(FN, dim=2)
        FN = FN / FNnorm.unsqueeze(2)

        eN = FN[:, 0, :] + FN[:, 1, :]

        b3 = eN / torch.norm(eN, dim=1).unsqueeze(1)

        # 2nd frame: their cross product
        b2 = torch.cross(b3, b1)

        # concatenage all local frames
        b1 = b1.unsqueeze(1)
        b2 = b2.unsqueeze(1)
        b3 = b3.unsqueeze(1)
        localFrames = torch.cat((b1, b2, b3), dim=1)  # Nv x 3 x 3

        # normalize features
        sort_midpoint_6hfIdx_pos = sort_midpoint_6hfIdx_fv[:, :, :, :3]  # half flap vertex position Nv x 6 x 4 x 3
        sort_midpoint_6hfIdx_feature = sort_midpoint_6hfIdx_fv[:, :, :, 3:]  # half flap features Nv x 6 x 4 x 3
        sort_midpoint_6hfIdx_pos = sort_midpoint_6hfIdx_pos - sort_midpoint_6hfIdx_pos[:, :, 0, :].unsqueeze(
            2)  # translate
        sort_midpoint_6hfIdx_pos = sort_midpoint_6hfIdx_pos.view(sort_midpoint_6hfIdx_pos.size(0), 24, 3)
        sort_midpoint_6hfIdx_pos = torch.bmm(sort_midpoint_6hfIdx_pos, torch.transpose(localFrames, 1, 2))
        sort_midpoint_6hfIdx_pos = sort_midpoint_6hfIdx_pos.view(sort_midpoint_6hfIdx_pos.size(0), 6, 4, 3)

        normalize_length = torch.norm(sort_midpoint_6hfIdx_pos[:, 0, 1, :], dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(
            1)
        if normalizeScale:
            sort_midpoint_6hfIdx_pos = sort_midpoint_6hfIdx_pos / normalize_length

        if normalizeFeature:  # if also normalize the feature using local frames
            assert (sort_midpoint_6hfIdx_feature.size(3) == 3 or sort_midpoint_6hfIdx_feature.size(3) == 6)
            if sort_midpoint_6hfIdx_feature.size(3) == 3:
                sort_midpoint_6hfIdx_feature = sort_midpoint_6hfIdx_feature.view(sort_midpoint_6hfIdx_feature.size(0),
                                                                                 24, 3)
                sort_midpoint_6hfIdx_feature = torch.bmm(sort_midpoint_6hfIdx_feature,
                                                         torch.transpose(localFrames, 1, 2))
                sort_midpoint_6hfIdx_feature = sort_midpoint_6hfIdx_feature.view(sort_midpoint_6hfIdx_feature.size(0),
                                                                                 6, 4, 3)
                if normalizeScale:
                    sort_midpoint_6hfIdx_feature = sort_midpoint_6hfIdx_feature / normalize_length
            elif sort_midpoint_6hfIdx_feature.size(3) == 6:
                sort_midpoint_6hfIdx_feature1 = sort_midpoint_6hfIdx_feature[:, :, :, :3].view(
                    sort_midpoint_6hfIdx_feature.size(0),
                    24, 3)
                sort_midpoint_6hfIdx_feature1 = torch.bmm(sort_midpoint_6hfIdx_feature1,
                                                          torch.transpose(localFrames, 1, 2))
                sort_midpoint_6hfIdx_feature1 = sort_midpoint_6hfIdx_feature1.view(sort_midpoint_6hfIdx_feature.size(0),
                                                                                   6, 4, 3)
                if normalizeScale:
                    sort_midpoint_6hfIdx_feature1 = sort_midpoint_6hfIdx_feature1 / normalize_length
                sort_midpoint_6hfIdx_feature2 = sort_midpoint_6hfIdx_feature[:, :, :, 3:].view(
                    sort_midpoint_6hfIdx_feature.size(0), 24, 3)
                sort_midpoint_6hfIdx_feature2 = torch.bmm(sort_midpoint_6hfIdx_feature2,
                                                          torch.transpose(localFrames, 1, 2))
                sort_midpoint_6hfIdx_feature2 = sort_midpoint_6hfIdx_feature2.view(sort_midpoint_6hfIdx_feature.size(0),
                                                                                   6, 4, 3)
                if normalizeScale:
                    sort_midpoint_6hfIdx_feature2 = sort_midpoint_6hfIdx_feature2 / normalize_length
                sort_midpoint_6hfIdx_feature = torch.cat((sort_midpoint_6hfIdx_feature1, sort_midpoint_6hfIdx_feature2),
                                                         dim=3)

        sort_midpoint_6hfIdx_normalize = torch.cat((sort_midpoint_6hfIdx_pos, sort_midpoint_6hfIdx_feature),
                                                   dim=3)  # Nv x 6 x 4 x 6
        return sort_midpoint_6hfIdx_normalize, localFrames, normalize_length.squeeze(1)

    def v26hf(self, fv, hfIdx, poolmat, even_num, normalizeFeature=True, normalizeScale=True):
        midpoint_6hf = self.getMidpointSixHalfflap(poolmat, even_num, fv.shape[0])
        half_edge_length = self.calHalfEdgeLength(fv, hfIdx)
        midpoint_6edge_length = half_edge_length[midpoint_6hf]
        # longest edge
        _, sort_indices = torch.sort(midpoint_6edge_length, dim=1, descending=True)
        sort_midpoint_6hf = midpoint_6hf[torch.arange(midpoint_6hf.shape[0])[:, None], sort_indices]
        sort_midpoint_6hfIdx = hfIdx[sort_midpoint_6hf]  # Nv x 6 x 4
        sort_midpoint_6hfIdx_fv = fv[sort_midpoint_6hfIdx]  # Nv x 6 x 4 x Din

        sort_midpoint_6hfIdx_fv_normalize, localFrames, normalize_length = \
            self.sixVertexNormalization(sort_midpoint_6hfIdx_fv, normalizeFeature=normalizeFeature,
                                        normalizeScale=normalizeScale)
        sort_midpoint_6hfIdx_fv_normalize = sort_midpoint_6hfIdx_fv_normalize.view(
            sort_midpoint_6hfIdx_fv_normalize.size(0), 6, -1)
        sort_midpoint_6hfIdx_fv_normalize = sort_midpoint_6hfIdx_fv_normalize[:, :, 3:]  # remove the first 3 components as they are always (0,0,0)
        if not normalizeScale:
            normalize_length = None
        return sort_midpoint_6hfIdx_fv_normalize, localFrames, normalize_length

    def local2Global(self, hf_local, LFs, normalize_length):
        '''
        LOCAL2GLOBAL turns position features (the first three elements) described in the local frame of an half-flap to world coordinates
        '''
        if normalize_length is None:
            hf_local_pos = hf_local[:, :3]
        else:
            hf_local_pos = hf_local[:, :3] * normalize_length.squeeze(1)  # get the vertex position features
        hf_feature = hf_local[:, 3:]  # get the high-dim features
        c0 = hf_local_pos[:, 0].unsqueeze(1)
        c1 = hf_local_pos[:, 1].unsqueeze(1)
        c2 = hf_local_pos[:, 2].unsqueeze(1)
        hf_global_pos = c0 * LFs[:, 0, :] + c1 * LFs[:, 1, :] + c2 * LFs[:, 2, :]
        hf_global = torch.cat((hf_global_pos, hf_feature), dim=1)
        return hf_global

    def halfEdgePool(self, fhe):
        '''
        average pooling of half edge features
        '''
        fhe = fhe.unsqueeze(0).unsqueeze(0)
        fe = self.pool(fhe)
        fe = fe.squeeze(0).squeeze(0)
        return fe

    def edgeMidPoint(self, fv, hfIdx):
        '''
        get the mid point position of each edge
        '''
        Ve0 = fv[hfIdx[:, 0], :3]
        Ve1 = fv[hfIdx[:, 1], :3]
        Ve = (Ve0 + Ve1) / 2.0
        Ve = self.halfEdgePool(Ve)
        return Ve

    def getLaplaceCoordinate(self, V, HF, poolMat, dof):
        """
        get the vectors of the differential coordinates
        Inputs:
            hfList: half flap list (see self.getHalfFlap)
            poolMats: vertex one-ring pooling matrix (see self.getFlapPool)
            dofs: degrees of freedom per vertex (see self.getFlapPool)
        """
        dV_he = V[HF[:, 0], :] - V[HF[:, 1], :]
        dV_v = torch.spmm(poolMat, dV_he)
        dV_v /= dof.unsqueeze(1)
        LC = dV_v
        return LC

    def normalize_columns(self, tensor, epsilon=1e-10):
        '''
        tensor: n x 6 x c
        '''
        mean = torch.mean(tensor, dim=1).unsqueeze(1)  # n x 1 x c
        variance = torch.var(tensor, dim=1).unsqueeze(1)  # n x 1 x c
        normalized_tensor = (tensor - mean) / torch.sqrt(variance + epsilon)

        return normalized_tensor

    def forward(self, fv, HF, poolMat, DOF):
        outputs = []
        fv_input_pos = fv[:, :3]
        outputs.append(fv_input_pos)

        # subdivision starts
        for ii in range(self.numSubd):
            even_vertices_num = fv_input_pos.shape[0]

            fe = self.edgeMidPoint(fv_input_pos, HF[ii])  # midpoint subdivision
            fv = torch.cat((fv_input_pos, fe), dim=0)  # nV x Dout
            # compute the vector of 1st order vertex laplace coordinate
            fv_lc = self.getLaplaceCoordinate(fv, HF[ii+1], poolMat[ii+1], DOF[ii+1])
            # compute the vector of 2th order laplace coordinates
            fv_lc2 = self.getLaplaceCoordinate(fv_lc, HF[ii+1], poolMat[ii+1], DOF[ii+1])
            fv = torch.cat((fv, fv_lc, fv_lc2), dim=1)  # nV_next x Dout

            fv_input_pos = fv[:, :3]

            midpoint_6nhf_normalize, localFrames, normalize_length = self.v26hf(fv, HF[ii + 1],
                                                                                poolMat[ii + 1],
                                                                                even_vertices_num,
                                                                                normalizeScale=self.scale)
            # edge feature embdedding
            midpoint_6edge_feat = self.edge_feature_embedding(midpoint_6nhf_normalize)

            # graph attention aggregation
            midpoint_6edge_vec = midpoint_6nhf_normalize[:, :, 3:6]
            norm_midpoint_6edge_feat = self.normalize_columns(midpoint_6edge_feat)

            midpoint_6edge_vec_pos_feat = torch.cat((midpoint_6edge_vec, norm_midpoint_6edge_feat),
                                                    dim=2)
            init_attention = self.graph_attention_aggregation(midpoint_6edge_vec_pos_feat)
            norm_attention = torch.nn.functional.softmax(init_attention, dim=1)

            midpoint_atten_feat = torch.sum(norm_attention * midpoint_6edge_feat, dim=1)
            # vertex reposition
            midpoint_res = self.vertex_reposition(midpoint_atten_feat)
            midpoint_res = self.local2Global(midpoint_res, localFrames, normalize_length)
            fv_input_pos[even_vertices_num:, :] += midpoint_res

            outputs.append(fv_input_pos)

        return outputs
