import torch

def normalizeUnitCube(V):
    '''
    NORMALIZEUNITCUBE normalize a shape to the bounding box by 0.5,0.5,0.5

    Inputs:
        V (|V|,3) torch array of vertex positions

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    V = V - torch.min(V,0)[0].unsqueeze(0)

    V = V / torch.max(V.view(-1)) / 2.0
    return V

def my_normalizeUnitCube(V):
    '''
    NORMALIZEUNITCUBE normalize a shape to the bounding box by 1.0,1.0,1.0

    Inputs:
        V (|V|,3) torch array of vertex positions

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    scale =[]
    scale.append(torch.min(V,0)[0])
    V = V - torch.min(V,0)[0].unsqueeze(0)
    scale.append(torch.max(V.view(-1)))
    V = V / torch.max(V.view(-1))
    return V, scale

def my_normalizeUnitCubeScale(V, scale):
    '''
    NORMALIZEUNITCUBE normalize a shape to the bounding box by 1.0,1.0,1.0

    Inputs:
        V (|V|,3) torch array of vertex positions

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''

    V = V - scale[0].unsqueeze(0)
    V = V / scale[1]
    return V


def my_backNormalizeUnitCubeScale(V, scale):
    '''
    NORMALIZEUNITCUBE normalize a shape to the bounding box by 1.0,1.0,1.0

    Inputs:
        V (|V|,3) torch array of vertex positions

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    V = V * scale[1]
    V = V + scale[0].unsqueeze(0)

    return V



