from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
from models.models import *
from include import *
import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pretrain_net', type=str, required=True, help='path of pretrained net')
    parser.add_argument('-t', '--test_obj_path', type=str, required=True, help='test mesh obj path')
    parser.add_argument('-o', '--output_dir_path', type=str, default=None, help='output_dir_path')
    parser.add_argument('-ns', '--numSubd', type=int, default=2, help='num of subdivision in testing')
    parser.add_argument('-di', '--din', type=int, default=9, help='num dimension of Din in training net')
    parser.add_argument('-do', '--dout', type=int, default=64, help='num dimension of Dout in training net')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='GPU id, e.g. -1, 0,1,2,3')
    args_ = parser.parse_args()
    if args_.output_dir_path is None:
        args_.output_dir_path = os.path.join('data_meshes/refined/',
                                             os.path.basename(args_.pretrain_net)[:-4])
    return args_


def test_obj(net, T, args):

    out_test_dir = args.output_dir_path
    os.path.exists(out_test_dir) or os.makedirs(out_test_dir)
    mesh_name = os.path.basename(args.test_obj_path)[:-4]

    numSubd = args.numSubd
    for mIdx in range(T.nM):
        x = T.getInputData(mIdx)
        scale = T.getScale(mIdx)
        outputs = net(x, T.hfList[mIdx], T.poolMats[mIdx], T.dofs[mIdx])
        assert len(outputs) == numSubd + 1
        for ii in range(1, numSubd + 1):
            pred_path = os.path.join(out_test_dir, mesh_name +'_subd' + str(ii) + '.obj')
            pred_x = outputs[ii].cpu() * scale[1] + scale[0].unsqueeze(0)
            tgp.writeOBJ(pred_path, pred_x, T.meshes[mIdx][ii].F.to('cpu'))
            print('write pred_mesh to: ', pred_path)


def load_data(args):
    # load testing set
    pkl_file_path = args.test_obj_path[:-4] + '_subd' + str(args.numSubd) + '.pkl'
    if os.path.exists(pkl_file_path):
        T = pickle.load(open(pkl_file_path, 'rb'))
        print('Loading testing data from ' + pkl_file_path)
    else:
        print('Calculate half-flap of ' + args.test_obj_path)
        T = TestMeshes([args.test_obj_path], args.numSubd)
        pickle.dump(T, file=open(pkl_file_path, "wb"))
        print('Saving testing data to ' + pkl_file_path)

    if T is None:
        raise Exception('No test data!')

    T.computeParameters()

    return T

def set_device(gpu):
    if torch.cuda.is_available():
        test_device = 'cuda'
        if gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        else:
            import subprocess
            import json
            cmd = 'gpustat --json'
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode()
            gpu_info = json.loads(result)['gpus']
            min_memory_gpu_index = np.argmin([gpu['memory.used'] for gpu in gpu_info])
            os.environ['CUDA_VISIBLE_DEVICES'] = str(min_memory_gpu_index)
            os.system('echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES')
    else:
        test_device = 'cpu'
    print('Test device:', test_device)
    return test_device

if __name__ == '__main__':
    args = parse_args()

    t_device = set_device(args.gpu)

    net = NMRNet(Din=args.din, Dout=args.dout, numSubd=args.numSubd)
    net = net.to(t_device)
    net.load_state_dict(torch.load(args.pretrain_net, map_location=torch.device(t_device)))
    net.eval()

    T = load_data(args)
    T.toDevice(t_device)

    test_obj(net, T, args)





