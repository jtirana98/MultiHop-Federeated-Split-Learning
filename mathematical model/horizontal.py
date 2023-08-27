import numpy as np
from numpy import linalg as LA
import pandas as pd
import math
import random
import time
import random
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import argparse
import fully_utilized_resnet
import fully_utilized_vgg

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='vgg19', help='select model resnet101/vgg19')
    parser.add_argument('--parts', '-p', type=int, default=2, help='run fifo with load balancer')
    parser.add_argument('--splitting_points', '-S', type=str, default='4,23', help='give an input in the form of s1,s2')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    random.seed(42)
    P = args.parts
    splitting_points = args.splitting_points
    points = list(splitting_points.split(','))
    point_a = int(points[0])
    point_b = int(points[1])

    model_type = args.model
    if model_type == 'resnet101':
        filename = '../optimize_split/real_data/resnet101_CIFAR.xlsx'
        bytes = fully_utilized_resnet.init_bytes("dataload.data")
        fully_utilized_resnet.bytes = bytes
        layers = fully_utilized_resnet.init_layers("model.data")
        layers_dataowner = fully_utilized_resnet.init_layers("resnet_1_d1.dat")
        data_owners_computation = [fully_utilized_resnet.Model_part(0, 1, 0, 0, layers_dataowner), fully_utilized_resnet.Model_part(1, 2, 0, 0, layers_dataowner)]
        compute_node_computation = fully_utilized_resnet.Model_part(point_a, point_b, 0, 0, layers)

    elif model_type == 'vgg19':
        filename = '../optimize_split/real_data/vgg19_CIFAR.xlsx'
        bytes = fully_utilized_vgg.init_bytes("vgg_data.data")
        fully_utilized_vgg.bytes = bytes
        layers = fully_utilized_vgg.init_layers("vgg_cn.data")
        layers_dataowner = fully_utilized_vgg.init_layers("vgg_iot.data")
        data_owners_computation = [fully_utilized_vgg.Model_part(0, 1, 0, 0, layers_dataowner), fully_utilized_vgg.Model_part(1, 2, 0, 0, layers_dataowner)]
        compute_node_computation = fully_utilized_vgg.Model_part(point_a, point_b, 0, 0, layers)
    
    N = point_b - point_a


    df_memory = pd.read_excel(io=filename, sheet_name='memory', header=None)
    df_vm = pd.read_excel(io=filename, sheet_name='VM', header=None)
    df_laptop = pd.read_excel(io=filename, sheet_name='laptop', header=None)

        
    # find proc delay
    proc_f = np.zeros((P,N))
    proc_b = np.zeros((P,N))

    vm_data = df_vm.values.tolist()
    laptop_data = df_laptop.values.tolist()

    rr = []
    for j in range(P):
        rr.append(random.randint(1,3))
    k = 0
    for i in range(point_a, point_b):
        for j in range(P):
            proc_f[j,k] = rr[j]*vm_data[i][0]
            proc_b[j,k] = rr[j]*(vm_data[i][1] + vm_data[i][2])
        k += 1
    
    
    print('----------------------------- HORIZONTAL SCALING ------------------------------------')
    data_owners = [50,100,150]
    num_of_batches = 16
    local_epocs = 2

    tfwd = []
    tbwd = []
    tepoch = []
    for p in range(P):
        print(f'client {p+1}')
        totf = 0
        totbwd = 0
        for j in range(N):
            totf += proc_f[p,j]
            totbwd += proc_b[p,j]
        tfwd.append(totf)
        tbwd.append(totbwd)
        print(f' {totf+totbwd}')
        tepoch.append(totf+totbwd)
    
    print(tepoch)

    order = np.argsort(tepoch)
    tepoch.sort()
    print(tepoch)
    print(order)

    times = []
    for p in range(P):
        times.append(tepoch[0]/tepoch[p])        
    
    for d in data_owners:
        print('-----------')
        pp = np.rint(d/sum(times))
        print(pp)

        distribution = [pp]
        for p in range(1,P):
            distribution.append(np.rint(pp*times[p]))
        
        if sum(distribution) > d:
            order = np.argsort(distribution)
            k = 0
            while sum(distribution) > d:
                distribution[k] -= 1        
                k += (k+1)%p
        
        if sum(distribution) < d:
            order = np.argsort(distribution)
            k = 0
            while sum(distribution) < d:
                distribution[-k] += 1        
                k += (k+1)%p

        print(distribution)
        print(sum(distribution))

        # lets compute epoch's delay
        first_batch = fully_utilized_resnet.all_hops([data_owners_computation[0]], 0) + data_owners_computation[0].send_a
        last_batch =  fully_utilized_resnet.all_hops([data_owners_computation[1]], 0)+ fully_utilized_resnet.all_hops([data_owners_computation[1]], 1) + compute_node_computation.send_g
        allepoch = []
        for p in range(P):
            epoch = first_batch + last_batch + distribution[p]*tepoch[p]*local_epocs*num_of_batches
            print(epoch)
            allepoch.append(epoch)
        print(f'The delay of the epoch is: {(max(allepoch)/1000)/60}')

if __name__ == '__main__':
    main()