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
    parser.add_argument('--splitting_points', '-S', type=str, default='3,22', help='give an input in the form of s1,s2')
    parser.add_argument('--seed', '-s', type=int, default=42, help='run fifo with load balancer')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    random.seed(args.seed)
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

    memory_data = df_memory.values.tolist()    
    # find proc delay
    proc_f = np.zeros((P,N))
    proc_b = np.zeros((P,N))

    vm_data = df_vm.values.tolist()
    laptop_data = df_laptop.values.tolist()

    '''
    mem_ = [17,17,9]
    mem_ = np.array([
        [32,32,32,32,32],
        [32,32,32,32,16],
        [32,32,32,16,16],
        [32,32,16,16,16],
        [32,16,16,16,16],
        [16,16,16,16,16],
        [32,32,32,32,8],
        [32,32,32,8,8],
        [32,32,8,8,8],
        [32,8,8,8,8],
        [8,8,8,8,8],
        [8,8,8,8,16],
        [8,8,8,16,16],
        [8,8,16,16,16],
        [8,16,16,16,16],
        [8,16,16,16,32],
        [8,16,16,32,32],
        [8,16,32,32,32],
        [16,8,8,8,32],
        [16,8,8,32,32],
    ])

    samples = [
        [1,1,1,1,1],
        [1,1,1,1,2],
        [1,1,1,2,2],
        [1,1,2,2,2],
        [1,2,2,2,2],
        [2,2,2,2,2],
        [1,1,1,1,4],
        [1,1,1,4,4],
        [1,1,4,4,4],
        [1,4,4,4,4],
        [4,4,4,4,4],
        [4,4,4,4,2],
        [4,4,4,2,2],
        [4,4,2,2,2],
        [4,2,2,2,2],
        [4,2,2,2,1],
        [4,2,2,1,1],
        [4,2,1,1,1],
        [2,4,4,4,1],
        [2,4,4,1,1],
    ]
    '''
    
    
    mem_ = np.array([
        [16,16,16,16,16],
        [16,16,16,16,8],
        [16,16,16,8,8],
        [16,16,8,8,8],
        [16,8,8,8,8],
        [8,8,8,8,8],
        [16,16,16,16,4],
        [16,16,16,4,4],
        [16,16,4,4,4],
        #[16,4,4,4,4],
        #[4,4,4,4,4],
        ##[4,4,4,4,8],
        #[4,4,4,8,9],
        [4,4,8,8,9],
        [4,8,8,8,8],
        [4,8,8,8,16],
        [4,8,8,16,16],
        [4,8,16,16,16],
        [8,4,4,4,16],
        [8,4,4,16,16],
    ])

    samples = [
        [1,1,1,1,1],
        [1,1,1,1,2],
        [1,1,1,2,2],
        [1,1,2,2,2],
        [1,2,2,2,2],
        [2,2,2,2,2],
        [1,1,1,1,2],
        [1,1,1,2,2],
        [1,1,2,2,2],
        #[1,2,2,2,2],
        #[2,2,2,2,2],
        ##[2,2,2,2,2],
        #[2,2,2,2,2],
        [2,2,2,2,2],
        [2,2,2,2,2],
        [2,2,2,2,1],
        [2,2,2,1,1],
        [2,2,1,1,1],
        [2,2,2,2,1],
        [2,2,2,1,1],
    ]
    
    '''
    mem_ = np.array([
        [32,32,32],
        [32,32,16],
        [32,32,8],
        [32,16,8],
        #[16,16,16],
        #[16,16,8],
        #[16,8,8]
    ])

    samples = [
        [1,1,1],
        [1,1,2],
        [1,1,4],
        [1,2,4],
        #[2,2,2],
        #[2,2,4],
        #[2,4,4]
    ]
    '''
    '''
    samples = [
        [1,1,1],
        [1,1,2],
        [1,1,2],
        [1,2,2],
    ]

    mem_ = np.array([
        [16,16,16],
        [16,16,8],
        [16,16,4],
        [16,8,4],
        
    ])
    '''

    '''
    # mikra
    samples = [
        [1,1,1,1],
        [1,1,1,2],
        [1,1,2,2],
        [1,2,2,2],
        [2,2,2,2],
        [1,1,1,2],
        [1,1,2,2],
        [1,2,2,2],
        [2,2,2,2],
        [1,2,2,2],
        [1,1,2,2],
        [1,2,2,2]
    ]

    mem_ = np.array([
        [16,16,16,16],
        [16,16,16,8],
        [16,16,8,8],
        [16,8,8,8],
        [8,8,8,8],
        [16,16,16,4],
        [16,16,4,4],
        [16,4,4,4],
        [8,8,8,4],
        [16,8,8,4],
        [16,16,8,4],
        [16,8,4,4]
    ])
    '''
    # megala
    '''
    samples = [
        [1,1,1,1],
        [1,1,1,2],
        [1,1,2,2],
        [1,2,2,2],
        [2,2,2,2],
        [1,1,1,4],
        [1,1,4,4],
        [1,4,4,4],
        [2,2,2,4],
        [1,2,2,4],
        [1,1,2,4],
        [1,2,4,4]
    ]

    mem_ = np.array([
        [32,32,32,32],
        [32,32,32,16],
        [32,32,16,16],
        [32,16,16,16],
        [16,16,16,16],
        [32,32,32,8],
        [32,32,8,8],
        [32,8,8,8],
        [16,16,16,8],
        [32,16,16,8],
        [32,32,16,8],
        [32,16,8,8]
    ])
    '''
    # end megala
    
    for ii in range(len(samples)):
        rr = []
        for j in range(P):
            rr.append(random.randint(1,3))
            #rr[-1] = 1
            rr[j] = samples[ii][j]

        k = 0
        for i in range(point_a, point_b):
            for j in range(P):
                proc_f[j,k] = rr[j]*vm_data[i][0]
                proc_b[j,k] = rr[j]*(vm_data[i][1] + vm_data[i][2])
            k += 1
        
        demand = np.zeros(N)                 # memory demands for each layer
        k = 0
        for i in range(point_a, point_b):
            demand[k] = ((memory_data[i][0] + memory_data[i][1])/1024)/1024 # prefer GB
            k += 1
        print(f'The memory demands {demand.sum()}')
        print('----------------------------- HORIZONTAL SCALING ------------------------------------')
        data_owners = [50,100,150]
        num_of_batches = 16
        local_epocs = 2

        tfwd = []
        tbwd = []
        tepoch = []
        for p in range(P):
            print(f'node {p+1}')
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
        print(times)

        data_owners = [150]
        random.seed(100)
        for d in data_owners:
            mem = [0 for i in range(P)]
            
            #while sum(mem) < d:
            for p in range(P):
                #mem[p] += random.randint(1,d/2)
                #mem[p] = d
                mem[p] = np.rint((mem_[ii][p]/demand.sum()))
                #mem[p] = d
            
            print('-----------')
            pp = np.rint(d/sum(times))
            #print(pp)
            if pp > mem[0]:
                pp = mem[0]
            
            
            print(f'memory {mem}')
            
            distribution = [pp]
            for p in range(1,P):
                if pp*times[p] < mem[p]:
                    distribution.append(np.rint(pp*times[p]))
                else:
                    distribution.append(np.rint(mem[p]*times[p]))
            
            #print('prin')
            #print(distribution)

            if sum(distribution) > d:
                #order = np.argsort(distribution)
                k = 0
                while sum(distribution) > d:
                    distribution[k] -= 1        
                    k = (k+1)%P
            
            if sum(distribution) < d:
                #order = np.argsort(distribution)
                k = 0
                while sum(distribution) < d:
                    #print(k)
                    if distribution[-k] < mem[-k]:
                        distribution[-k] += 1    
                    k = (k+1)%P
            
            #for p in range(P):
                
            print('----------- memory demands -----------')
            for p in range(P):
                print(mem[p]*demand.sum())

            print('----------- memory usage -----------')
            for p in range(P):
                print(distribution[p]*demand.sum())

            print(distribution)
            print(sum(distribution))

            # lets compute epoch's delay
            first_batch = fully_utilized_resnet.all_hops([data_owners_computation[0]], 0) + data_owners_computation[0].send_a
            last_batch =  fully_utilized_resnet.all_hops([data_owners_computation[1]], 0)+ fully_utilized_resnet.all_hops([data_owners_computation[1]], 1) + compute_node_computation.send_g
            allepoch = []
            for p in range(P):
                epoch = distribution[p]*tepoch[p]*local_epocs*num_of_batches
                print(epoch)
                allepoch.append(epoch)
            print(f'The delay of the epoch is: {(max(allepoch)/1000)/60}')

if __name__ == '__main__':
    main()