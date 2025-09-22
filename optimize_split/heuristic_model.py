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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='select model resnet101/vgg19')
    parser.add_argument('--splitting_points', '-S', type=str, default='2,36', help='give an input in the form of s1,s2')
    parser.add_argument('--seed', '-s', type=int, default=42, help='run fifo with load balancer')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    splitting_points = args.splitting_points
    points = list(splitting_points.split(','))
    point_a = int(points[0])
    point_b = int(points[1])
    data = 150
    model_type = args.model
    if model_type == 'resnet101':
        filename = 'real_data/resnet101_CIFAR.xlsx'
    elif model_type == 'vgg19':
        filename = 'real_data/vgg19_CIFAR.xlsx'

    df_memory = pd.read_excel(io=filename, sheet_name='memory', header=None)
    df_vm = pd.read_excel(io=filename, sheet_name='VM', header=None)
    df_laptop = pd.read_excel(io=filename, sheet_name='laptop', header=None)

    vm_data = df_vm.values.tolist()
    laptop_data = df_laptop.values.tolist()

    memory_data = df_memory.values.tolist()
    N = point_b - point_a

    # find memory demands
    d = np.zeros(N)                 # memory demands for each layer
    k = 0
    for i in range(point_a, point_b):
        d[k] = (((memory_data[i][0] + memory_data[i][1])/1024)/1024)*data # prefer MB
        k += 1
    

    mem_array = {
        3: np.array([100, 100, 100]),
        4: np.array([100, 100, 100, 100]),
        5: np.array([100, 100, 100, 100, 100])

    }
    
    samples = {
        3: [1, 2, 2],
        4: [1, 2, 2, 2],
        5: [1, 1, 2, 2, 2]

    }

    # samples = {
    #     3: [2, 2, 1],
    #     4: [2, 2, 2, 1],
    #     5: [2, 2, 2, 1, 1]

    # }

    for P in [3, 4, 5]:
        print('NEWWW')
        rr = [1 for _ in range(P)]

        proc_f = np.zeros((P,N))
        proc_b = np.zeros((P,N))

        for j in range(P):
            rr[j] = samples[P][j]

        k = 0
        for i in range(point_a, point_b):
            for j in range(P):
                proc_f[j,k] = rr[j]*vm_data[i][0]
                proc_b[j,k] = rr[j]*(vm_data[i][1] + vm_data[i][2])
            k += 1

        # finding the avg processing time
        sum_all = 0
        for j in range(P):
            prof_p = 0
            for i in range(N):
                prof_p += proc_f[j,i] + proc_b[j,i]
            
            sum_all += prof_p
        
        avg_all = sum_all/P
        avg_parts = avg_all / P # this is our goal

        cuts = []
        cur_cut = 0
        # iterate helpers 
        for j in range(P):
            prof_p = 0
            cut_c = cur_cut
            
            for i in range(cur_cut, N):
                if prof_p + (proc_f[j,i]+proc_b[j,i]) < avg_parts:
                    cut_c = i
                    prof_p += proc_f[j,i]+proc_b[j,i]
                else:
                    cuts.append(cut_c)
                    cur_cut = cut_c+1
                    break
        
        print(cuts)

        max_pf = 0
        max_pb = 0
        
        for j in range(P):
            prof_pf = 0
            prof_pb = 0

            if j == 0:
                start_ = 0
            else:
                start_ = cuts[j-1]
            
            if j < P-1:
                end_ = cuts[j]
            else:
                end_ = N-1

            for i in range(start_, end_):
                prof_pf += proc_f[j,i]
                prof_pb += proc_b[j,i]
            
            print(f'helper {j} has {prof_pf} {prof_pb}')

            if prof_pf > max_pf:
                max_pf = prof_pf

            if prof_pb > max_pb:
                max_pb = prof_pb
        
        
        num_of_batches = 16
        local_epocchs = 2

        
        Ttotal = num_of_batches*(max_pf + max_pb)
        Ttotal = Ttotal*local_epocchs*(data-1)
        print(f'total cost when: {data}: {(Ttotal/1000)/60}')





if __name__ == '__main__':
    main()