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
    parser.add_argument('--model', '-m', type=str, default='vgg19', help='select model resnet101/vgg19')
    parser.add_argument('--parts', '-p', type=int, default=2, help='run fifo with load balancer')
    parser.add_argument('--splitting_points', '-S', type=str, default='1,24', help='give an input in the form of s1,s2')
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
        filename = 'real_data/resnet101_CIFAR.xlsx'
    elif model_type == 'vgg19':
        filename = 'real_data/vgg19_CIFAR.xlsx'

    df_memory = pd.read_excel(io=filename, sheet_name='memory', header=None)
    df_vm = pd.read_excel(io=filename, sheet_name='VM', header=None)
    df_laptop = pd.read_excel(io=filename, sheet_name='laptop', header=None)

    memory_data = df_memory.values.tolist()
    N = point_b - point_a

    # find memory demands
    d = np.zeros(N)                 # memory demands for each layer
    k = 0
    for i in range(point_a, point_b):
        d[k] = ((memory_data[i][0] + memory_data[i][1])/1024)/1024 # prefer MB
        k += 1
    
    mem = np.array([16 for i in range(P)])      # memory capacity for each compute node
    

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
            ''''
            min_f = min(vm_data[i][0], laptop_data[i][0])
            max_f = max(vm_data[i][0], laptop_data[i][0])
            
            min_f = int(min_f)
            max_f = int(max_f)
            
            if min_f == max_f:
                max_f = max_f+1

            
            proc_f[j,k] = random.randrange(min_f, max_f)
            min_b = min(vm_data[i][1]+vm_data[i][2], laptop_data[i][1]+laptop_data[i][2])
            max_b = max(vm_data[i][1]+vm_data[i][2], laptop_data[i][1]+laptop_data[i][2])
            
            min_b = int(min_b)
            max_b = int(max_b)
            
            if min_b == max_b:
                max_b = max_b+1

            proc_b[j,k] = random.randrange(min_b, max_b)
            '''
            proc_f[j,k] = rr[j]*vm_data[i][0]
            proc_b[j,k] = rr[j]*(vm_data[i][1] + vm_data[i][2])
        k += 1

    ones_P = np.ones((P,1))
    ones_N = np.ones((N,1))

    print(f'There are {N} possible layers')
    start_time = time.time()
    m = gp.Model("")
    x = m.addMVar(shape = (P,N), vtype=GRB.BINARY, name="x")
    Lf = m.addMVar(shape=(P), name="lforward")
    Lb = m.addMVar(shape=(P), name="lforward")
    Lf_max = m.addMVar(shape=(1), name="lforwardMAX")
    Lb_max = m.addMVar(shape=(1), name="lforwardMAX")
   

    #m.addConstr(x @ ones_N == ones_P) #cconstraint-1: layer can only be assigned to one compute node
    for i in range(N):
        m.addConstr(qsum(x[:,i]) == 1)
    
    
    for p in range(P):
        m.addConstr(qsum(x[p,:]) >= 1)
    
    #constraint-2: only sequential layers
    '''
    for p in range(P):
        for j in range(N):
            for k in range(1,j):
                m.addConstr((x[p,k-1]-x[p,j])*x[p,j] <= (x[p,k]-x[p,j])*x[p,j])
    '''
    
    for p in range(P):
        for j in range(N):
            for k in range(1,j):
                m.addConstr((x[p,k-1]-x[p,j])*x[p,j] <= (x[p,k]-x[p,j])*x[p,j])


    m.addConstr(x @ d <= mem) #constraint-3: memory constraint on cn
    #m.addConstr(Lf == qsum(x@(proc_f.T)))    #constraint-4a: forward delay for each cn
    #m.addConstr(Lb == qsum(x@proc_b.T))    #constraint-4b: backward delay for each cn

    for p in range(P):
        m.addConstr(Lf[p] == qsum(x[p,j] *proc_f[p,j] for j in range(N)))
        m.addConstr(Lb[p] == qsum(x[p,j] *proc_b[p,j] for j in range(N)))

    # objetive
    for p in range(P):
        m.addConstr(Lf_max >= Lf[p])
        m.addConstr(Lb_max >= Lb[p])

    m.setObjective(Lf_max+Lb_max, GRB.MINIMIZE)
    m.optimize()
    end_time = time.time()
    print(np.abs(np.rint(x.X)))
    print(m.ObjVal)
    print('model parts:')
    maxf = 0
    maxb = 0
    for p in range(P):
        procf = 0
        procb = 0
        for i in range(N):
            procf += np.abs(np.rint(x[p,i].X))*proc_f[p,i]
            procb += np.abs(np.rint(x[p,i].X))*proc_b[p,i]
        if maxf < procf:
            maxf = procf

        if maxb < procb:
            maxb = procb
        print(f'{p}: {procf}  {procb} \t\t TOTAL {procf+procb}')
    print(f'MAXX {maxf}  {maxb}  {maxf+maxb}')
    print(f'Total time {end_time-start_time}')
    print(np.abs(np.rint(x.x))@d)


    xx = np.abs(np.rint(x.x))
    splitting_points = []
    for p in range(P):
        start = False
        end = False
        pair = [point_a, point_a]
        for j in range(N):
            if not start and xx[p,j] == 1:
                start = True
                pair[0] = pair[0] + j
            if start and xx[p,j] == 0:
                end = True
                pair[1] = pair[1] + j
                splitting_points.append(pair)
                break
        if not end:
            pair[1] = pair[1] + N
        print(f'{pair[0]} {pair[1]}')
if __name__ == '__main__':
    main()