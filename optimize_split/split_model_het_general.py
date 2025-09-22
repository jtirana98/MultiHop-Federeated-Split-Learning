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
        3: np.array([10000, 10000, 10000]),
        4: np.array([10000, 10000, 10000, 10000]),
        5: np.array([10000, 10000, 10000, 10000, 10000])

    }
    
    samples = {
        3: [1, 2, 2],
        4: [1, 2, 2, 2],
        5: [1, 1, 2, 2, 2]

    }

    for P in [3, 4, 5]:
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
        
        for p in range(P):
            for j in range(N):
                for k in range(1,j):
                    m.addConstr((x[p,k-1]-x[p,j])*x[p,j] <= (x[p,k]-x[p,j])*x[p,j])


        m.addConstr(x @ d <= mem_array[P]) #constraint-3: memory constraint on cn

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


        # print(np.abs(np.rint(x.X)))
        # print(m.ObjVal)
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
            # print(f'{p}: {procf}  {procb} \t\t TOTAL {procf+procb}')
        # print(f'MAXX {maxf}  {maxb}  {maxf+maxb}')
        # print(f'Total time {end_time-start_time}')
        print('Memory usage:')
        for p in range(P):
            memm = 0
            for j in range(N):
                memm += np.abs(np.rint(x.x[p,j]))*d[j]
            # print(memm)
        #print(((np.abs(np.rint(x.x))@d)))


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
            # print(f'{pair[0]} {pair[1]}')

        num_of_batches = 16
        local_epocchs = 2

        
        Ttotal = num_of_batches*(maxf + maxb)
        Ttotal = Ttotal*local_epocchs*(data-1)
        print(f'total cost when: {data}: {(Ttotal/1000)/60}')


if __name__ == '__main__':
    main()
