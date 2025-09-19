import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='vgg19', help='select model resnet101/vgg19')
    parser.add_argument('--parts', '-p', type=int, default=5, help='run fifo with load balancer')
    parser.add_argument('--splitting_points', '-S', type=str, default='3,22', help='give an input in the form of s1,s2')
    parser.add_argument('--seed', '-s', type=int, default=42, help='run fifo with load balancer')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    P = args.parts
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

    memory_data = df_memory.values.tolist()
    N = point_b - point_a

    # find memory demands
    d = np.zeros(N)                 # memory demands for each layer
    k = 0
    for i in range(point_a, point_b):
        d[k] = (((memory_data[i][0] + memory_data[i][1])/1024)/1024)*data # prefer MB
        k += 1
    
    mem = np.array([160 for i in range(P)])      # memory capacity for each compute node
    mem = np.array([32,32,16])

    # find proc delay
    proc_f = np.zeros((P,N))
    proc_b = np.zeros((P,N))

    vm_data = df_vm.values.tolist()
    laptop_data = df_laptop.values.tolist()

    rr = []
    samples = [1,1,2]
    '''
    mem = np.array([
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
    
    mem = np.array([
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
    samples = [
        [1,1,1,1],
        [1,1,1,2],
        #[1,1,2,2],
        #[1,2,2,2],
        #[2,2,2,2],
        [1,1,1,2],
        #[1,1,2,2],
        #[1,2,2,2],
        #[2,2,2,2],
        #[1,2,2,2],
        [1,1,2,2],
        #[1,2,2,2]
    ]

    mem = np.array([
        [16,16,16,16],
        [16,16,16,8],
        #[16,16,8,8],
        #[16,8,8,8],
        #[8,8,8,8],
        [16,16,16,4],
        #[16,16,4,4],
        #[16,4,4,4],
        #[8,8,8,4],
        #[16,8,8,4],
        [16,16,8,4],
        #[16,8,4,4]
    ])
    '''
    '''
    mem = np.array([
        [16,16,16,16,16],
        [16,16,16,16,8],
        [16,16,16,8,8],
        [16,16,8,8,8],
        [16,8,8,8,8],
        [8,8,8,8,8],
        [16,16,16,16,4],
        [16,16,16,4,4],
        [16,16,4,4,5],
        [16,5,5,5,5],
        #[4,4,4,4,4],
        #[4,4,4,4,8],
        #[4,4,4,8,8],
        [4,5,8,8,8],
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
        [1,2,2,2,2],
        [2,2,2,2,2],
        #[2,2,2,2,2],
        #[2,2,2,2,2],
        #[2,2,2,2,2],
        [2,2,2,2,2],
        [2,2,2,2,1],
        [2,2,2,1,1],
        [2,2,1,1,1],
        [2,2,2,2,1],
        [2,2,2,1,1],
    ]
    '''
    '''
    mem = np.array([
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

    mem = np.array([
        [16,16,16],
        [16,16,8],
        [16,16,4],
        [16,8,4]
    ])
    '''
    
    # mikra
    '''
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

    mem = np.array([
        [16,16,16,16],
        [16,16,16,8],
        [16,16,8,8],
        [16,8,8,8],
        [8,8,8,9],
        [16,16,16,4],
        [16,16,4,4],
        [16,4,4,5],
        [8,8,8,5],
        [16,8,8,5],
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

    mem = np.array([
        [32,32,32,32],
        [32,32,32,16],
        [32,32,16,16],
        [32,16,16,16],
        [16,16,16,16],
        [32,32,32,8],
        [32,32,8,8],
        [32,8,8,8],
        [16,16,16,9],
        [32,16,16,8],
        [32,32,16,8],
        [32,16,8,8]
    ])
    '''
    for ii in range(len(samples)):
        rr = [1 for _ in range(P)]
        for j in range(P):
            rr[j] = samples[ii][j]
        k = 0
        for i in range(point_a, point_b):
            for j in range(P):
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


        m.addConstr(x @ d <= mem[ii]) #constraint-3: memory constraint on cn
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
        print('Memory usage:')
        for p in range(P):
            memm = 0
            for j in range(N):
                memm += np.abs(np.rint(x.x[p,j]))*d[j]
            print(memm)
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
            print(f'{pair[0]} {pair[1]}')

        num_of_batches = 16
        local_epocchs = 2

        
        Ttotal = num_of_batches*(maxf + maxb)
        Ttotal = Ttotal*local_epocchs*(data-1)
        print(f'total cost when: {data}: {(Ttotal/1000)/60}')
if __name__ == '__main__':
    main()