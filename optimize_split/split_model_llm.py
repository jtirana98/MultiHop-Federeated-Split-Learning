import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import argparse

fwd_compute_node = [
    0.014,
    0.14,
    0.138,
    0.135,
    0.133,
    0.133,
    0.132,
    0.131,
    0.13,
    0.132,
    0.132,
    0.133,
    0.00012
]

bwd_compute_node = [
    0,
    0.064,
    0.08,
    0.77,
    0.075,
    0.076,
    0.076,
    0.076,
    0.075,
    0.076,
    0.075,
    0.073,
    0.0014,
]

fwd_client = fwd_compute_node*2
bwd_client = bwd_compute_node*2

memory_data = [
    739,
    33.2,
    33.2,
    33.2,
    33.2,
    33.2,
    33.2,
    33.2,
    33.2,
    33.2,
    33.2,
    33.2,
    0.008983105
]



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts', '-p', type=int, default=4, help='run fifo with load balancer')
    parser.add_argument('--splitting_points', '-S', type=str, default='1,12', help='give an input in the form of s1,s2')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    P = args.parts
    splitting_points = args.splitting_points
    points = list(splitting_points.split(','))
    point_a = int(points[0])
    point_b = int(points[1])

   
    N = point_b - point_a

    # find memory demands
    d = np.zeros(N)                 # memory demands for each layer
    k = 0
    for i in range(point_a, point_b):
        d[k] = memory_data[i] /1024 # convert to GB
        k += 1
    
    mem = np.array([16 for i in range(P)])      # memory capacity for each compute node
    

    # find proc delay
    proc_f = np.zeros((P,N))
    proc_b = np.zeros((P,N))

    
    k = 0
    for i in range(point_a, point_b):
        for j in range(P):
            proc_f[j,k] = fwd_compute_node[i]
            proc_b[j,k] = bwd_compute_node[i]
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
    print(x.X)
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

    num_of_batches = 16
    local_epocchs = 2

    data=50
    Ttotal = num_of_batches*(maxf + maxb)
    Ttotal = Ttotal*local_epocchs*(data-1)
    print(f'total cost when: {data}: {(Ttotal/60)}')
if __name__ == '__main__':
    main()