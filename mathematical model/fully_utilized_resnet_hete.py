import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import random 

num_of_compute_nodes = [2,3,4,5]
#splits = [[5,21,33],[5,17,26,33],[5,14,21,28,33],[5,12,18,24,30,33]]
splits = [[2,25,36],[2,18,31,36],[2,16,26,33,36],[2,12,20,28,33,36]]

num_of_batches = 16
local_epocs = 2

layers = []
bytes = []

types_badwidth = [lambda a : ((a*0.000008)/8)*1000, lambda a : ((a*0.000000008)/7.13)*1000] # [ cn <-> rpi, cn <-> cn ]  

class Layer:
    def __init__(self, proc_forw, proc_back, proc_opti, send_a, send_g):
        self.proc_forw = proc_forw
        self.proc_back = proc_back
        self.proc_opti = proc_opti
        self.send_a = send_a
        self.send_g = send_g

class Model_part:
    def __init__(self, start, end, bandwith1, bandwith2, layers_):
        self.proc_forw = 0
        self.proc_back = 0
        self.proc_opti = 0
        self.send_a = 0
        self.send_g = 0
        global layers
        for i in range(start, end):
            self.proc_forw = self.proc_forw + layers_[i].proc_forw
            self.proc_back = self.proc_back + layers_[i].proc_back
            self.proc_opti = self.proc_opti + layers_[i].proc_opti
        
        # compute transfer time
        self.send_a = types_badwidth[bandwith2](layers_[end-1].send_a)
        self.send_g = types_badwidth[bandwith1](layers_[start-1].send_g)

def init_bytes(filename):
    txt_file = open(filename, "r")
    file_content = txt_file.read()

    content_list = file_content.split("\n")
    txt_file.close()

    res = []
    for b in content_list:
        res.append(float(b))
    return res

def init_layers(filename):
    layers_ = []
    
    txt_file = open(filename, "r")
    file_content = txt_file.read()

    content_list = file_content.split("\n")
    txt_file.close()
    
    i = 0
    for metrics in content_list:
        attributes = metrics.split(",")
        tok = 2
        if (len(attributes) > 3):
            tok = 3
            layers_.append(Layer(float(attributes[0]), float(attributes[1]), float(attributes[2]), bytes[int(attributes[3])], -1))
        else:
            layers_.append(Layer(float(attributes[0]), float(attributes[1]), 0, bytes[int(attributes[2])], -1))
        
        if(i>0):
            layers_[-2].send_g=bytes[int(attributes[tok])]
        i += 1

    return layers_

def mymax(thelist, op):
    max = 0

    for i in range(len(thelist)):
        proc = 0
        if(op == 1):
            proc = thelist[i].proc_back  + thelist[i].proc_opti 
        else:
            proc = thelist[i].proc_forw

        if(max < proc):
            max = proc

    return max
    
def all_hops(thelist, op):
    total = 0
    for i in range(len(thelist)):
        if(op == 1):
            total = total+ thelist[i].proc_back + thelist[i].proc_opti
        else:
            total = total+ thelist[i].proc_forw
    return total

def get_res(data_owners =300):
    data_owners = [150]
    global bytes, layers
    bytes = init_bytes("dataload.data")
    layers = init_layers("model.data")
    layers_dataowner = init_layers("resnet_1_d1.dat")
    layers_dataowner2 = init_layers("resnet_1_d1.dat")
    layers_dataowner3 = init_layers("resnet_1_d1.dat")

    compute_nodes_computation = []
    data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner), Model_part(1, 2, 0, 0, layers_dataowner)]
    total_cost = []
    exp = 0
    for num_c in num_of_compute_nodes:
        print(f"------------- nodes {num_c} ----------------")
        if (num_c > 1):
             data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner2), Model_part(1, 2, 0, 0, layers_dataowner2)]
        
        compute_nodes_computation = []
        total_cost_ = []
        
        for i in range(num_c):
            bd1 = 1
            bd2 = 1
            if (i==0):
                bd1 = 0
            if (i == num_c-1):
                bd2 = 0
            compute_nodes_computation.append(Model_part(splits[exp][i], splits[exp][i+1], bd1, bd2, layers))
            #if num_c < 2:
            #    compute_nodes_computation[-1].proc_back = 1000
            #    compute_nodes_computation[-1].proc_opti = 0

            print(f'HHH {i} {compute_nodes_computation[-1].proc_forw} {compute_nodes_computation[-1].proc_back} {compute_nodes_computation[-1].proc_opti}')
            print(f'HH2: {i} {compute_nodes_computation[-1].send_a} {compute_nodes_computation[-1].send_g}')
        flag = 0
        for d in data_owners: # ready to compute cost
            Thr_f = mymax(compute_nodes_computation, 0)
            Thr_b = mymax(compute_nodes_computation, 1)
            
            
            if (flag == 0):
                print(f"f: {Thr_f}" )
                print(f"b: {Thr_b}")
                flag = 1
            
            if d <= 5:
                continue

            
            TAB = (d-1)*Thr_f + all_hops(compute_nodes_computation, 0)
            TCE = (d-1)*Thr_b + all_hops(compute_nodes_computation, 1)

            #KEEP IT first_batch = all_hops([data_owners_computation[0]], 0) + data_owners_computation[0].send_a + all_hops(compute_nodes_computation, 0) + all_hops(compute_nodes_computation, 1)
            #KEEP IT last_batch =  all_hops([data_owners_computation[1]], 0)+ all_hops([data_owners_computation[1]], 1) + compute_nodes_computation[-1].send_g
            first_batch = 0
            last_batch = 0
            Ttotal = num_of_batches*(TAB + TCE)
            Ttotal = Ttotal*local_epocs+first_batch+last_batch
            
            #aggr = 44602+25212 
            aggr = 8200
            if (num_c < 2):
                aggr = 171300000
                
            #KEEP IT Ttotal = Ttotal + types_badwidth[1](aggr)*(d) + types_badwidth[0](aggr)
            total_cost_.append(Ttotal)

        #for i in range(len(total_cost_)):
            #print(f'total cost when: {d}: {total_cost_[i]}')
            print(f'total cost when: {d}: {(Ttotal/1000)/60}')
        exp += 1
        total_cost.append(total_cost_)
        return total_cost

def main():
    get_res()

if __name__ == "__main__":
    main()
