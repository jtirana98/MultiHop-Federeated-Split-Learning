import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt

num_of_compute_nodes = [1,2,3]
splits = [[2,35],[2,25,35],[2,15,25,35]]

num_of_batches = 80
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
            self.proc_forw = self.proc_forw + layers[i].proc_forw
            self.proc_back = self.proc_back + layers[i].proc_back
            self.proc_opti = self.proc_opti + layers[i].proc_opti
        
        # compute transfer time
        self.send_a = types_badwidth[bandwith2](layers[end-1].send_a)
        self.send_g = types_badwidth[bandwith1](layers[start-1].send_g)

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
        layers_.append(Layer(float(attributes[0]), float(attributes[1]), -1, bytes[int(attributes[2])], -1))
        if(i>0):
            layers_[-2].send_g=bytes[int(attributes[2])]
        i += 1

    return layers_

def mymax(thelist, op):
    max = 0
    for i in range(len(thelist)):
        proc = 0
        if(op == 1):
            if(thelist[i].send_g > thelist[i].proc_back):
                proc = thelist[i].proc_back 
            else:
                proc = thelist[i].proc_back 
        else:
            if(thelist[i].send_a > thelist[i].proc_forw):
                proc = thelist[i].proc_forw 
            else:
                proc = thelist[i].proc_forw 

        if(max < proc):
            max = proc

    return max
    
def all_hops(thelist, op):
    total = 0
    for i in range(len(thelist)):
        if(op == 1):
            total = total+ thelist[i].proc_back 
        else:
            total = total+ thelist[i].proc_forw 
    return total

def main():
    global bytes, layers
    bytes = init_bytes("dataload.data")
    layers = init_layers("model.data")
    layers_dataowner = init_layers("model_iot.py")

    compute_nodes_computation = []
    data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner), Model_part(1, 2, 0, 0, layers_dataowner)]
    total_cost = []
    exp = 0
    for num_c in num_of_compute_nodes:
        compute_nodes_computation = []
        total_cost_ = []
        print(f"nodes {num_c}")
        for i in range(num_c):
            bd1 = 1
            bd2 = 1
            if (i==0):
                bd1 = 0
            if (i == num_c-1):
                bd2 = 0
            compute_nodes_computation.append(Model_part(splits[exp][i], splits[exp][i+1], bd1, bd2, layers))
            print(f'HHH {i} {compute_nodes_computation[-1].proc_forw} {compute_nodes_computation[-1].proc_back}')
            print(f'HH2: {i} {compute_nodes_computation[-1].send_a} {compute_nodes_computation[-1].send_g}')
        for d in range(0,51,5): # ready to compute cost
            # check if it will be fully utilized

            if(False):
                total_cost_.append(-1)
                continue
            
            Thr_f = mymax(compute_nodes_computation, 0)
            Thr_b = mymax(compute_nodes_computation, 1)
            if (d == 0):
                print(f"f: {Thr_f}" )
                print(Thr_b)
            TAB = (d-1)*Thr_f + all_hops(compute_nodes_computation, 0)
            TCE = (d-1)*Thr_b + all_hops(compute_nodes_computation, 1)

            TAB_free = TAB - (num_c-1)*Thr_f
            TCE_free = TCE - (num_c-1)*Thr_b

            first_batch = all_hops(compute_nodes_computation, 0) + (d-1)*Thr_f
            #all_hops([data_owners_computation[0]], 0) +
            Ttotal = (num_of_batches-2)*(TAB_free + TCE_free) + TAB_free + TCE
            total_cost_.append(Ttotal*local_epocs+first_batch)

        for i in range(len(total_cost_)):
            print(total_cost_[i])
        exp += 1
        total_cost.append([total_cost])

if __name__ == "__main__":
    main()
