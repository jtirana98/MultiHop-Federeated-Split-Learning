import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt

num_of_compute_nodes = [1,2,3]
splits = [[10,35],[2,26,35],[2,16,26,35]]
#25
num_of_batches = 16
local_epocs = 2

layers = []
bytes = []

types_badwidth = [lambda a : ((a*0.000008)/8)*1000, lambda a : ((a*0.000000008)/7.13)*1000, lambda a : ((a*0.000008)/2)*1000] # [ cn <-> rpi, cn <-> cn ]  

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
    global bytes, layers
    bytes = init_bytes("dataload.data")
    layers = init_layers("model.data")
    layers_dataowner = init_layers("resnet_1_d1.dat") #d1
    layers_dataowner2 = init_layers("resnet_2_d1.dat") #d1

    layers_dataowner3 = init_layers("resnet_1_d2.dat") #d2
    layers_dataowner4 = init_layers("resnet_2_d2.dat") #d2

    compute_nodes_computation = []
    data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner), Model_part(1, 2, 0, 0, layers_dataowner)]
    data_owners_computation2 = [Model_part(0, 1, 0, 0, layers_dataowner3), Model_part(1, 2, 0, 0, layers_dataowner3)]
    total_cost = []
    exp = 0
    for num_c in num_of_compute_nodes:
        print(f"nodes {num_c}")
        if (num_c > 1):
            data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner2), Model_part(1, 2, 0, 0, layers_dataowner2)]
            data_owners_computation2 = [Model_part(0, 1, 0, 0, layers_dataowner4), Model_part(1, 2, 0, 0, layers_dataowner4)]
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

            #print(f'HHH {i} {compute_nodes_computation[-1].proc_forw} {compute_nodes_computation[-1].proc_back} {compute_nodes_computation[-1].proc_opti}')
            #print(f'HH2: {i} {compute_nodes_computation[-1].send_a} {compute_nodes_computation[-1].send_g}')
        for d in range(0,data_owners+1,10): # ready to compute cost
            # check if it will be fully utilized
            #print(d)
            d = 30
            Thr_f = mymax(compute_nodes_computation, 0) 
            Thr_b = mymax(compute_nodes_computation, 1)

            if num_c < 2:
               Thr_b = 1000

            if (d == 10):
                print(f"f: {Thr_f}" )
                print(f"b: {Thr_b}")
            
            if d <= 5:
                continue

            
            TAB = Thr_f
            TCE = Thr_b
            
            d1 =  0
            for d1 in [30, 20, 15, 10, 0]:
                d2 = d - d1
                first_batch = (d1*all_hops([data_owners_computation[0]], 0) + d2*all_hops([data_owners_computation2[0]], 0))/d + data_owners_computation[0].send_a + all_hops(compute_nodes_computation, 0) + all_hops(compute_nodes_computation, 1)
                last_batch =  (d1*all_hops([data_owners_computation[1]], 0)+ d1*all_hops([data_owners_computation[1]], 1) + d2*all_hops([data_owners_computation2[1]], 0)+ d2*all_hops([data_owners_computation2[1]], 1))/d +  + compute_nodes_computation[-1].send_g
                Ttotal = (num_of_batches*d*local_epocs -1 )*(TAB + TCE)
                Ttotal = Ttotal+first_batch+last_batch
                
                aggr = 44602+25212
                if (num_c <2):
                    aggr = 171300000
                
                Ttotal = Ttotal + types_badwidth[1](aggr)*(d) + types_badwidth[0](aggr)
                #print(d)
                print(f'{d1} : {Ttotal}')
                total_cost_.append(Ttotal)
            break

        #for i in range(len(total_cost_)):
        #    print(total_cost_[i])
        exp += 1
        total_cost.append(total_cost_)


    print("SL")
    data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner), Model_part(1, 2, 0, 0, layers_dataowner)]
    compute_nodes_computation = []
    compute_nodes_computation.append(Model_part(splits[0][0], splits[0][1], 0, 0, layers))
    
    forw_data = all_hops([data_owners_computation[0]], 0) + all_hops([data_owners_computation[0]], 1) + data_owners_computation[0].send_a
    forw_comp = all_hops([compute_nodes_computation[0]],0) + compute_nodes_computation[0].send_a

    back_data = all_hops([data_owners_computation[1]], 1) + all_hops([data_owners_computation[1]], 0) + data_owners_computation[0].send_g
    back_comp = all_hops([compute_nodes_computation[0]],1) + compute_nodes_computation[0].send_g

    total1 = forw_data + forw_comp + back_data + back_comp
    total1 = total1*16

    data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner3), Model_part(1, 2, 0, 0, layers_dataowner3)]
    forw_data = all_hops([data_owners_computation[0]], 0) + all_hops([data_owners_computation[0]], 1) + data_owners_computation[0].send_a    
    back_data = all_hops([data_owners_computation[1]], 1) + all_hops([data_owners_computation[1]], 0) + data_owners_computation[0].send_g


    total2 = forw_data + forw_comp + back_data + back_comp
    total2 = total2*16

    model_tran = types_badwidth[0]((171300000))
    for d in range(0,data_owners+1,50):
        if d < 5:
            continue
        d = 300

        d1 =  0
        for d1 in [300, 200, 150, 100, 0]:
            d2 = d - d1
            value = d1*total1 + d2*total2+ d*model_tran - model_tran
            print(f'{d1} {value}')

        break
    '''
    
    print("SL - d1")
    data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner), Model_part(1, 2, 0, 0, layers_dataowner)]
    compute_nodes_computation = []
    compute_nodes_computation.append(Model_part(splits[0][0], splits[0][1], 0, 0, layers))
    
    forw_data = all_hops([data_owners_computation[0]], 0) + all_hops([data_owners_computation[0]], 1) + data_owners_computation[0].send_a
    forw_comp = all_hops([compute_nodes_computation[0]],0) + compute_nodes_computation[0].send_a

    back_data = all_hops([data_owners_computation[1]], 1) + all_hops([data_owners_computation[1]], 0) + data_owners_computation[0].send_g
    back_comp = all_hops([compute_nodes_computation[0]],1) + compute_nodes_computation[0].send_g

    total = forw_data + forw_comp + back_data + back_comp
    total = total*16
    model_tran = types_badwidth[0]((171300000))
    for d in range(0,data_owners+1,50):
        if d < 5:
            continue
        
        value = d*(total + model_tran) - model_tran
        
        
        print(f'{d} {value}')

    print("SL - d2")
    data_owners_computation = [Model_part(0, 1, 0, 0, layers_dataowner3), Model_part(1, 2, 0, 0, layers_dataowner3)]
    compute_nodes_computation = []
    compute_nodes_computation.append(Model_part(splits[0][0], splits[0][1], 0, 0, layers))
    
    forw_data = all_hops([data_owners_computation[0]], 0) + all_hops([data_owners_computation[0]], 1) + data_owners_computation[0].send_a
    forw_comp = all_hops([compute_nodes_computation[0]],0) + compute_nodes_computation[0].send_a

    back_data = all_hops([data_owners_computation[1]], 1) + all_hops([data_owners_computation[1]], 0) + data_owners_computation[0].send_g
    back_comp = all_hops([compute_nodes_computation[0]],1) + compute_nodes_computation[0].send_g

    total = forw_data + forw_comp + back_data + back_comp
    total = total*16
    model_tran = types_badwidth[0]((171300000))
    for d in range(0,data_owners+1,50):
        if d < 5:
            continue
        
        value = d*(total + model_tran) - model_tran
        
        print(f'{d} {value}')

    return total_cost
'''
#171300000
#
def main():
    get_res()

if __name__ == "__main__":
    main()
