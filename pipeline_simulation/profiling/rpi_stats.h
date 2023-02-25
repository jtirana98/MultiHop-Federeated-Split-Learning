#ifndef _RPI_SRATS_H_
#define _RPI_SRATS_H_

class rpi_stats{
 public:
    int rpi_fm1;   
    int rpi_fm1_v; 
    int rpi_fbm2;  
    int rpi_fbm2_v;
    int rpi_bm1;   
    int rpi_bm1_v; 

    int rpi_to_vm = 8;
    int vm_to_rpi = 8;

    rpi_stats(int i) {
        if(i == 1) {  //d2
            rpi_fm1     = 290.5;
            rpi_fm1_v   = 2;
            rpi_fbm2    = 43.8;
            rpi_fbm2_v  = 6;
            rpi_bm1     = 299.66;
            rpi_bm1_v   = 21;
        }
        else {
            rpi_fm1     = 291;
            rpi_fm1_v   = 38;
            rpi_fbm2    = 25;
            rpi_fbm2_v  = 7;
            rpi_bm1     = 265;
            rpi_bm1_v   = 28;
        }
    }
};

#endif
