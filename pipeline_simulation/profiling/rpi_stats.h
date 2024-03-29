#ifndef _RPI_SRATS_H_
#define _RPI_SRATS_H_

class rpi_stats{
 public:
    double rpi_fm1;   
    double rpi_fm1_v; 
    double rpi_fbm2;  
    double rpi_fbm2_v;
    double rpi_bm1;   
    double rpi_bm1_v; 

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
            rpi_fm1     = 242.58;
            rpi_fm1_v   = 38;
            rpi_fbm2    = 29;
            rpi_fbm2_v  = 7;
            rpi_bm1     = 174.25;
            rpi_bm1_v   = 28;
        }
    }
};

#endif
