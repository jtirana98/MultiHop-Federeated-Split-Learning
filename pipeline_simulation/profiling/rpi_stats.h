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

    double rpi_to_vm = 8;
    double vm_to_rpi = 8;

    rpi_stats(int i) {
        if(i == 1) {
            rpi_fm1     = 236.3;
            rpi_fm1_v   = 2.05;
            rpi_fbm2    = 17.76;
            rpi_fbm2_v  = 6.25;
            rpi_bm1     = 133.26;
            rpi_bm1_v   = 21.03;
        }
        else {
            rpi_fm1     = 291.423;
            rpi_fm1_v   = 38.20;
            rpi_fbm2    = 25.38;
            rpi_fbm2_v  = 7.23;
            rpi_bm1     = 265.15;
            rpi_bm1_v   = 28.09;
        }
    }
};

#endif
