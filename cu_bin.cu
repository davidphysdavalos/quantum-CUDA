#include <iostream>
#include "math.h"


__device__  void dec_bin(int a,int bin[],int num) {
  int res=0;
  for(int i=0;i<num;i++) {
    res=a%2;
    bin[i]=res;
    a=(a-res)/2;
  }
}


__device__ int bin_dec(int bin[],int num) {
  int dec=0;
  for(int i=0;i<num;i++) {
    dec=dec+(powf(2,i)*bin[i]);
  }
  return dec;
}



  