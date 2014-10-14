#include <iostream>
#include <itpp/itbase.h>
#include "../libs/cpp/itpp_ext_math.cpp"
#include <math.h>
#include <tclap/CmdLine.h>
#include <device_functions.h>
#include <cuda.h>
#include "tools.cpp"
#include "../libs/cpp/spinchain.cpp"
#include "cuda_functions.cu"
#include "cuda_utils.cu"
#include "ev_routines.cu"


int main(int argc,char* argv[]) {
  cudaSetDevice(0);
  itpp::RNG_randomize();
  int nqubits=3;
  int l=pow(2,nqubits);
  itpp::cvec state = itppextmath::RandomState(l);
  itpp::cvec cstate(l);
  for(int i=0;i<l;i++) {
    cstate(i)=state(i);
  }
  double ising=1.0;
  itpp::vec b(3); b(0)=0.0; b(1)=0; b(2)=0;
  double *dev_R,*dev_I;
  evcuda::itpp2cuda(state,&dev_R,&dev_I);
  evcuda::apply_floquet(state,dev_R,dev_I,ising,b);
  
  for(int i=0;i<nqubits;i++) { 
    spinchain::apply_ising_z(cstate,ising,i,(i+1)%nqubits);
  }
  for(int i=0;i<nqubits;i++) { 
    spinchain::apply_magnetic_kick(cstate,b,i);
  }
  evcuda::cuda2itpp(state,dev_R,dev_I);
  cout << state<<endl;
  cout << cstate;
}