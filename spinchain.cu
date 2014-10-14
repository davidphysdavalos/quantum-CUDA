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


TCLAP::CmdLine cmd("Command description message", ' ', "0.1");
TCLAP::ValueArg<string> optionArg("o","option", "Option" ,false,"nichts", "string",cmd);
TCLAP::ValueArg<int> nqubits("q","qubits", "Number of qubits",false, 3,"int",cmd);
TCLAP::ValueArg<int> numt("","t", "Number of time iterartions",false, 1,"int",cmd);
TCLAP::ValueArg<int> position("","position", "The position of something",false, 0,"int",cmd);
TCLAP::ValueArg<int> whichq("","which", "Which qubits in densmat",false, 1,"int",cmd);
TCLAP::ValueArg<int> x("","x", "Size of the x-dimention",false, 0,"int",cmd);
TCLAP::ValueArg<int> y("","y", "Size of the y-dimention",false, 0,"int",cmd);
//TCLAP::ValueArg<int> position2("","position2", "The position of something",false, 3,"int",cmd);
TCLAP::ValueArg<double> ising("","ising_z", "Ising interaction in the z-direction",false, 0,"double",cmd);
TCLAP::ValueArg<double> deltav("","delta", "Some small delta",false, 1,"double",cmd);
TCLAP::ValueArg<double> trotternum("","trotter", "Number of steps for trotter-suzuki algorithm",false, 1,"double",cmd);
TCLAP::ValueArg<double> bx("","bx", "Magnetic field in x direction",false, 0,"double",cmd);
TCLAP::ValueArg<double> by("","by", "Magnetic field in y direction",false, 0,"double",cmd);
TCLAP::ValueArg<double> bz("","bz", "Magnetic field in z direction",false, 0,"double",cmd);

int main(int argc,char* argv[]) {
  cudaSetDevice(0);
  itpp::RNG_randomize();
  int nqubits=nqubits.getValue();
  int l=pow(2,nqubits);
  itpp::cvec state = itppextmath::RandomState(l);
  itpp::cvec cstate(l);
  for(int i=0;i<l;i++) {
    cstate(i)=state(i);
  }
  double ising=1.0;
  itpp::vec b(3); b(0)=1.0; b(1)=0; b(2)=0;
  double *dev_R,*dev_I;
  evcuda::itpp2cuda(state,&dev_R,&dev_I);
  evcuda::apply_floquet(state,dev_R,dev_I,ising,b);
  
  evcuda::apply_sumdx(state,dev_R,dev_I,dev_sumdxR,dev_sumdxI);
  evcuda::cuda2itpp(state,dev_R,dev_I);
  cout << state<<endl;
  cout << cstate;
}