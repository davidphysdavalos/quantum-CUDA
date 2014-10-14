#include <iostream>
#include <itpp/itbase.h>
#include "itpp_ext_math.cpp"
#include <math.h>
#include <tclap/CmdLine.h>
#include <device_functions.h>
#include <cuda.h>
#include "tools.cpp"
#include "spinchain.cpp"
#include "cuda_functions.cu"
#include "cuda_utils.cu"
#include "cfp_routines.cu"
#include <tclap/CmdLine.h>
TCLAP::CmdLine cmd("Command description message", ' ', "0.1");
TCLAP::ValueArg<int> x("","x", "Size of the x-dimention",false, 2,"int",cmd);
TCLAP::ValueArg<int> y("","y", "Size of the y-dimention",false, 2,"int",cmd);
TCLAP::ValueArg<double> ising("","ising_z", "Ising interaction in the z-direction",false, 1,"double",cmd);
TCLAP::ValueArg<double> bx("","bx", "Magnetic field in x direction",false, 0,"double",cmd);
TCLAP::ValueArg<double> by("","by", "Magnetic field in y direction",false, 0,"double",cmd);
TCLAP::ValueArg<double> bz("","bz", "Magnetic field in z direction",false, 0,"double",cmd);
TCLAP::ValueArg<int> numt("","t", "Number of time steps",false, 1,"int",cmd);
TCLAP::SwitchArg second_order("","second_order","Do second order Trotter", cmd, false);


int main(int argc,char* argv[]) {
  cudaSetDevice(0);
  cout.precision(16);
  cmd.parse(argc,argv);
  itpp::RNG_randomize();
  int nx=x.getValue();
  int ny=y.getValue();
  int nqubits = nx*ny;

  int l=pow(2,nqubits);
  itpp::cvec state = itppextmath::RandomState(l);
  itpp::cvec cstate = state;
  double J=ising.getValue();
//   ising=0.;
  itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
//   b=0.;

  int int_ts=numt.getValue();



//   cout << "Iniciando la construcción de la matriz" << endl;
  for (int i=0; i<cfpmath::pow_2(nqubits); i++){
//     cout << "Probando estado " << i <<  endl;
    state=0.;
    state(i)=1;
//     cout << "Por Evolucionado el  estado " << i <<  endl;
    if (second_order.getValue()){
      itppcuda::apply_floquet2d_trotter2g(state,J,b,nx, int_ts);
    } else {
      itppcuda::apply_floquet2d_trotter1g(state,J,b,nx, int_ts);
    }
//     itppcuda::apply_floquet2d_trotter2g(state,J,b,nx, int_ts);
//     cout << "Evolucionado el  estado " << i <<  endl;
//     cout << state << endl;
      for (int j=0; j<cfpmath::pow_2(nqubits); j++){
         cout << real(state(j)) << " " << imag(state(j)) << endl;
//       cout << i << " " << j << " " << real(state(j)) << " " << imag(state(j)) << endl;
      }
    }
//   cout<<state;
//   itppcuda::apply_floquet_trotter1g(cstate,J,b,int_ts);
//   cout<<cstate;
//   cout<<itpp::norm(state-cstate) << endl;

  
//   apply_floquet2d_trotter2g(itpp::cvec& state,double J,itpp::vec b,int xlen,double numtrotter)
//   int i=3;
//   cout << double(i)/2;
//   for(int i=0;i<nqubits;i++) { 
//     spinchain::apply_ising_z(cstate,J,i,(i+1)%nqubits);
//   }
//   for(int i=0;i<nqubits;i++) { 
//     spinchain::apply_magnetic_kick(cstate,b,i);
//   }
//   cout << state<<endl;
//   cout << cstate;
}
