#include <iostream>
#include <fstream>
#include <itpp/itbase.h>
#include <itpp_ext_math.cpp>
#include <math.h>
#include <tclap/CmdLine.h>
#include <device_functions.h>
#include <cuda.h>
#include "tools.cpp"
#include <spinchain.cpp>
#include "cuda_functions.cu"
#include "cuda_utils.cu"
#include "ev_routines.cu"
#include "cfp_routines.cu"
#include <tclap/CmdLine.h>




  TCLAP::CmdLine cmd("Command description message", ' ', "0.1");
  TCLAP::ValueArg<string> optionArg("o","option", "Option" ,false,"nichts", "string",cmd);
  TCLAP::ValueArg<int> x("","x", "Size of the x-dimention",false, 2,"int",cmd);
  TCLAP::ValueArg<int> qubits("q","qubits", "Number of qubits",false, 3,"int",cmd);
  TCLAP::ValueArg<double> ising("","ising_z", "Ising interaction in the z-direction",false, 1,"double",cmd);
  TCLAP::ValueArg<double> bx("","bx", "Magnetic field in x direction",false, 0,"double",cmd);
  TCLAP::ValueArg<double> by("","by", "Magnetic field in y direction",false, 0,"double",cmd);
  TCLAP::ValueArg<double> bz("","bz", "Magnetic field in z direction",false, 0,"double",cmd);
  TCLAP::ValueArg<int> numt("","t", "Number of time steps",false, 1,"int",cmd);
  TCLAP::SwitchArg second_order("","second_order","Do second order Trotter", cmd, false);	
  TCLAP::ValueArg<double> k("","k", "qusimomentum number",false,0,"double",cmd);
  TCLAP::ValueArg<int> dev("","dev", "Gpu to be used, 0 for k20, 1 for c20",false, 0,"int",cmd);
  TCLAP::ValueArg<int> symx("","symx", "If simetry on sigma_x is to be used ",false, 0,"int",cmd);
  TCLAP::ValueArg<string> filename("","savefile", "Name of eigenvalues total savefile" ,false,"nichts", "string",cmd);
  
    int main(int argc,char* argv[]) {
		
      cout.precision(17);
      cudaSetDevice(dev.getValue());
      itpp::RNG_randomize();
      cmd.parse(argc,argv);
      string option=optionArg.getValue();
      
      itpp::cmat vec=evcuda::invariant_vectors_chain(qubits.getValue(),k.getValue());
      
      for (int i=0; i<vec.rows(); i++){
      for (int j=0; j<vec.cols(); j++) {
				cout<< real(vec(i,j)) <<" ";
      
	}
	cout<<endl;
   //pow(qubits.getValue(),2)   
  }
      
  }