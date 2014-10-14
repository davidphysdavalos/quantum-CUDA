// {{{ Includes 
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

// }}}
// Tclap {{{
  TCLAP::CmdLine cmd("Command description message", ' ', "0.1");
  TCLAP::ValueArg<string> optionArg("o","option", "Option" ,false,"nichts", "string",cmd);
  TCLAP::ValueArg<int> x("","x", "Size of the x-dimention",false, 2,"int",cmd);
  TCLAP::ValueArg<int> nqubits("q","qubits", "Number of qubits",false, 3,"int",cmd);
  TCLAP::ValueArg<double> ising("","ising_z", "Ising interaction in the z-direction",false, 1,"double",cmd);
  TCLAP::ValueArg<double> bx("","bx", "Magnetic field in x direction",false, 0,"double",cmd);
  TCLAP::ValueArg<double> by("","by", "Magnetic field in y direction",false, 0,"double",cmd);
  TCLAP::ValueArg<double> bz("","bz", "Magnetic field in z direction",false, 0,"double",cmd);
  TCLAP::ValueArg<int> numt("","t", "Number of time steps",false, 1,"int",cmd);
  TCLAP::SwitchArg second_order("","second_order","Do second order Trotter", cmd, false);
  TCLAP::ValueArg<double> kx("","kx", "Momentum of the proyector",false,0,"double",cmd);
  TCLAP::ValueArg<double> ky("","ky", "Momentum of the proyector",false,0,"double",cmd);
  TCLAP::ValueArg<int> dev("","dev", "Gpu to be used, 0 for k20, 1 for c20",false, 0,"int",cmd);
  TCLAP::ValueArg<int> symx("","symx", "If simetry on sigma_x is to be used ",false, 0,"int",cmd);
  TCLAP::ValueArg<string> filename("","savefile", "Name of eigenvalues total savefile" ,false,"nichts", "string",cmd);
  // }}}
  int main(int argc,char* argv[]) {
    // Set initial stuff {{{
      cout.precision(17);
      cudaSetDevice(dev.getValue());
      itpp::RNG_randomize();
      cmd.parse(argc,argv);
      string option=optionArg.getValue();
      
      int l=pow(2,nqubits.getValue());    
      
      double *dev_R,*dev_I;
      double *dev_rotR,*dev_rotI;
      evcuda::cmalloc(&dev_R,&dev_I,l);
      evcuda::cmalloc(&dev_rotR,&dev_rotI,l);
      
      if(option=="test_completeness") {                       // {{{
	int *A=new int[l];
	int y=nqubits.getValue()/x.getValue();
	itpp::cmat check(l,l);
	check=0.;
	itpp::cvec rstate,lstate,temp;
	itpp::cvec vector(l);
	for(int k_x=0;k_x<x.getValue();k_x++) {
	  for(int k_y=0;k_y<y;k_y++) {
	    for(int i=0;i<l;i++) {
	      A[i]=2;
	    }
	    find_states_take2_total(A,nqubits.getValue(),x.getValue(),k_x,k_y,l);
	    int cont=0,rcont=0;
	    
	    for(int i=0;i<l;i++) {
	      cont+=A[i];
	    }
	    
	    itpp::cmat eigenvectors(cont,l);
	    for(int vec=0;vec<cont;vec++) {
	      int flag=0;
	      vector=0.;
	      for(int i=0;i<l;i++) {
		if(A[i]==1 && flag==0) {  
		  vector(i)=1.;
		  flag=1;
		  A[i]=0;
		  // 	      if(vec==34 || vec==53) {
		    // 		cout<<i<<" "<<k_x<<" "<<k_y<<endl;
		  // 	      }
		}
	      }
	      //cout<<vector<<endl;
	      //evcuda::itpp2cuda(vector,dev_R,dev_I);
	      evcuda::proyector_vertical_itpp(vector,dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),k_y);
	      evcuda::proyector_horizontal_itpp(vector,dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),k_x);	  	 
	      if(itpp::norm(vector)>1e-13) {
		eigenvectors.set_row(rcont,vector/itpp::norm(vector));
		rcont++;
	      }
	    }
	    
	    cout<<"Cont "<<rcont<<" "<<k_x<<" "<<k_y<<endl;
	    
	    //cout<<cont;
	    for(int i=0;i<rcont;i++) {
	      rstate=eigenvectors.get_row(i);
	      lstate=itpp::conj(rstate);
	      for(int j=0;j<l;j++) {
		temp=check.get_row(j);
		check.set_row(j,(rstate(j)*lstate)+temp);
	      }
	    }
	  } 
	}
	double error=itpp::norm(itpp::eye_c(l)-check);
	cout<<error<<endl;
	// }}}    
      } 
      else if(option=="find_eigenvalues") {          // {{{
	cudaFree(dev_rotR);
	cudaFree(dev_rotI);
	int nqu=nqubits.getValue();
	int xlen=x.getValue();
	int numthreads,numblocks,i_ver,i_hor;
	choosenumblocks(l,numthreads,numblocks);
	int *S=new int[l];
	for(int m=0;m<l;m++) {
	  S[m]=2;
	}	
	
	if(symx.getValue()!=0) {
	  find_states_take2_total(S,nqu,xlen,kx.getValue(),ky.getValue(),l,symx.getValue());
	}
	else {
	  find_states_take2_total(S,nqu,xlen,kx.getValue(),ky.getValue(),l);
	}
	double *R=new double[l];
	double *I=new double[l];
	int rcont=0;
	
	
	for(int i=0;i<l;i++) {
	  if(S[i]==1) {
	    S[rcont]=i;
	    rcont++;
	  }
	}
	
	double *dotR=new double[rcont];
	double *dotI=new double[rcont];
	// 	
	double *dev_dotR,*dev_dotI;
	int *dev_S;
	cudaMalloc((void**)&dev_S,rcont*sizeof(int)); 
	cudaMemcpy(dev_S,S,rcont*sizeof(int),cudaMemcpyHostToDevice);
	evcuda::cmalloc(&dev_dotR,&dev_dotI,rcont);
	
	itpp::cmat umat(rcont,rcont);
	
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	double icos,isin,kcos,ksin,bx,by,bz;
	set_parameters(ising.getValue(),b,icos,isin,kcos,ksin,bx,by,bz);
	double rotation=itpp::pi/4;
	double rcos=cos(rotation),rsin=sin(rotation);
	
	for(int i=0;i<rcont;i++) {
	  to_zero<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
	  special_both_proyector<<<1,1>>>(dev_R,dev_I,xlen,nqu,rcont,kx.getValue(),ky.getValue(),S[i]);
	  times_norm<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
	  //cudaCheckError("kicki",i);
	  
	  // 	 cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
	  // 	 cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
	  // 	 
	  // 	 for(int m=0;m<l;m++) {
	    // 	    state(m)=std::complex<double>(R[m],I[m]);
	  // 	}
	  // 	cout<<state<<"norm:"<<itpp::norm(state)<<" sum:"<<itpp::sum(state)<<endl;
	  
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,-1.,0.,rcos,rsin,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    i_hor=(q+1)%xlen+(q/xlen)*xlen;
	    i_ver=(q+xlen)%nqu;
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_hor,dev_R,dev_I,icos,isin,l);
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_ver,dev_R,dev_I,icos,isin,l);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,1.,0.,rcos,rsin,l);
	    //cudaCheckError("kick",i);
	  }
	  to_zero<<<numblocks,numthreads>>>(dev_dotR,dev_dotI,rcont);
	  
	  proyected_dot<<<numblocks,numthreads>>>(dev_R,dev_I,dev_dotR,dev_dotI,xlen,nqu,rcont,kx.getValue(),ky.getValue(),dev_S);	  
	  //cudaCheckError("kick",i);
	  
	  //evcuda::itpp2cuda(eigenvectors.get_row(i),dev_R,dev_I);
	  //evcuda::apply_floquet2d(rstate,dev_R,dev_I,ising.getValue(),b,x.getValue());
	  
	  
	  
	  
	  cudaMemcpy(dotR,dev_dotR,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(dotI,dev_dotI,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  //cout<<i<<endl;
	  
	  for(int m=0;m<rcont;m++) {
	    umat(m,i)=std::complex<double>(dotR[m],dotI[m]);
	  }
	  //matriz<<endl;
	  
	}
	
	//cout<<umat<<endl;
	
	// 	     cout << "Prueba de unitariedad " << 
	// 	         itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(rcont))
	// 	 	<<", " << itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(rcont))
	// 	 	<< endl; 
	//cout<<umat<<endl;
	itpp::cvec eigenvalues(rcont);
	// 	
	itpp::eig(umat,eigenvalues);
	
	
	for(int i=0;i<rcont;i++) {
	  cout<<argument(eigenvalues(i))<<" "<<1-norm(eigenvalues(i))<<endl;
	}
	cudaFree(dotR);
	cudaFree(dotI);		
      }  // }}}
      else if(option=="find_eigenvalues_chain") {          // {{{
	cudaFree(dev_rotR);
	cudaFree(dev_rotI);
	cudaFree(dev_R);
	cudaFree(dev_I);
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	
	itpp::cmat umat=evcuda::umat_block_chain(nqubits.getValue(),kx.getValue(),b,ising.getValue());
        
	int rcont=umat.rows();
	
	//cout<<umat<<endl;
	
	// 	     cout << "Prueba de unitariedad " << 
	// 	         itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(rcont))
	// 	 	<<", " << itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(rcont))
	// 	 	<< endl; 
	//cout<<umat<<endl;
	itpp::cvec eigenvalues(rcont);
	// 	
	itpp::eig(umat,eigenvalues);
	
	ofstream eigen;
	eigen.precision(17);
	string file=filename.getValue();
	eigen.open(file.c_str());
	
	for(int i=0;i<rcont;i++) {
	  cout<<argument(eigenvalues(i))<<" "<<1-norm(eigenvalues(i))<<endl;
	}

	eigen.close();
	
	
      } // }}}
      else if(option=="write_umat") {    // {{{
	cudaFree(dev_rotR);
	cudaFree(dev_rotI);
	int nqu=nqubits.getValue();
	int xlen=x.getValue();
	int numthreads,numblocks,i_ver,i_hor;
	choosenumblocks(l,numthreads,numblocks);
	int *S=new int[l];
	for(int m=0;m<l;m++) {
	  S[m]=2;
	}	
	
	if(symx.getValue()!=0) {
	  find_states_take2_total(S,nqu,xlen,kx.getValue(),ky.getValue(),l,symx.getValue());
	}
	else {
	  find_states_take2_total(S,nqu,xlen,kx.getValue(),ky.getValue(),l);
	}
	double *R=new double[l];
	double *I=new double[l];
	int rcont=0;
	
	
	for(int i=0;i<l;i++) {
	  if(S[i]==1) {
	    S[rcont]=i;
	    rcont++;
	  }
	}
	//cout<<"RCONT "<<rcont<<endl;
	// 	  cout<<endl; 
	// 	}
	//cout<<eigenvectors<<endl;
	//cout<<"rcont "<<rcont<<" "<<kx.getValue()<<" "<<ky.getValue()<<endl;
	//cout<<"NORM: "<<itpp::norm(eigenvectors.get_row(1))<<endl;
	double *dotR=new double[rcont];
	double *dotI=new double[rcont];
	// 	
	double *dev_dotR,*dev_dotI;
	int *dev_S;
	cudaMalloc((void**)&dev_S,rcont*sizeof(int)); 
	cudaMemcpy(dev_S,S,rcont*sizeof(int),cudaMemcpyHostToDevice);
	evcuda::cmalloc(&dev_dotR,&dev_dotI,rcont);
	
	//itpp::cmat umat(rcont,rcont);
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	double icos,isin,kcos,ksin,bx,by,bz;
	set_parameters(ising.getValue(),b,icos,isin,kcos,ksin,bx,by,bz);
	double rotation=itpp::pi/4;
	double rcos=cos(rotation),rsin=sin(rotation);
	
	//itpp::cvec state(l);
	//para diagonalizar se usa cat umat__.dat | diagonalize -d rcont 
	//contar lineas wc -l dat
	ofstream matriz;
	matriz.precision(17);
	string file=filename.getValue();
	matriz.open(file.c_str());
	
	for(int i=0;i<rcont;i++) {
	  to_zero<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
	  special_both_proyector<<<1,1>>>(dev_R,dev_I,xlen,nqu,rcont,kx.getValue(),ky.getValue(),S[i]);
	  times_norm<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
	  //cudaCheckError("kicki",i);
	  
	  // 	 cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
	  // 	 cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
	  // 	 
	  // 	 for(int m=0;m<l;m++) {
	    // 	    state(m)=std::complex<double>(R[m],I[m]);
	  // 	}
	  // 	cout<<state<<"norm:"<<itpp::norm(state)<<" sum:"<<itpp::sum(state)<<endl;
	  
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,-1.,0.,rcos,rsin,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    i_hor=(q+1)%xlen+(q/xlen)*xlen;
	    i_ver=(q+xlen)%nqu;
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_hor,dev_R,dev_I,icos,isin,l);
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_ver,dev_R,dev_I,icos,isin,l);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,1.,0.,rcos,rsin,l);
	    //cudaCheckError("kick",i);
	  }
	  to_zero<<<numblocks,numthreads>>>(dev_dotR,dev_dotI,rcont);
	  
	  proyected_dot<<<numblocks,numthreads>>>(dev_R,dev_I,dev_dotR,dev_dotI,xlen,nqu,rcont,kx.getValue(),ky.getValue(),dev_S);	  
	  //cudaCheckError("kick",i);
	  
	  //evcuda::itpp2cuda(eigenvectors.get_row(i),dev_R,dev_I);
	  //evcuda::apply_floquet2d(rstate,dev_R,dev_I,ising.getValue(),b,x.getValue());
	  
	  
	  
	  
	  cudaMemcpy(dotR,dev_dotR,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(dotI,dev_dotI,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  //cout<<i<<endl;
	  
	  for(int m=0;m<rcont;m++) {
	    matriz<<dotR[m]<<" "<<dotI[m]<<endl;
	    //umat(m,i)=std::complex<double>(dotR[m],dotI[m]);
	  }
	  //matriz<<endl;
	  
	}
	
	
	cudaFree(dotR);
	cudaFree(dotI);
	matriz.close();
	
	
      } // }}}
      else if(option=="find_eigenvalues_write") {    // {{{
	cudaFree(dev_rotR);
	cudaFree(dev_rotI);
	int nqu=nqubits.getValue();
	int xlen=x.getValue();
	int numthreads,numblocks,i_ver,i_hor;
	choosenumblocks(l,numthreads,numblocks);
	int *S=new int[l];
	for(int m=0;m<l;m++) {
	  S[m]=2;
	}	
	
	if(symx.getValue()!=0) {
	  find_states_take2_total(S,nqu,xlen,kx.getValue(),ky.getValue(),l,symx.getValue());
	}
	else {
	  find_states_take2_total(S,nqu,xlen,kx.getValue(),ky.getValue(),l);
	}
	double *R=new double[l];
	double *I=new double[l];
	int rcont=0;
	
	
	for(int i=0;i<l;i++) {
	  if(S[i]==1) {
	    S[rcont]=i;
	    rcont++;
	  }
	}
	
	double *dotR=new double[rcont];
	double *dotI=new double[rcont];
	// 	
	double *dev_dotR,*dev_dotI;
	int *dev_S;
	cudaMalloc((void**)&dev_S,rcont*sizeof(int)); 
	cudaMemcpy(dev_S,S,rcont*sizeof(int),cudaMemcpyHostToDevice);
	evcuda::cmalloc(&dev_dotR,&dev_dotI,rcont);
	
	itpp::cmat umat(rcont,rcont);
	
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	double icos,isin,kcos,ksin,bx,by,bz;
	set_parameters(ising.getValue(),b,icos,isin,kcos,ksin,bx,by,bz);
	double rotation=itpp::pi/4;
	double rcos=cos(rotation),rsin=sin(rotation);
	
	for(int i=0;i<rcont;i++) {
	  to_zero<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
	  special_both_proyector<<<1,1>>>(dev_R,dev_I,xlen,nqu,rcont,kx.getValue(),ky.getValue(),S[i]);
	  times_norm<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
	  //cudaCheckError("kicki",i);
	  
	  // 	 cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
	  // 	 cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
	  // 	 
	  // 	 for(int m=0;m<l;m++) {
	    // 	    state(m)=std::complex<double>(R[m],I[m]);
	  // 	}
	  // 	cout<<state<<"norm:"<<itpp::norm(state)<<" sum:"<<itpp::sum(state)<<endl;
	  
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,-1.,0.,rcos,rsin,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    i_hor=(q+1)%xlen+(q/xlen)*xlen;
	    i_ver=(q+xlen)%nqu;
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_hor,dev_R,dev_I,icos,isin,l);
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_ver,dev_R,dev_I,icos,isin,l);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,1.,0.,rcos,rsin,l);
	    //cudaCheckError("kick",i);
	  }
	  to_zero<<<numblocks,numthreads>>>(dev_dotR,dev_dotI,rcont);
	  
	  proyected_dot<<<numblocks,numthreads>>>(dev_R,dev_I,dev_dotR,dev_dotI,xlen,nqu,rcont,kx.getValue(),ky.getValue(),dev_S);	  
	  //cudaCheckError("kick",i);
	  
	  //evcuda::itpp2cuda(eigenvectors.get_row(i),dev_R,dev_I);
	  //evcuda::apply_floquet2d(rstate,dev_R,dev_I,ising.getValue(),b,x.getValue());
	  
	  
	  
	  
	  cudaMemcpy(dotR,dev_dotR,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(dotI,dev_dotI,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  //cout<<i<<endl;
	  
	  for(int m=0;m<rcont;m++) {
	    umat(m,i)=std::complex<double>(dotR[m],dotI[m]);
	  }
	  //matriz<<endl;
	  
	}
	
	//cout<<umat<<endl;
	
	// 	     cout << "Prueba de unitariedad " << 
	// 	         itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(rcont))
	// 	 	<<", " << itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(rcont))
	// 	 	<< endl; 
	//cout<<umat<<endl;
	itpp::cvec eigenvalues(rcont);
	// 	
	itpp::eig(umat,eigenvalues);
	
	ofstream eigen;
	eigen.precision(17);
	string file=filename.getValue();
	eigen.open(file.c_str());
	
	for(int i=0;i<rcont;i++) {
	  eigen<<argument(eigenvalues(i))<<" "<<1-norm(eigenvalues(i))<<endl;
	}
	cudaFree(dotR);
	cudaFree(dotI);
	eigen.close();
	
	
      } // }}}
      else if(option=="find_eigenvalues_diagonalization_check") {   // {{{
	cudaFree(dev_rotR);
	cudaFree(dev_rotI);
	int nqu=nqubits.getValue();
	int xlen=x.getValue();
	int numthreads,numblocks,i_ver,i_hor;
	choosenumblocks(l,numthreads,numblocks);
	int *S=new int[l];
	for(int m=0;m<l;m++) {
	  S[m]=2;
	}	
	
	if(symx.getValue()!=0) {
	  find_states_take2_total(S,nqu,xlen,kx.getValue(),ky.getValue(),l,symx.getValue());
	}
	else {
	  find_states_take2_total(S,nqu,xlen,kx.getValue(),ky.getValue(),l);
	}
	double *R=new double[l];
	double *I=new double[l];
	int rcont=0;
	
	
	for(int i=0;i<l;i++) {
	  if(S[i]==1) {
	    S[rcont]=i;
	    rcont++;
	  }
	}
	//cout<<"RCONT "<<rcont<<endl;
	// 	  cout<<endl; 
	// 	}
	//cout<<eigenvectors<<endl;
	//cout<<"rcont "<<rcont<<" "<<kx.getValue()<<" "<<ky.getValue()<<endl;
	//cout<<"NORM: "<<itpp::norm(eigenvectors.get_row(1))<<endl;
	double *dotR=new double[rcont];
	double *dotI=new double[rcont];
	// 	
	double *dev_dotR,*dev_dotI;
	int *dev_S;
	cudaMalloc((void**)&dev_S,rcont*sizeof(int)); 
	cudaMemcpy(dev_S,S,rcont*sizeof(int),cudaMemcpyHostToDevice);
	evcuda::cmalloc(&dev_dotR,&dev_dotI,rcont);
	
	itpp::cmat umat(rcont,rcont);
	
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	double icos,isin,kcos,ksin,bx,by,bz;
	set_parameters(ising.getValue(),b,icos,isin,kcos,ksin,bx,by,bz);
	double rotation=itpp::pi/4;
	double rcos=cos(rotation),rsin=sin(rotation);
	
	for(int i=0;i<rcont;i++) {
	  to_zero<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
	  special_both_proyector<<<1,1>>>(dev_R,dev_I,xlen,nqu,rcont,kx.getValue(),ky.getValue(),S[i]);
	  times_norm<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
	  //cudaCheckError("kicki",i);
	  
	  // 	 cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
	  // 	 cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
	  // 	 
	  // 	 for(int m=0;m<l;m++) {
	    // 	    state(m)=std::complex<double>(R[m],I[m]);
	  // 	}
	  // 	cout<<state<<"norm:"<<itpp::norm(state)<<" sum:"<<itpp::sum(state)<<endl;
	  
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,-1.,0.,rcos,rsin,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    i_hor=(q+1)%xlen+(q/xlen)*xlen;
	    i_ver=(q+xlen)%nqu;
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_hor,dev_R,dev_I,icos,isin,l);
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_ver,dev_R,dev_I,icos,isin,l);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,1.,0.,rcos,rsin,l);
	    //cudaCheckError("kick",i);
	  }
	  to_zero<<<numblocks,numthreads>>>(dev_dotR,dev_dotI,rcont);
	  
	  proyected_dot<<<numblocks,numthreads>>>(dev_R,dev_I,dev_dotR,dev_dotI,xlen,nqu,rcont,kx.getValue(),ky.getValue(),dev_S);	  
	  //cudaCheckError("kick",i);
	  
	  //evcuda::itpp2cuda(eigenvectors.get_row(i),dev_R,dev_I);
	  //evcuda::apply_floquet2d(rstate,dev_R,dev_I,ising.getValue(),b,x.getValue());
	  
	  
	  
	  
	  cudaMemcpy(dotR,dev_dotR,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(dotI,dev_dotI,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  //cout<<i<<endl;
	  
	  for(int m=0;m<rcont;m++) {
	    umat(m,i)=std::complex<double>(dotR[m],dotI[m]);
	  }
	  //matriz<<endl;
	  
	}
	
	//cout<<umat<<endl;
	
	// 	     cout << "Prueba de unitariedad " << 
	// 	         itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(rcont))
	// 	 	<<", " << itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(rcont))
	// 	 	<< endl; 
	//cout<<umat<<endl;
	itpp::cvec eigenvalues(rcont);
	itpp::cmat eigenvectors(rcont,rcont);
	// 	
	itpp::eig(umat,eigenvalues,eigenvectors);
	
	double error=itpp::norm(umat-eigenvectors*itpp::diag(eigenvalues)*itpp::hermitian_transpose(eigenvectors));
	cout<<"ERROR "<<error<<endl; 
	
	cudaFree(dotR);
	cudaFree(dotI);
	
	
	
      } // }}}
      else if(option=="ublock_check") {   // {{{
	int nqu=nqubits.getValue();
	int xlen=x.getValue();
	int numthreads,numblocks;
	choosenumblocks(l,numthreads,numblocks);
	int* S=new int[l];
	itpp::cmat eigenvectors=evcuda::invariant_vectors(nqubits.getValue(),x.getValue(),kx.getValue(),ky.getValue(),symx.getValue());
	
	int rcont=eigenvectors.rows();
	
	itpp::cvec small_state=itppextmath::RandomState(rcont);
	//itpp::cvec small_state(rcont); small_state=0.; small_state(0)=std::complex<double>(1.,0.);
	small_state=small_state/itpp::norm(small_state);
	cout<<"SMALL "<<itpp::norm(small_state)<<endl;
	
	itpp::cvec big_state = itpp::transpose(eigenvectors)*small_state;
	
	
	evcuda::itpp2cuda(big_state,dev_R,dev_I);
	
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	double icos,isin,kcos,ksin,bx,by,bz;
	set_parameters(ising.getValue(),b,icos,isin,kcos,ksin,bx,by,bz);
	itpp::vec br(3); br(0)=0.; br(1)=itpp::pi/4; br(2)=0.;
// 	double rotation=itpp::pi/4;
// 	double rcos=cos(rotation),rsin=sin(rotation);
	
	
	//evcuda::apply_magnetic_kick(dev_R,dev_I,-1*br,nqu);
	evcuda::apply_floquet2d(dev_R,dev_I,b,ising.getValue(),nqu,xlen);
	//evcuda::apply_magnetic_kick(dev_R,dev_I,1*br,nqu);
	
	evcuda::cuda2itpp(big_state,dev_R,dev_I);
	
	
	itpp::cmat umat=evcuda::umat_block(nqubits.getValue(),x.getValue(),kx.getValue(),ky.getValue(),symx.getValue(),b,ising.getValue());
	
	
	//cout<<"small_state "<<small_state<<endl;
	cout<<"UMAT "<<itpp::round_to_zero(umat,1e-15)<<endl;
	
	//cout<<"E-VECTORS "<<itpp::round_to_zero(eigenvectors,1e-15)<<endl;
	
	small_state=umat*small_state;
	itpp::cvec direct_state=itpp::transpose(eigenvectors)*small_state;
	//cout<<"small_state "<<small_state<<endl;
	
	
	
	
	cout<<"BIG "<<itpp::norm(big_state)<<endl;
	cout<<itpp::round_to_zero(big_state)<<endl;
	
	cout<<"Direct "<<itpp::norm(direct_state)<<endl;
	cout<<itpp::round_to_zero(direct_state)<<endl;
	
	
	//cout<<"EIGEN "<<itppextmath::proportionality_test(small2_state,big_state)<<endl;
	cout<<"ERROR B-D "<<itpp::norm(direct_state-big_state)<<endl;
	
	
	
      } // }}}
      else if(option=="blockstate_corr") {   // {{{
	int nqu=nqubits.getValue();
	int xlen=x.getValue();
	int numthreads,numblocks;
	choosenumblocks(l,numthreads,numblocks);
	int* S=new int[l];
	double *inR=new double[l];
	double *inI=new double[l];
	double *rotR=new double[l];
	double *rotI=new double[l];
	itpp::cmat eigenvectors1=evcuda::invariant_vectors(nqubits.getValue(),x.getValue(),1,1,0);
	itpp::cmat eigenvectors2=evcuda::invariant_vectors(nqubits.getValue(),x.getValue(),1,2,0);

	int rcont1=eigenvectors1.rows();
	int rcont2=eigenvectors2.rows();
	
	
	itpp::cvec small_state=itppextmath::RandomState(rcont1);
	itpp::cvec state = itpp::transpose(eigenvectors1)*small_state;
	small_state=itppextmath::RandomState(rcont2);
	state=state+itpp::transpose(eigenvectors2)*small_state;
	state=state/itpp::norm(state);
	
	//itpp::cvec state=itppextmath::RandomState(l);
	evcuda::itpp2cuda(state,dev_R,dev_I);
	double *dev_inR,*dev_inI;
         evcuda::cmalloc(&dev_inR,&dev_inI,l);
// 	//evcuda::apply_sumdx_itpp(state,dev_R,dev_I,dev_zeroR,dev_zeroI);
//  	itpp::cvec zerostate(l);
// 	double res;
//         choosenumblocks(l,numthreads,numblocks); 
//         sumsigma_x<<<numblocks,numthreads>>>(dev_R,dev_I,dev_zeroR,dev_zeroI,nqu,l);
// 	
// 	
// 	
// 	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
// 
// 	
// 	
// 	for(int t=0;t<numt.getValue();t++) {
// 	  sumsigma_x<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,nqu,l);
// 
// 	  
//  	  //evcuda::apply_floquet2d(dev_R,dev_I,b,ising.getValue(),nqu,xlen);
// 	  //evcuda::apply_floquet2d(dev_zeroR,dev_zeroI,b,ising.getValue(),nqu,xlen);
// // 	
// //           
//            evcuda::cuda2itpp(state,dev_rotR,dev_rotI);
//  	  evcuda::cuda2itpp(zerostate,dev_zeroR,dev_zeroI);
// 	
// 	  res=std::norm(itpp::dot(itpp::conj(zerostate),state));
//           cout<<sqrt(res)/nqu<<endl;
// 	}
//       }
	
	double bx2,by2,bz2,kcos,ksin,icos,isin;
    itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
    set_parameters(ising.getValue(),b,icos,isin,kcos,ksin,bx2,by2,bz2);
    itpp::cvec finalstate(l);
    itpp::cvec zerostate(l);
    double res;
//     double res,norm;
    int i_hor,i_ver;
    
    sumsigma_x<<<numblocks,numthreads>>>(dev_R,dev_I,dev_inR,dev_inI,nqubits.getValue(),l);
    //cudaCheckError("sum_dx",1);
    
    
    
    
    for(int n=0;n<numt.getValue();n++) {
      //se aplica M
      sumsigma_x<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,nqubits.getValue(),l);
      
      cudaMemcpy(inR,dev_inR,l*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(inI,dev_inI,l*sizeof(double),cudaMemcpyDeviceToHost);   
      cudaMemcpy(rotR,dev_rotR,l*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(rotI,dev_rotI,l*sizeof(double),cudaMemcpyDeviceToHost);
      
      //se aplica U a in
      for(int i=0;i<nqubits.getValue();i++) {
	i_hor=(i+1)%xlen+(i/xlen)*xlen;
	i_ver=(i+xlen)%nqubits.getValue();
	Ui_kernel<<<numblocks,numthreads>>>(i,i_hor,dev_inR,dev_inI,icos,isin,l);
	Ui_kernel<<<numblocks,numthreads>>>(i,i_ver,dev_inR,dev_inI,icos,isin,l);
	//cudaCheckError("ising",i);
      }
      for(int i=0;i<nqubits.getValue();i++) {
	Uk_kernel<<<numblocks,numthreads>>>(i,dev_inR,dev_inI,bx2,by2,bz2,kcos,ksin,l);
	//cudaCheckError("kick",i);
      }
      
      //se aplica la  U 
      for(int i=0;i<nqubits.getValue();i++) {
	i_hor=(i+1)%xlen+(i/xlen)*xlen;
	i_ver=(i+xlen)%nqubits.getValue();
	Ui_kernel<<<numblocks,numthreads>>>(i,i_hor,dev_R,dev_I,icos,isin,l);
	Ui_kernel<<<numblocks,numthreads>>>(i,i_ver,dev_R,dev_I,icos,isin,l);
	//cudaCheckError("ising",i);
      }
      for(int i=0;i<nqubits.getValue();i++) {
	Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx2,by2,bz2,kcos,ksin,l);
	//cudaCheckError("kick",i);
      }
      
     
      for(int i=0;i<l;i++) {
	finalstate(i)=std::complex<double>(rotR[i],rotI[i]);
	zerostate(i)=std::complex<double>(inR[i],inI[i]);
      }
      res=std::norm(itpp::dot(itpp::conj(zerostate),finalstate));
      cout<<sqrt(res)/nqubits.getValue()<<endl;
    }
  } // }}} 
      else if(option=="ublock_test") {   // {{{
	
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	
	itpp::cmat umat=evcuda::umat_block(nqubits.getValue(),x.getValue(),kx.getValue(),ky.getValue(),symx.getValue(),b,ising.getValue());
	int rcont=umat.rows();
	
	cout << "Prueba de unitariedad " << 
	itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(rcont))
	<<", " << itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(rcont))
	<< endl;
	
	// }}}	
      } // }}}
      else if(option=="eigenvalues_symmetry") {                 // {{{
	int nqu=nqubits.getValue();
	int xlen=x.getValue();
	int numthreads,numblocks,i_ver,i_hor;
	choosenumblocks(l,numthreads,numblocks);
	int *S=new int[l];
	cout<<"Starts"<<endl;
	itpp::cmat eigenvectors=evcuda::invariant_vectors(nqubits.getValue(),x.getValue(),kx.getValue(),ky.getValue(),symx.getValue());
	double *R=new double[l];
	double *I=new double[l];
	int dimension_invariant_space=eigenvectors.rows();
	
	cout<<"dimension_invariant_space "<<dimension_invariant_space<<" "<<kx.getValue()<<" "<<ky.getValue()<<endl;
	//cout<<"NORM: "<<itpp::norm(eigenvectors.get_row(1))<<endl;
	double *dotR=new double[dimension_invariant_space];
	double *dotI=new double[dimension_invariant_space];
	
	double *dev_dotR,*dev_dotI;
	int *dev_S;
	cudaMalloc((void**)&dev_S,dimension_invariant_space*sizeof(int)); 
	cudaMemcpy(dev_S,S,dimension_invariant_space*sizeof(int),cudaMemcpyHostToDevice);
	
	evcuda::cmalloc(&dev_dotR,&dev_dotI,dimension_invariant_space);
	
	itpp::cmat umat(dimension_invariant_space,dimension_invariant_space);
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	double icos,isin,kcos,ksin,bx,by,bz;
	set_parameters(ising.getValue(),b,icos,isin,kcos,ksin,bx,by,bz);
	
	
	for(int i=0;i<dimension_invariant_space;i++) {
	  
	  
	  for(int m=0;m<l;m++) {                                  // {{{
	    R[m]=real(eigenvectors(i,m));
	    I[m]=imag(eigenvectors(i,m));
	  }                                                       // }}}
	  
	  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
	  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);      
	  
	  
	  for(int q=0;q<nqu;q++) {                                // {{{
	    i_hor=(q+1)%xlen+(q/xlen)*xlen;
	    i_ver=(q+xlen)%nqu;
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_hor,dev_R,dev_I,icos,isin,l);
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_ver,dev_R,dev_I,icos,isin,l);
	  }                                                       // }}}
	  
	  for(int q=0;q<nqu;q++) {                                // {{{
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
	    //cudaCheckError("kick",i);
	  }                                                       // }}}
	  to_zero<<<numblocks,numthreads>>>(dev_dotR,dev_dotI,dimension_invariant_space);
	  proyected_dot<<<numblocks,numthreads>>>(dev_R,dev_I,dev_dotR,dev_dotI,xlen,nqu,dimension_invariant_space,kx.getValue(),ky.getValue(),dev_S);	  
	  cudaMemcpy(dotR,dev_dotR,dimension_invariant_space*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(dotI,dev_dotI,dimension_invariant_space*sizeof(double),cudaMemcpyDeviceToHost);
	  for(int m=0;m<dimension_invariant_space;m++) {
	    umat(m,i)=std::complex<double>(dotR[m],dotI[m]);
	  }
	  
	}
	// 	cout<<umat<<endl;
	double prueba_unitariedad=0;
	prueba_unitariedad = itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(dimension_invariant_space))
	+ itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(dimension_invariant_space));
	if (prueba_unitariedad > 10e-10){
	  
	  cerr << "Alerta, prueba de unitariedad demasiado alta" << prueba_unitariedad << endl;
	  cerr << "1 " << itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(dimension_invariant_space)) << endl;
	  cerr << "1 " << itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(dimension_invariant_space)) << endl;
	  abort();
	}
	itpp::cvec eigenvalues(dimension_invariant_space);
	
	itpp::eig(umat,eigenvalues);
	
	//cout<<eigenvalues << endl;
	for(int i=0;i<dimension_invariant_space;i++) {
	  cout<<argument(eigenvalues(i))<<"          "<<1-norm(eigenvalues(i))<< ", " << eigenvalues(i) << endl;
	}
	cudaFree(dotR);
	cudaFree(dotI);  
	
	
      }  // }}}
      else if(option=="test_PkpUPkis0") {                       // {{{
	// ./eigenvalues --option ph_y_pv_conmutan --qubits 12 --x 4
	itpp::cvec psi, psiHV, psiVH, psip, THpsip, TVpsip, Pkpsi;
	int q=nqubits.getValue();
	int d=cfpmath::pow_2(q);
	
	psi = itppextmath::RandomState(d);
	psi=0.;
	psi(1)=1.; 
	int nx = x.getValue();
	// // 	int ny = q/nx; 
	
	
	int kv=1,kh=1;
	Pkpsi = psi;
	
	evcuda::proyector_horizontal_itpp(Pkpsi, dev_R,dev_I,dev_rotR,dev_rotI,nx,kh);
	evcuda::proyector_vertical_itpp(Pkpsi, dev_R,dev_I,dev_rotR,dev_rotI,nx,kv);
	
	cout<<norm(Pkpsi)<<endl;
	double Ising=.1241324;
	itpp::vec b(3);
	b(0)=.4325;
	b(1)=.52435;
	b(2)=.34;
	psi=Pkpsi;
	itppcuda::apply_floquet2d(Pkpsi, Ising, b, nx);
	cout<<round_to_zero(Pkpsi,1e-15)<<endl;
	
	for(int i=0;i<d;i++) { if(std::norm(round_to_zero(Pkpsi,1e-15)(i))!=0) {cout<<i<<" ";}}
	cout<<endl;
	
	
	evcuda::proyector_horizontal_itpp(Pkpsi, dev_R,dev_I,dev_rotR,dev_rotI,nx,0);
	evcuda::proyector_vertical_itpp(Pkpsi, dev_R,dev_I,dev_rotR,dev_rotI,nx,0);
	cout<<norm(Pkpsi)<<endl;
	
	cout << "Error en la proporcionalidad es "  << itpp::norm(Pkpsi)<< endl;
	
	
      } // }}}
      else if(option=="find_eigenvalues_take3") {               // {{{
	int numthreads,numblocks;
	choosenumblocks(l,numthreads,numblocks);
	itpp::cmat eigenvectors=evcuda::invariant_vectors(nqubits.getValue(),x.getValue(),kx.getValue(),ky.getValue());
	//cout<<eigenvectors<<endl;
	//cout<<"AQUI "<<eigenvectors.rows()<<endl;
	int rcont=eigenvectors.rows();
	itpp::cvec doted(rcont);
	itpp::cvec rstate,lstate,temp;
	cout<<"Cont "<<rcont<<endl;
	//      cout<<eigenvectors<<endl<<"---------------------------------------------"<<endl;
	//       cout << "Prueba de ortonormalidad " << 
	//          itpp::norm(eigenvectors*itpp::hermitian_transpose(eigenvectors) - itpp::eye_c(rcont))
	//  	<< endl; 
	double *dotR,*dotI;
	evcuda::cmalloc(&dotR,&dotI,rcont);
	itpp::cmat umat(rcont,rcont);
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	for(int i=0;i<rcont;i++) {
	  rstate=eigenvectors.get_row(i);
	  doted=0.;
	  evcuda::itpp2cuda(doted,dotR,dotI);
	  
	  evcuda::itpp2cuda(rstate,dev_R,dev_I);
	  //        evcuda::apply_floquet(rstate,dev_R,dev_I,ising.getValue(),b);
	  evcuda::apply_floquet2d_itpp(rstate,dev_R,dev_I,ising.getValue(),b,x.getValue());
	  //evcuda::cuda2itpp(rstate,dev_R,dev_I);
	  //cudaCheckError("kick",i);
	  //cout<<rstate<<endl;
	  for(int j=0;j<rcont;j++) {
	    lstate=eigenvectors.get_row(j);
	    
	    evcuda::itpp2cuda(lstate,dev_rotR,dev_rotI);
	    timed_dot<<<numblocks,numthreads>>>(j,dev_R,dev_I,dev_rotR,dev_rotI,dotR,dotI,l);
	    cudaCheckError("kick",i);
	  }
	  //cout<<i<<endl;
	  evcuda::cuda2itpp(doted,dotR,dotI);
	  umat.set_row(i,doted);
	}
	
	
	
	//      cout << "Prueba de unitariedad " << 
	//          itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(rcont))
	//  	<<", " << itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(rcont))
	//  	<< endl; 
	//cout<<umat<<endl;
	itpp::cvec eigenvalues(rcont);
	
	itpp::eig(umat,eigenvalues);
	
	//cout<<eigenvalues << endl;
	for(int i=0;i<rcont;i++) {
	  cout<<argument(eigenvalues(i))<<"          "<<1-norm(eigenvalues(i))<<endl;
	}
      } // }}}  
      else if(option=="find_eigenvalues_take3_ugly") {          // {{{
	int nqu=nqubits.getValue();
	int xlen=x.getValue();
	int numthreads,numblocks,i_ver,i_hor;
	choosenumblocks(l,numthreads,numblocks);
	itpp::cmat eigenvectors=evcuda::invariant_vectors(nqubits.getValue(),x.getValue(),kx.getValue(),ky.getValue());
	double *R=new double[l];
	double *I=new double[l];
	double *rotR=new double[l];
	double *rotI=new double[l];
	int rcont=eigenvectors.rows();
	// 	for(int i=0;i<rcont;i++) {
	  // 	for(int m=0;m<l;m++) {
	    // 	    cout<<real(eigenvectors(i,m));
	// 	    cout<<"  i"<<imag(eigenvectors(i,m))<<" ";
	// 	  }
	// 	  cout<<endl;
	// 	}
	cout<<rcont<<endl;
	
	double *dotR=new double[l];
	double *dotI=new double[l];
	
	double *dev_dotR,*dev_dotI;
	evcuda::cmalloc(&dev_dotR,&dev_dotI,rcont);
	itpp::cmat umat(rcont,rcont);
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	double icos,isin,kcos,ksin,bx,by,bz;
	set_parameters(ising.getValue(),b,icos,isin,kcos,ksin,bx,by,bz);
	
	
	for(int i=0;i<rcont;i++) {
	  
	  
	  for(int m=0;m<l;m++) {
	    R[m]=real(eigenvectors(i,m));
	    I[m]=imag(eigenvectors(i,m));
	  }
	  
	  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
	  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);      
	  
	  
	  for(int q=0;q<nqu;q++) {
	    i_hor=(q+1)%xlen+(q/xlen)*xlen;
	    i_ver=(q+xlen)%nqu;
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_hor,dev_R,dev_I,icos,isin,l);
	    Ui_kernel<<<numblocks,numthreads>>>(q,i_ver,dev_R,dev_I,icos,isin,l);
	  }
	  
	  for(int q=0;q<nqu;q++) {
	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
	    //cudaCheckError("kick",i);
	  }
	  to_zero<<<numblocks,numthreads>>>(dev_dotR,dev_dotI,rcont);
	  
	  
	  
	  //evcuda::itpp2cuda(eigenvectors.get_row(i),dev_R,dev_I);
	  //evcuda::apply_floquet2d(rstate,dev_R,dev_I,ising.getValue(),b,x.getValue());
	  
	  for(int j=0;j<rcont;j++) {
	    for(int m=0;m<l;m++) {
	      rotR[m]=real(eigenvectors(j,m));
	      rotI[m]=imag(eigenvectors(j,m));
	    }
	    
	    cudaMemcpy(dev_rotR,rotR,l*sizeof(double),cudaMemcpyHostToDevice);
	    cudaMemcpy(dev_rotI,rotI,l*sizeof(double),cudaMemcpyHostToDevice);      
	    
	    timed_dot<<<numblocks,numthreads>>>(j,dev_R,dev_I,dev_rotR,dev_rotI,dev_dotR,dev_dotI,l);
	    //cudaCheckError("kick",i);
	  }
	  
	  
	  cudaMemcpy(dotR,dev_dotR,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(dotI,dev_dotI,rcont*sizeof(double),cudaMemcpyDeviceToHost);
	  
	  
	  for(int m=0;m<rcont;m++) {
	    umat(m,i)=std::complex<double>(dotR[m],dotI[m]);
	  }
	  
	}
	
	
	
	//      cout << "Prueba de unitariedad " << 
	//          itpp::norm(umat*itpp::hermitian_transpose(umat) - itpp::eye_c(rcont))
	//  	<<", " << itpp::norm(itpp::hermitian_transpose(umat)*umat - itpp::eye_c(rcont))
	//  	<< endl; 
	//cout<<umat<<endl;
	itpp::cvec eigenvalues(rcont);
	
	itpp::eig(umat,eigenvalues);
	
	//cout<<eigenvalues << endl;
	for(int i=0;i<rcont;i++) {
	  cout<<argument(eigenvalues(i))<<"          "<<1-norm(eigenvalues(i))<<endl;
	}
	cudaFree(dotR);
	cudaFree(dotI);
	
	
      } // }}}
      else if(option=="test_completeness_horizontal") {         // {{{
	int *A=new int[l];
	itpp::cmat check(l,l);
	check=0.;
	itpp::cvec rstate,lstate,temp;
	for(int k=0;k<x.getValue();k++) {
	  for(int i=0;i<l;i++) {
	    A[i]=2;
	  }
	  find_states_horizontal(A,nqubits.getValue(),x.getValue(),k,l);
	  int cont=0;
	  for(int i=0;i<l;i++) {
	    cont+=A[i];
	  }
	  itpp::cmat eigenvectors(cont,l);
	  itpp::cvec vector(l);
	  
	  for(int vec=0;vec<cont;vec++) {
	    int flag=0;
	    vector=0.;
	    for(int i=0;i<l;i++) {
	      if(A[i]==1 && flag==0) {
		vector(i)=1.;
		flag=1;
		A[i]=0;
	      }
	    }
	    //evcuda::itpp2cuda(vector,dev_R,dev_I);
	    evcuda::proyector_horizontal_itpp(vector,dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),k);
	    
	    eigenvectors.set_row(vec,vector/itpp::norm(vector));
	  }
	  
	  //cout<<eigenvectors<<endl<<"---------------------------------------------"<<endl;
	  //cout<<cont;
	  for(int i=0;i<cont;i++) {
	    rstate=eigenvectors.get_row(i);
	    lstate=itpp::conj(rstate);
	    for(int j=0;j<l;j++) {
	      temp=check.get_row(j);
	      check.set_row(j,(rstate(j)*lstate)+temp);
	    }
	  }
	  
	  
	}
	cout<<check<<endl;
	double error=itpp::norm(itpp::eye_c(l)-check);
	cout<<error<<endl;
	
      } // }}}
      else if(option=="test_completeness_vertical") {           // {{{
	int *A=new int[l];
	itpp::cmat check(l,l);
	check=0.;
	int y=nqubits.getValue()/x.getValue();
	itpp::cvec rstate,lstate,temp;
	for(int k=0;k<y;k++) {
	  for(int i=0;i<l;i++) {
	    A[i]=2;
	  }
	  find_states_vertical(A,nqubits.getValue(),x.getValue(),k,l);
	  int cont=0;
	  for(int i=0;i<l;i++) {
	    cont+=A[i];
	  }
	  itpp::cmat eigenvectors(cont,l);
	  itpp::cvec vector(l);
	  
	  for(int vec=0;vec<cont;vec++) {
	    int flag=0;
	    vector=0.;
	    for(int i=0;i<l;i++) {
	      if(A[i]==1 && flag==0) {
		vector(i)=1.;
		flag=1;
		A[i]=0;
	      }
	    }
	    //evcuda::itpp2cuda(vector,dev_R,dev_I);
	    evcuda::proyector_vertical_itpp(vector,dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),k);
	    
	    eigenvectors.set_row(vec,vector/itpp::norm(vector));
	  }
	  
	  //cout<<eigenvectors<<endl<<"---------------------------------------------"<<endl;
	  //cout<<cont;
	  for(int i=0;i<cont;i++) {
	    rstate=eigenvectors.get_row(i);
	    lstate=itpp::conj(rstate);
	    for(int j=0;j<l;j++) {
	      temp=check.get_row(j);
	      check.set_row(j,(rstate(j)*lstate)+temp);
	    }
	  }
	  
	  
	}
	cout<<check<<endl;
	double error=itpp::norm(itpp::eye_c(l)-check);
	cout<<error<<endl;
      } // }}}    
      else if(option=="test_vh_eigenvector") {                  // {{{
	// ./eigenvalues --option ph_y_pv_conmutan --qubits 12 --x 4
	itpp::cvec psi, psiHV, psiVH, psip, THpsip, TVpsip;
	int q=nqubits.getValue();
	int d=cfpmath::pow_2(q);
	
	psi = itppextmath::RandomState(d);
	psi=0.;
	psi(84)=1.;
	int nx = x.getValue();
	// 	int ny = q/nx; 
	double error=0;
	
	//for (int kv=0; kv < ny; kv ++){ for (int kh=0; kh < nx; kh ++){ // {{{
	  int kv=1,kh=1;
	  psip = psi;
	  
	  cout<<psip<<endl;
	  
	  evcuda::proyector_horizontal_itpp(psip, dev_R,dev_I,dev_rotR,dev_rotI,nx,kh);
	  cout<<psip<<endl;
	  evcuda::proyector_vertical_itpp(psip, dev_R,dev_I,dev_rotR,dev_rotI,nx,kv);
	  THpsip = psip; TVpsip = psip; 
	  
	  cout<<psip<<endl;
	  cout<<itpp::norm(psip)<<endl;
	  
	  evcuda::apply_horizontal_rotation_itpp(THpsip,dev_R,dev_I,dev_rotR,dev_rotI,nx); 
	  evcuda::apply_vertical_rotation_itpp(TVpsip,dev_R,dev_I,dev_rotR,dev_rotI,nx); 
	  
	  error += abs(itppextmath::proportionality_test(THpsip,psip));
	  error += abs(itppextmath::proportionality_test(TVpsip,psip));
	  //}} // }}}
	  cout << "Error en la proporcionalidad es "  << error << endl;
      } // }}}
      else if(option=="test_ph_and_pv_conmutation") {           // {{{
	// ./eigenvalues --option ph_y_pv_conmutan --qubits 12 --x 4
	itpp::cvec psi, psiHV, psiVH;
	int q=nqubits.getValue();
	int d=cfpmath::pow_2(q);
	
	psi = itppextmath::RandomState(d);
	int nx = x.getValue();
	int ny = q/nx; 
	
	for (int kv=0; kv < ny; kv ++){ for (int kh=0; kh < nx; kh ++){ 
	  psiHV = psi; psiVH = psi;
	  evcuda::proyector_vertical_itpp(psiHV, dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),kv);
	  evcuda::proyector_horizontal_itpp(psiHV, dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),kh);
	  
	  evcuda::proyector_horizontal_itpp(psiVH, dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),kh);
	  evcuda::proyector_vertical_itpp(psiVH, dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),kv);
	  
	  std::cout << "Checando que conmuten, norm(psiHV - psiVH) = " << itpp::norm(psiHV - psiVH) << std::endl; 
	}}
	
      } // }}}
      else if(option=="find_eigenvalues_horizontal") {          // {{{
	int *A=new int[l];
	itpp::cvec rstate,lstate,temp;
	for(int i=0;i<l;i++) {
	  A[i]=2;
	}
	find_states_horizontal(A,nqubits.getValue(),x.getValue(),kx.getValue(),l);
	int cont=0,rcont=0;
	for(int i=0;i<l;i++) {
	  cont+=A[i];
	}
	itpp::cmat eigenvectors(cont,l);
	itpp::cvec vector(l);
	
	for(int vec=0;vec<cont;vec++) {
	  int flag=0;
	  vector=0.;
	  for(int i=0;i<l;i++) {
	    if(A[i]==1 && flag==0) {
	      vector(i)=1.;
	      flag=1;
	      A[i]=0;
	    }
	  }
	  //evcuda::itpp2cuda(vector,dev_R,dev_I);
	  evcuda::proyector_horizontal_itpp(vector,dev_R,dev_I,dev_rotR,dev_rotI,x.getValue(),kx.getValue());
	  
	  if(itpp::norm(vector)>1e-13) {
	    eigenvectors.set_row(rcont,vector/itpp::norm(vector));
	    rcont++;
	  }
	}
	//cout<<"CONT "<<rcont<<endl;
	//cout<<eigenvectors<<endl<<"---------------------------------------------"<<endl;
	itpp::cmat umat(rcont,rcont);
	itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	for(int i=0;i<rcont;i++) {
	  rstate=eigenvectors.get_row(i);
	  evcuda::itpp2cuda(rstate,dev_R,dev_I);
	  evcuda::apply_floquet_itpp(rstate,dev_R,dev_I,ising.getValue(),b);
	  //evcuda::apply_floquet2d(rstate,dev_R,dev_I,ising.getValue(),b,x.getValue());
	  evcuda::cuda2itpp(rstate,dev_R,dev_I);
	  cudaCheckError("kick",i);
	  //cout<<rstate<<endl;
	  for(int j=0;j<rcont;j++) {
	    lstate=eigenvectors.get_row(j);
	    umat(i,j)=itpp::dot(itpp::conj(lstate),rstate);
	  }
	}
	//cout<<umat<<endl;
	itpp::cvec eigenvalues(rcont);
	
	itpp::eig(umat,eigenvalues);
	
	//cout<<eigenvalues << endl;
	for(int i=0;i<rcont;i++) {
	  cout<<argument(eigenvalues(i))<<"          "<<1-norm(eigenvalues(i))<<endl;
	}
      }                 // }}}
      else if(option=="trU_phasespace") {                  // {{{
	  // ./eigenvalues --option ph_y_pv_conmutan --qubits 12 --x 4
	  itpp::cvec psi,psit;
	  int q=nqubits.getValue();
	  int l=pow(2,q);
	  
	  psi = itppextmath::RandomState(l);
	  psit=psi;
	  
	  itpp::vec b(3); b(0)=0.; b(1)=0.; b(2)=0.;
	  
	  
	  for(double is=itpp::pi/4.;is<=itpp::pi/4.;is+=0.025) {
      for(double bxi=0;bxi<=itpp::pi/2;bxi+=itpp::pi/640) {
	  b(0)=bxi;
	  evcuda::itpp2cuda(psi,dev_R,dev_I);
	  
	  for(int t=0;t<12;t++) {
	  evcuda::apply_floquet2d(dev_R,dev_I,b,is,q,x.getValue(),1);
	  evcuda::cuda2itpp(psit,dev_R,dev_I);
	  cout<<norm(itpp::dot(itpp::conj(psi),psit))<<" ";  
	  } 
	cout<<endl;
      }
      
      
	  }
	  
	} // }}}
	else if(option=="trU_phasespace_qvariance") {                  // {{{
	  // ./eigenvalues --option ph_y_pv_conmutan --qubits 12 --x 4
	  itpp::cvec psi,psit;
	  int q=nqubits.getValue();
	  int l=pow(2,q);
	  psi = itppextmath::RandomState(l);
	  psit=psi; 
	  itpp::vec b(3); b(0)=0.; b(1)=0.; b(2)=0.;
	  b(0)=0.17;
	  double is=0.54
	  ;
	  evcuda::itpp2cuda(psit,dev_R,dev_I);
	  evcuda::apply_floquet2d(dev_R,dev_I,b,is,q,x.getValue(),3);
	  evcuda::cuda2itpp(psit,dev_R,dev_I);
	  cout<<nqubits.getValue()<<" "<<norm(itpp::dot(itpp::conj(psi),psit));  
	  
	  cout<<endl;
	  
	  
	} // }}}
	else if(option=="trU_phasespace_block") {                  // {{{
	  // ./eigenvalues --option ph_y_pv_conmutan --qubits 12 --x 4
	  itpp::cvec psit(l);
	  int q=nqubits.getValue();
	  int l=pow(2,q);
	  
	  itpp::vec b(3); b(0)=0.; b(1)=0.; b(2)=0.;
	  
	  itpp::cmat eigenvectors1=evcuda::invariant_vectors(nqubits.getValue(),x.getValue(),1,1,0);
	itpp::cmat eigenvectors2=evcuda::invariant_vectors(nqubits.getValue(),x.getValue(),1,2,0);

	int rcont1=eigenvectors1.rows();
	int rcont2=eigenvectors2.rows();
	
	
	itpp::cvec small_state=itppextmath::RandomState(rcont1);
	itpp::cvec state = itpp::transpose(eigenvectors1)*small_state;
	small_state=itppextmath::RandomState(rcont2);
	state=state+itpp::transpose(eigenvectors2)*small_state;
	state=state/itpp::norm(state);
	

	  for(double is=0.;is<=0.8;is+=0.005) {
      for(double bxi=0.;bxi<=1.6;bxi+=0.005) {
	  b(0)=bxi;
	  evcuda::itpp2cuda(state,dev_R,dev_I);
	  
	  for(int t=0;t<12;t++) {
	  evcuda::apply_floquet2d(dev_R,dev_I,b,is,q,x.getValue(),1);
	  evcuda::cuda2itpp(psit,dev_R,dev_I);

	  cout<<norm(itpp::dot(itpp::conj(state),psit))<<" ";  

	    
	  } 
	cout<<endl;
      }
      
      
	  }
	  
	} // }}}
      else if(option=="trU") {                  // {{{
	  // ./eigenvalues --option ph_y_pv_conmutan --qubits 12 --x 4
	  itpp::cvec psi,psit;
	  int q=nqubits.getValue();
	  int l=pow(2,q);
	  
	  psi = itppextmath::RandomState(l);
	  psit=psi;
	  
	  itpp::vec b(3); b(0)=bx.getValue(); b(1)=by.getValue(); b(2)=bz.getValue();
	  
	  
	  for(int t=0;t<numt.getValue();t++) {
	  itppcuda::apply_floquet2d(psit,ising.getValue(),b,x.getValue(),1);
	  cout<<t+1<<" "<<norm(itpp::dot(itpp::conj(psi),psit))<<endl;  
	  }
	
      }      // }}}
      // Set final stuff {{{
	cudaFree(dev_R);
	cudaFree(dev_I);
	cudaFree(dev_rotR);
	cudaFree(dev_rotI);
	// }}}
      }
      
      
      
