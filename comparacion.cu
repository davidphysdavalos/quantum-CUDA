#include <iostream>
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
#include <time.h>



double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
	return diffms;
} 

void comparacion_magnetic(itpp::vec b,int numt,int nqubits) {
  int l=pow(2,nqubits);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  
  double* R=new double[l];
  double* I=new double[l];
  
  randomstate(l,R,I);
  
  double *dev_R;
  double *dev_I;
  
  
  itpp::cvec state(l);
  for (int ind=0; ind<l; ind++){
     state(ind) = complex<double>(R[ind],I[ind]);
  }
  
  cudaMalloc((void**)&dev_R,l*sizeof(double));
  cudaMalloc((void**)&dev_I,l*sizeof(double));
  
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  
  double theta=sqrt(b(0)*b(0)+b(1)*b(1)+b(2)*b(2));
  double mcos=cos(theta);
  double msen=sin(theta);
  //cout << "COS  " <<mcos<<endl;
  //cout << "SEN  " <<msen<<endl;
  double bx=b(0)/theta;
  double by=b(1)/theta;
  double bz=b(2)/theta;
  for(int t=0;t<numt;t++) {
    for(int i=0;i<nqubits;i++) {
      Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx,by,bz,mcos,msen,l);
      cudaCheckError("kick",i);
      spinchain::apply_magnetic_kick(state,b,i);
    }
  }
  
  itpp::cvec cudastate(l);
  
  cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
  
  
  for(int ind=0;ind<l;ind++) {
    cudastate(ind) = complex<double>(R[ind],I[ind]);
  }

  cudaFree(dev_R);
  cudaFree(dev_I);
  //cout << cudastate<< endl;
  //cout << state << endl;
  cout<<itpp::norm(cudastate-state)<<endl;
  
}

void comparacion_ising(double Ising,int numt,int nqubits) {
  int l=pow(2,nqubits);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  
  double* R=new double[l];
  double* I=new double[l];
  
  randomstate(l,R,I);
  
  double *dev_R;
  double *dev_I;
  
  itpp::cvec state(l);
  for (int ind=0; ind<l; ind++){
     state(ind) = complex<double>(R[ind],I[ind]);
  }
  
  cudaMalloc((void**)&dev_R,l*sizeof(double));
  cudaMalloc((void**)&dev_I,l*sizeof(double));
  
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  
  double mcos=cos(Ising);
  double msen=sin(Ising);
  for(int t=0;t<numt;t++) {
    for(int i=0;i<nqubits;i++) {
      Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,mcos,msen,l);
      
      spinchain::apply_ising_z(state,Ising,i,(i+1)%nqubits);
    }
  }
  itpp::cvec cudastate(l);
  
  cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
  
  
  for(int ind=0;ind<l;ind++) {
    cudastate(ind) = complex<double>(R[ind],I[ind]);
  }

  cudaFree(dev_R);
  cudaFree(dev_I);
  cout<<itpp::norm(cudastate-state)<<endl;
 
}




void comparacion_cadena(double Ising,itpp::vec b,int numt,int nqubits ) {
  int l=pow(2,nqubits);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  
  double* R=new double[l];
  double* I=new double[l];
  
  randomstate(l,R,I);  
  double *dev_R;
  double *dev_I;
  
  
  itpp::cvec state(l);
  for (int ind=0; ind<l; ind++){
     state(ind) = complex<double>(R[ind],I[ind]);
  }
  
  cudaMalloc((void**)&dev_R,l*sizeof(double));
  cudaMalloc((void**)&dev_I,l*sizeof(double));
  
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  
  double theta=sqrt(b(0)*b(0)+b(1)*b(1)+b(2)*b(2));
  double mcos=cos(theta);
  double msen=sin(theta);
  double bx=b(0)/theta;
  double by=b(1)/theta;
  double bz=b(2)/theta;
  double icos=cos(Ising);
  double isen=sin(Ising);
  for(int t=0;t<numt;t++) {
    for(int i=0;i<nqubits;i++) {
      Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,icos,isen,l);
      cudaCheckError("ising",i);
      spinchain::apply_ising_z(state,Ising,i,(i+1)%nqubits);
    }
    for(int i=0;i<nqubits;i++) {
      Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx,by,bz,mcos,msen,l);
      cudaCheckError("kick",i);
      spinchain::apply_magnetic_kick(state,b,i);
    }
  
    itpp::cvec cudastate(l);
    
    cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
    
    
    for(int ind=0;ind<l;ind++) {
      cudastate(ind) = complex<double>(R[ind],I[ind]);
    }

    cout<<itpp::norm(cudastate-state)<<endl;
  }
  
  cudaFree(dev_R);
  cudaFree(dev_I);    
}

void cadena_carlos_tiempo(double Ising,itpp::vec b,int numt,int nqubits) {
  int l=pow(2,nqubits);
  itpp::cvec state;
  state=itppextmath::RandomState(l);
  clock_t begin=clock();
  for(int n=0;n<numt;n++) {
    for(int i=0;i<nqubits;i++) {
      spinchain::apply_ising_z(state,Ising,i,(i+1)%nqubits);
    }
    for(int i=0;i<nqubits;i++) {
      spinchain::apply_magnetic_kick(state,b,i);
    }
  }
  clock_t end=clock();
  double tiempo=double(diffclock(end,begin));
  cout <<tiempo/numt<< endl;
}

  

void comparacion_sum_sigx(int nqubits) {
  //cambiado para ver cuando se aplican dos 
  int l=pow(2,nqubits);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  
  double* R=new double[l];
  double* I=new double[l];
  
  randomstate(l,R,I);  
  double *dev_R;
  double *dev_I;
  
  
  itpp::cvec state(l),zerostate(l);
  itpp::cvec cudastate(l);
  itpp::cmat sumsig=itpp::zeros_c(l,l);
  itpp::vec b(3); b(0)=1.; b(1)=0.; b(2)=0.;
  for (int ind=0; ind<l; ind++){
    state(ind) = complex<double>(R[ind],I[ind]);
  }
  zerostate=state;
  
  cudaSafeCall(cudaMalloc((void**)&dev_R,l*sizeof(double)),"malloc",1);
  cudaSafeCall(cudaMalloc((void**)&dev_I,l*sizeof(double)),"malloc",2);
  
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  
  double *sumdxR=new double[l];
  double *sumdxI=new double[l];
  double *dev_sumdxR;      
  double *dev_sumdxI;
  cudaMalloc((void**)&dev_sumdxR,l*sizeof(double));     
  cudaMalloc((void**)&dev_sumdxI,l*sizeof(double));
  
  
  //sumsigma_x<<<numblocks,numthreads>>>(dev_R,dev_I,dev_sumdxR,dev_sumdxI,nqubits,l);
  sigma_x<<<numblocks,numthreads>>>(dev_R,dev_I,dev_sumdxR,dev_sumdxI,2,l);
  cudaCheckError("rot",0);
//   for(int i=0;i<nqubits;i++) {
//     sumsig+=itppextmath::sigma(b,i,nqubits);
//   }
  sumsig=itppextmath::sigma(b,2,nqubits);
  
  
  cudaMemcpy(R,dev_sumdxR,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(I,dev_sumdxI,l*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i=0;i<l;i++) {
    cudastate(i)=std::complex<double>(R[i],I[i]);
  }
  
  state=sumsig*state;
  //cout<<"Carlos "<<itpp::dot(itpp::conj(zerostate),state)<<endl;
  //cout<<"Eduardo "<<itpp::dot(itpp::conj(zerostate),cudastate)<<endl;
  cout<<"Sx "<<itpp::norm(cudastate-state)<<endl;
}


void comparacion_sum_sigz(int nqubits) {
  //cambiado para ver cuando se aplican dos 
  int l=pow(2,nqubits);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  
  double* R=new double[l];
  double* I=new double[l];
  
  randomstate(l,R,I);  
  double *dev_R;
  double *dev_I;
  
  
  itpp::cvec state(l),zerostate(l);
  itpp::cvec cudastate(l);
  itpp::cmat sumsig=itpp::zeros_c(l,l);
  itpp::vec b(3); b(0)=0.; b(1)=0.; b(2)=1.;
  for (int ind=0; ind<l; ind++){
    state(ind) = complex<double>(R[ind],I[ind]);
  }
  zerostate=state;
  
  cudaSafeCall(cudaMalloc((void**)&dev_R,l*sizeof(double)),"malloc",1);
  cudaSafeCall(cudaMalloc((void**)&dev_I,l*sizeof(double)),"malloc",2);
  
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  
  double *sumdxR=new double[l];
  double *sumdxI=new double[l];
  double *dev_sumdxR;      
  double *dev_sumdxI;
  cudaMalloc((void**)&dev_sumdxR,l*sizeof(double));     
  cudaMalloc((void**)&dev_sumdxI,l*sizeof(double));
  
  
  sumsigma_z<<<numblocks,numthreads>>>(dev_R,dev_I,dev_sumdxR,dev_sumdxI,nqubits,l);

  cudaCheckError("rot",0);
  for(int i=0;i<nqubits;i++) {
    sumsig+=itppextmath::sigma(b,i,nqubits);
  }
  
  
  
  cudaMemcpy(R,dev_sumdxR,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(I,dev_sumdxI,l*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i=0;i<l;i++) {
    cudastate(i)=std::complex<double>(R[i],I[i]);
  }
  
  state=sumsig*state;
  //cout<<"Carlos "<<itpp::dot(itpp::conj(zerostate),state)<<endl;
  //cout<<"Eduardo "<<itpp::dot(itpp::conj(zerostate),cudastate)<<endl;
  cout<<"Sz "<<itpp::norm(cudastate-state)<<endl;
}



void comparacion_sum_sigy(int nqubits) {
  //cambiado para ver cuando se aplican dos 
  int l=pow(2,nqubits);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  
  double* R=new double[l];
  double* I=new double[l];
  
  randomstate(l,R,I);  
  double *dev_R;
  double *dev_I;
  
  
  itpp::cvec state(l),zerostate(l);
  itpp::cvec cudastate(l);
  itpp::cmat sumsig=itpp::zeros_c(l,l);
  itpp::vec b(3); b(0)=0.; b(1)=1.; b(2)=0.;
  for (int ind=0; ind<l; ind++){
    state(ind) = complex<double>(R[ind],I[ind]);
  }
  zerostate=state;
  
  cudaSafeCall(cudaMalloc((void**)&dev_R,l*sizeof(double)),"malloc",1);
  cudaSafeCall(cudaMalloc((void**)&dev_I,l*sizeof(double)),"malloc",2);
  
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  
  double *sumdxR=new double[l];
  double *sumdxI=new double[l];
  double *dev_sumdxR;      
  double *dev_sumdxI;
  cudaMalloc((void**)&dev_sumdxR,l*sizeof(double));     
  cudaMalloc((void**)&dev_sumdxI,l*sizeof(double));
  
  
  sumsigma_y<<<numblocks,numthreads>>>(dev_R,dev_I,dev_sumdxR,dev_sumdxI,nqubits,l);
  cudaCheckError("rot",0);
  for(int i=0;i<nqubits;i++) {
    sumsig+=itppextmath::sigma(b,i,nqubits);
  }
  
  
  
  cudaMemcpy(R,dev_sumdxR,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(I,dev_sumdxI,l*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i=0;i<l;i++) {
    cudastate(i)=std::complex<double>(R[i],I[i]);
  }
  
  state=sumsig*state;
  //cout<<"Carlos "<<itpp::dot(itpp::conj(zerostate),state)<<endl;
  //cout<<"Eduardo "<<itpp::dot(itpp::conj(zerostate),cudastate)<<endl;
  cout<<"Sy "<<itpp::norm(cudastate-state)<<endl;
}
    

void comparacion_rotacion_v (int nqubits, int x) {
  int l=pow(2,nqubits);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  
  double* R=new double[l];
  double* I=new double[l];
  
  randomstate(l,R,I);  
  double *dev_R;
  double *dev_I;
  
  
  itpp::cvec state(l);
  itpp::cvec cudastate(l);
  for (int ind=0; ind<l; ind++){
    state(ind) = complex<double>(R[ind],I[ind]);
  }
  cout<<state<<endl;
  
  cudaMalloc((void**)&dev_R,l*sizeof(double));
  cudaMalloc((void**)&dev_I,l*sizeof(double));
  
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  
  double *rotR=new double[l];
  double *rotI=new double[l];
  double *dev_rotR;      
  double *dev_rotI;
  cudaMalloc((void**)&dev_rotR,l*sizeof(double));     
  cudaMalloc((void**)&dev_rotI,l*sizeof(double));
  
  
  vertical_rotation<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,x,nqubits,l,2);
  state=spinchain::apply_vertical_rotation(state,x,2);
  cout<<state<<endl;
  
  cudaMemcpy(rotR,dev_rotR,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(rotI,dev_rotI,l*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i=0;i<l;i++) {
    cudastate(i)=std::complex<double>(rotR[i],rotI[i]);
  }
  cout<<cudastate<<endl;
  
  cout<<itpp::norm(cudastate-state)<<endl;
}
    

int main() {
  cout.precision(16);
  //comparacion_rotacion_v(4,2);
  
  itpp::vec b(3); b(0)=1.; b(1)=3.; b(2)=2.;
  //double ising=1.;
  for(int i=10;i<=15;i++) {
    cout << i << " ";
  comparacion_sum_sigx(i);
  comparacion_sum_sigy(i);
  comparacion_sum_sigz(i);
    //comparacion_cadena(ising,b,1,i);
    //comparacion_ising(ising,1,i);
    //comparacion_magnetic(b,1,i);
    //cadena_carlos_tiempo(ising,b,1,i);
  }
  
  
  
  
  
  //comparacion_trace();
//   double* R=new double[l];
//   double* I=new double[l];
//   
//   exit_randomstate(l,R,I);
//   
//   double *dev_R;
//   double *dev_I;
//   
//   cudaMalloc((void**)&dev_R,l*sizeof(double));
//   cudaMalloc((void**)&dev_I,l*sizeof(double));
//   
//   cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
//   cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
//   
    // empieza rutina carlosp 
  
//    int dim=pow(2,nqubits);
//    itpp::cvec state(dim);
// //    double x,y;
// //    ifstream myReadFile;
// //    myReadFile.open("/home/eduardo/infq/estado.dat");
//    for (int i=0; i<dim; i++){
// //      myReadFile >> x >> y ;
//      state(i) = complex<double>(R[i],I[i]) ;
// //       cout << x <<", "<<y<<endl;
//    }
//   // myReadFile.close();

//    itpp::vec b(3); b(0)=1; b(1)=1; b(2)=1;
//    comparacion(0,2,b,1,dev_R,dev_I);
   //ising abajo
//    for(int num=0;num<1;num++) {
//     spinchain::apply_chain(state,1,b); 
//    }
// //    for (int i=0; i<dim; i++){
// //      cout << real(state(i)) << " " << imag(state(i)) << endl;
// //    }
//  
//  
//  //termina rutina carlosp
//   
// 
// 
//   for(int num=0;num<1;num++) {
// //     for(int i=0;i<nqubits;i++) {
// //       Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,1);
// //     }
// //   
// //     for(int i=0;i<nqubits;i++) {
// //       Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,1,1,1);
// //     }
//     Uk_kernel<<<numblocks,numthreads>>>(0,dev_R,dev_I,1,1,1);
//   }
//     
    
//   cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
//   cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
//   
//   cudaFree(dev_R);
//   cudaFree(dev_I);
  
  
//   itpp::cvec cudastate(l);
//   for(int i=0;i<l;i++) {
//     cudastate(i) = complex<double>(R[i],I[i]);
//   }
//   std::complex<double> resultado = itpp::dot(state,conj(cudastate));
//   //cout<< resultado<<endl; 
//   
//   int count_r=0;
//   int count_i=0;
//   for(int i=0;i<l;i++) {
//     cout<<abs(real(state(i))-real(cudastate(i)))<<" "<<abs(imag(state(i))-imag(cudastate(i)))<<endl;
// //     if (abs(real(state(i))-real(cudastate(i)))>=1e-7) {
// //       count_r++;
// //     }
// //     if (abs(imag(state(i))-imag(cudastate(i)))>=1e-7) {
// //       count_i++;
// //     }
//   }
// //   cout<<count_r<<endl;
// //   cout<<count_i<<endl;

}


