#ifndef EVCUDA
# define EVCUDA
# include "tools.cpp"
# include "cuda_functions.cu"
namespace evcuda{ 

void cmalloc(double** dev_R,double** dev_I,int l) {
  double *source_R;
  double *source_I;
  cudaMalloc((void**)&source_R,l*sizeof(double));
  cudaMalloc((void**)&source_I,l*sizeof(double));
  *dev_R=source_R;
  *dev_I=source_I;
  }


void itpp2cuda_malloc(itpp::cvec& state,double** dev_R,double** dev_I) { 
  int l=state.size();
  double *R=new double[l];
  double *I=new double[l];
  double *source_R;
  double *source_I;
  for(int i=0;i<l;i++) {
    R[i]=real(state(i));
    I[i]=imag(state(i));
    }
  cudaMalloc((void**)&source_R,l*sizeof(double));
  cudaMalloc((void**)&source_I,l*sizeof(double));

  //cout<<source_R<<" "<<source_I<<endl;

  cudaMemcpy(source_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(source_I,I,l*sizeof(double),cudaMemcpyHostToDevice);

  //cout<<dev_R<<" "<<dev_I<<endl;
  *dev_R=source_R;
  *dev_I=source_I;

  delete[] R;
  delete[] I;
  }
void itpp2cuda(itpp::cvec& state,double* dev_R,double* dev_I) { 
  int l=state.size();
  double *R=new double[l];
  double *I=new double[l];
  for(int i=0;i<l;i++) {
    R[i]=real(state(i));
    I[i]=imag(state(i));
    }

  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);

  delete[] R;
  delete[] I;
  }
void cuda2itpp(itpp::cvec& state,double* dev_R,double* dev_I) { 
  int l=state.size();
  double *R=new double[l];
  double *I=new double[l];
  //cout<<dev_R<<" "<<dev_I<<endl;

  cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i=0;i<l;i++) {
    state(i)=std::complex<double>(R[i],I[i]);
    }

  delete[] R;
  delete[] I;
  } 
void apply_ising2d(double *dev_R, double *dev_I, double Ising, int qubits, int qubits_x){ 
  int numthreads, numblocks, xlen=qubits_x, i_hor, i_ver;
  double icos=cos(Ising);
  double isin=sin(Ising);
  int l=pow(2,qubits);
  choosenumblocks(l,numthreads,numblocks);
  for(int i=0;i<qubits;i++) {
    i_hor=(i+1)%xlen+(i/xlen)*xlen;
    i_ver=(i+xlen)%qubits;
    Ui_kernel<<<numblocks,numthreads>>>(i,i_hor,dev_R,dev_I,icos,isin,l);
    Ui_kernel<<<numblocks,numthreads>>>(i,i_ver,dev_R,dev_I,icos,isin,l);

    }
  } 
void apply_magnetic_kick(double *dev_R, double *dev_I, itpp::vec magnetic_field, int qubits){ 
  int numthreads, numblocks;
  int l=pow(2,qubits);
  choosenumblocks(l,numthreads,numblocks);
  double theta=norm(magnetic_field);
  if(theta==0) {
    return;
    }
  double bx=magnetic_field(0)/theta, by=magnetic_field(1)/theta, bz=magnetic_field(2)/theta;
  double kcos=cos(theta);
  double ksin=sin(theta);
  for(int i=0;i<qubits;i++) {
    Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx,by,bz,kcos,ksin,l);     
    }
  return; 
  }  

void apply_floquet2d(double *dev_R, double *dev_I, itpp::vec magnetic_field, double Ising, int qubits, int qubits_x, int t=1){ 
  int numthreads, numblocks, xlen=qubits_x, i_hor, i_ver;
  int l=pow(2,qubits);
  choosenumblocks(l,numthreads,numblocks);
  double icos=cos(Ising);
  double isin=sin(Ising);
  double theta=norm(magnetic_field);
  if(theta==0) {
    return;
    }
  double bx=magnetic_field(0)/theta, by=magnetic_field(1)/theta, bz=magnetic_field(2)/theta;
  double kcos=cos(theta);
  double ksin=sin(theta);
  for(int times=0;times<t;times++) {
    for(int i=0;i<qubits;i++) {
      i_hor=(i+1)%xlen+(i/xlen)*xlen;
      i_ver=(i+xlen)%qubits;
      Ui_kernel<<<numblocks,numthreads>>>(i,i_hor,dev_R,dev_I,icos,isin,l);
      Ui_kernel<<<numblocks,numthreads>>>(i,i_ver,dev_R,dev_I,icos,isin,l);

      }

    for(int i=0;i<qubits;i++) {
      Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx,by,bz,kcos,ksin,l);     
      }
    }
  return; 
  }  

void apply_floquet2d_hermit(double *dev_R, double *dev_I, itpp::vec magnetic_field, double Ising, int qubits, int qubits_x, int t=1){ 
  int numthreads, numblocks, xlen=qubits_x, i_hor, i_ver;
  int l=pow(2,qubits);
  choosenumblocks(l,numthreads,numblocks);
  double icos=cos(Ising);
  double isin=sin(Ising);
  double theta=norm(magnetic_field);
  if(theta==0) {
    return;
    }
  double bx=magnetic_field(0)/theta, by=magnetic_field(1)/theta, bz=magnetic_field(2)/theta;
  double kcos=cos(theta);
  double ksin=sin(theta);
  for(int times=0;times<t;times++) {
    for(int i=0;i<qubits;i++) {
      i_hor=(i+1)%xlen+(i/xlen)*xlen;
      i_ver=(i+xlen)%qubits;
      Ui_kernel<<<numblocks,numthreads>>>(i,i_hor,dev_R,dev_I,icos,-1*isin,l);
      Ui_kernel<<<numblocks,numthreads>>>(i,i_ver,dev_R,dev_I,icos,-1*isin,l);

      }

    for(int i=0;i<qubits;i++) {
      Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,-1*bx,-1*by,-1*bz,kcos,ksin,l);     
      }
    }
  return; 
  } 

void apply_vertical_rotation_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double* dev_rotR,double* dev_rotI,int x) { 
  int l=state.size();
  double *R=new double[l];
  double *I=new double[l];
  double *rotR=new double[l];
  double *rotI=new double[l];  
  int nqubits=log(l)/log(2);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  for(int i=0;i<l;i++) {
    R[i]=real(state(i));
    I[i]=imag(state(i));
    }
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  vertical_rotation<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,x,nqubits,l);
  //cudaCheckError("kick",i);
  cudaMemcpy(rotR,dev_rotR,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(rotI,dev_rotI,l*sizeof(double),cudaMemcpyDeviceToHost);

  for(int i=0;i<l;i++) {
    state(i)=std::complex<double>(rotR[i],rotI[i]);
    }
  } 

void apply_horizontal_rotation_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double* dev_rotR,double* dev_rotI,int x) { 
  int l=state.size();
  double *R=new double[l];
  double *I=new double[l];
  double *rotR=new double[l];
  double *rotI=new double[l];  
  int nqubits=log(l)/log(2);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  for(int i=0;i<l;i++) {
    R[i]=real(state(i));
    I[i]=imag(state(i));
    }
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  horizontal_rotation<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,x,nqubits,l);
  //cudaCheckError("kick",i);
  cudaMemcpy(rotR,dev_rotR,l*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(rotI,dev_rotI,l*sizeof(double),cudaMemcpyDeviceToHost);

  for(int i=0;i<l;i++) {
    state(i)=std::complex<double>(rotR[i],rotI[i]);
    }
  }                                                         // }}}

void apply_floquet_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double ising,itpp::vec b,int t=1) { 
  int l=state.size();
  int nqubits=log(l)/log(2);
  // cout << nqubits;
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  double icos,isin,kcos,ksin,bx,by,bz;
  set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);


  for(int n=0;n<t;n++) {
    for(int i=0;i<nqubits;i++) {		
      Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,icos,isin,l);
      //       cudaCheckError("ising",i);
      }

    for(int i=0;i<nqubits;i++) {
      Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
      //       cudaCheckError("kick",i);
      }
    }
  //cout<<dev_R<<" "<<dev_I<<endl;
  }

void apply_floquet_hermit_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double ising,itpp::vec b,int t=1) { 
  int l=state.size();
  int nqubits=log(l)/log(2);
  // cout << nqubits;
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  double icos,isin,kcos,ksin,bx,by,bz;
  set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);


  for(int n=0;n<t;n++) {
    for(int i=0;i<nqubits;i++) {
      Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,icos,-1*isin,l);
      //       cudaCheckError("ising",i);
      }

    for(int i=0;i<nqubits;i++) {
      Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,-1*bx,-1*by,-1*bz,kcos,ksin,l);
      //       cudaCheckError("kick",i);
      }
    }
  //cout<<dev_R<<" "<<dev_I<<endl;
  }

void apply_floquet2d_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double ising,itpp::vec b,int xlen,int t=1) { 
  int l=state.size();
  int nqubits=log(l)/log(2);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  double icos,isin,kcos,ksin,bx,by,bz;
  set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);
  int i_hor,i_ver;


  for(int n=0;n<t;n++) {
    for(int i=0;i<nqubits;i++) {
      i_hor=(i+1)%xlen+(i/xlen)*xlen;
      i_ver=(i+xlen)%nqubits;
      Ui_kernel<<<numblocks,numthreads>>>(i,i_hor,dev_R,dev_I,icos,isin,l);
      Ui_kernel<<<numblocks,numthreads>>>(i,i_ver,dev_R,dev_I,icos,isin,l);
      }

    for(int i=0;i<nqubits;i++) {
      Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
      //cudaCheckError("kick",i);
      }
    }


  } 

void apply_floquet2d_hermit_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double ising,itpp::vec b,int xlen,int t=1) { 
  int l=state.size();
  int nqubits=log(l)/log(2);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  double icos,isin,kcos,ksin,bx,by,bz;
  set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);
  int i_hor,i_ver;


  for(int n=0;n<t;n++) {
    for(int i=0;i<nqubits;i++) {
      i_hor=(i+1)%xlen+(i/xlen)*xlen;
      i_ver=(i+xlen)%nqubits;
      Ui_kernel<<<numblocks,numthreads>>>(i,i_hor,dev_R,dev_I,icos,-1*isin,l);
      Ui_kernel<<<numblocks,numthreads>>>(i,i_ver,dev_R,dev_I,icos,-1*isin,l);
      }

    for(int i=0;i<nqubits;i++) {
      Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,-1*bx,-1*by,-1*bz,kcos,ksin,l);
      //cudaCheckError("kick",i);
      }
    }
  }

void apply_sumdx_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double* dev_sumdxR,double* dev_sumdxI) {
  int l=state.size();
  int nqubits=log(l)/log(2);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks); 
  sumsigma_x<<<numblocks,numthreads>>>(dev_R,dev_I,dev_sumdxR,dev_sumdxI,nqubits,l);
  }

void proyector_horizontal_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double* dev_rotR,double* dev_rotI,int x, int k) { 
  int l=state.size();
  double *R=new double[l];
  double *I=new double[l];
  double *rotR=new double[l];
  double *rotI=new double[l];  
  int nqubits=log(l)/log(2);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  for(int i=0;i<l;i++) {
    R[i]=real(state(i));
    I[i]=imag(state(i));
    }
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  for(int i=1;i<x;i++) {
    horizontal_rotation<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,x,nqubits,l,i);
    //cudaCheckError("kick",i);
    cudaMemcpy(rotR,dev_rotR,l*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(rotI,dev_rotI,l*sizeof(double),cudaMemcpyDeviceToHost);
    for(int j=0;j<l;j++) {
      R[j]=R[j]+cos((2*itpp::pi*k*i)/x)*rotR[j]-sin((2*itpp::pi*k*i)/x)*rotI[j];
      I[j]=I[j]+sin((2*itpp::pi*k*i)/x)*rotR[j]+cos((2*itpp::pi*k*i)/x)*rotI[j];
      }
    }


  for(int i=0;i<l;i++) {
    state(i)=std::complex<double>(R[i]/x,I[i]/x);
    }
  }                                                         // }}}
void proyector_vertical_itpp(itpp::cvec& state,double* dev_R,double* dev_I,double* dev_rotR,double* dev_rotI,int x, int k) { 
  int l=state.size();
  double *R=new double[l];
  double *I=new double[l];
  double *rotR=new double[l];
  double *rotI=new double[l];  
  int nqubits=log(l)/log(2);
  int y=nqubits/x;
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  for(int i=0;i<l;i++) {
    R[i]=real(state(i));
    I[i]=imag(state(i));
    }
  cudaMemcpy(dev_R,R,l*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(double),cudaMemcpyHostToDevice);
  for(int i=1;i<y;i++) {
    vertical_rotation<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,x,nqubits,l,i);
    //cudaCheckError("kick",i);
    cudaMemcpy(rotR,dev_rotR,l*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(rotI,dev_rotI,l*sizeof(double),cudaMemcpyDeviceToHost);
    for(int j=0;j<l;j++) {
      R[j]=R[j]+cos(2*(itpp::pi*k*i)/y)*rotR[j]-sin(2*(itpp::pi*k*i)/y)*rotI[j];
      I[j]=I[j]+sin(2*(itpp::pi*k*i)/y)*rotR[j]+cos(2*(itpp::pi*k*i)/y)*rotI[j];
      }
    }


  for(int i=0;i<l;i++) {
    state(i)=std::complex<double>(R[i]/y,I[i]/y);
    }
  }                                                         // }}}

//corregir los argumentos y calculos extra aqui
itpp::cmat invariant_vectors(int nqubits,int x,int kx,int ky,int symx=0) {
  double *dev_R,*dev_I,*dev_rotR,*dev_rotI;
  int l=pow(2,nqubits);
  cmalloc(&dev_R,&dev_I,l);
  cmalloc(&dev_rotR,&dev_rotI,l);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  int *St=new int[l];
  //      int y=nqubits/x;
  itpp::cvec vector(l);
  int *S=new int[l];
  for(int i=0;i<l;i++) {
    S[i]=2;
    }
  if(symx!=0) {
    find_states_take2_total(S,nqubits,x,kx,ky,l,symx);
    }
  else {
    find_states_take2_total(S,nqubits,x,kx,ky,l);
    }
  int cont=0,rcont=0;

  for(int i=0;i<l;i++) {
    cont+=S[i];
    }
  itpp::cmat eigenvectors(cont,l);
  for(int vec=0;vec<cont;vec++) {
    int flag=0;
    vector=0.;
    for(int i=0;i<l;i++) {
      if(S[i]==1 && flag==0) {  
	vector(i)=1.;
	flag=1;
	S[i]=0;
	St[vec]=i;
	// 	      if(vec==34 || vec==53) {
	// 		cout<<i<<" "<<k_x<<" "<<k_y<<endl;
	// 	      }
	}
      }
    //evcuda::itpp2cuda(vector,dev_R,dev_I);
    itpp2cuda(vector,dev_R,dev_I);
    horizontal_proyector<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,x,nqubits,l,kx);	  	 
    vertical_proyector<<<numblocks,numthreads>>>(dev_rotR,dev_rotI,dev_R,dev_I,x,nqubits,l,ky);
    cuda2itpp(vector,dev_R,dev_I);



    if(itpp::norm(vector)>1e-13) {
      eigenvectors.set_row(rcont,vector/itpp::norm(vector));
      rcont++;
      }
    else {
      St[vec]=-1;
      }
    }
  int track=0;
  for(int i=0;i<cont;i++) {
    if(St[i]>-1) {
      S[track]=St[i];
      track++;
      }
    }
  delete[] St;
  cudaFree(dev_R);
  cudaFree(dev_I);
  cudaFree(dev_rotR);
  cudaFree(dev_rotI);
  return eigenvectors.get_rows(0,rcont-1);

  }

itpp::cmat invariant_vectors_chain(int nqubits,int kx) {
  double *dev_R,*dev_I,*dev_rotR,*dev_rotI;
  int l=pow(2,nqubits),x=nqubits;
  cmalloc(&dev_R,&dev_I,l);
  cmalloc(&dev_rotR,&dev_rotI,l);
  int numthreads;
  int numblocks;
  choosenumblocks(l,numthreads,numblocks);
  //      int y=nqubits/x;
  itpp::cvec vector(l);
  int *S=new int[l];
  for(int i=0;i<l;i++) {
    S[i]=2;
    }
  find_states_total_horizontal(S,nqubits,kx,l);

  int cont=0,rcont=0;

  for(int i=0;i<l;i++) {
    cont+=S[i];
    }
  itpp::cmat eigenvectors(cont,l);
  for(int vec=0;vec<cont;vec++) {
    int flag=0;
    vector=0.;
    for(int i=0;i<l;i++) {
      if(S[i]==1 && flag==0) {  
	vector(i)=1.;
	flag=1;
	S[i]=0;
	// 	      if(vec==34 || vec==53) {
	// 		cout<<i<<" "<<k_x<<" "<<k_y<<endl;
	// 	      }
	}
      }
    //evcuda::itpp2cuda(vector,dev_R,dev_I);
    itpp2cuda(vector,dev_R,dev_I);
    horizontal_proyector<<<numblocks,numthreads>>>(dev_R,dev_I,dev_rotR,dev_rotI,x,nqubits,l,kx);	  	 
    cuda2itpp(vector,dev_rotR,dev_rotI);



    if(itpp::norm(vector)>1e-13) {
      eigenvectors.set_row(rcont,vector/itpp::norm(vector));
      rcont++;
      }
    }


  delete[] S;
  cudaFree(dev_R);
  cudaFree(dev_I);
  cudaFree(dev_rotR);
  cudaFree(dev_rotI);   
  return eigenvectors.get_rows(0,rcont-1);
  }

itpp::cmat umat_block(int nqubits,int x,int kx,int ky,int symx,itpp::vec b,double ising) {
  int nqu=nqubits;
  int xlen=x;
  int numthreads,numblocks,i_ver,i_hor;
  int l=pow(2,nqubits);
  choosenumblocks(l,numthreads,numblocks);
  double *dev_R,*dev_I;
  evcuda::cmalloc(&dev_R,&dev_I,l);
  int *S=new int[l];
  for(int m=0;m<l;m++) {
    S[m]=2;
    }	

  if(symx!=0) {
    find_states_take2_total(S,nqu,xlen,kx,ky,l,symx);
    }
  else {
    find_states_take2_total(S,nqu,xlen,kx,ky,l);
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


  double icos,isin,kcos,ksin,bx,by,bz;
  set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);
  double rotation=itpp::pi/4;
  double rcos=cos(rotation),rsin=sin(rotation);

  for(int i=0;i<rcont;i++) {
    to_zero<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
    special_both_proyector<<<1,1>>>(dev_R,dev_I,xlen,nqu,rcont,kx,ky,S[i]);
    times_norm<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
//cudaCheckError("kicki",i);

// 	 cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
// 	 cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
// 	 
// 	 for(int m=0;m<l;m++) {
// 	    state(m)=std::complex<double>(R[m],I[m]);
// 	}
// 	cout<<state<<"norm:"<<itpp::norm(state)<<" sum:"<<itpp::sum(state)<<endl;


/*for(int q=0;q<nqu;q++) {
    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,-1.,0.,rcos,rsin,l);
    //cudaCheckError("kick",i);
    }*/  

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

    // 	  for(int q=0;q<nqu;q++) {
    // 	    Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,0.,1.,0.,rcos,rsin,l);
    // 	    //cudaCheckError("kick",i);
    // 	  } 

    to_zero<<<numblocks,numthreads>>>(dev_dotR,dev_dotI,rcont);

    proyected_dot<<<numblocks,numthreads>>>(dev_R,dev_I,dev_dotR,dev_dotI,xlen,nqu,rcont,kx,ky,dev_S);	  
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
  delete[] S;
  cudaFree(dev_R);
  cudaFree(dev_I);
  cudaFree(dev_dotR);
  cudaFree(dev_dotI);
  return umat;
  } 
  
itpp::cmat umat_block_chain(int nqubits,int kx,itpp::vec b,double ising) {
  int nqu=nqubits;
  int numthreads,numblocks;
  int l=pow(2,nqubits);
  choosenumblocks(l,numthreads,numblocks);
  double *dev_R,*dev_I;
  evcuda::cmalloc(&dev_R,&dev_I,l);
  int *S=new int[l];
  for(int m=0;m<l;m++) {
    S[m]=2;
    }	


    find_states_total_horizontal(S,nqu,kx,l);
    
    
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


  double icos,isin,kcos,ksin,bx,by,bz;
  set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);


  for(int i=0;i<rcont;i++) {
    to_zero<<<numblocks,numthreads>>>(dev_R,dev_I,l); 
    special_chain_proyector<<<1,1>>>(dev_R,dev_I,nqu,rcont,kx,S[i]);
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
      Ui_kernel<<<numblocks,numthreads>>>(q,(q+1)%nqubits,dev_R,dev_I,icos,isin,l);
      }

    for(int q=0;q<nqu;q++) {
      Uk_kernel<<<numblocks,numthreads>>>(q,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
      //cudaCheckError("kick",i);
      }


    to_zero<<<numblocks,numthreads>>>(dev_dotR,dev_dotI,rcont);

    proyected_dot_chain<<<numblocks,numthreads>>>(dev_R,dev_I,dev_dotR,dev_dotI,nqu,rcont,kx,dev_S);	  
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
  delete[] S;
  cudaFree(dev_R);
  cudaFree(dev_I);
  cudaFree(dev_dotR);
  cudaFree(dev_dotI);
  return umat;
  }  
  
} 
#endif                                                    // EVCUDA
