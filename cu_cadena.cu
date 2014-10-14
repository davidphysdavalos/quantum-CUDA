#include <iostream>
#include "cu_complex.cu"
#include "bin.cpp"
#include "math_functions.h"
#include "device_functions.h"

#define TOTAL_DIM 32
// 3.14159

const int nqubits=5;
int l=pow(2,nqubits);
int bin[nqubits];

float* CNOT(float L[],int x, int y)  {
  int nqu=2;
  int l2=pow(2,nqu);
  float* CL=new float[l];
  for(int i=0;i<l2;i++) {
    dec_bin(i,bin,nqu);
    bin[y]=(bin[x]+bin[y])%2;
    int n=bin_dec(bin,nqu);
    CL[n]=L[i];
  }
  return CL;
}

float* Hadamard(float L[],int x) {
  int nqu=2;
  int l2=pow(2,nqu);
  float* HL=new float[l];
  for(int i=0;i<l2;i++) {
    dec_bin(i,bin,nqu);
    if (bin[x]==0) {
      bin[x]=1;
      int n=bin_dec(bin,nqu); 
      HL[i]=HL[i]+L[i];
      HL[n]=HL[n]+L[i];
    }
    if (bin[x]==1) {
      bin[0]=0;
      int n=bin_dec(bin,nqu);
      float temp=(L[i]*(-1));
      HL[i]=HL[i]+(temp);
      HL[n]=HL[n]+L[i];
    }
  }
  for(int i=0;i<l;i++) {
    HL[i]=HL[i]*(1/sqrt(2));
  }
  return HL;
}

void randomstate(int length,float R[],float I[]) {
  float norm=0;
  srand(time(0));
  for(int i=0;i<length;i++) {
    R[i]=rand()/float(RAND_MAX);
    I[i]=rand()/float(RAND_MAX);
    norm=norm+pow(R[i],2)+pow(I[i],2);
  }
  norm=1/(sqrt(norm));
  for(int i=0;i<length;i++) {
    R[i]=R[i]*norm;
    I[i]=I[i]*norm;
  }
}

void alist(float A1[],float A2[],float B1[],float B2[],float Res1[],float Res2[],int a,int b) {
  for(int i=0;i<a;i++) {
    for(int k=0;k<b;k++) {
      Res1[(i*b)+k]=(A1[i]*B1[k])-(A2[i]*B2[k]);
      Res2[(i*b)+k]=(A1[i]*B2[k])+(A2[i]*B1[k]);
    }
  }
}

__global__ void Ui_kernel(int k,float R[],float I[],float j) {
  int index=threadIdx.x + blockIdx.x * blockDim.x;
  if (index<TOTAL_DIM) {
    Complex i(0,1);
    Complex a=Complex(R[index],I[index])*cosf(j);
    int sigz=(((index/__float2int_rz(powf(2,k)))%2)+((index/__float2int_rz(powf(2,k+1)))%2))%2;
    Complex b=Complex(R[index],I[index])*powf(-1,sigz)*sinf(j);
    b=b*i;
    R[index]=(a-b).real;
    I[index]=(a-b).imag;
  }
}

__global__ void Uk_kernel(int k,float R[],float I[],float bx,float bz,float theta) {
  int index=threadIdx.x + blockIdx.x * blockDim.x;
//   int index=blockIdx.x;
  if (((index/__float2int_rz(powf(2,k)))%2==0) && (index<TOTAL_DIM)) {
    Complex i(0,1);
    int i2=__float2int_rz(powf(2,k));
    Complex a=Complex(R[index],I[index])*cosf(theta);
    Complex b=Complex(R[index],I[index])*bz;
    Complex c=Complex(R[index+i2],I[index+i2])*bx;
    b=b+c;
    b=(b*i)*sinf(theta);
    Complex a2=Complex(R[index+i2],I[index+i2])*cosf(theta);
    Complex b2=Complex(R[index+i2],I[index+i2])*bz*-1;
    Complex c2=Complex(R[index],I[index])*bx;
    b2=b2+c2;
    b2=(b2*i)*sinf(theta);
    R[index]=(a-b).real;
    I[index]=(a-b).imag;
    R[index+i2]=(a2-b2).real;
    I[index+i2]=(a2-b2).imag;
  }
}


    
    


      

      
      
int main() {
  float R11[]={1,0,0,0};
  float I1[]={0,0,0,0};
  float* R1=Hadamard(R11,0);
  R1=CNOT(R1,0,1);
  
  float *dev_R;
  float *dev_I;
  float* R=new float[l];
  float* I=new float[l];
  
//   float R2[]={0.7071,0.7071};
//   float I2[]={0,0};
  int l2=pow(2,nqubits-2);
  float* R2=new float[l2];
  float* I2=new float[l2];
  float* Rcu=new float[l];
  float* Icu=new float[l];
//   float* I2=new float[l2];
  randomstate(l2,R2,I2);

  
  
  alist(R1,I1,R2,I2,R,I,4,l2);
  for(int i=0;i<l;i++) {
//     cout<< R[i]<<" "<<I[i]<<endl;
    cout<< "Estado inicial " << R[i]<<" i"<<I[i]<<endl;
  }
  
  cudaMalloc((void**)&dev_R,l*sizeof(float));
  cudaMalloc((void**)&dev_I,l*sizeof(float));
  
  cudaMemcpy(dev_R,R,l*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_I,I,l*sizeof(float),cudaMemcpyHostToDevice);
  
  int iteraciones=1;
  for(int num=0;num<iteraciones;num++) {
    for(int i=0;i<nqubits-1;i++) {
//       Ui_kernel<<<(l+500)/501,501>>>(i,dev_R,dev_I,1);
    }
  
    for(int i=0;i<nqubits;i++) {
      Uk_kernel<<<(l+500)/501,501>>>(i,dev_R,dev_I,0.707106,0.707106,1.414213);
    }
  }
  
//   cudaMemcpy(R,dev_R,l*sizeof(float),cudaMemcpyDeviceToHost);
//   cudaMemcpy(I,dev_I,l*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(Rcu,dev_R,l*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(Icu,dev_I,l*sizeof(float),cudaMemcpyDeviceToHost);
  
  for(int i=0;i<l;i++) {
    cout<<Rcu[i]<<" "<<Icu[i]<<endl;
  }
  
  cudaFree(dev_R);
  cudaFree(dev_I);
}



