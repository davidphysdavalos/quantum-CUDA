#ifndef CUDAITPP
#define CUDAITPP
#include "tools.cpp"
#include "cuda_functions.cu"
namespace itppcuda{ // {{{
	void itpp2cuda(itpp::cvec& state,double** dev_R,double** dev_I) { // {{{
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

		delete R;
		delete I;
	} // }}}
	void cuda2itpp(itpp::cvec& state,double* dev_R,double* dev_I) { // {{{
		int l=state.size();
		double *R=new double[l];
		double *I=new double[l];
		//cout<<dev_R<<" "<<dev_I<<endl;

		cudaMemcpy(R,dev_R,l*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(I,dev_I,l*sizeof(double),cudaMemcpyDeviceToHost);
		for(int i=0;i<l;i++) {
			state(i)=std::complex<double>(R[i],I[i]);
		}

		delete R;
		delete I;
	} // }}}
	void apply_floquet(itpp::cvec& state,double ising,itpp::vec b,int t=1) { // {{{
		double* dev_R;
		double* dev_I;
		int l=state.size();
		int nqubits=log(l)/log(2);
		// cout << nqubits;
		int numthreads;
		int numblocks;
		choosenumblocks(l,numthreads,numblocks);
		double icos,isin,kcos,ksin,bx,by,bz;
		set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);

		itpp2cuda(state,&dev_R,&dev_I);

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

		cuda2itpp(state,dev_R,dev_I);
		cudaFree(dev_R);
		cudaFree(dev_I);

	} // }}}
	void apply_floquet2d(itpp::cvec& state,double ising,itpp::vec b,int xlen,int t=1) { // {{{
		double* dev_R;
		double* dev_I;
		int l=state.size();
		int nqubits=log(l)/log(2);
		int numthreads;
		int numblocks;
		choosenumblocks(l,numthreads,numblocks);
		double icos,isin,kcos,ksin,bx,by,bz;
		set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);
		int i_hor,i_ver;

		itpp2cuda(state,&dev_R,&dev_I);

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

		cuda2itpp(state,dev_R,dev_I);
		cudaFree(dev_R);
		cudaFree(dev_I);

	} // }}}
	void apply_floquet_trotter1g(itpp::cvec& state,double ising,itpp::vec b,double numtrotter,int t=1) { // {{{
		double* dev_R;
		double* dev_I;
		int l=state.size();
		int nqubits=log(l)/log(2);
		// cout << nqubits;
		int numthreads;
		int numblocks;
		choosenumblocks(l,numthreads,numblocks);
		double icos,isin,kcos,ksin,bx,by,bz;
		set_parameters_trotter(2*ising,b,icos,isin,kcos,ksin,bx,by,bz,numtrotter);

		itpp2cuda(state,&dev_R,&dev_I);

		for(int n=0;n<t;n++) {
			for(int j=0;j<numtrotter;j++) {
				for(int i=0;i<nqubits;i++) {
					Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,icos,isin,l);
					//       cudaCheckError("ising",i);
				}

				for(int i=0;i<nqubits;i++) {
					Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
					//       cudaCheckError("kick",i);
				}

			}
		}
		//cout<<dev_R<<" "<<dev_I<<endl;

		cuda2itpp(state,dev_R,dev_I);
		cudaFree(dev_R);
		cudaFree(dev_I);

	} // }}}
	void apply_floquet2d_trotter1g(itpp::cvec& state,double ising,itpp::vec b,int xlen,double numtrotter,int t=1) { // {{{
		double* dev_R;
		double* dev_I;
		int l=state.size();
		int nqubits=log(l)/log(2);
		int numthreads;
		int numblocks;
		choosenumblocks(l,numthreads,numblocks);
		double icos,isin,kcos,ksin,bx,by,bz;
		set_parameters_trotter(2*ising,b,icos,isin,kcos,ksin,bx,by,bz,numtrotter);
		int i_hor,i_ver;

		itpp2cuda(state,&dev_R,&dev_I);

		for(int n=0;n<t;n++) {
			for(int j=0;j<numtrotter;j++) {
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

		cuda2itpp(state,dev_R,dev_I);
		cudaFree(dev_R);
		cudaFree(dev_I);

		} // }}}
	}
	void apply_floquet_trotter2g(itpp::cvec& state,double ising,itpp::vec b,double numtrotter,int t=1) { // {{{
		double* dev_R;
		double* dev_I;
		int l=state.size();
		int nqubits=log(l)/log(2);
		// cout << nqubits;
		int numthreads;
		int numblocks;
		choosenumblocks(l,numthreads,numblocks);
		double icos,isin,kcos,ksin,bx,by,bz;
		set_parameters_trotter(ising,b,icos,isin,kcos,ksin,bx,by,bz,numtrotter);

		itpp2cuda(state,&dev_R,&dev_I);

		for(int n=0;n<t;n++) {
			for(int j=0;j<numtrotter;j++) {
				for(int i=0;i<nqubits;i++) {
					Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,icos,isin,l);
					//       cudaCheckError("ising",i);
				}

				for(int i=0;i<nqubits;i++) {
					Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,bx,by,bz,kcos,ksin,l);
					//       cudaCheckError("kick",i);
				}
				
				for(int i=0;i<nqubits;i++) {
					Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,icos,isin,l);
					//       cudaCheckError("ising",i);
				}
			}
		}
		//cout<<dev_R<<" "<<dev_I<<endl;

		cuda2itpp(state,dev_R,dev_I);
		cudaFree(dev_R);
		cudaFree(dev_I);

	} // }}}
	void apply_floquet2d_trotter2g(itpp::cvec& state,double ising,itpp::vec b,int xlen,double numtrotter,int t=1) { // {{{
// 	  cout << "Entre en la rutina, tamano estado " << state.size() << endl;
		double* dev_R;
		double* dev_I;
		int l=state.size();
		int nqubits=log(l)/log(2);
		int numthreads;
		int numblocks;
		choosenumblocks(l,numthreads,numblocks);
		double icos,isin,kcos,ksin,bx,by,bz;
		set_parameters_trotter(ising,b,icos,isin,kcos,ksin,bx,by,bz,numtrotter);
		int i_hor,i_ver;

		itpp2cuda(state,&dev_R,&dev_I);

		for(int n=0;n<t;n++) {
			for(int j=0;j<numtrotter;j++) {
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
				
				
				for(int i=0;i<nqubits;i++) {
					i_hor=(i+1)%xlen+(i/xlen)*xlen;
					i_ver=(i+xlen)%nqubits;
					Ui_kernel<<<numblocks,numthreads>>>(i,i_hor,dev_R,dev_I,icos,isin,l);
					Ui_kernel<<<numblocks,numthreads>>>(i,i_ver,dev_R,dev_I,icos,isin,l);
					}
				
		}

		cuda2itpp(state,dev_R,dev_I);
		cudaFree(dev_R);
		cudaFree(dev_I);

		} // }}}
	}
	
} // }}}
#endif // CUDAITPP
