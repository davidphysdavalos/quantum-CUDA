# Configuration details. Adjust to your needs with 
# LOGNAME and, for example, ifeq ($(HOSTNAME),jesus) 
LDLIBS = -litpp
NVCCFLAGS= -arch=sm_13
# Carlos Pineda {{{
ifeq ($(LOGNAME),carlosp)
    INCLUDES := -I. -I../cpp/ -I../
endif
# }}}
# Eduardo Villaseñor {{{
ifeq ($(LOGNAME),eduardo)
  INCLUDES := -I. -I ../libs/cpp/
  ifeq ($(HOSTNAME),jesus) 
    INCLUDES := -I. -I ../libs/cpp/
  endif
endif
# }}}

# Kambalam {{{
ifeq ($(LOGNAME),cfpz)
  INCLUDES := -I. -I ../libs/cpp/ -I../../libs/include/ -I../../libs/tclap-1.2.1/include/ -L../../libs/lib/
  ifeq ($(HOSTNAME),jesus)
    INCLUDES := -I. -I ../libs/cpp/
  endif
endif
# }}}

# David Davalos {{{
ifeq ($(LOGNAME),david)
    INCLUDES := -I ~/cuda -I ~/libs/cpp
endif
# }}}

# Toca:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cfpz_g/cfpz/libs/lib
# module load mkl

% :: %.cu
	nvcc $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDLIBS)

test :: 
	./cuda_spinchain -o color_map2d_stdev_fast_in70 -q 16 --x 4 --ising_z 0.3 --dev 0 

mycuda_test :: mycuda_test.cu
	nvcc $(NVCCFLAGS) $< -o $@ $(LDLIBS)
diagonalize :: diagonalize.cpp
	g++ $(INCLUDES) $< -o diagonalize $(LDLIBS)
cuda_spinchain_single :: cuda_spinchain.cu
	nvcc $(INCLUDES) $< -o cuda_spinchain $(LDLIBS)
eigenvalues ::
