// {{{ includes and using
#include <iostream>
#include <tclap/CmdLine.h>
#include <itpp/base/timing.h>


// #include <RMT.cpp>
// #include <purity_RMT.cpp>
#include <cpp/itpp_ext_math.cpp>
#include <cpp/spinchain.cpp>
#include <cpp/dev_random.cpp>
using namespace std;
using namespace itpp;
using namespace itppextmath;
using namespace cfpmath;
using namespace spinchain;
// }}}
// {{{ tlacp
  TCLAP::CmdLine cmd("Command description message", ' ', "0.1");

  TCLAP::ValueArg<string> optionArg("o","option", "Option" ,false,"nichts", "string",cmd);
  TCLAP::ValueArg<int> qubits("q","qubits", "Number of qubits",false, 4,"int",cmd);
  TCLAP::ValueArg<int> position("","position", "The position of something",false, 1,"int",cmd);
  TCLAP::ValueArg<int> position2("","position2", "The position of something",false, 3,"int",cmd);
  TCLAP::ValueArg<double> Ising("","ising_z", "Ising interaction in the z-direction",false, 1.4,"double",cmd);
  TCLAP::ValueArg<double> coupling("","coupling", "Ising interaction for the coupling",false, 0.01,"double",cmd);
  TCLAP::ValueArg<double> bx("","bx", "Magnetic field in x direction",false, 1.4,"double",cmd);
  TCLAP::ValueArg<double> by("","by", "Magnetic field in y direction",false, 1.4,"double",cmd);
  TCLAP::ValueArg<double> bz("","bz", "Magnetic field in z direction",false, 1.4,"double",cmd);
  TCLAP::SwitchArg no_general_report("","no_general_report",
      "Print the general report", cmd);
  TCLAP::ValueArg<unsigned int> seed("s","seed",
      "Random seed [0 for urandom]",false, 243243,"unsigned int",cmd);
// }}}
std::complex<double> Im(0,1);

  Real_Timer timer;

int main(int argc, char* argv[]) { //{{{
// 	Random semilla_uran;
// 	itpp::RNG_reset(semilla_uran.strong());
//   	cout << PurityRMT::QubitEnvironmentHamiltonian(3,0.) <<endl;
// 	cout << RMT::FlatSpectrumGUE(5,.1) <<endl;
//
  cmd.parse( argc, argv );
  int error=0;
  cout.precision(17); cerr.precision(17);
// Set seed for random {{{ 
  unsigned int semilla=seed.getValue();
  if (semilla == 0){
    Random semilla_uran; semilla=semilla_uran.strong();
  } 
  RNG_reset(semilla);
  // }}}
// Report on the screen {{{ 
  if(!no_general_report.getValue()){
    cout << "#linea de comando: "; 
    for(int i=0;i<argc;i++){ 
      cout <<argv[i]<<" " ;
    } cout << endl ;
    cout << "#semilla = " << semilla << endl; 
    error += system("echo \\#hostname: $(hostname)");
    error += system("echo \\#comenzando en: $(date)");
    error += system("echo \\#uname -a: $(uname -a)");
    error += system("echo \\#working dir: $(pwd)");
  }
  // }}}

  string option=optionArg.getValue();
  if (option=="get_speed_chain"){ // {{{// {{{
    int dim=pow_2(qubits.getValue());
    cvec state=RandomState(dim);
    vec b(3); b=randu(3)-.5;
    double Ising=randu();
    int iterations=1;
    double time_per_iteration=0., elapsedtime=0. ;


    while (elapsedtime<1){
      timer.reset(); 
      timer.start(); 
      for (int i=0; i< iterations; i++){
        apply_chain(state,Ising,b);
      }
      timer.stop(); 
      elapsedtime = timer.get_time(); 
//       cout << "Tiempo parcial, con  "<< iterations<< " iteraciones, " << elapsedtime << endl;
      iterations=2*iterations;
    }
    iterations=iterations/2;
    time_per_iteration=elapsedtime/iterations;
    cout << "# Hecho. Tiempo, " << elapsedtime << endl;
    cout << "# Hecho. iterations=, " << iterations << endl;
    cout << "# Hecho. tiempo por iteracion, " << time_per_iteration << endl;
    cout << qubits.getValue() << " " <<  time_per_iteration << endl;
      //}}}
  } else {// {{{
    vec b(3);
    b(0)=-0.404;b(1)=0.0844;b(2)=0.361;
    cout <<   exp_minus_i_b_sigma(b) << endl;
    // b = {-0.404, 0.084, 0.361};
    // InputForm[MatrixExp[-I b.Table[Pauli[i], {i, 3}]]]
    // {{0.8534308186484069 - 0.34318420526987775*I, 
    // -0.07985449651709063 + 0.3840621022964837*I}, 
    // {0.0798544965170907 + 0.3840621022964837*I, 
    // 0.8534308186484068 + 0.3431842052698777*I}}
  }//}}}
  //}}}
  return 0;
}//}}}
