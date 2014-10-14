// {{{ includes and using
#include <iostream>
#include <tclap/CmdLine.h>
#include <itpp/base/timing.h>


// #include <RMT.cpp>
// #include <purity_RMT.cpp>
#include <itpp_ext_math.cpp>
#include <spinchain.cpp>
#include <dev_random.cpp>
using namespace std;
using namespace itpp;
using namespace itppextmath;
using namespace cfpmath;
using namespace spinchain;
// }}}
// {{{ tlacp
  TCLAP::CmdLine cmd("Command description message", ' ', "0.1");

//   TCLAP::ValueArg<int> dimension("d","dimension", "dimension of the matrix to diagonalize",false, 4,"int",cmd);
  TCLAP::ValueArg<string> fileArg("f","file", "File that contains the matrix" ,false,"nichts", "string",cmd);
  TCLAP::ValueArg<string> saveArg("s","save", "File that will contain spectra" ,false,"nichts", "string",cmd);
  TCLAP::SwitchArg no_general_report("","no_general_report",
      "Print the general report", cmd);
  TCLAP::SwitchArg dcheck("","dcheck",
      "Print the diagonalization error", cmd);
  TCLAP::ValueArg<unsigned int> seed("","seed",
      "Random seed [0 for urandom]",false, 243243,"unsigned int",cmd);
// }}}
std::complex<double> Im(0,1);

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

//   int dim=dimension.getValue();
//   dim = system("echo \\#working dir: $(pwd)");

//   std::ifstream inFile("/tmp/umat3x4_kx1_ky2_bx05_is05_symx0.dat"); 
  string file=fileArg.getValue();
  string save=saveArg.getValue();
  std::ifstream inFile; inFile.open(file.c_str()); 
  int dim2= std::count(std::istreambuf_iterator<char>(inFile), std::istreambuf_iterator<char>(), '\n');
  inFile.close();
  int dim=cfpmath::isqrt(dim2);



//   cout << "Lineas " << dim << endl;
//   abort();
//   cin >> dim ;
  cmat X(dim,dim);
  double x,y;
  std::ifstream inFile2; inFile2.open(file.c_str()); 
  
//   std::ifstream inFile("/tmp/umat3x4_kx1_ky2_bx05_is05_symx0.dat"); 
  for (int ir=0; ir<dim; ir++){
    for (int ic=0; ic<dim; ic++){
//       cout << ir << ", " << ic << endl;
      inFile2 >> x  >> y ;
      X(ic,ir)=x+std::complex<double>(0,1)*y;
//       cout << X(ir,ic) << endl;
    }
  }

  inFile2.close();
  cvec lambda(dim);
  itpp::cmat eigenvectors(dim,dim);
  
  if(dcheck.getValue()){    
    eig(X,lambda,eigenvectors); 
  }
  else {
    eig(X,lambda);
  }
  
  
  complex<double> z;
  std::ofstream outFile;
  outFile.precision(17);
  outFile.open(save.c_str());
  
  for (int ir=0; ir<dim; ir++){
    z=lambda(ir);
    outFile  << arg(z) <<" " << norm(z)-1 << endl;
  }
  outFile.close();
  
  if(dcheck.getValue()){
    double d_error=itpp::norm(X-eigenvectors*itpp::diag(lambda)*itpp::hermitian_transpose(eigenvectors));
    cout<<"#ERROR "<<d_error<<endl;
  }
  // {{{ Final report
  if(!no_general_report.getValue()){
    error += system("echo \\#terminando:    $(date)");
  }
  // }}}

  //}}}
  return 0;
}//}}}
