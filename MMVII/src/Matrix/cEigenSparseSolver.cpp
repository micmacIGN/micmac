

#include "MMVII_EigenWrap.h"
#include "ExternalInclude/Eigen/Sparse"
#include "ExternalInclude/Eigen/Dense"

#include "ExternalInclude/Eigen/Eigenvalues"
#include "ExternalInclude/Eigen/Householder"  // HouseholderQR.h"


using namespace Eigen;

namespace MMVII
{

	/*
int m=1000000, n = 10000;
VectorXd x(n), b(m);
SparseMatrix<double> A(m,n);
// fill A and b
LeastSquaresConjugateGradient<SparseMatrix<double> > lscg;
lscg.compute(A);
x = lscg.solve(b);
std::cout << "#iterations:     " << lscg.iterations() << std::endl;
std::cout << "estimated error: " << lscg.error()      << std::endl;
// update b, and solve again
x = lscg.solve(b);
*/
template<class Type> cDenseVect<Type> EigenSolveLsqGC
                                      (
				           const std::vector<cEigenTriplet<Type> > & aV3,
                                           const std::vector<Type> & aVec,
					   int   aNbVar
				      )
{
   int aNbEq= aVec.size();

   Eigen::SparseMatrix<Type> aSpMat(aNbEq,aNbVar);
   aSpMat.setFromTriplets(aV3.begin(), aV3.end());
   LeastSquaresConjugateGradient< Eigen::SparseMatrix<Type> >  aLSCG;

   cDenseVect<Type> aRes(aNbVar);
   cNC_EigenColVectWrap<Type> aWRes(aRes);

   cConst_EigenColVectWrap  aWVec(aVec);
   Type rhsNorm2 = (aSpMat.adjoint()*aWVec.EW()).squaredNorm();
   std::cout<<"rhsNorm2 JM: "<<rhsNorm2<<std::endl;
   /* before convergence, rhsNorm2 is 9.14     => threshold 4.50939e-31
    *                then rhsNorm2 is 0.0097   => threshold 4.80211e-34
    *  after convergence, rhsNorm2 is 4.44e-08 => threshold 2.18926e-39, but minimal achived is 3.08152e-33
    *
    * fix tol to cap at 1e-4 ? threshold 4.9e-36
    * not sufficient since we reached 1.88079e-37 on previous iteration, and only 3.08152e-33 now... and after it diverges
    *
    * https://en.wikipedia.org/wiki/Conjugate_gradient_method : or the method may even start diverging
    * Limit iterations to 3? but the final success test is still false...
    */
   Type tol = std::numeric_limits<Type>::epsilon();
   if (rhsNorm2 < 1e-4)
       tol *= sqrt(1e-4/rhsNorm2);

   aLSCG.setTolerance(tol);
   aLSCG.setMaxIterations(3);
   aLSCG.compute(aSpMat);

   if (EigenDoTestSuccess())
   {
      if (aLSCG.info()!=Eigen::Success)
      {
          ON_EIGEN_NO_SUCC("LeastSquaresConjugateGradient::compute");
      }
   }


   aWRes.EW() = aLSCG.solve(aWVec.EW());



   //std::cout<<"EigenSolveLsqGC with matrix:\n"<<Eigen::Matrix<Type, Dynamic, Dynamic>(aSpMat)<<std::endl;
   //std::cout<<"and vector:\n"<<aWVec.EW().transpose()<<std::endl;

   if (EigenDoTestSuccess())
   {
      if (aLSCG.info()!=Eigen::Success)
      {
          ON_EIGEN_NO_SUCC("LeastSquaresConjugateGradient::solve");
      }
   }

   return aRes;
}



template<class Type> cDenseVect<Type> EigenSolveCholeskyarseFromV3
                                      (
				           const std::vector<cEigenTriplet<Type> > & aV3,
                                           const cDenseVect<Type> & aVec
				      )
{
   int aN= aVec.Sz();

   Eigen::SparseMatrix<Type> aSpMat(aN,aN);
   aSpMat.setFromTriplets(aV3.begin(), aV3.end());
   Eigen::SimplicialCholesky< Eigen::SparseMatrix<Type>  > aChol(aSpMat);  // performs a Cholesky factorization of A


   cConst_EigenColVectWrap  aWVec(aVec);

   cDenseVect<Type> aRes(aN);
   cNC_EigenColVectWrap<Type> aWRes(aRes);

   aWRes.EW() = aChol.solve(aWVec.EW());

   //std::cout<<"EigenSolveCholeskyarseFromV3 with matrix:\n"<<Eigen::Matrix<Type, Dynamic, Dynamic>(aSpMat)<<std::endl;
   //std::cout<<"and vector:\n"<<aWVec.EW().transpose()<<std::endl;

   if (EigenDoTestSuccess())
   {
      if (aChol.info()!=Eigen::Success)
      {
          ON_EIGEN_NO_SUCC("SimplicialCholesky::solve");
      }
   }

   return aRes;
}



#define INSTANTIATE_EIGEN_SPARSE(Type)\
template cDenseVect<Type> EigenSolveCholeskyarseFromV3(const std::vector<cEigenTriplet<Type> > &,const cDenseVect<Type> & aVec);\
template cDenseVect<Type> EigenSolveLsqGC (const std::vector<cEigenTriplet<Type> > &, const std::vector<Type> & aVec, int   aNbVar);


INSTANTIATE_EIGEN_SPARSE(tREAL4)
INSTANTIATE_EIGEN_SPARSE(tREAL8)
INSTANTIATE_EIGEN_SPARSE(tREAL16)





	/*

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n) {}
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename)  {}

int xmain(int argc, char** argv)
{
  if(argc!=2) {
    std::cerr << "Error: expected one and only one argument.\n";
    return -1;
  }

  int n = 300;  // size of the image
  int m = n*n;  // number of unknowns (=number of pixels)

  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  buildProblem(coefficients, b, n);

  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
  Eigen::VectorXd x = chol.solve(b);         // use the factorization to solve for the given right hand side

  // Export the result to a file:
  saveAsBitmap(x, n, argv[1]);

  return 0;
}
*/

#if (0)
static void x2main()
{
int m=1000000, n = 10000;
VectorXd x(n), b(m);
SparseMatrix<double> A(m,n);
// fill A and b
LeastSquaresConjugateGradient<SparseMatrix<double> > lscg;
lscg.compute(A);
x = lscg.solve(b);
std::cout << "#iterations:     " << lscg.iterations() << std::endl;
std::cout << "estimated error: " << lscg.error()      << std::endl;
// update b, and solve again
x = lscg.solve(b);
}

static void xmain()
{
MatrixXf m = MatrixXf::Random(3,2);

std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
JacobiSVD<MatrixXf > svd(m,ComputeThinU | ComputeThinV);
/*

std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
*/
Vector3f rhs(1, 0, 0);
std::cout << "Now consider this rhs vector:" << std::endl << rhs << std::endl;
std::cout << "A least-squares solution of m*x = rhs is:" << std::endl << svd.solve(rhs) << std::endl;
/*
*/
	/*
   Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
   std::cout << "Here is the matrix A:\n" << A << std::endl;
   Eigen::VectorXf b = Eigen::VectorXf::Random(3);
   std::cout << "Here is the right hand side b:\n" << b << std::endl;
   std::cout << "The least-squares solution is:\n"
        << A.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(b) << std::endl;
	*/
}
#endif


};


