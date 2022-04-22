#include "include/MMVII_all.h"
// #include "include/MMVII_Tpl_Images.h"

#include "MMVII_EigenWrap.h"
#include "ExternalInclude/Eigen/Sparse"


namespace MMVII
{

template<class Type> cDenseVect<Type> EigenSolveCholeskyarseFromV3
                                      (
				           const std::vector<cEigenTriplet<Type> > & aV3,
                                           const cDenseVect<Type> & aVec
				      )
{
   int aN= aVec.Sz();

   Eigen::SparseMatrix<Type,RowMajor> aSpMat(aN,aN);
   aSpMat.setFromTriplets(aV3.begin(), aV3.end());
   Eigen::SimplicialCholesky< Eigen::SparseMatrix<Type>  > aChol(aSpMat);  // performs a Cholesky factorization of A


   cConst_EigenColVectWrap  aWVec(aVec);

   cDenseVect<Type> aRes(aN);
   cNC_EigenColVectWrap<Type> aWRes(aRes);
   aWRes.EW() = aChol.solve(aWVec.EW());

   return aRes;
}



#define INSTANTIATE_EIGEN_SPARSE(Type)\
template cDenseVect<Type> EigenSolveCholeskyarseFromV3(const std::vector<cEigenTriplet<Type> > &,const cDenseVect<Type> & aVec);


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

};

/*
*/

