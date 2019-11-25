#include "include/MMVII_all.h"
/** \file TestEigen.cpp
    \brief File to test eigen library



class  	Eigen::SimplicialLDLT< _MatrixType, _UpLo, _Ordering >
 	A direct sparse LDLT Cholesky factorizations without square root. More...
 
class  	Eigen::SimplicialLLT< _Matri
*/
#include "ExternalInclude/Eigen/Dense"
#include "ExternalInclude/Eigen/Core"
#include "ExternalInclude/Eigen/Eigenvalues" 
#include "ExternalInclude/Eigen/SparseCholesky"

// #include "External/eigen-git-mirror-master/unsupported/Eigen/src/SparseExtra/MarketIO.h"
#include "ExternalInclude/MarketIO.h"
#include "ExternalInclude/Eigen/LU"


using Eigen::MatrixXd;
using Eigen::Map;
using namespace Eigen;

namespace MMVII
{

/// MMVII Appli for Testing boost serialization service
/**
     Probably obsolete
*/

class cAppli_MMVII_TestEigen : public cMMVII_Appli
{
     public :
        cAppli_MMVII_TestEigen(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) ;
        int Exe() override ;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override {return anArgObl;}
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;// override {return anArgOpt;}

        void T1(); ///< Un test de la doc
        void TestRawData(); ///< Test sur l'import des raw data
        void TCho();
        void BenchCho();
        void TestInv();
    private :

        bool mTestRawD;
        int  mNbCho;
        std::string mNameMatMark;
};

cCollecSpecArg2007 & cAppli_MMVII_TestEigen::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
   return anArgOpt
          <<   AOpt2007(mNbCho,"NbCho","Size Mat in choleski",{})
          <<   AOpt2007(mNameMatMark,"NMM","Name of Matrix Market for Bench",{})
          <<   AOpt2007(mTestRawD,"TIRD","Test Import Raw Data",{})
   ;

}


cAppli_MMVII_TestEigen::cAppli_MMVII_TestEigen (const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli  (aVArgs,aSpec),
  mTestRawD     (false),
  mNbCho        (-1)
{
}

int cAppli_MMVII_TestEigen::Exe()
{
  if (mTestRawD)
  {
     TestRawData(); 
  }
  while (mNbCho>0)
  {
     TCho();
     getchar();
  }
  if (IsInit(&mNameMatMark))
    BenchCho();

  return EXIT_SUCCESS;
}

void cAppli_MMVII_TestEigen::T1()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  StdOut()  << m << "\n";;

}

void cAppli_MMVII_TestEigen::TestInv()
{
   // for (int aK=0 ; aK 
}



void cAppli_MMVII_TestEigen::TestRawData()
{
    cPt2di aSz(3,2);

    cIm2D<double> aIm(aSz);
    cDataIm2D<double> & aDIM=aIm.DIm();

    for (const auto & aP : aDIM)
        aDIM.SetV(aP,10*aP.x() + aP.y());

    {
       StdOut() << "RowMajor SzY SzX \n";
       Map<Matrix<double,Dynamic,Dynamic,RowMajor> > aMap(aDIM.RawDataLin(),aSz.y(),aSz.x());
       StdOut() << aMap << "\n";
    }
    {
       StdOut() << "ColumnMajor SzX SzY \n";
       Map<Matrix<double,Dynamic,Dynamic> > aMap(aDIM.RawDataLin(),aSz.x(),aSz.y());
       StdOut() << aMap << "\n";
    }
}

void cAppli_MMVII_TestEigen::BenchCho()
{
    SparseMatrix<double> aS2;
    loadMarket(aS2,mNameMatMark);

    StdOut() << " NB " << aS2.rows() << " " << aS2.cols() << "\n";
    double aT1 = SecFromT0();
    SimplicialLDLT<SparseMatrix<double> > aDLT(aS2);
    double aT2 = SecFromT0();
    StdOut() << "DONE " <<  aT2 - aT1 << "\n";
}

void cAppli_MMVII_TestEigen::TCho()
{
    int aSzBrd = 3;

    Matrix<double,Dynamic,Dynamic>  m(mNbCho,mNbCho);
    VectorXd aSol(mNbCho);

    for (int aX=0 ; aX<mNbCho ; aX++)
    {
        aSol(aX) =   RandUnif_0_1();
        for (int aY=aX ; aY<mNbCho ; aY++)
        {
            if (aX==aY)
            {
                m(aX,aY) = 2* mNbCho + RandUnif_0_1();
            }
            else if ((std::abs(aX-aY) <aSzBrd) && ( RandUnif_0_1() < 0.5))
            {
                m(aX,aY) = m(aY,aX) = RandUnif_0_1();
            }
            else
            {
                m(aX,aY) = m(aY,aX) = 0.0;
            }
        }
    }
    VectorXd aB = m * aSol;
    StdOut() << m << "\n";

    SelfAdjointEigenSolver<MatrixXd> aSAES(m);
    StdOut() << "The eigenvalues of A are: " << aSAES.eigenvalues().transpose() << "\n";
    StdOut() << aSAES.eigenvectors() << "\n";
    double aDetEV =  aSAES.eigenvalues().prod();

    std::vector<Triplet<double> > aVT; 
    for (int aX=0 ; aX<mNbCho ; aX++)
    {
        // for (int aY=aX ; aY<mNbCho ; aY++)
        // for (int aY=0 ; aY<mNbCho ; aY++)
        for (int aY=0 ; aY<=aX ; aY++)  // !!!  => les algo sur les sparse matrix doivent faire des supoistion triang sup ou inf
        {
            if (m(aX,aY) != 0.0)
               aVT.push_back(Triplet<double>(aX,aY,m(aX,aY)));
        }
    }
    SparseMatrix<double> aSM(mNbCho,mNbCho);
    aSM.setFromTriplets(aVT.begin(), aVT.end());


    SimplicialLDLT<SparseMatrix<double> > aDLT(aSM);
    // aDLT.compute(m);
    StdOut() << "Det= " << aDLT.determinant()  << " " << aDetEV << "\n";

    VectorXd aSolCho = aDLT.solve(aB);

    StdOut() << "Check Sol " << (aSolCho - aSol).norm() << "\n";
}
    





tMMVII_UnikPApli Alloc_MMVII_TestEigen(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_TestEigen(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpec_TestEigen
(
     "TestEigen",
      Alloc_MMVII_TestEigen,
      "This command execute some experiments eigen (matrix manipulation) library",
      {eApF::Test},
      {eApDT::None},
      {eApDT::Console},
      __FILE__
);

/*
*/




};
