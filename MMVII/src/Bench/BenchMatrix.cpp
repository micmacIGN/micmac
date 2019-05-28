#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"



namespace MMVII
{


/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

static double FTestMatr(const cPt2di & aP)
{
    return 1 + 1/(aP.x()+2.45) + 1/(aP.y()*aP.y() + 3.14);
}

static double FTestVect(const int & aK)
{
    return  (aK+3.0) / (aK+17.899 + 1e-2 * aK * aK);
}


template <class TypeMatr,class TypeVect>  void TplBenchMatrix(int aSzX,int aSzY,int aSzMil)
{
    TypeMatr aM(aSzX,aSzY);
    TypeMatr aMt(aSzY,aSzX);
    for (const auto & aP : aM)
    {
        aM.V_SetElem(aP.x(),aP.y(),FTestMatr(aP));
        aMt.V_SetElem(aP.y(),aP.x(),FTestMatr(aP));
    }
    // Bench Col mult
    {
        cDenseVect<TypeVect> aVIn(aSzX),aVOut(aSzY),aVOut2(aSzY);
        for (int aX=0 ; aX<aSzX ; aX++)
        {
            aVIn(aX) = FTestVect(aX);
        }
        aM.MulColInPlace(aVOut,aVIn);

        // Verifie mul col avec alloc
        cDenseVect<TypeVect>  aVOut3 = aM.MulCol(aVIn);
        MMVII_INTERNAL_ASSERT_bench(aVOut.L2Dist(aVOut3)<1e-5,"Bench Matrixes");
        aVOut3 = aM*aVIn;
        MMVII_INTERNAL_ASSERT_bench(aVOut.L2Dist(aVOut3)<1e-5,"Bench Matrixes");
/*
*/
        // Verifier que MulLine est equiv a MulCol de la transposee
        aMt.MulLineInPlace(aVOut2,aVIn);

        MMVII_INTERNAL_ASSERT_bench(aVOut.L2Dist(aVOut2)<1e-5,"Bench Matrixes");

        for (int aY=0 ; aY<aSzY ; aY++)
        {
             double aV1 =  aVOut(aY);
             double aV2 =  aM.MulColElem(aY,aVIn);
             double aV3 = 0 ;
             for (int aX=0 ; aX<aSzX ; aX++)
                 aV3 +=  FTestVect(aX) * FTestMatr(cPt2di(aX,aY));
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Bench Matrixes");
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV2-aV3)<1e-5,"Bench Matrixes");
        }
    }
    // Bench Line mult
    {
        cDenseVect<TypeVect> aVIn(aSzY),aVOut(aSzX);
        for (int aY=0 ; aY<aSzY ; aY++)
        {
            aVIn(aY) = FTestVect(aY);
        }
        aM.MulLineInPlace(aVOut,aVIn);
        cDenseVect<TypeVect>  aVOut3 = aM.MulLine(aVIn);
        MMVII_INTERNAL_ASSERT_bench(aVOut.L2Dist(aVOut3)<1e-5,"Bench line Matrixes");
        aVOut3 = aVIn * aM;
        MMVII_INTERNAL_ASSERT_bench(aVOut.L2Dist(aVOut3)<1e-5,"Bench line Matrixes");

        for (int aX=0 ; aX<aSzX ; aX++)
        {
             double aV1 =  aVOut(aX);
             double aV2 =  aM.MulLineElem(aX,aVIn);
             double aV3 = 0 ;
             for (int aY=0 ; aY<aSzY ; aY++)
                 aV3 +=  FTestVect(aY) * FTestMatr(cPt2di(aX,aY));
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Bench Matrixes");
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV2-aV3)<1e-5,"Bench Matrixes");
        }
    }

    // Bench read/write col/line
    for (int aK=0 ; aK<std::max(aSzX,aSzY) ; aK++)
    {
        cDenseVect<TypeVect> aVLine1(aSzX),aVCol1(aSzY),aVLine2(aSzX),aVCol2(aSzY);
        aVLine1.DIm().InitRandom();
        aVCol1.DIm().InitRandom();

        if (aK<aSzX)
        {
            aM.WriteCol(aK,aVCol1);
            aM.ReadColInPlace(aK,aVCol2);
            MMVII_INTERNAL_ASSERT_bench(aVCol1.L2Dist(aVCol2) <1e-5,"Bench line Matrixes");
        }

        if (aK<aSzY)
        {
            aM.WriteLine(aK,aVLine1);
            aM.ReadLineInPlace(aK,aVLine2);
            MMVII_INTERNAL_ASSERT_bench(aVLine1.L2Dist(aVLine2) <1e-5,"Bench line Matrixes");
        }
        
    }
    // Test mul mat
    {
        TypeMatr aRes(aSzX,aSzY); aRes.DIm().InitRandom();
        TypeMatr aM1(aSzMil,aSzY);  aM1.DIm().InitRandom();
        TypeMatr aM2(aSzX,aSzMil);  aM2.DIm().InitRandom();
 
        aRes.MatMulInPlace(aM1,aM2);
        TypeMatr aRes2 = aM1 * aM2;
        double aD = aRes2.DIm().L2Dist(aRes.DIm());
        MMVII_INTERNAL_ASSERT_bench(aD<1e-5,"Bench Mat*Mat");

        {
           cDenseVect<TypeVect> aVCol(aSzX); aVCol.DIm().InitRandom();

           cDenseVect<TypeVect> aVLine1 = aRes.MulCol(aVCol);
           MMVII_INTERNAL_ASSERT_bench(aSzY==aVLine1.Sz(),"Bench line Matrixes");
           cDenseVect<TypeVect> aVLine2 = aM1.MulCol(aM2.MulCol(aVCol));
           MMVII_INTERNAL_ASSERT_bench(aVLine1.L2Dist(aVLine2)<1e-5,"Bench mul  Matrixes");
        }

        {
           cDenseVect<TypeVect> aVLine(aSzY); aVLine.DIm().InitRandom();

           cDenseVect<TypeVect> aVCol1 = aRes.MulLine(aVLine);
           MMVII_INTERNAL_ASSERT_bench(aSzX==aVCol1.Sz(),"Bench line Matrixes");
           cDenseVect<TypeVect> aVCol2 = aM2.MulLine(aM1.MulLine(aVLine));
           MMVII_INTERNAL_ASSERT_bench(aVCol2.L2Dist(aVCol1)<1e-5,"Bench mul  Matrixes");
        }
    }
    // Test AddtAB
    {
        TypeMatr aMxy(aSzX,aSzY,eModeInitImage::eMIA_Null);
        TypeMatr aMyySym(aSzY,aSzY,eModeInitImage::eMIA_Null);
        TypeMatr aMyyUp(aSzY,aSzY,eModeInitImage::eMIA_Null);
        
        cDenseVect<TypeVect> aLine(aSzX),aLine2(aSzX),aCol(aSzY),aCol3(aSzY);
        for (int aK=0 ; aK<aSzX ; aK++)
        {
            aLine(aK) = FTestVect(aK);
            aLine2(aK) = 2*FTestVect(aK);
        }
        for (int aK=0 ; aK<aSzY ; aK++)
        {
            aCol(aK) = -FTestVect(aK);
            aCol3(aK) = 3*FTestVect(aK);
        }
        aMxy.Add_tAB(aCol,aLine);
        aMxy.Add_tAB(aCol3,aLine2);
        for (const auto & aP : aM)
        {
             double aV1 =  aMxy.V_GetElem(aP.x(),aP.y());
             double aV2 = 5 * FTestVect(aP.x()) * FTestVect(aP.y());
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Bench add tAA");
        }

        aMyySym.Add_tAA(aCol,false);
        aMyySym.Add_tAA(aCol3,false);
        for (const auto & aP : aMyySym)
        {
             double aV1 =  aMyySym.V_GetElem(aP.x(),aP.y());
             double aV2 = 10 * FTestVect(aP.x()) * FTestVect(aP.y());
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Bench add tAA");
        }
       
        aMyyUp.Add_tAA(aCol);
        aMyyUp.Add_tAA(aCol3);
        MMVII_INTERNAL_ASSERT_bench(aMyyUp.TriangSupicity()<1e-5,"Bench add tAA");

        aMyyUp.SelfSymetrizeBottom();
        MMVII_INTERNAL_ASSERT_bench(aMyySym.DIm().L2Dist(aMyyUp.DIm())<1e-5,"Bench add tAA");

        // Test Sub then Add come back to initial values
        aMyySym.Add_tAA(aCol,false);
        aMyySym.Sub_tAA(aCol,false);
        MMVII_INTERNAL_ASSERT_bench(aMyySym.DIm().L2Dist(aMyyUp.DIm())<1e-5,"Bench add tAA");
    }
}

template <class Type>  void TplBenchDenseMatr(int aSzX,int aSzY)
{
     static int aCpt=0; aCpt++;

    // Operators 
    {
        cDenseMatrix<Type> aM1(aSzX,aSzY,eModeInitImage::eMIA_Rand);
        cDenseMatrix<Type> aM2(aSzX,aSzY,eModeInitImage::eMIA_Rand);
        Type aCste = RandUnif_0_1();


        // Test QR decomp
        {
             cDenseMatrix<Type> aM1Bis = aM1.Dup();
             cResulQR_Decomp<Type>  aDec = aM1Bis.QR_Decomposition();

             // Check do not modify
             MMVII_INTERNAL_ASSERT_bench(aM1.DIm().L2Dist(aM1Bis.DIm())<1e-5,"Bench QR  Matrixes");
             // Check Q is Unitary
             MMVII_INTERNAL_ASSERT_bench(aDec.Q_Matrix().Unitarity()<1e-5,"Bench QR  Matrixes");
             // Check R is triangular
             MMVII_INTERNAL_ASSERT_bench( aDec.R_Matrix().TriangSupicity() <1e-5,"Bench QR  Matrixes");
             // Check Q*R == M
             MMVII_INTERNAL_ASSERT_bench(aM1.DIm().L2Dist(aDec.OriMatr().DIm())<1e-5,"Bench QR  Matrixes");
        }
        
        {  //  Test sur transp transp = Id
           double aDtt = aM1.DIm().L2Dist(aM1.Transpose().Transpose().DIm());
           MMVII_INTERNAL_ASSERT_bench(aDtt<1e-5,"Bench diff  Matrixes");
        }
        cDenseMatrix<Type> aM1T = aM1.Transpose();
 
        // Test operator on matrixes

        cDenseMatrix<Type> aMDif = aM1-aM2;
        cDenseMatrix<Type> aMSom = aM1+aM2;
        cDenseMatrix<Type> aMMulG = aM1 * aCste;
        cDenseMatrix<Type> aMMulD = aCste * aM1;

        for (const auto & aP : aMDif.DIm())
        {
             Type aV1 = aM1.GetElem(aP);
             Type aV2 = aM2.GetElem(aP);
             Type aVt3 = aM1T.GetElem(cPt2di(aP.y(),aP.x()));
             MMVII_INTERNAL_ASSERT_bench(std::abs(aVt3-aV1)<1e-5,"Bench transp  Matrixes");

             Type aTestDif = aV1-aV2 - aMDif.GetElem(aP);
             MMVII_INTERNAL_ASSERT_bench(std::abs(aTestDif)<1e-5,"Bench diff  Matrixes");

             Type aTestSom = aV1+aV2 - aMSom.GetElem(aP);
             MMVII_INTERNAL_ASSERT_bench(std::abs(aTestSom)<1e-5,"Bench som  Matrixes");

             Type aTestMulG = aV1*aCste - aMMulG.GetElem(aP);
             Type aTestMulD = aV1*aCste - aMMulD.GetElem(aP);
             MMVII_INTERNAL_ASSERT_bench(std::abs(aTestMulG)<1e-5,"Bench mul  Matrixes");
             MMVII_INTERNAL_ASSERT_bench(std::abs(aTestMulD)<1e-5,"Bench mul  Matrixes");
        }
    }

    //  Test symetization + self adjoint eigen
    {
        int aNb = aSzX;
        cDenseMatrix<Type> aM(aNb,eModeInitImage::eMIA_Rand);
         
        {
        // As they are both distance to complementary orthog space, their sum must equal the norm
           double aV = std::hypot(aM.Symetricity(),aM.AntiSymetricity()) ;
           MMVII_INTERNAL_ASSERT_bench(std::abs(aM.DIm().L2Norm()/aV-1)<1e-5,"Bench mul  Matrixes");

           cDenseMatrix<Type> aSim = aM.Symetrize(); 
           cDenseMatrix<Type> aASim = aM.AntiSymetrize(); 

           MMVII_INTERNAL_ASSERT_bench(aSim.Symetricity()<1e-5,"Bench mul  Matrixes");
           MMVII_INTERNAL_ASSERT_bench(aASim.AntiSymetricity()<1e-5,"Bench mul  Matrixes");
           
           cDenseMatrix<Type> aM2 = aSim+aASim;
           MMVII_INTERNAL_ASSERT_bench(aM.DIm().L2Dist(aM2.DIm()) <1e-5,"Bench mul  Matrixes");
           
           //   Bench sur Eigene value
           cResulSymEigenValue<Type> aRSEV = aSim.SymEigenValue();

           // Test it's really orthognal
           MMVII_INTERNAL_ASSERT_bench(aRSEV.EigenVectors().Unitarity() <1e-5,"Bench unitarity EigenValue");
               // Test Eigen value are given in growing order
           const cDenseVect<Type>   & aEVals = aRSEV.EigenValues();
           for (int aK=1 ; aK<aNb ; aK++)
           {
               MMVII_INTERNAL_ASSERT_bench(aEVals(aK-1)<=aEVals(aK),"Bench unitarity EigenValue");
           }
           cDenseMatrix<Type> aCheckEV =  aRSEV.OriMatr();
           MMVII_INTERNAL_ASSERT_bench(aCheckEV.DIm().L2Dist(aSim.DIm())<1e-5,"Bench unitarity EigenValue");

        }
    

        cDenseMatrix<Type> aId(aNb,eModeInitImage::eMIA_MatrixId);
        cDenseMatrix<Type> aMInv = aM.Inverse();
        cDenseMatrix<Type> aCheck = aM*aMInv;

        double aD = aId.DIm().L2Dist(aCheck.DIm());
        double aDTest = 1e-4 * pow(10,-(4-sizeof(Type))/2.0) * pow(aNb,1.0) ;  // Accuracy expectbale vary with tREAL4, tREAL8 ...

  
        if (aD>aDTest)
        {
            std::cout << "D=" << aD << " DTest=" << aDTest << " Cpt=" << aCpt << "\n";
            MMVII_INTERNAL_ASSERT_bench(aD<aDTest,"Bench inverse  Matrixes");
        }


        //  aM.Inverse(1e-19,25);

        // cDenseMatrix<Type> aM1(aNb,aNb,eModeInitImage::eMIA_Rand);
    }
}

void TQR(int aX,int aY)
{
    cDenseMatrix<double> aM(aX,aY,eModeInitImage::eMIA_Rand);
    aM.QR_Decomposition();
}

template <class Type> class cStrStat2
{
    public :
       cStrStat2(int aSz);
       void Add(const cDenseVect<Type> & );
       void Normalise();
       const double              Pds() const;
       const cDenseVect<Type>  & Moy() const;
       const cDenseMatrix<Type>& Cov() const;
       const cResulSymEigenValue<Type> & DoEigen();
    private :
       double                    mPds;
       cDenseVect<Type>          mMoy;
       cDenseMatrix<Type>        mCov;
       cResulSymEigenValue<Type> mEigen;
};



void BenchStatCov()
{
    int aSz=3;
    int aNb = 6;
    cDenseVect<double> aMoyV(aSz);
    std::list<cDenseVect<double> >  aLV;
    cDenseMatrix<double> aCov1(aSz,aSz,eModeInitImage::eMIA_Null);
    for (int aK=0 ; aK<aNb ; aK++)
    {
        cDenseVect<double> aV(aSz,eModeInitImage::eMIA_Rand);
        aLV.push_back(aV);
        aCov1.Add_tAA(aV);
    }
}

void BenchDenseMatrix0()
{
    {
      cPt2di aSz(3,8);
      cIm2D<double> aI1(aSz,nullptr,eModeInitImage::eMIA_Rand);
      cIm2D<double> aI1B(aSz,nullptr,eModeInitImage::eMIA_Rand);
      cIm2D<double> aI3(aSz,nullptr,eModeInitImage::eMIA_Rand);
      cIm2D<double> aI3B = aI3.Dup();

      CopyIn(aI1B.DIm(),aI1.DIm());  // I2 == I1
      AddIn(aI3.DIm(),aI1B.DIm());  //  I3 == I3B + I1
      DivCsteIn(aI3.DIm(),3.14);          // I3 == (I3B+I1) / 3.14
      for (const auto & aP : aI1.DIm())
      {
          double aDif= std::abs(aI3.DIm().GetV(aP)-(aI3B.DIm().GetV(aP) + aI1.DIm().GetV(aP)) / 3.14); 
          MMVII_INTERNAL_ASSERT_bench(aDif<1e-5,"Bench Op Im");
          // std::cout << aI3.DIm().GetV(aP) << " " << (aI3B.DIm().GetV(aP) + aI1.DIm().GetV(aP)) / 3.14 << "\n";
          // std::cout << aDif << "\n";
      }
      
    }

    TQR(3,3);
    TQR(3,5);
    TQR(5,3);

    cDenseVect<double> aV0(2);
    cDenseVect<double> aV1(2);
    aV0(0) = 10; aV0(1) = 20;
    aV1(0) = 13; aV1(1) = 24;
    MMVII_INTERNAL_ASSERT_bench(std::abs(aV0.L1Dist(aV1)-3.5)<1e-5,"Bench Matrixes");
    MMVII_INTERNAL_ASSERT_bench(std::abs(aV0.L2Dist(aV1)-5.0/sqrt(2))<1e-5,"Bench Matrixes");
   


    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL4 > (3,2,4);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL16 > (3,2,1);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL16 > (3,2,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL8 > (2,3,10);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL4>,tREAL4 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL4>,tREAL8 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL4>,tREAL16 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL16>,tREAL4 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL16>,tREAL8 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL16>,tREAL16 > (2,3,2);


    TplBenchMatrix<cDenseMatrix<tREAL4>,tREAL4 > (7,2,7);
    TplBenchMatrix<cDenseMatrix<tREAL4>,tREAL8 > (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL4>,tREAL16> (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL8>,tREAL4 > (7,2,7);
    TplBenchMatrix<cDenseMatrix<tREAL8>,tREAL8 > (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL8>,tREAL16> (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL16>,tREAL4 > (7,2,7);
    TplBenchMatrix<cDenseMatrix<tREAL16>,tREAL8 > (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL16>,tREAL16> (7,2,3);

    TplBenchMatrix<cDenseMatrix<tREAL4>,tREAL4 > (2,5,1);
    TplBenchMatrix<cDenseMatrix<tREAL8>,tREAL4 > (2,5,2);
    TplBenchMatrix<cDenseMatrix<tREAL16>,tREAL4 > (2,5,9);


    for (int aK=0 ; aK<8 ; aK++)
    {
        TplBenchDenseMatr<tREAL4>(1<<aK,1<<aK);
        TplBenchDenseMatr<tREAL4>(1<<aK,1<<aK);
    }

    TplBenchDenseMatr<tREAL4>(1,1);
    TplBenchDenseMatr<tREAL4>(2,2);
    TplBenchDenseMatr<tREAL4>(4,4);
    TplBenchDenseMatr<tREAL4>(8,8);
    TplBenchDenseMatr<tREAL4>(16,16);
    TplBenchDenseMatr<tREAL4>(32,32);
    TplBenchDenseMatr<tREAL4>(64,64);

    TplBenchDenseMatr<tREAL8>(4,6);
    TplBenchDenseMatr<tREAL4>(100,100);
    TplBenchDenseMatr<tREAL4>(100,100);
    TplBenchDenseMatr<tREAL4>(100,100);
    TplBenchDenseMatr<tREAL8>(100,100);
    TplBenchDenseMatr<tREAL16>(100,100);
}

};
