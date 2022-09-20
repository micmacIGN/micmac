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


// SzMil =>  Taille intermediaire pour multiplication matrice

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

        aMyySym.Weighted_Add_tAA(-2,aCol,false);

        aMyySym.Add_tAA(aCol,false);
        aMyySym.Add_tAA(aCol3,false);
        for (const auto & aP : aMyySym)
        {
             double aV1 =  aMyySym.V_GetElem(aP.x(),aP.y());
             double aV2 = 8 * FTestVect(aP.x()) * FTestVect(aP.y());
             MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Bench add tAA");
        }
       
        aMyyUp.Weighted_Add_tAA(-2,aCol);
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
    {  // Bench sparse
       int aNb = aSzX+aSzY;
       TypeMatr aM1(aNb,aNb,eModeInitImage::eMIA_Null);
       TypeMatr aM2(aNb,aNb,eModeInitImage::eMIA_Null);
       cDenseVect<typename TypeMatr::tVal> aDV1(aNb,eModeInitImage::eMIA_Null);
       cDenseVect<typename TypeMatr::tVal> aDV2(aNb,eModeInitImage::eMIA_Null);
 
       for (int aK=0 ; aK<4 ; aK++)
       {
           cSparseVect<typename TypeMatr::tVal> aSV;
           cDenseVect<typename TypeMatr::tVal> aDV(aNb,eModeInitImage::eMIA_Null);
           for (int aX=0 ; aX<aNb ;aX++)
           {
               if (HeadOrTail())
               {
                   int anInd = (aX+1) % aNb;
                   typename TypeMatr::tVal aVal = RandUnif_0_1();
                   aSV.AddIV(anInd,aVal);
                   aDV(anInd) = aVal;
               }
           }
           typename TypeMatr::tVal aW = RandUnif_0_1();
           aM1.Weighted_Add_tAA(aW,aDV,(aK%2));
           WeightedAddIn(aDV1.DIm(),aW,aDV.DIm());

           aM2.Weighted_Add_tAA(aW,aSV,(aK%2));
           aDV2.WeightedAddIn(aW,aSV);
       }
       double aDistM = aM1.DIm().L2Dist(aM2.DIm());
       double aDistV = aDV1.DIm().L2Dist(aDV2.DIm());
       
       MMVII_INTERNAL_ASSERT_bench(aDistM<1e-5,"Bench SparseVect");
       MMVII_INTERNAL_ASSERT_bench(aDistV<1e-5,"Bench SparseVect");
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
        // cDenseMatrix<Type> aM(aNb,eModeInitImage::eMIA_Rand);
        cDenseMatrix<Type> aM= cDenseMatrix<Type>::RandomSquareRegMatrix(cPt2di(aNb,aNb),false,1e-5,1e-2);

         
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
  

        double aDTest = 1e-4 * pow(10,-(4-sizeof(Type))/2.0) * pow(aNb,1.0) ;  // Accuracy expectbale vary with tREAL4, tREAL8 ...
    
        // Singular value decomposition
        {
             cResulSVDDecomp<Type>  aSVD = aM.SVD();
             MMVII_INTERNAL_ASSERT_bench(aSVD.MatU().Unitarity()<aDTest,"SVD-U Unit");
             MMVII_INTERNAL_ASSERT_bench(aSVD.MatV().Unitarity()<aDTest,"SVD-V Unit");
             MMVII_INTERNAL_ASSERT_bench(aM.DIm().L2Dist(aSVD.OriMatr().DIm())<aDTest,"SVD Reconstr");

             Type aDU =  aSVD.MatU().Det();
             Type aDV =  aSVD.MatV().Det();
             // Are detminant of unitary matrix equal to +-1 ?
             double aDifU = std::abs(std::abs(aDU)-1) / Cube(aNb);
             double aDifV = std::abs(std::abs(aDV)-1) / Cube(aNb);
             MMVII_INTERNAL_ASSERT_bench(aDifU<1e-5,"Bench det unit matrix");
             MMVII_INTERNAL_ASSERT_bench(aDifV<1e-5,"Bench det unit matrix");

             // double aD1 = aSVD.MatU().Det() * aSVD.MatV().Det();
             double aDifDet = RelativeDifference(aDU*aDV*aSVD.SingularValues().ProdElem(),aM.Det());
             aDifDet /= Cube(aNb);
             MMVII_INTERNAL_ASSERT_bench(aDifDet<1e-5,"Bench det unit matrix");
        }
        
        cDenseMatrix<Type> aId(aNb,eModeInitImage::eMIA_MatrixId);
        cDenseMatrix<Type> aMInv = aM.Inverse();
        cDenseMatrix<Type> aCheck = aM*aMInv;

        cDenseMatrix<Type> aMInv2 = aM.Solve(aId);
        cDenseMatrix<Type> aCheck2 = aM*aMInv2;

        double aD = aId.DIm().L2Dist(aCheck.DIm());
        double aD2 = aId.DIm().L2Dist(aCheck2.DIm());
  
        if (aD>aDTest)
        {
            StdOut() << "D=" << aD << " DTest=" << aDTest << " Cpt=" << aCpt << "\n";
            MMVII_INTERNAL_ASSERT_bench(false,"Bench inverse  Matrixes");
        }
        if (aD2>aDTest)
        {
            StdOut() << "D=" << aD2 << " DTest=" << aDTest << " Cpt=" << aCpt << "\n";
            MMVII_INTERNAL_ASSERT_bench(false,"Bench inverse  Matrixes");
        }

        {  // Bench Solve Vect
           cDenseVect<Type> aV(aNb,eModeInitImage::eMIA_Rand);
           cDenseVect<Type> aVSol = aM.SolveColumn(aV);
           cDenseVect<Type> aVCheck = aM * aVSol;
           MMVII_INTERNAL_ASSERT_bench( aVCheck.L2Dist(aV) < aDTest ,"Bench Solve Vect  Matrixes");
           // Check transpose
           aVSol = aM.SolveLine(aV);
           aVCheck = aVSol * aM;
           MMVII_INTERNAL_ASSERT_bench( aVCheck.L2Dist(aV) < aDTest ,"Bench SolveLine Vect  Matrixes");
        }
        {  // Bench Solve Mat
           cDenseMatrix<Type> aMTest(3,aNb,eModeInitImage::eMIA_Rand);
           cDenseMatrix<Type> aMSol = aM.Solve(aMTest);
           cDenseMatrix<Type> aMCheck = aM * aMSol;
           double aD =  aMCheck.DIm().L2Dist(aMTest.DIm());
           MMVII_INTERNAL_ASSERT_bench(aD < aDTest ,"Bench Solve Vect  Matrixes");
        }

        //  aM.Inverse(1e-19,25);

        // cDenseMatrix<Type> aM1(aNb,aNb,eModeInitImage::eMIA_Rand);
    }

    // Bench sur kernel and rank-deficient matrix
    {
        int aNb = aSzX;
        // int aSzK = std::min(aNb,1 + round_ni(RandUnif_N(2));
        int aSzK = 1;
        cDenseMatrix<Type> aMat =   cDenseMatrix<Type>::RandomSquareRankDefMatrix(cPt2di(aNb,aNb),aSzK);

        cDenseVect<Type> aK = aMat.Kernel();
        cDenseVect<Type> aMK = aMat * aK;
        Type aNormK  =  aK.L2Norm();
        Type aNormMK  =  aMK.L2Norm();
        double aRatio = ((aNormMK / aNormK) / Cube(aNb)) / tNumTrait<Type>::Accuracy();

        MMVII_INTERNAL_ASSERT_bench(aRatio<1e-4,"Kernel compute");
    }
}

void TQR(int aX,int aY)
{
    cDenseMatrix<double> aM(aX,aY,eModeInitImage::eMIA_Rand);
    aM.QR_Decomposition();
}

void BenchStatCov(int aSz,int aNb)
{
    cStrStat2<double> aSt2Init(aSz);
    std::list<cDenseVect<double> >  aLV;
    for (int aK=0 ; aK<aNb ; aK++)
    {
        cDenseVect<double> aV(aSz,eModeInitImage::eMIA_Rand);
        aLV.push_back(aV);
        aSt2Init.Add(aV);
    }
    // Test with uncentered moment
    cResulSymEigenValue<double>  aVpInit = aSt2Init.DoEigen();
    {
       double aV0 = aVpInit.EigenValues()(0);
       if (aNb<aSz)
       {
          MMVII_INTERNAL_ASSERT_bench(std::abs(aV0)<1e-8,"cStrStat2");
       }
       else // In fact in could be by "chance" close to 0 ...
       {
          MMVII_INTERNAL_ASSERT_bench(std::abs(aV0)>1e-6,"Test cStrStat2");
       }
    }
    // Test with centered moment
    aSt2Init.Normalise();
    aVpInit = aSt2Init.DoEigen();
    {
       double aV0 = aVpInit.EigenValues()(0);
       if (aNb<=aSz)
       {
          MMVII_INTERNAL_ASSERT_bench(std::abs(aV0)<1e-10,"cStrStat2");
       }
       else // In fact in could be by "chance" close to 0 ...
       {
          MMVII_INTERNAL_ASSERT_bench(std::abs(aV0)>1e-7,"Test cStrStat2");
       }
    }

    cStrStat2<double> aSt2Orthog(aSz);
    cDenseVect<double> aTmp(aSz);
    for (const auto & aV : aLV)
    {
        aSt2Init.ToNormalizedCoord(aTmp,aV);
        aSt2Orthog.Add(aTmp);
        for (int aX=0 ; aX < aSz ; aX++)
        {
            double aDif= std::abs(aTmp(aX)-aSt2Init.KthNormalizedCoord(aX,aV));
            MMVII_INTERNAL_ASSERT_bench(aDif<1e-5,"Test cStrStat2 ");
        }
    }
    aSt2Orthog.Cov().SelfSymetrizeBottom();

    // Variable should be uncorrelated
    MMVII_INTERNAL_ASSERT_bench(aSt2Orthog.Cov().Diagonalicity()<1e-5,"Test cStrStat2 ");
    // coordinate  should be centered
    MMVII_INTERNAL_ASSERT_bench(aSt2Orthog.Moy().DIm().L2Norm()<1e-5,"Test cStrStat2 ");

    aSt2Orthog.Normalise(false);
    const cResulSymEigenValue<double> & aVpOrt = aSt2Orthog.DoEigen();
    // Eigen value should be the same as init
    MMVII_INTERNAL_ASSERT_bench(aVpInit.EigenValues().L2Dist(aVpOrt.EigenValues())<1e-5,"Test cStrStat2 ");

}
void BenchStatCov()
{
    BenchStatCov(3,2);
    BenchStatCov(3,6);
    BenchStatCov(3,8);
    BenchStatCov(3,300);
}

template <class Type> static Type Residual
                                  (
                                     const cLinearOverCstrSys<Type>& aSys,
                                     const cDenseVect<Type> & aSol,
                                     const std::vector<Type>               aLWeight,
                                     const std::vector<cDenseVect<Type> >  aLVec,
                                     const std::vector<Type>               aLVal
                                  )
{
    Type aRes = 0.0;
    for (int aK=0 ; aK<int(aLVec.size()) ; aK++)
    {
        aRes += aSys.Residual(aSol,aLWeight[aK],aLVec[aK],aLVal[aK]);
    }
    return aRes;
}

template <class Type> void BenchSysSur(cLinearOverCstrSys<Type>& aSys,bool Exact)
{
   int  aNbVar = aSys.NbVar();
   // cLinearOverCstrSys<Type>& aSys = aSysLsq;

   for (int aNbIter=0 ; aNbIter<2; aNbIter++)
   {
      cLeasSqtAA<Type>  aSpSys(aNbVar);
      aSys.Reset();
      std::vector<Type>               aLWeight;
      std::vector<cDenseVect<Type> >  aLVec;
      std::vector<Type>               aLVal;

      int aNbEq = Exact ? aNbVar : 6 * aNbVar;

      cDenseMatrix<Type> aMatrReg = cDenseMatrix<Type>::RandomSquareRegMatrix(cPt2di(aNbVar,aNbVar),false,1e-1,1e-1);
      for (int aK=0 ; aK<aNbEq ; aK++)
      {
          Type aWeight = 1+aK;
          cDenseVect<Type> aDV(aNbVar,eModeInitImage::eMIA_Rand);
          // If we test exact system, with no reduncy, we must be sure that system is not degenerated
          if (Exact)
          {
              aDV = aMatrReg.ReadCol(aK);
          }
          cSparseVect<Type> aSV(aNbVar);
          for (int aK=0 ; aK<aNbVar ; aK++)
          {
              int aI = (aK+1) % aNbVar;
              aSV.AddIV(aI,aDV(aI));
          }
          
          Type aVal = RandUnif_0_1();
          aLVec.push_back(aDV);
          aLVal.push_back(aVal);
          aLWeight.push_back(aWeight);
          aSys.AddObservation(aWeight,aDV,aVal);
          aSpSys.AddObservation(aWeight,aSV,aVal);
      }
      cDenseVect<Type> aSol = aSys.Solve();
      cDenseVect<Type> aSpSol = aSpSys.Solve();
      double aDist= aSol.DIm().L2Dist(aSpSol.DIm());
      MMVII_INTERNAL_ASSERT_bench(aDist<1e-5,"Bench Op Im");

      Type aR0 = Residual(aSys,aSol,aLWeight,aLVec,aLVal);
      Type aDTest = sqrt(std::numeric_limits<Type>::epsilon()) * 100;
      // StdOut() << "aR0aR0aR0aR0aR0 " << aR0 << "\n";

      if (Exact)
      {
         MMVII_INTERNAL_ASSERT_bench(aR0<(aDTest*aNbVar),"Bench Op Im");
         for (int aK=0 ; aK<aNbVar ; aK++)
         {
             Type aDif = std::abs( aLVal[aK] - aSol.DotProduct(aLVec[aK]));
             /* StdOut() << "ddDddDDDD " << aDif << " " << E2Str(tElemNumTrait<Type>::TyNum()) 
                   << " Eps: " << std::numeric_limits<Type>::epsilon() << " " <<  aDTest << "\n"; */
             MMVII_INTERNAL_ASSERT_bench(aDif<aDTest,"Bench Op Im");
             // StdOut() << "ScaAal " <<  aLVal[aK] - aSol.DotProduct(aLVec[aK]) << "\n";
         }
      }
      else
      {
          // Check that solution is really the minimum
          for (int aK=0 ; aK<100 ; aK++)
          {
             double aEps = 1e-2;
             cDenseVect<Type> aNewV(aNbVar);
             for (int aK=0 ; aK<aNbVar ; aK++)
                aNewV(aK) = aSol(aK) + aEps * RandUnif_C();

             Type aRN = Residual(aSys,aNewV,aLWeight,aLVec,aLVal);
             // StdOut() << "RRRRR  " << aRN-aR0 << "\n";

             MMVII_INTERNAL_ASSERT_bench(aRN>=aR0-1e-5,"Bench residual ");

             // double aD = aSol.DIm().L2Dist(aNewV.DIm());
             // StdOut() << " D=" << aD << "Dif " << (aRN-aR0) /Square(aD) << "\n";
          }
      }
   }
}

template <class Type> void BenchObsFixVar
                           (
                              cLinearOverCstrSys<Type>& aSys1, // 1 by 1
                              cLinearOverCstrSys<Type>& aSys2, // By Sparse
                              cLinearOverCstrSys<Type>& aSys3  // By dense
                           )
{
/*  Will Parse all the var, and compute a vector using different mode
 *  of fxing var, test they give the same result*/

   int  aNbVar = aSys1.NbVar();
   cSparseVect<Type> aSV2A(aNbVar);  // Fix odd variable
   cSparseVect<Type> aSV2B(aNbVar);  // Fix even variable
   cDenseVect<Type>  aDV3(aNbVar);    // create the variable hand-craftly
   for (int aK=0 ; aK<aNbVar; aK++)
   {
      Type aVal = RandUnif_0_1();
      aSys1.AddObsFixVar(1.0,aK,aVal); // fix all var 1 by 1
      if (aK%2)
         aSV2A.AddIV(aK,aVal);
      else
         aSV2B.AddIV(aK,aVal);
      aDV3(aK) = aVal;
   }
   aSys2.AddObsFixVar(1.0,aSV2A);  // fix odd var in S2
   aSys2.AddObsFixVar(1.0,aSV2B);  // fix even var in S2
   aSys3.AddObsFixVar(1.0,aDV3);  // fix all var in S3

   cDenseVect<Type>  aV1 = aSys1.Solve();
   // cDenseVect<Type>  aV2 = aSys2.Solve().DIm();
   // cDenseVect<Type>  aV3 = aSys3.Solve().DIm();
   double aD1 = aDV3.DIm().L2Dist(aV1.DIm());
   double aD2 = aDV3.DIm().L2Dist(aSys2.Solve().DIm());
   double aD3 = aDV3.DIm().L2Dist(aSys3.Solve().DIm());

   MMVII_INTERNAL_ASSERT_bench((aD1+aD2+aD3)<1e-5,"Bench Add Obs Fix Var");
}

template <class Type,class TypeSys> void OneTplBenchLsq(int aNbVar)
{
   {
      TypeSys  aSysLsq(aNbVar);
      BenchSysSur(aSysLsq,true);
      BenchSysSur(aSysLsq,false);
   }

   {
       TypeSys  aSysLsq1(aNbVar);  // 1 by 1
       TypeSys  aSysLsq2(aNbVar);  // By Sparse
       TypeSys  aSysLsq3(aNbVar);  // By dense
       BenchObsFixVar(aSysLsq1,aSysLsq2,aSysLsq3);
   }

}

template <class Type,class TypeSys> void TplBenchLsq()
{
     // there is unexplained error in LeastSquaresConjugateGradient, activate this flag to track it
     bool  TrackBugEigenSucc = false;
     
     //  Check  all Least Square give the same solution
     for (int aK=0 ; aK<20 ; aK++)
     {
         int aNbVar = 1+ RandUnif_N(10);
         int aNbEq =  2 * aNbVar + 3 ;

         if (TrackBugEigenSucc)
            StdOut()  << "NBVARR=" << aNbVar << " NbE="<<aNbEq << "\n";

         // random param for sparse normal syst
	 cParamSparseNormalLstSq aParam
		                 (
				       8.0*RandUnif_0_1(), 
                                       round_ni(aNbVar*pow(RandUnif_0_1(), 4.0)),      // Max range Dense fix
                                       round_ni(0.5*aNbVar*pow(RandUnif_0_1(), 2.0))   // tempo
                                 );
         // the different type of solver, that should return the same result
	 std::vector<cLeasSq<Type>*> aVSys = {
	                                            cLeasSq<Type>::AllocDenseLstSq(aNbVar),
		                                    cLeasSq<Type>::AllocSparseGCLstSq(aNbVar),
	                                            cLeasSq<Type>::AllocSparseNormalLstSq(aNbVar,aParam)
	                                     };
         // juste make several time the test , becaude chek also reseting
	 for (int aNbTest=0 ; aNbTest<3 ; aNbTest++)
	 {

              std::vector< cSparseVect<Type> > aVSV;
	      for (int aK=0 ; aK<= 3*aNbEq ; aK++)
	      {
                  // sparse vector with density K/aNbEq
                  cSparseVect<Type> aVCoeff = cSparseVect<Type>::RanGenerate(aNbVar,double(aK)/aNbEq);
                  aVSV.push_back(aVCoeff);
                  Type  aCste =  RandUnif_C();
                  Type  aW    =   0.5 + RandUnif_0_1();
	          for (auto & aSys : aVSys)
		      aSys->AddObservation(aW,aVCoeff,aCste);
	      }

              static int aCpt = 0; aCpt++;
                  // the value CPT==6 is probably to change depending on context
              bool  CptBug = (aCpt==6);
              if (TrackBugEigenSucc)
              {
                  if (CptBug)
                  {
                       cDenseMatrix<Type>  aMat = aVSys[0]->V_tAA() ;
                       cResulSymEigenValue<Type> aSVD = aMat.SymEigenValue();
                       cDenseVect<Type>   aVP = aSVD.EigenValues() ;

                       StdOut() << "EIGENVAL " << aVP << "\n";
                  }
              }

	      std::vector<cDenseVect<Type> > aVSol;
	      for (int aKSys=0 ; aKSys<int(aVSys.size()) ;  aKSys++)
	      {
                  if (TrackBugEigenSucc)
                      StdOut() << "KSYS " << aKSys << " CPT=" << aCpt << "\n";
	          auto & aSys  =  aVSys[aKSys];
                  // with conj grad, change error handling
                  if (aKSys==1) 
                  {
                     if (TrackBugEigenSucc) // if we track, put a warning
                         PushErrorEigenErrorLevel(eLevelCheck::Warning);
                     else
                         PushErrorEigenErrorLevel(eLevelCheck::NoCheck); // else just ignore
                 }
                  aVSol.push_back(aSys->Solve());
                  if ((aKSys==1)&& (!TrackBugEigenSucc)) // restore eventually
                     PopErrorEigenErrorLevel();
	          Type aDist = aVSol[0].L2Dist(aVSol.back()) / tNumTrait<Type>::Accuracy() ;
                  if ((aDist>=1e-2) || (TrackBugEigenSucc && CptBug))
                  {
                      for (const auto& aCoefs : aVSV)
                      {
                            for (const auto& aIV : aCoefs.IV())
                                StdOut() << "[" << aIV.mInd << "," << aIV.mVal << "] ";
                            StdOut() << "\n";
                      }
                      StdOut() << "KS=" << aKSys << " DDddd " << aDist 
                               << " Acc=" << tNumTrait<Type>::Accuracy() << "\n";
                      StdOut() << " NBV=" << aNbVar << " NbE=" << aNbEq << "\n"; 

                      cDenseMatrix<Type>  atAA = aVSys[0]->V_tAA();
                      cDenseVect<Type>    atARhs = aVSys[0]->V_tARhs();
                       
	              for (int aKS2=0 ; aKS2<int(aVSys.size()) ;  aKS2++)
                      {
                          cDenseVect<Type> aRes = atAA * aVSys[aKS2]->Solve() - atARhs;
                          StdOut() << "NNNN " << aRes.L2Norm()   << " " << atARhs.L2Norm() << "\n";
                      }
  
                      MMVII_INTERNAL_ASSERT_bench(false,"Cmp Least Square");
                  }

	      }
	      for (int aKSys=0 ; aKSys<int(aVSys.size()) ;  aKSys++)
              {
	          auto & aSys  =  aVSys[aKSys];
		  aSys->Reset();
              }
	 }

	 DeleteAllAndClear(aVSys);
     }


     OneTplBenchLsq<Type,TypeSys>(1);
     OneTplBenchLsq<Type,TypeSys>(2);
     OneTplBenchLsq<Type,TypeSys>(4);
}

/** Test/Bench to see how the solver are robust to rank deficient least-square
    
     The (surprising !) conclusion being that  the dense least square using normal equation
    is the more robust.
*/

void BenchLsqDegenerate()
{
    int aKer = RandUnif_N(5);  // Sz of kernel (dim of null vector)
    int aSz =  aKer + 1 +RandUnif_N(10);  // Sz of matric
    int aNbV=  1 +RandUnif_N(aSz*2);  // Number of equation

    //  Allocate a rank deficient matric corresponding to parameters
    cResulSVDDecomp<tREAL8> aRSVD = cDenseMatrix<tREAL8>::RandomSquareRankDefSVD(cPt2di(aSz,aSz),aKer);
    cDenseMatrix<tREAL8>  aMatr = aRSVD.OriMatr() ;


    std::vector<cLeasSq<tREAL8>*> aVSys;
    cParamSparseNormalLstSq  aParam;

    aVSys.push_back(cLeasSq<tREAL8>::AllocSparseGCLstSq(aSz));   //  0 => sparse, memo eq
    aVSys.push_back(cLeasSq<tREAL8>::AllocSparseNormalLstSq(aSz,aParam));  // 1 => sparse normal matrix
    aVSys.push_back(cLeasSq<tREAL8>::AllocDenseLstSq(aSz));    // 2 => dense matrix, normal eq

    for (int aK=0 ; aK<aNbV ; aK++)
    {
        // random vector
        cDenseVect<tREAL8> aV(aSz,eModeInitImage::eMIA_Rand);

        aV = aMatr * aV; // all vector belongs to image(aMatr)
        double aVal = RandUnif_C();

        for (const auto & aPtrSys : aVSys)
            aPtrSys->AddObservation(1.0,aV,aVal);
    }
    
    cDenseMatrix<tREAL8>  atAA = aVSys[2]->V_tAA();
    cDenseVect<tREAL8>  atARhs = aVSys[2]->V_tARhs();
    for (int aK=0 ; aK<int(aVSys.size()) ; aK++ )
    {
        // check if  tAA * Sol = tARhs  or not ...
        const auto & aPtrSys = aVSys[aK];
        cDenseVect<tREAL8>  aSol = aPtrSys->Solve();
        cDenseVect<tREAL8> aDif  = atAA * aSol - atARhs;
        //  stop only if DenseLeastSq, as other are not robust (to investigate ...)
        if ((aDif.L2Norm()>=1e-5) && (aK==2))
        {
             StdOut() << aK  << "   NNNN " <<  aDif.L2Norm() 
                      << " Ker="   <<aKer << " Sz="   <<aSz << " NbV="   <<aNbV
                      << "\n";
            MMVII_INTERNAL_ASSERT_bench(false,"lsq with degenerate sys");
        }
    }
    {
        cDecSumSqLinear<tREAL8> aDSSL;
        cDenseVect<tREAL8>  aV0 = cDenseVect<tREAL8>::Cste(aSz,0.0);
        aDSSL.Set(aV0,atAA,atARhs);
        cLeasSqtAA<tREAL8> aSys2 = aDSSL.OriSys();
        tREAL8 aDV =  aSys2.V_tARhs().L2Dist(atARhs) ;
        tREAL8 aDM =   aSys2.V_tAA().L2Dist(atAA);
    
         // StdOut() << "Ddddd " << aDV << " " << aDM  << " \n";
         MMVII_INTERNAL_ASSERT_bench(aDV<1e-5,"cDecSumSqLinear vector");
         MMVII_INTERNAL_ASSERT_bench(aDM<1e-5,"cDecSumSqLinear matrix");
         ///  StdOut() << "mmmMMMMmm " << aSys2.V_tAA().L2Dist(atAA) << " \n";
    }
    DeleteAllAndClear(aVSys);
}

void BenchLsq()
{
     {
         PushErrorEigenErrorLevel(eLevelCheck::NoCheck);
         for (int aK=0 ; aK<200 ; aK++)
         {
            BenchLsqDegenerate();
         }
         PopErrorEigenErrorLevel();
         // reactivate if you want to check 
         if (0)
         {
             for (int aK=0 ; aK<200 ; aK++)
             {
                BenchLsqDegenerate();
             }
         }
         // StdOut() << "EEEEEEEeeeeeeeeeeeeeeee\n"; getchar();
     }

     TplBenchLsq<tREAL4 ,cLeasSqtAA<tREAL4> >();
     TplBenchLsq<tREAL8 ,cLeasSqtAA<tREAL8> >();
     TplBenchLsq<tREAL16,cLeasSqtAA<tREAL16> >();
}

template <class Type,const int DimX,const int DimY> void BenchMatPt()
{
     cDenseMatrix<Type> aMat0(DimX,DimY,eModeInitImage::eMIA_Rand);
     cDenseMatrix<Type> aMatL(DimX,DimY,eModeInitImage::eMIA_Rand);
     cDenseMatrix<Type> aMatC(DimX,DimY,eModeInitImage::eMIA_Rand);

     {
         // Check the matrix are in fact different at the begining
         double aDL = aMat0.DIm().L1Dist(aMatL.DIm());
         double aDC = aMat0.DIm().L1Dist(aMatC.DIm());
         MMVII_INTERNAL_ASSERT_bench(aDL>1e-5,"BenchMatPt");
         MMVII_INTERNAL_ASSERT_bench(aDC>1e-5,"BenchMatPt");
     }

     {
        // Copy aMat0 in MatL and MatC by Line and Col
        for (int aX=0 ; aX<DimX ; aX++)
        {
            SetCol(aMatC,aX,cPtxd<Type,DimY>::Col(aMat0,aX));
        }
        for (int aY=0 ; aY<DimY ; aY++)
        {
            SetLine(aY,aMatL,cPtxd<Type,DimX>::Line(aY,aMat0));
        }

        double aDL = aMat0.DIm().L1Dist(aMatL.DIm());
        double aDC = aMat0.DIm().L1Dist(aMatC.DIm());
        MMVII_INTERNAL_ASSERT_bench(aDL==0,"BenchMatPt");
        MMVII_INTERNAL_ASSERT_bench(aDC==0,"BenchMatPt");
     }

     // Check MulCol and MulLine
     {
          cDenseMatrix<Type> aMat2(DimY,DimX,eModeInitImage::eMIA_Rand);
          cDenseMatrix<Type>  aMat02 = aMat0 * aMat2;

          for (int aY=0; aY<DimY ; aY++)
          {
            cPtxd<Type,DimY> aPMul02;

            MulLine(aPMul02,cPtxd<Type,DimX>::Line(aY,aMat0),aMat2);
            cPtxd<Type,DimY> aLine02=cPtxd<Type,DimY>::Line(aY,aMat02);
            MMVII_INTERNAL_ASSERT_bench(NormInf(aPMul02 - aLine02)<1e-5,"Mul Line Mat");


            MulCol(aPMul02,aMat0,cPtxd<Type,DimX>::Col(aMat2,aY));
            cPtxd<Type,DimY> aCol02=cPtxd<Type,DimY>::Col(aMat02,aY);
            MMVII_INTERNAL_ASSERT_bench(NormInf(aPMul02 - aCol02)<1e-5,"Mul Line Mat");
          }
     }

     // Check   Pt * Mat and Mat * Pt
     {
         cDenseMatrix<Type> aSq0(DimX,DimX,eModeInitImage::eMIA_Rand);
         cDenseMatrix<Type> aSq1(DimX,DimX,eModeInitImage::eMIA_Rand);

         cDenseMatrix<Type> aSq01 = aSq0 * aSq1;

         for (int aK=0; aK<DimX ; aK++)
         {
            cPtxd<Type,DimX> aLine0 = cPtxd<Type,DimX>::Line(aK,aSq0) * aSq1;
            cPtxd<Type,DimX> aCol0 =  aSq0 * cPtxd<Type,DimX>::Col(aSq1,aK) ;

            MMVII_INTERNAL_ASSERT_bench(NormInf(aLine0- cPtxd<Type,DimX>::Line(aK,aSq01))<1e-5,"Pt*Mat");
            MMVII_INTERNAL_ASSERT_bench(NormInf(aCol0 - cPtxd<Type,DimX>::Col (aSq01,aK))<1e-5,"Mat*Pt");
         }
     }

     // Chek solve,  COL :   Mat * ? = C   , LINE : ? * Mat = L
     {
         static int aCpt=0; aCpt++;
         Type Eps = tNumTrait<Type>::Eps();  // Accurracy depends of the type
         Type aAccur = sqrt(Eps);
         

         cDenseMatrix<Type> aSqM = cDenseMatrix<Type>::RandomSquareRegMatrix(cPt2di(DimX,DimX),false,1e-5,1e-2);
         cPtxd<Type,DimX> aPt = cPtxd<Type,DimX>::PRandC();
         cPtxd<Type,DimX> aSCol = SolveCol(aSqM,aPt);
         cPtxd<Type,DimX> aSLine = SolveLine(aPt,aSqM);

         Type aCheckCol =  Norm1(aPt-aSqM*aSCol);
         Type aChekLine = Norm1(aPt-aSLine*aSqM);

         if (NeverHappens()) // (aCpt==-1) 
         {

            StdOut() << "BENCH MATRIX " << aCpt << " " << tNumTrait<Type>::NameType() 
                     << " " << tNumTrait<Type>::Eps()
                     <<  " " << aCheckCol << " " << aChekLine<< "\n";
         }

         MMVII_INTERNAL_ASSERT_bench(aCheckCol<aAccur,"Solev Mat*?");
         MMVII_INTERNAL_ASSERT_bench(aChekLine<aAccur,"Solve ?*Mat");
     }
     
     // std::cout << "DLLLLCC " << aDL << " " << aDC << "\n";
}

void BenchDenseMatrix0(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Matrix0")) return;

    Bench_EigenDecompos(aParam);


    for (int aK=0 ; aK<100 ; aK++)
    {
       BenchMatPt<tREAL4,3,2>();
       BenchMatPt<tREAL8,4,3>();
       BenchMatPt<tREAL16,4,3>();
    }

    BenchLsq();
    BenchStatCov();
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
   


    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL8 > (2,3,10);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL4>,tREAL4 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL16>,tREAL16 > (2,3,2);
/*
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL4 > (3,2,4);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL16 > (3,2,1);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL8>,tREAL16 > (3,2,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL4>,tREAL8 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL4>,tREAL16 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL16>,tREAL4 > (2,3,2);
    TplBenchMatrix<cUnOptDenseMatrix<tREAL16>,tREAL8 > (2,3,2);
*/


    TplBenchMatrix<cDenseMatrix<tREAL4>,tREAL4 > (7,2,7);
    TplBenchMatrix<cDenseMatrix<tREAL8>,tREAL8 > (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL16>,tREAL16> (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL4>,tREAL4 > (2,5,1);
/*
    TplBenchMatrix<cDenseMatrix<tREAL8>,tREAL4 > (2,5,2);
    TplBenchMatrix<cDenseMatrix<tREAL16>,tREAL4 > (2,5,9);
    TplBenchMatrix<cDenseMatrix<tREAL4>,tREAL8 > (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL4>,tREAL16> (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL8>,tREAL4 > (7,2,7);
    TplBenchMatrix<cDenseMatrix<tREAL8>,tREAL16> (7,2,3);
    TplBenchMatrix<cDenseMatrix<tREAL16>,tREAL4 > (7,2,7);
    TplBenchMatrix<cDenseMatrix<tREAL16>,tREAL8 > (7,2,3);

*/

    {
        int aNb = std::min(60,20+aParam.Level()*5);
        for (int aK=1 ; aK<aNb ; aK++)
        {
// StdOut() << "KKKKKK " << aK << "\n";
           TplBenchDenseMatr<tREAL4>(aK,aK);
           TplBenchDenseMatr<tREAL8>(aK,aK);
        }
    }

    {
        int aNb = std::min(7,4+aParam.Level());
        for (int aKx=0 ; aKx<aNb ; aKx++)
        {
            for (int aKy=0 ; aKy<aNb ; aKy++)
            {
                TplBenchDenseMatr<tREAL4>(1<<aKx,1<<aKy);
                TplBenchDenseMatr<tREAL8>(1<<aKx,1<<aKy);
            }
        }
    }

    TplBenchDenseMatr<tREAL4>(1,1);
    TplBenchDenseMatr<tREAL4>(2,2);
    TplBenchDenseMatr<tREAL4>(4,4);
    TplBenchDenseMatr<tREAL4>(8,8);
    TplBenchDenseMatr<tREAL4>(16,16);
    TplBenchDenseMatr<tREAL4>(32,32);
    TplBenchDenseMatr<tREAL4>(64,64);

    TplBenchDenseMatr<tREAL8>(4,6);
    TplBenchDenseMatr<tREAL4>(50,50);
    TplBenchDenseMatr<tREAL8>(40,40);
    TplBenchDenseMatr<tREAL16>(35,35);

    aParam.EndBench();
}

};
