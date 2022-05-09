#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_GenNameAlloc.h"
#include "Formulas_CamStenope.h"
#include "Formulas_Geom2D.h"


/*
La compil:

    make distclean
    make -j x
    ./MMVII GenCodeSymDer
    make -j x
    ./MMVII Bench 5

*/

using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace NS_GenerateCode
{

template <typename TypeFormula> std::string NameFormula(const TypeFormula & anEq,bool WithDerive)
{
   return  anEq.FormulaName() +  std::string(WithDerive ?"VDer":"Val");
}

// EqBaseFuncDist
std::string  NameEqDist(const cPt3di & aDeg,bool WithDerive,bool ForBase )
{
   cMMVIIUnivDist aDist(aDeg.x(),aDeg.y(),aDeg.z(),ForBase);
   cEqDist<cMMVIIUnivDist> anEq(aDist); 

   return NameFormula(anEq,WithDerive);
}



/* **************************** */
/*      BENCH  PART             */
/* **************************** */

template<class TyProj> void OneBenchProjToDirBundle(cParamExeBench & aParam)
{
   // Just to force compile with these tricky classes
   if (NeverHappens())
   {
       std::vector<double> aV;
       std::vector<cFormula<double>> aVF;
       cPt2dr aP(0,0);

       TyProj::Proj(aV);
       TyProj::Proj(aVF);
       TyProj::ToDirBundle(aP);
   }
   // Generate random point aPt0, project aVIm0, inverse aPt1, and check collinearity between Pt1 and Pt0
   for (int aK=0 ; aK<10000 ; )
   {
       cPt3dr aP000(0,0,0);
       cPt3dr aPt0 =  cPt3dr::PRandUnitDiff(aP000);
       if (TyProj::DegreeDef(aPt0)>1e-5)
       {

          std::vector<double> aVIm0 =  TyProj::Proj(aPt0.ToStdVector());
          cPt3dr aPt1 =  TyProj::ToDirBundle(cPt2dr::FromStdVector(aVIm0));
          MMVII_INTERNAL_ASSERT_bench(std::abs(Cos(aPt0,aPt1)-1.0)<1e-8,"Proj/ToDirBundle");
          aK++;
       }
   }
   if (aParam.Show())
   {
      StdOut() << "NAME=" << TyProj::NameProj() << "\n";
   }
}

void BenchProjToDirBundle(cParamExeBench & aParam)
{
   if (aParam.Show())
   {
       StdOut()<<"cName2Calc T0:"<<cName2Calc<double>::CalcFromName("toto",10,true)<<"\n";
       StdOut()<<"cName2Calc T1:"<<cName2Calc<double>::CalcFromName("EqDistDist_Rad3_Dec1_XY1",10,true)<<"\n";
   }

   OneBenchProjToDirBundle<cProjStenope> (aParam);
   OneBenchProjToDirBundle<cProjFE_EquiDist> (aParam);
   OneBenchProjToDirBundle<cProjStereroGraphik> (aParam);
   OneBenchProjToDirBundle<cProjOrthoGraphic> (aParam);
   OneBenchProjToDirBundle<cProjFE_EquiSolid> (aParam);
}



class cAppliGenCode ;  // class for main application

/*  ============================================== */
/*                                                 */
/*             cAppliGenCode                       */
/*                                                 */
/*  ============================================== */

class cAppliGenCode : public cMMVII_Appli
{
     public :

       // =========== Declaration ========
                 // --- Method to be a MMVII application
        cAppliGenCode(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        cAppliBenchAnswer BenchAnswer() const override ; ///< Has it a bench, default : no
        int  ExecuteBench(cParamExeBench &) override ;

//      private :
        template <typename tDist> void GenCodesFormula(const tDist & aDist,bool WithDerive);

       // =========== Data ========
            // Mandatory args
        std::string mDirGenCode;
        void GenerateOneDist(const cPt3di & aDeg) ;
};


cAppliGenCode::cAppliGenCode
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli(aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppliGenCode::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   
   return 
      anArgObl  ;
/*
         << Arg2007(mModeMatch,"Matching mode",{AC_ListVal<eModeEpipMatch>()})
         << Arg2007(mNameIm1,"Name Input Image1",{eTA2007::FileImage})
         << Arg2007(mNameIm2,"Name Input Image1",{eTA2007::FileImage})
   ;
*/
}

cCollecSpecArg2007 & cAppliGenCode::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
      anArgOpt;
}


template <typename tFormula> void cAppliGenCode::GenCodesFormula(const tFormula & aFormula,bool WithDerive)
{
   int aSzBuf=1;
   // std::string aNF = anEq.FormulaName() +   std::string(WithDerive ?"VDer":"Val");
   std::string aNF =  NameFormula(aFormula,WithDerive);

   NS_SymbolicDerivative::cCoordinatorF<double> 
   aCEq(aNF,aSzBuf,aFormula.VNamesUnknowns(),aFormula.VNamesObs()); // Gives the liste of names

   // Set header in a place to compilation path of MMVII
   aCEq.SetHeaderIncludeSymbDer("include/SymbDer/SymbDer_Common.h"); 
   aCEq.SetDirGenCode(mDirGenCode);

   auto aXY= aFormula.formula(aCEq.VUk(),aCEq.VObs()); // Give ths list of atomic formula
   if (WithDerive)
      aCEq.SetCurFormulasWithDerivative(aXY);
   else
      aCEq.SetCurFormulas(aXY);
   auto [aClassName,aFileName] = aCEq.GenerateCode("CodeGen_");
   cGenNameAlloc::Add(aClassName,aFileName);
};

void cAppliGenCode::GenerateOneDist(const cPt3di & aDeg) 
{
   cMMVIIUnivDist           aDist(aDeg.x(),aDeg.y(),aDeg.z(),false);
   cEqDist<cMMVIIUnivDist>  anEqDist(aDist);  // Distorsion function 2D->2D
   cEqIntr<cMMVIIUnivDist>  anEqIntr(aDist);  // Projection 3D->2D


   GenCodesFormula(anEqDist,false);  //  Dist without derivative
   GenCodesFormula(anEqDist,true);   //  Dist with derivative
   GenCodesFormula(anEqIntr,false);  //  Proj without derivative
   GenCodesFormula(anEqIntr,true);   //  Proj with derivative

   // Generate the base of all functions
   cMMVIIUnivDist           aDistBase(aDeg.x(),aDeg.y(),aDeg.z(),true);
   cEqDist<cMMVIIUnivDist>  anEqBase(aDistBase);
   GenCodesFormula(anEqBase,false);
}


int cAppliGenCode::Exe()
{
   cGenNameAlloc::Reset();
   mDirGenCode = TopDirMMVII() + "src/GeneratedCodes/";

   {
       GenerateOneDist(cPt3di(3,1,1));
       GenerateOneDist(cPt3di(2,0,0));
       GenerateOneDist(cPt3di(5,1,1));
   }

   for (const auto WithDer : {true,false})
   {
       // cDist2DConservation aD2C;
       GenCodesFormula(cDist2DConservation(),WithDer);
       GenCodesFormula(cRatioDist2DConservation(),WithDer);
   }
/*
   cMMVIIUnivDist           aDist(3,1,1,false);
   cEqDist<cMMVIIUnivDist>  anEqDist(aDist);
   cEqIntr<cMMVIIUnivDist>  anEqIntr(aDist);

   cGenNameAlloc::Reset();

   GenCodesFormula(anEqDist,false);
   GenCodesFormula(anEqDist,true);
   GenCodesFormula(anEqIntr,false);
   GenCodesFormula(anEqIntr,true);
   // GenCodesFormula<cMMVIIUnivDist>(aDist,true,false);

   cMMVIIUnivDist           aDistBase(3,1,1,true);
   cEqDist<cMMVIIUnivDist>  anEqBase(aDistBase);
   GenCodesFormula(anEqBase,false);
   cGenNameAlloc::GenerateFile(mDirGenCode+"cName2CalcRegisterAll.cpp","include/SymbDer/SymbDer_Common.h","");
*/

   cGenNameAlloc::GenerateFile(mDirGenCode+"cName2CalcRegisterAll.cpp","include/SymbDer/SymbDer_Common.h","");
   return EXIT_SUCCESS;
}

cAppliBenchAnswer cAppliGenCode::BenchAnswer() const 
{
   return cAppliBenchAnswer(true,1.0);
}


int  cAppliGenCode::ExecuteBench(cParamExeBench & aParam) 
{
   BenchProjToDirBundle(aParam);
   return EXIT_SUCCESS;
}

/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */


tMMVII_UnikPApli Alloc_GenCode(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{


   return tMMVII_UnikPApli(new cAppliGenCode(aVArgs,aSpec));
}

} // NS_GenerateCode

using namespace NS_GenerateCode;
namespace MMVII
{

cCalculator<double> * EqDist(const cPt3di & aDeg,bool WithDerive,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameEqDist(aDeg,WithDerive,false),aSzBuf);
}

cCalculator<double> * EqBaseFuncDist(const cPt3di & aDeg,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameEqDist(aDeg,false,true),aSzBuf);
}

cCalculator<double> * EqConsDist(bool WithDerive,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameFormula(cDist2DConservation(),WithDerive),aSzBuf);
}

cCalculator<double> * EqConsRatioDist(bool WithDerive,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameFormula(cRatioDist2DConservation(),WithDerive),aSzBuf);
}



std::vector<cDescOneFuncDist>   DescDist(const cPt3di & aDeg)
{
   cMMVIIUnivDist  aDist(aDeg.x(),aDeg.y(),aDeg.z(),false);

   return aDist.VDescParams();
 
}


cSpecMMVII_Appli  TheSpecGenSymbDer
(
     "GenCodeSymDer",
      NS_GenerateCode::Alloc_GenCode,
      "Generation of code for symbolic derivatives",
      {eApF::ManMMVII},
      {eApDT::ToDef},
      {eApDT::ToDef},
      __FILE__
);

};

