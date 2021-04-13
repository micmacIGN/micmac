#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "Formulas_CamStenope.h"

using namespace NS_SymbolicDerivative;
using namespace MMVII;


namespace NS_GenerateCode
{

/* **************************** */
/*      BENCH  PART             */
/* **************************** */

template<class TyProj> void OneBenchProjToDirBundle()
{
   // Just to force compile with these tricky classes
   if (NeverHappens())
   {
       std::vector<double> aV;
       std::vector<cFormula<double>> aVF;
       cPt2dr aP;

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
   std::cout << "NAME=" << TyProj::NameProj() << "\n";
}

void BenchProjToDirBundle()
{
   if (true)
   {
     std::cout << "T0:" << cName2Calc<double>::CalcFromName("toto",10,true) << "\n";
     std::cout << "T1:" << cName2Calc<double>::CalcFromName("EqDistDist_Rad3_Dec1_XY1",10,true) << "\n";
   }

   OneBenchProjToDirBundle<cProjStenope> ();
   OneBenchProjToDirBundle<cProjFE_EquiDist> ();
   OneBenchProjToDirBundle<cProjStereroGraphik> ();
   OneBenchProjToDirBundle<cProjOrthoGraphic> ();
   OneBenchProjToDirBundle<cProjFE_EquiSolid> ();
}



class cAppli ;  // class for main application

/*  ============================================== */
/*                                                 */
/*             cAppli                              */
/*                                                 */
/*  ============================================== */

class cAppli : public cMMVII_Appli
{
     public :

       // =========== Declaration ========
                 // --- Method to be a MMVII application
        cAppli(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        cAppliBenchAnswer BenchAnswer() const override ; ///< Has it a bench, default : no
        int  ExecuteBench(cParamExeBench &) override ;

     private :
        template <typename tDist> void GenCodesDist(const tDist & aDist,bool WithDerive,bool PP);

       // =========== Data ========
            // Mandatory args
        std::string mDirGenCode;
};


/*  ============================================== */
/*                                                 */
/*              cAppli                             */
/*                                                 */
/*  ============================================== */

cAppli::cAppli
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli(aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppli::ArgObl(cCollecSpecArg2007 & anArgObl)
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

cCollecSpecArg2007 & cAppli::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
      anArgOpt;
}



template <typename tDist> void cAppli::GenCodesDist(const tDist & aDist,bool WithDerive,bool PP)
{

   int SzBuf=0;
   cEqDist<tDist> anEq(aDist); 

   NS_SymbolicDerivative::cCoordinatorF<double> 
   aCEq(anEq.FormulaName(),SzBuf,anEq.VNamesUnknowns(),anEq.VNamesObs());

   // Set header in a place to compilation path of MMVII
   aCEq.SetHeaderIncludeSymbDer("include/SymbDer/SymbDer_Common.h"); 
   aCEq.SetUseAllocByName(true);
   aCEq.SetDirGenCode(mDirGenCode);

   auto aXY= anEq.formula(aCEq.VUk(),aCEq.VObs());
   if (WithDerive)
      aCEq.SetCurFormulasWithDerivative(aXY);
   else
      aCEq.SetCurFormulas(aXY);
   aCEq.GenerateCode("CodeGen" + std::string(WithDerive ?"VDer":"Val")+"_");
};




int cAppli::Exe()
{
   mDirGenCode = TopDirMMVII() + "src/GeneratedCodes/";
   cMMVIIUnivDist aDist(3,1,1);

   GenCodesDist<cMMVIIUnivDist>(aDist,false,false);
   GenCodesDist<cMMVIIUnivDist>(aDist,true,false);
   return EXIT_SUCCESS;
}

cAppliBenchAnswer cAppli::BenchAnswer() const 
{
   return cAppliBenchAnswer(true,1.0);
}

int  cAppli::ExecuteBench(cParamExeBench & aParam) 
{
   BenchProjToDirBundle();
   return EXIT_SUCCESS;
}

/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */


tMMVII_UnikPApli Alloc_GenCode(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{


   return tMMVII_UnikPApli(new cAppli(aVArgs,aSpec));
}

} // NS_GenerateCode
namespace MMVII
{


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

