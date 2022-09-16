#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_GenNameAlloc.h"
#include "Formulas_ImagesDeform.h"
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
// using namespace MMVII;

namespace MMVII
{

std::string NameFormulaOfStr(const std::string & aName,bool WithDerive)
{
   return  aName +  std::string(WithDerive ?"VDer":"Val");
}
template <typename TypeFormula> std::string NameFormula(const TypeFormula & anEq,bool WithDerive)
{
   return  NameFormulaOfStr(anEq.FormulaName(),WithDerive);
   //return  anEq.FormulaName() +  std::string(WithDerive ?"VDer":"Val");
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
   cHelperProj<TyProj> aProj;
   // Just to force compile with these tricky classes
   if (NeverHappens())
   {
       std::vector<double> aV;
       std::vector<cFormula<double>> aVF;
       cPt2dr aP(0,0);

       TyProj::Proj(aV);
       TyProj::Proj(aVF);
       TyProj::ToDirBundle(aV);

       aProj.ToDirBundle(aP);
       aProj.Proj(aProj.ToDirBundle(aP));
   }
   // Generate random point aPt0, project aVIm0, inverse aPt1, and check collinearity between Pt1 and Pt0
   cPt3dr AxeK(0,0,1);
   for (int aK=0 ; aK<10000 ; )
   {
       cPt3dr aP000(0,0,0);
       cPt3dr aPt3d =  cPt3dr::PRandUnitDiff(aP000);
       if (TyProj::DegreeDef(aPt3d)>1e-5)
       {
          // 1- test inversion
          cPt2dr aProj2 =  aProj.Proj(aPt3d);
          cPt3dr aRay3d =  aProj.ToDirBundle(aProj2);
   
          MMVII_INTERNAL_ASSERT_bench(std::abs(Cos(aPt3d,aRay3d)-1.0)<1e-8,"Inversion Proj/ToDirBundle");

          // 2- test radiality  , conservation of angles :  aRay2, aRay3d, AxeK  must be coplanar
          cPt3dr aRay2(aProj2.x(),aProj2.y(),1.0);
	  double aDet =  Scal(AxeK,aRay2^aRay3d) ;
	     //   StdOut() <<  "DETtt " << aDet << "\n";
          MMVII_INTERNAL_ASSERT_bench(std::abs(aDet)<1e-8,"Proj/ToDirBundle");

          // 3- test radiality  , conservation of distance , image of circle is a cylinder

          cPt2dr aQ2 =  aProj2 * FromPolar(1.0,RandUnif_C()*10);
          cPt3dr aQ3 =  aProj.ToDirBundle(aQ2);
	  double aDif = Norm2(AxeK-aQ3) - Norm2(AxeK-aRay3d);
          MMVII_INTERNAL_ASSERT_bench(std::abs(aDif)<1e-8,"Proj/ToDirBundle");

          aK++;
       }

   }
   std::vector<double> aV00{0,0};
   cPt3dr aPtZ = cPt3dr::FromStdVector(TyProj::ToDirBundle(aV00));
   MMVII_INTERNAL_ASSERT_bench(Norm2(aPtZ-AxeK)<1e-8,"Proj/ToDirBundle");

   if  (1)
   {
        cCalculator<double> *  aCalcDirDev = EqCPProjDir(TyProj::TypeProj(),true,10);
	StdOut()  << " aCalcDirDev " << aCalcDirDev << "\n";

	delete aCalcDirDev;
   }


   if (aParam.Show())
   {
      StdOut() << "NAME=" << E2Str(TyProj::TypeProj()) << "\n";
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

        /// tCompute is a fake paremeter, it'used to force the value
        template <typename tDist,typename tCompute> void GenCodesFormula(tCompute *,const tDist & aDist,bool WithDerive);

       // =========== Data ========
            // Mandatory args
        std::string mDirGenCode;
        void GenerateOneDist(const cPt3di & aDeg) ;
        template <typename tProj> void GenerateCodeProjCentralPersp();

	eProjPC  mTypeProj;
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
      anArgOpt
         << AOpt2007(mTypeProj,"TypeProj","Type of projection for specific generation",{AC_ListVal<eProjPC>()})
      ;
}


template <typename tFormula,typename tCompute> void cAppliGenCode::GenCodesFormula(tCompute *,const tFormula & aFormula,bool WithDerive)
{
   int aSzBuf=1;
   // std::string aNF = anEq.FormulaName() +   std::string(WithDerive ?"VDer":"Val");
   std::string aNF =  NameFormula(aFormula,WithDerive);

   NS_SymbolicDerivative::cCoordinatorF<tCompute> 
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


   GenCodesFormula((tREAL8*)nullptr,anEqDist,false);  //  Dist without derivative
   GenCodesFormula((tREAL8*)nullptr,anEqDist,true);   //  Dist with derivative
   GenCodesFormula((tREAL8*)nullptr,anEqIntr,false);  //  Proj without derivative
   GenCodesFormula((tREAL8*)nullptr,anEqIntr,true);   //  Proj with derivative

   // Generate the base of all functions
   cMMVIIUnivDist           aDistBase(aDeg.x(),aDeg.y(),aDeg.z(),true);
   cEqDist<cMMVIIUnivDist>  anEqBase(aDistBase);
   GenCodesFormula((tREAL8*)nullptr,anEqBase,false);
}

template <typename tProj> void cAppliGenCode::GenerateCodeProjCentralPersp()
{
   for (const auto WithDer : {true,false})
   {
       GenCodesFormula((tREAL8*)nullptr,cGenCode_ProjDir<tProj>(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cGenCode_ProjInv<tProj>(),WithDer);
   }
}


int cAppliGenCode::Exe()
{
   if (IsInit(&mTypeProj))
   {
       // will process later ...
       return EXIT_SUCCESS;
   }
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
       GenCodesFormula((tREAL8*)nullptr,cDist2DConservation(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cRatioDist2DConservation(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cNetworConsDistProgCov(cPt2di(2,2)),WithDer);
       for (const auto WithSimUk : {true,false})
           GenCodesFormula((tREAL8*)nullptr, cNetWConsDistSetPts(cPt2di(2,2),WithSimUk),WithDer);

       GenCodesFormula((tREAL8*)nullptr,cDeformImHomotethy(),WithDer);
   }

   GenerateCodeProjCentralPersp<cProjStenope>();
   GenerateCodeProjCentralPersp<cProjFE_EquiDist>();
   GenerateCodeProjCentralPersp<cProjStereroGraphik>();
   GenerateCodeProjCentralPersp<cProjOrthoGraphic>();
   GenerateCodeProjCentralPersp<cProjFE_EquiSolid>(); //  ->  asin
   // GenCodesFormula((tREAL8*)nullptr,cGenCode_ProjDir<cProjStenope>(),false);

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


cCalculator<double> * EqDist(const cPt3di & aDeg,bool WithDerive,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameEqDist(aDeg,WithDerive,false),aSzBuf);
}

cCalculator<double> * EqBaseFuncDist(const cPt3di & aDeg,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameEqDist(aDeg,false,true),aSzBuf);
}


//    Cons distance
template <class Type> cCalculator<Type> * TplEqConsDist(bool WithDerive,int aSzBuf)
{ 
    return cName2Calc<Type>::CalcFromName(NameFormula(cDist2DConservation(),WithDerive),aSzBuf);
}

cCalculator<double> * EqConsDist(bool WithDerive,int aSzBuf)
{ 
    return TplEqConsDist<double>(WithDerive,aSzBuf);
}

//  cons ratio dist
cCalculator<double> * EqConsRatioDist(bool WithDerive,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameFormula(cRatioDist2DConservation(),WithDerive),aSzBuf);
}

//  Network for points  
cCalculator<double> * EqNetworkConsDistProgCov(bool WithDerive,int aSzBuf,const cPt2di& aSzN)
{ 
    return cName2Calc<double>::CalcFromName(NameFormula(cNetworConsDistProgCov(aSzN),WithDerive),aSzBuf);
}

cCalculator<double> * EqNetworkConsDistFixPoints(bool WithDerive,int aSzBuf,const cPt2di& aSzN,bool WithSimUK)
{ 
    return cName2Calc<double>::CalcFromName(NameFormula(cNetWConsDistSetPts(aSzN,WithSimUK),WithDerive),aSzBuf);
}

cCalculator<double> * EqDeformImHomotethy(bool WithDerive,int aSzBuf)
{
     return cName2Calc<double>::CalcFromName(NameFormula(cDeformImHomotethy(),WithDerive),aSzBuf);
}

//  Projection
cCalculator<double> * EqCPProjDir(eProjPC  aType,bool WithDerive,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameFormulaOfStr(FormulaName_ProjDir(aType),WithDerive),aSzBuf);
}

cCalculator<double> * EqCPProjInv(eProjPC  aType,bool WithDerive,int aSzBuf)
{ 
    return cName2Calc<double>::CalcFromName(NameFormulaOfStr(FormulaName_ProjInv(aType),WithDerive),aSzBuf);
}


std::vector<cDescOneFuncDist>   DescDist(const cPt3di & aDeg)
{
   cMMVIIUnivDist  aDist(aDeg.x(),aDeg.y(),aDeg.z(),false);

   return aDist.VDescParams();
}



cSpecMMVII_Appli  TheSpecGenSymbDer
(
     "GenCodeSymDer",
      Alloc_GenCode,
      "Generation of code for symbolic derivatives",
      {eApF::ManMMVII},
      {eApDT::ToDef},
      {eApDT::ToDef},
      __FILE__
);

};

