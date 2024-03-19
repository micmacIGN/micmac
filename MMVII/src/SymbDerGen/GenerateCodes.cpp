#include "ComonHeaderSymb.h"
#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_GenNameAlloc.h"
#include "Formulas_ImagesDeform.h"
#include "Formulas_TrianglesDeform.h"
#include "Formulas_CamStenope.h"
#include "Formulas_Geom2D.h"
#include "Formulas_Radiom.h"
#include "Formulas_Geom3D.h"
#include "Formulas_BlockRigid.h"
#include "Formulas_GenSensor.h"
#include "Formulas_RPC.h"
#include "Formulas_Topo.h"
#include "MMVII_Sys.h"
#include "MMVII_Geom2D.h"

#include "MMVII_PCSens.h"
#include "MMVII_2Include_Serial_Tpl.h"


      //  cPt3di  Deg.x=Rad  Deg.y=Dec  Deg.z=Gen
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



  /****************************************************/
  /****************************************************/
  /****************************************************/
  /****************************************************/


extern const std::vector<cPt3di>  TheVectDegree;

std::vector<cDescOneFuncDist>   DescDist(const cPt3di & aDeg,bool isFraserMode)
{
   cMMVIIUnivDist  aDist(aDeg.x(),aDeg.y(),aDeg.z(),false,isFraserMode);

   return aDist.VDescParams();
}

     //   PUSHB
std::vector<cDescOneFuncDist>   Polyn2DDescDist(int aDegree)
{
   cDistPolyn2D aDist(aDegree,false,true) ;
   return aDist.mVDesc;
}


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
std::string  NameEqDist(const cPt3di & aDeg,bool WithDerive,bool ForBase,bool isFraserMode)
{
   cMMVIIUnivDist aDist(aDeg.x(),aDeg.y(),aDeg.z(),ForBase,isFraserMode);
   cEqDist<cMMVIIUnivDist> anEq(aDist); 

   return NameFormula(anEq,WithDerive);
}

template <typename tProj> std::string Tpl_NameEqColinearityCamPPC(eProjPC  aType,const cPt3di & aDeg,bool WithDerive,bool isFraserMode)
{
   MMVII_INTERNAL_ASSERT_tiny(tProj::TypeProj()==aType,"incoherence in Tpl_NameEqProjCam");

   cMMVIIUnivDist aDist(aDeg.x(),aDeg.y(),aDeg.z(),false,isFraserMode);
   cEqColinearityCamPPC<cMMVIIUnivDist,tProj>  anEq(aDist);

   return NameFormula(anEq,WithDerive);
}

std::string NameEqColinearityCamPPC(eProjPC  aType,const cPt3di & aDeg,bool WithDerive,bool isFraserMode)
{
    switch (aType)
    {
        case eProjPC::eStenope        :   return Tpl_NameEqColinearityCamPPC<cProjStenope>       (aType,aDeg,WithDerive,isFraserMode);
        case eProjPC::eFE_EquiDist    :   return Tpl_NameEqColinearityCamPPC<cProjFE_EquiDist>   (aType,aDeg,WithDerive,isFraserMode);
        case eProjPC::eFE_EquiSolid   :   return Tpl_NameEqColinearityCamPPC<cProjFE_EquiSolid>  (aType,aDeg,WithDerive,isFraserMode);
        case eProjPC::eStereroGraphik :   return Tpl_NameEqColinearityCamPPC<cProjStereroGraphik>(aType,aDeg,WithDerive,isFraserMode);
        case eProjPC::eOrthoGraphik   :   return Tpl_NameEqColinearityCamPPC<cProjOrthoGraphic>  (aType,aDeg,WithDerive,isFraserMode);
        case eProjPC::eEquiRect       :   return Tpl_NameEqColinearityCamPPC<cProj_EquiRect>     (aType,aDeg,WithDerive,isFraserMode);

        default :;

    }    ;

    MMVII_INTERNAL_ERROR("Unhandled proj in NameEqProjCam");
    return "";
}


/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

cCalculator<double> *  StdAllocCalc(const std::string & aName,int aSzBuf,bool SVP=false,bool ReUse=false)
{
    if (aSzBuf<=0)
       aSzBuf =  cMMVII_Appli::CurrentAppli().NbProcAllowed();

    if (ReUse)
    {
        static std::map<std::string,cCalculator<double> * >  TheMapS2C;
        if (TheMapS2C.find(aName) == TheMapS2C.end())
        {
            cCalculator<double> * aResult =  cName2Calc<double>::CalcFromName(aName,aSzBuf,SVP);
            TheMapS2C[aName] = aResult;
            cMMVII_Appli::AddObj2DelAtEnd(aResult);
        }
	return TheMapS2C[aName];
    }

    return  cName2Calc<double>::CalcFromName(aName,aSzBuf,SVP);
}


     //=============   Photogrammetry ============

void TestResDegree(cCalculator<double> * aCalc,const cPt3di & aDeg,const std::string & aFonc)
{
     if (aCalc==nullptr)
     {
         StdOut() << " *  Generated Degree Are " <<   TheVectDegree << std::endl;
	 MMVII_UsersErrror
         (
	      eTyUEr::eBadDegreeDist,
	      "Required degree " + ToStr(aDeg) + " for distorsion  in "+aFonc+" has not been generated"
         );
     }
}

     //   PUSHB
NS_SymbolicDerivative::cCalculator<double> * EqColinearityCamGen(int  aDeg,bool WithDerive,int aSzBuf,bool ReUse)
{
     bool SVP =  false; // we generate an error if dont exist
     return StdAllocCalc(NameFormula(cEqColinSensGenPolyn2D(aDeg,false),WithDerive),aSzBuf,SVP,ReUse);
}

NS_SymbolicDerivative::cCalculator<double> * EqDistPol2D(int  aDeg,bool WithDerive,int aSzBuf,bool ReUse) // PUSHB
{
     bool SVP =  false; // we generate an error if dont exist
     return StdAllocCalc(NameFormula(cEqDistPolyn2D(aDeg,false),WithDerive),aSzBuf,SVP,ReUse);
}

NS_SymbolicDerivative::cCalculator<double> * RPC_Proj(bool WithDerive,int aSzBuf,bool ReUse) // PUSHB
{
     bool SVP =  false; // we generate an error if dont exist
     return StdAllocCalc(NameFormula(cFormula_RPC_RatioPolyn(),WithDerive),aSzBuf,SVP,ReUse);
}
											     


     //  distorion
cCalculator<double> * EqDist(const cPt3di & aDeg,bool WithDerive,int aSzBuf,bool isFraserMode)
{ 
    cCalculator<double> * aRes =  StdAllocCalc(NameEqDist(aDeg,WithDerive,false,isFraserMode),aSzBuf,true);
    TestResDegree(aRes,aDeg,"EqDist");
    return aRes;
}
cCalculator<double> * EqBaseFuncDist(const cPt3di & aDeg,int aSzBuf,bool isFraserMode)
{ 
    return StdAllocCalc(NameEqDist(aDeg,false,true,isFraserMode),aSzBuf);
}

     //  Projection
cCalculator<double> * EqCPProjDir(eProjPC  aType,bool WithDerive,int aSzBuf)
{ 
    return StdAllocCalc(NameFormulaOfStr(FormulaName_ProjDir(aType),WithDerive),aSzBuf);
}

cCalculator<double> * EqCPProjInv(eProjPC  aType,bool WithDerive,int aSzBuf)
{ 
    return StdAllocCalc(NameFormulaOfStr(FormulaName_ProjInv(aType),WithDerive),aSzBuf);
}

     //  Projection+distorsion+ Foc/PP

cCalculator<double> * EqColinearityCamPPC(eProjPC  aType,const cPt3di & aDeg,bool WithDerive,int aSzBuf,bool ReUse,bool isFraserMode)
{
	//  true->  SVP
     cCalculator<double> * aRes = StdAllocCalc(NameEqColinearityCamPPC(aType,aDeg,WithDerive,isFraserMode),aSzBuf,true,ReUse);

    TestResDegree(aRes,aDeg,"EqColinearityCamPPC");
    /*
     if (aRes==nullptr)
     {
         StdOut() << " *  Generated Degree Are " <<   TheVectDegree << std::endl;
	 MMVII_UsersErrror
         (
	      eTyUEr::eBadDegreeDist,
	      "Required degree for distorsion  EqColinearityCamPPC has not been generated"
         );
     }
     */

     return aRes;
}

     //    Radiometry

cCalculator<double> * EqRadiomCalibRadSensor(int aNbDeg,bool WithDerive,int aSzBuf,bool WithCste,int aDegPolSens)
{ 
    return StdAllocCalc(NameFormula(cRadiomCalibRadSensor(aNbDeg,WithCste,aDegPolSens),WithDerive),aSzBuf);
}

cCalculator<double> * EqRadiomCalibPolIma(int aNbDeg,bool WithDerive,int aSzBuf)
{ 
    return StdAllocCalc(NameFormula(cRadiomCalibPolIma(aNbDeg),WithDerive),aSzBuf);
}
cCalculator<double> * EqRadiomEqualisation(int aDegSens,int aDegIm,bool WithDerive,int aSzBuf,bool WithCste,int aDegPolSens)
{ 
    return StdAllocCalc(NameFormula(cRadiomEqualisation(true,aDegSens,aDegIm,WithCste,aDegPolSens),WithDerive),aSzBuf);
}

cCalculator<double> * EqRadiomStabilization(int aDegSens,int aDegIm,bool WithDerive,int aSzBuf,bool WithCste,int aDegPolSens)
{ 
    return StdAllocCalc(NameFormula(cRadiomEqualisation(false,aDegSens,aDegIm,WithCste,aDegPolSens),WithDerive),aSzBuf);
}



const std::vector<cDescOneFuncDist> & VDesc_RadiomCPI(int aDegree,int aDRadElim)
{
    static std::map<std::pair<int,int>,std::vector<cDescOneFuncDist>>  aDico;
    std::pair<int,int> aKey(aDegree,aDRadElim);

    if (aDico.find(aKey) == aDico.end())
       aDico[aKey] = cRadiomCalibPolIma(aDegree,aDRadElim).VDesc();

    return aDico[aKey];
}

      // To delete soon
      cCalculator<double> * EqRadiomVignettageLinear(int aNbDeg,bool WithDerive,int aSzBuf)
      { 
          return StdAllocCalc(NameFormula(cRadiomVignettageLinear(aNbDeg),WithDerive),aSzBuf);
      }


     //=============   Tuto/Bench/Network ============

     //    Cons distance
template <class Type> cCalculator<Type> * TplEqConsDist(bool WithDerive,int aSzBuf)
{ 
    return StdAllocCalc(NameFormula(cDist2DConservation(),WithDerive),aSzBuf);
}

cCalculator<double> * EqConsDist(bool WithDerive,int aSzBuf)
{ 
    return TplEqConsDist<double>(WithDerive,aSzBuf);
}

     //  cons ratio dist
cCalculator<double> * EqConsRatioDist(bool WithDerive,int aSzBuf)
{ 
    return StdAllocCalc(NameFormula(cRatioDist2DConservation(),WithDerive),aSzBuf);
}

     //  Network for points  
cCalculator<double> * EqNetworkConsDistProgCov(bool WithDerive,int aSzBuf,const cPt2di& aSzN)
{ 
    return StdAllocCalc(NameFormula(cNetworConsDistProgCov(aSzN),WithDerive),aSzBuf);
}

cCalculator<double> * EqNetworkConsDistFixPoints(bool WithDerive,int aSzBuf,const cPt2di& aSzN,bool WithSimUK)
{ 
    return StdAllocCalc(NameFormula(cNetWConsDistSetPts(aSzN,WithSimUK),WithDerive),aSzBuf);
}

cCalculator<double> * EqNetworkConsDistFixPoints(bool WithDerive,int aSzBuf,int aNbPts)
{ 
    return StdAllocCalc(NameFormula(cNetWConsDistSetPts(aNbPts,true),WithDerive),aSzBuf);
}

cCalculator<double> * EqDeformImHomotethy(bool WithDerive,int aSzBuf)
{
     return StdAllocCalc(NameFormula(cDeformImHomotethy(),WithDerive),aSzBuf);
}

cCalculator<double> * EqDeformImAffinity(bool WithDerive,int aSzBuf)
{
     return StdAllocCalc(NameFormula(cDeformImAffinity(),WithDerive),aSzBuf);
}

cCalculator<double> *EqDeformTri(bool WithDerive, int aSzBuf)
{
    return StdAllocCalc(NameFormula(cTriangleDeformation(), WithDerive), aSzBuf);
}

cCalculator<double> *EqDeformTriTranslation(bool WithDerive, int aSzBuf)
{
    return StdAllocCalc(NameFormula(cTriangleDeformationTranslation(), WithDerive), aSzBuf);
}

cCalculator<double> *EqDeformTriRadiometry(bool WithDerive, int aSzBuf)
{
    return StdAllocCalc(NameFormula(cTriangleDeformationRadiometry(), WithDerive), aSzBuf);
}

// dist3d
//    Cons distance
template <class Type> cCalculator<Type> * TplEqDist3D(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cDist3D(),WithDerive),aSzBuf);
}

cCalculator<double> * EqDist3D(bool WithDerive,int aSzBuf)
{
    return TplEqDist3D<double>(WithDerive,aSzBuf);
}

// dist3d with dist parameter
//    Cons distance
template <class Type> cCalculator<Type> * TplEqDist3DParam(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cDist3DParam(),WithDerive),aSzBuf);
}

cCalculator<double> * EqDist3DParam(bool WithDerive,int aSzBuf)
{
    return TplEqDist3DParam<double>(WithDerive,aSzBuf);
}

cCalculator<double> * EqBlocRig(bool WithDerive,int aSzBuf,bool ReUse)  // RIGIDBLOC
{
    return StdAllocCalc(NameFormula(cFormulaBlocRigid(),WithDerive),aSzBuf,false,ReUse);
}

cCalculator<double> * EqBlocRig_RatE(bool WithDerive,int aSzBuf,bool ReUse)  // RIGIDBLOC
{
    return StdAllocCalc(NameFormula(cFormulaRattBRExist(),WithDerive),aSzBuf,false,ReUse);
}

// topo subframe with dist parameter
template <class Type> cCalculator<Type> * TplEqTopoSubFrame(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cTopoSubFrame(),WithDerive),aSzBuf);
}

cCalculator<double> * EqTopoSubFrame(bool WithDerive,int aSzBuf)
{
    return TplEqTopoSubFrame<double>(WithDerive,aSzBuf);
}

// topo az
template <class Type> cCalculator<Type> * TplEqTopoAz(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cFormulaTopoHz(),WithDerive),aSzBuf);
}

cCalculator<double> * EqTopoHz(bool WithDerive,int aSzBuf)
{
    return TplEqTopoAz<double>(WithDerive,aSzBuf);
}

// topo zen
template <class Type> cCalculator<Type> * TplEqTopoZen(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cFormulaTopoZen(),WithDerive),aSzBuf);
}

cCalculator<double> * EqTopoZen(bool WithDerive,int aSzBuf)
{
    return TplEqTopoZen<double>(WithDerive,aSzBuf);
}

// topo dist
template <class Type> cCalculator<Type> * TplEqTopoDist(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cFormulaTopoDist(),WithDerive),aSzBuf);
}

cCalculator<double> * EqTopoDist(bool WithDerive,int aSzBuf)
{
    return TplEqTopoDist<double>(WithDerive,aSzBuf);
}

// topo dX
template <class Type> cCalculator<Type> * TplEqTopoDX(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cFormulaTopoDX(),WithDerive),aSzBuf);
}

cCalculator<double> * EqTopoDX(bool WithDerive,int aSzBuf)
{
    return TplEqTopoDX<double>(WithDerive,aSzBuf);
}

// topo dY
template <class Type> cCalculator<Type> * TplEqTopoDY(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cFormulaTopoDY(),WithDerive),aSzBuf);
}

cCalculator<double> * EqTopoDY(bool WithDerive,int aSzBuf)
{
    return TplEqTopoDY<double>(WithDerive,aSzBuf);
}

// topo dZ
template <class Type> cCalculator<Type> * TplEqTopoDZ(bool WithDerive,int aSzBuf)
{
    return StdAllocCalc(NameFormula(cFormulaTopoDZ(),WithDerive),aSzBuf);
}

cCalculator<double> * EqTopoDZ(bool WithDerive,int aSzBuf)
{
    return TplEqTopoDZ<double>(WithDerive,aSzBuf);
}



cCalculator<double> * EqSumSquare(int aNb,bool WithDerive,int aSzBuf,bool ReUse)
{
    return StdAllocCalc(NameFormula(cFormulaSumSquares(8),WithDerive),aSzBuf,false,ReUse);
}

/* **************************** */
/*      BENCH  PART             */
/* **************************** */


typedef std::pair<cPt2dr,cPt3dr>  tPair23;

/** Generate a pair P2/P3 mutually homologous and in validity domain for proj */
template<class TyProj> tPair23  GenerateRandPair4Proj()
{
   const cDefProjPerspC * aProj = cDefProjPerspC::ProjOfType(TyProj::TypeProj());
   tPair23 aRes(cPt2dr(0,0), cPt3dr::PRandUnitDiff(cPt3dr(0,0,0),1e-3));
   while (aProj->P3DIsDef(aRes.second)<1e-5)
       aRes.second =  cPt3dr::PRandUnitDiff(cPt3dr(0,0,0),1e-3);
   cHelperProj<TyProj> aPropPt;
   aRes.first =  aPropPt.Proj(aRes.second);
   aRes.second =  aPropPt.ToDirBundle(aRes.first);

   delete aProj;

   return aRes;
}

template<class TyProj> void OneBenchProjToDirBundle(cParamExeBench & aParam)
{
   cHelperProj<TyProj> aPropPt;
   const cDefProjPerspC *    aDefProf = cDefProjPerspC::ProjOfType(TyProj::TypeProj());
   // Just to force compile with these tricky classes
   if (NeverHappens())
   {
       std::vector<double> aV;
       std::vector<cFormula<double>> aVF;
       cPt2dr aP(0,0);

       TyProj::Proj(aV);
       TyProj::Proj(aVF);
       TyProj::ToDirBundle(aV);

       aPropPt.ToDirBundle(aP);
       aPropPt.Proj(aPropPt.ToDirBundle(aP));
   }
   // Generate random point aPt0, project aVIm0, inverse aPt1, and check collinearity between Pt1 and Pt0
   cPt3dr AxeK(0,0,1);
   for (int aK=0 ; aK<10000 ; )
   {
       {
          tPair23  aP23 = GenerateRandPair4Proj<TyProj>();

          // 1- test inversion
          cPt2dr aProj2 =  aPropPt.Proj(aP23.second);
          cPt3dr aRay3d =  aPropPt.ToDirBundle(aP23.first);
   
          MMVII_INTERNAL_ASSERT_bench(Norm2(aP23.second-aRay3d)<1e-8,"Inversion Proj/ToDirBundle");

	  if ( aDefProf->HasRadialSym()) // 2- test radiality  => to skeep for non physical proj like 360 synthetic image
	  {
          // 2.1  , conservation of angles :  aRay2, aRay3d, AxeK  must be coplanar
              cPt3dr aRay2(aProj2.x(),aProj2.y(),1.0);
	      double aDet =  Scal(AxeK,aRay2^aRay3d) ;
              MMVII_INTERNAL_ASSERT_bench(std::abs(aDet)<1e-8,"Proj/ToDirBundle");

          // 2.2- test radiality  , conservation of distance , image of circle is a cylinder

              cPt2dr aQ2 =  aProj2 * FromPolar(1.0,RandUnif_C()*10);
              cPt3dr aQ3 =  aPropPt.ToDirBundle(aQ2);
	      double aDif = Norm2(AxeK-aQ3) - Norm2(AxeK-aRay3d);
              MMVII_INTERNAL_ASSERT_bench(std::abs(aDif)<1e-8,"Proj/ToDirBundle");
	  }

          aK++;
       }

   }
   std::vector<double> aV00{0,0};
   cPt3dr aPtZ = cPt3dr::FromStdVector(TyProj::ToDirBundle(aV00));
   MMVII_INTERNAL_ASSERT_bench(Norm2(aPtZ-AxeK)<1e-8,"Proj/ToDirBundle");

   if  (1)  // to skeep if code not generated ...
   {
	cDataMapCalcSymbDer<tREAL8,3,2> aProjDir
        (
            EqCPProjDir(TyProj::TypeProj(),false,10),
            EqCPProjDir(TyProj::TypeProj(),true,10),
            std::vector<tREAL8>(),
            true
        );
	cDataMapCalcSymbDer<tREAL8,2,3> aProjInv
        (
            EqCPProjInv(TyProj::TypeProj(),false,10),
            EqCPProjInv(TyProj::TypeProj(),true,10),
            std::vector<tREAL8>(),
            true
        );
        for (int aK=0 ; aK<10000 ; )
        {
            size_t aNb = 1+ RandUnif_N(100);
	    std::vector<cPt2dr>  aVP2;  // input for projinv
	    std::vector<cPt3dr>  aVP3;  // out put forproj

	    while (aVP2.size() < aNb)
	    {
                tPair23  aP23 = GenerateRandPair4Proj<TyProj>();
		aVP2.push_back(aP23.first);
		aVP3.push_back(aP23.second);
	    }
	    // static cast because Values(POut,PIn) is defined in class and hides definition in upper class ..
	    const std::vector<cPt2dr> & aVQ2 = static_cast<cDataMapping<tREAL8,3,2>&>(aProjDir).Values(aVP3);
	    const std::vector<cPt3dr> & aVQ3 = static_cast<cDataMapping<tREAL8,2,3>&>(aProjInv).Values(aVP2);
	    for (size_t aKp=0 ;  aKp< aNb ; aKp++)
	    {
                MMVII_INTERNAL_ASSERT_bench(Norm2(aVP2[aKp]-aVQ2[aKp])<1e-8,"Proj/ToDirBundle");
                MMVII_INTERNAL_ASSERT_bench(Norm2(aVP3[aKp]-aVQ3[aKp])<1e-8,"Proj/ToDirBundle");
	    }
	    aK += aNb;
        }
   }


   if (aParam.Show())
   {
      StdOut() << "NAME=" << E2Str(TyProj::TypeProj()) 
	       << " formula(test) =" << NameEqColinearityCamPPC(TyProj::TypeProj(),cPt3di(3,1,1),false,true)
	       << "\n";
     
   // std::string NameEqProjCam(eProjPC  aType,const cPt3di & aDeg,bool WithDerive)
   }

   delete aDefProf;
}

void BenchProjToDirBundle(cParamExeBench & aParam)
{
   if (aParam.Show())
   {
       StdOut()<<"cName2Calc T0:"<<StdAllocCalc("toto",10,true)<<std::endl;
       StdOut()<<"cName2Calc T1:"<<StdAllocCalc("EqDistDist_Rad3_Dec1_XY1",10,true)<<std::endl;
   }

   OneBenchProjToDirBundle<cProjStenope> (aParam);
   OneBenchProjToDirBundle<cProjFE_EquiDist> (aParam);
   OneBenchProjToDirBundle<cProjStereroGraphik> (aParam);
   OneBenchProjToDirBundle<cProjOrthoGraphic> (aParam);
   OneBenchProjToDirBundle<cProjFE_EquiSolid> (aParam);
   OneBenchProjToDirBundle<cProj_EquiRect> (aParam);
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
        void GenerateOneDist(const cPt3di & aDeg,bool isFraserMode) ;
        template <typename tProj> void GenerateCodeProjCentralPersp();
        template <typename tProj> void GenerateCodeCamPerpCentrale(const cPt3di &,bool IsFraserMode);


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
   aCEq.SetHeaderIncludeSymbDer("SymbDer/SymbDer_Common.h");
   aCEq.SetDirGenCode(mDirGenCode);

   auto aXY= aFormula.formula(aCEq.VUk(),aCEq.VObs()); // Give ths list of atomic formula
   if (WithDerive)
      aCEq.SetCurFormulasWithDerivative(aXY);
   else
      aCEq.SetCurFormulas(aXY);
   auto [aClassName,aFileName] = aCEq.GenerateCode("CodeGen_");
   cGenNameAlloc::Add(aClassName,aFileName);
};

void cAppliGenCode::GenerateOneDist(const cPt3di & aDeg,bool isFraserMode) 
{
   cMMVIIUnivDist           aDist(aDeg.x(),aDeg.y(),aDeg.z(),false,isFraserMode);
   cEqDist<cMMVIIUnivDist>  anEqDist(aDist);  // Distorsion function 2D->2D


   GenCodesFormula((tREAL8*)nullptr,anEqDist,false);  //  Dist without derivative
   GenCodesFormula((tREAL8*)nullptr,anEqDist,true);   //  Dist with derivative
   // GenCodesFormula((tREAL8*)nullptr,anEqIntr,false);  //  Proj without derivative
   // GenCodesFormula((tREAL8*)nullptr,anEqIntr,true);   //  Proj with derivative

   // Generate the base of all functions
   cMMVIIUnivDist           aDistBase(aDeg.x(),aDeg.y(),aDeg.z(),true,isFraserMode);
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

template <typename tProj> void cAppliGenCode::GenerateCodeCamPerpCentrale(const cPt3di & aDeg,bool isFraserMode)
{
   for (const auto WithDer : {true,false})
   {
       cMMVIIUnivDist aDist(aDeg.x(),aDeg.y(),aDeg.z(),false,isFraserMode);  // Distorsion function 2D->2D
       cEqColinearityCamPPC<cMMVIIUnivDist,tProj>  anEq(aDist);
       GenCodesFormula((tREAL8*)nullptr,anEq,WithDer);
   }
}

void AddData(const cAuxAr2007 & anAux,cDataPerspCamIntrCalib &);

void GenerateXMLSpec()
{
   {
      std::string aName = PrefixSpecifXML +"TTTT.xml";
      // cPerspCamIntrCalib * cPerspCamIntrCalib::FromFile(const std::string &);
      cDataPerspCamIntrCalib aDPC;
      SaveInFile (aDPC,aName);
   }
}

const std::vector<cPt3di>  
      TheVectDegree
      {
			   {0,0,0},  // no dist at all
			   {0,0,1},  // pure linear as used in 11 Param
			   {2,0,0},
			   {3,0,0},
			   {3,1,1},
			   {5,1,1},
			   {5,1,2},
			   {5,2,2},
			   {7,2,5}
      };

const std::vector<cPt3di>  
      TheVectDegreeNoFraser
      {
			   {0,0,1},  // pure linear as used in 11 Param
			   {3,1,1},
			   {5,1,1},
			   {2,0,0}
      };


int cAppliGenCode::Exe()
{
   if (IsInit(&mTypeProj))
   {
       // will process later ...
       return EXIT_SUCCESS;
   }
   cGenNameAlloc::Reset();
   mDirGenCode = TopDirMMVII() + "src/GeneratedCodes/";

   // ================  CODE FOR PHOTOGRAMMETRY =====================

        // ---   Colinearity for stantard camera -----------------
   for (const auto & aDeg :  TheVectDegree)
   {
       GenerateOneDist(aDeg,true);
       GenerateCodeCamPerpCentrale<cProjStenope>(aDeg,true);
   }

       //  ---  Here we generate the degree for SIA-cylindric systematisms -----------------
   
   for (const auto & aDegSIA :  TheVectDegreeNoFraser)
   {
       GenerateOneDist(aDegSIA,false);
       GenerateCodeCamPerpCentrale<cProjStenope>(aDegSIA,false);
   }

   GenerateCodeCamPerpCentrale<cProjFE_EquiDist>(cPt3di(3,1,1),true);

   for (const auto WithDer : {true,false})
   {
           GenCodesFormula((tREAL8*)nullptr,cFormula_RPC_RatioPolyn(),WithDer);  
   }


   for (const auto WithDer : {true,false})
   {
       // PUSHB
       std::vector<int>  aVDegEqCol   {0,1,2,3};
       for (const auto & aDegree : aVDegEqCol)
       {
           GenCodesFormula((tREAL8*)nullptr,cEqColinSensGenPolyn2D(aDegree),WithDer);  
           GenCodesFormula((tREAL8*)nullptr,cEqDistPolyn2D(aDegree),WithDer);  
           // StdOut() << "FffffffFFFF " << cEqDistPolyn2D(aDegree).FormulaName() << "\n";
       }
   }

   //=======================   Other code radiom/rigid ....

   for (const auto WithDer : {true,false})
   {
       GenCodesFormula((tREAL8*)nullptr,cFormulaSumSquares(8),WithDer); // example for contraint

       GenCodesFormula((tREAL8*)nullptr,cFormulaBlocRigid(),WithDer); // RIGIDBLOC
       GenCodesFormula((tREAL8*)nullptr,cFormulaRattBRExist(),WithDer); // RIGIDBLOC

       // cDist2DConservation aD2C;
       GenCodesFormula((tREAL8*)nullptr,cDist2DConservation(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cRatioDist2DConservation(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cNetworConsDistProgCov(cPt2di(2,2)),WithDer);
       for (const auto WithSimUk : {true,false})
           GenCodesFormula((tREAL8*)nullptr, cNetWConsDistSetPts(cPt2di(2,2),WithSimUk),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cNetWConsDistSetPts(3,true),WithDer);

       GenCodesFormula((tREAL8*)nullptr,cDist3D(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cDist3DParam(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cTopoSubFrame(),WithDer);

       GenCodesFormula((tREAL8*)nullptr,cFormulaTopoHz(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cFormulaTopoZen(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cFormulaTopoDist(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cFormulaTopoDX(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cFormulaTopoDY(),WithDer);
       GenCodesFormula((tREAL8*)nullptr,cFormulaTopoDZ(),WithDer);

       GenCodesFormula((tREAL8*)nullptr,cDeformImHomotethy()       ,WithDer);

       GenCodesFormula((tREAL8 *)nullptr, cTriangleDeformation(), WithDer);
       GenCodesFormula((tREAL8 *)nullptr, cTriangleDeformationTranslation(), WithDer);
       GenCodesFormula((tREAL8 *)nullptr, cTriangleDeformationRadiometry(), WithDer);


       //  ===============   CODE FOR RADIOMETRY =========================================

       GenCodesFormula((tREAL8*)nullptr,cRadiomVignettageLinear(5)       ,WithDer);
       std::vector<int>  aVDegSens {5};
       std::vector<int>  aVDegIm   {0,1,2,3};

       for (auto  aDegIm : aVDegIm)
       {
           if (!WithDer)  // Generator doesnt like multipe genera : he is quite touchy ...
               GenCodesFormula((tREAL8*)nullptr,cRadiomCalibPolIma(aDegIm)       ,WithDer);
       }
       for (auto  aDegSens : aVDegSens)
       {

           for (const auto WithCste : {true,false})
           {
               for (const auto & aDegIm : aVDegIm)
               {
		       //  true/false => Eq/Stab
                   GenCodesFormula((tREAL8*)nullptr,cRadiomEqualisation(true ,aDegSens,aDegIm,WithCste) ,WithDer);
                   GenCodesFormula((tREAL8*)nullptr,cRadiomEqualisation(false,aDegSens,aDegIm,WithCste) ,WithDer);
                   GenCodesFormula((tREAL8*)nullptr,cRadiomEqualisation(true ,aDegSens,aDegIm,WithCste,2) ,WithDer);
                   GenCodesFormula((tREAL8*)nullptr,cRadiomEqualisation(false,aDegSens,aDegIm,WithCste,2) ,WithDer);
                   GenCodesFormula((tREAL8*)nullptr,cRadiomEqualisation(true ,aDegSens,aDegIm,WithCste,3) ,WithDer);
                   GenCodesFormula((tREAL8*)nullptr,cRadiomEqualisation(false,aDegSens,aDegIm,WithCste,3) ,WithDer);
		   /*
		   */
               }
               if (!WithDer)
               {
                  GenCodesFormula((tREAL8*)nullptr,cRadiomCalibRadSensor(aDegSens,WithCste)       ,WithDer);
                  GenCodesFormula((tREAL8*)nullptr,cRadiomCalibRadSensor(aDegSens,WithCste,2)       ,WithDer);
                  GenCodesFormula((tREAL8*)nullptr,cRadiomCalibRadSensor(aDegSens,WithCste,3)       ,WithDer);
	       }
           }
            
/*
           GenCodesFormula((tREAL8*)nullptr,cRadiomCalibPolIma(0)       ,WithDer);
           GenCodesFormula((tREAL8*)nullptr,cRadiomCalibPolIma(1)       ,WithDer);
           GenCodesFormula((tREAL8*)nullptr,cRadiomCalibPolIma(2)       ,WithDer);
*/
       }

       
       GenCodesFormula((tREAL8*)nullptr,cDeformImAffinity()       ,WithDer);
   }

   GenerateCodeProjCentralPersp<cProjStenope>();
   GenerateCodeProjCentralPersp<cProjFE_EquiDist>();
   GenerateCodeProjCentralPersp<cProjStereroGraphik>();
   GenerateCodeProjCentralPersp<cProjOrthoGraphic>();
   GenerateCodeProjCentralPersp<cProjFE_EquiSolid>(); //  ->  asin
   GenerateCodeProjCentralPersp<cProj_EquiRect>(); //  ->  asin


/*
   {
   }

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
   cGenNameAlloc::GenerateFile(mDirGenCode+"cName2CalcRegisterAll.cpp","SymbDer/SymbDer_Common.h","");
*/

   cGenNameAlloc::GenerateFile(mDirGenCode+"cName2CalcRegisterAll.cpp","SymbDer/SymbDer_Common.h","");
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

tMMVII_UnikPApli Alloc_GenCode(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliGenCode(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecGenSymbDer
(
     "GenCodeSymDer",
      Alloc_GenCode,
      "Generation of code for symbolic derivatives",
      {eApF::ManMMVII,eApF::NoGui},
      {eApDT::ToDef},
      {eApDT::ToDef},
      __FILE__
);

};

