#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_MACRO.h"
#include "include/MMVII_TplSymbImage.h"
#include "Formulas_CamStenope.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

std::vector<std::string>  NamesP3(const std::string& aPref) {return {aPref+"_x",aPref+"_y",aPref+"_z"};}
std::vector<std::string>  NamesP2(const std::string& aPref) {return {aPref+"_x",aPref+"_y"};}
std::vector<std::string>  NamesMatr(const std::string& aPref,const cPt2di & aSz)
{
     std::vector<std::string> aRes;
     for (int aY=0 ; aY<aSz.y() ; aY++)
         for (int aX=0 ; aX<aSz.x() ; aX++)
             aRes.push_back(aPref+"_"+ToStr(aX)+"_"+ToStr(aY));

     return aRes;
}

std::vector<std::string>  NamesPose(const std::string& aNameC ,const std::string&  aNameOmega)
{
	return Append(NamesP3(aNameC),NamesP3(aNameOmega));
}

std::vector<std::string>  NamesIntr(const std::string& aPref)
{
	return {"F_"+aPref,"PPx_"+aPref,"PPy_"+aPref};
}



std::string FormulaName_ProjDir(eProjPC aProj) {return  "Dir_" + E2Str(aProj);}
std::string FormulaName_ProjInv(eProjPC aProj) {return  "Inv_" + E2Str(aProj);}

std::vector<std::string> FormalBilinIm2D_NameObs(const std::string & aPrefix)
{
   return std::vector<std::string> 
          {
              "PtX0_" + aPrefix,
              "PtY0_" + aPrefix,
              "Im00_" + aPrefix,
              "Im10_" + aPrefix,
              "Im01_" + aPrefix,
              "Im11_" + aPrefix
          };
}

/* ******************************** */
/*                                  */
/*         cDescOneFuncDist         */
/*                                  */
/* ******************************** */



cDescOneFuncDist::cDescOneFuncDist(eTypeFuncDist aType,const cPt2di aDegXY) :
   mType    (aType),
   mDegMon  (-1,-1),
   mNum     (-1)
{
   int aNum = aDegXY.x();
   if (mType==eTypeFuncDist::eRad)
   {
       mName = "K" + ToStr(aNum);  //  K1 K2 K3 ...
       mDegTot = 1 + 2 * (aNum);   //  X (X^2+Y^2) ^N
       mNum = aNum;
   }
   else if ((mType==eTypeFuncDist::eDecX) ||  (mType==eTypeFuncDist::eDecY))
   {
       int aDec = (mType==eTypeFuncDist::eDecX)?-1:0;
       // p1,p2  as usual, and by generalization p3 p4 p5 ...
       mName = "p" + ToStr(2*aNum+aDec);
       mDegTot =  2 * (aNum);  // Derivates of X (X^2+Y^2) ^N
       mNum = aNum;
   }
   else
   {
      mDegMon = aDegXY;
      mDegTot = mDegMon.x() + mDegMon.y();
      if ((mType==eTypeFuncDist::eMonX) && (mDegTot==1))
      {
          mName = ( mDegMon.x() == 1) ? "b1" : "b2";  // Usual convention
      }
      else
      {
          mName =  std::string((mType==eTypeFuncDist::eMonX) ? "x" : "y")
                 + "_" + ToStr(mDegMon.x())
                 + "_" + ToStr(mDegMon.y()) ;
      }
   }
}

double cDescOneFuncDist::MajNormJacOfRho(double aRho) const
{
   switch(mType)
   {
       case eTypeFuncDist::eRad :
           return mDegTot * powI(aRho,mDegTot-1);  // Rho ^degT
       case eTypeFuncDist::eDecX :
       case eTypeFuncDist::eDecY :
            return mDegTot*(mDegTot+1) * powI(aRho,mDegTot-1);  // d/drho (Rho ^(degT+1))

       case eTypeFuncDist::eMonX :
       case eTypeFuncDist::eMonY :
            return Norm2(mDegMon) * powI(aRho,mDegTot-1); // 
       default :
          ;
   }

   MMVII_INTERNAL_ERROR("Bad num in cDescOneFuncDist::MajNormJacOfRho");
   return 0.0;
}

double  MajNormJacOfRho
        (
             const double & aRho,
             const std::vector<cDescOneFuncDist> & aVDesc,
             const std::vector<double> & aVCoef
        )
{
    MMVII_INTERNAL_ASSERT_medium(aVDesc.size()==aVCoef.size(),"Sz in MajNormJacOfRho");
    double aRes =0.0;
    
    for (size_t aK=0 ; aK<aVDesc.size() ; aK++)
        aRes += aVCoef[aK] * aVDesc[aK].MajNormJacOfRho(aRho);

   return aRes;
}


/* ******************************** */
/*                                  */
/*           cDefProjPerspC         */
/*                                  */
/* ******************************** */

bool  cDefProjPerspC::HasRadialSym() const { return false; }

const cDefProjPerspC & cDefProjPerspC::ProjOfType(eProjPC eProj)
{
    static cProjStenope        TheProjStenope;
    static cProjFE_EquiDist    TheProjFE_EquiDist;
    static cProjStereroGraphik TheProjFE_StereroGraphik;
    static cProjOrthoGraphic   TheProjFE_OrthoGraphic;
    static cProjFE_EquiSolid   TheProjFE_EquiSolid;
    static cProj_EquiRect      TheProjFE_EquiRect;
    static std::vector<const cDefProjPerspC *> TheVProj;

    if (TheVProj.empty())
    {
        TheVProj.resize(size_t(eProjPC::eNbVals),nullptr);
	TheVProj.at(size_t(eProjPC::eStenope))        = & TheProjStenope;
	TheVProj.at(size_t(eProjPC::eFE_EquiDist))    = & TheProjFE_EquiDist;
	TheVProj.at(size_t(eProjPC::eStereroGraphik)) = & TheProjFE_StereroGraphik;
	TheVProj.at(size_t(eProjPC::eOrthoGraphik))   = & TheProjFE_OrthoGraphic;
	TheVProj.at(size_t(eProjPC::eFE_EquiSolid))   = & TheProjFE_EquiSolid;
	TheVProj.at(size_t(eProjPC::eEquiRect))       = & TheProjFE_EquiRect;
    }

    return *(TheVProj.at(size_t(eProj)));
}


};//  namespace MMVII

