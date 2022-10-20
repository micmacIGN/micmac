
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

std::string NamePow(const std::string aExpr,int aExposant) 
{  
    if (aExposant==0) return "";
    if (aExposant==1) return aExpr;
    return aExpr + "^" + ToStr(aExposant);
}


cDescOneFuncDist::cDescOneFuncDist(eTypeFuncDist aType,const cPt2di aDegXY) :
   mType    (aType),
   mDegMon  (-1,-1),
   mNum     (-1)
{
   int aNum = aDegXY.x();
   if (mType==eTypeFuncDist::eRad)
   {
       mName = "K" + ToStr(aNum);  //  K1 K2 K3 ...
       mLongName = "(X,Y) * "+  NamePow("(X^2+Y^2)",aNum);
				   //  mLongName
       mDegTot = 1 + 2 * (aNum);   //  X (X^2+Y^2) ^N
       mNum = aNum;
   }
   else if ((mType==eTypeFuncDist::eDecX) ||  (mType==eTypeFuncDist::eDecY))
   {
       bool aDecX = (mType==eTypeFuncDist::eDecX);
       int aShiftNum = aDecX?-1:0;  // DecX p1 , DecY p2 ... 
       // p1,p2  as usual, and by generalization p3 p4 p5 ...
       // DecX  [R2^N+2NX^2R2^(N-1),2NXYR2^(N-1)] :N=1=>[R2+2X^2,2XY]=[3X2+Y2,2XY]
       // DecY   [2XYNR2^(N-1), R2^N+2NY^2R^(N-1)  ] :N=1=>[2XY,R2+2Y^2]=[2XY,X2+3Y2]
       if (aDecX)
       {
           if (aNum==1)
               mLongName = "(3X2+Y2,2XY)";
	   else
               mLongName = "d/dX {(X,Y) * (X^2+Y^2) ^" +ToStr(aNum) + "}";
       }
       else
       {
           if (aNum==1)
               mLongName = "(2XY,X2+3Y2)";
	   else
               mLongName = "d/dY {(X,Y) * (X^2+Y^2) ^"+ToStr(aNum) + "}";
       }

       mName = "p" + ToStr(2*aNum+aShiftNum);
       mDegTot =  2 * (aNum);  // Derivates of X (X^2+Y^2) ^N
       mNum = aNum;
   }
   else
   {
      mDegMon = aDegXY;
      mDegTot = mDegMon.x() + mDegMon.y();
      bool isX = (mType==eTypeFuncDist::eMonX);
      if ((isX) && (mDegTot==1))
      {
          mName = ( mDegMon.x() == 1) ? "b1" : "b2";  // Usual convention
      }
      else
      {
          mName =  std::string((mType==eTypeFuncDist::eMonX) ? "x" : "y")
                 + "_" + ToStr(mDegMon.x())
                 + "_" + ToStr(mDegMon.y()) ;
      }
      std::string aStrMon="";
      if (mDegMon.x()==0)
        aStrMon = NamePow("Y",mDegMon.y());
      else if (mDegMon.y()==0)
        aStrMon = NamePow("X",mDegMon.x());
      else
        aStrMon = NamePow("X",mDegMon.x()) + "*" + NamePow("Y",mDegMon.y());

      mLongName = isX ? ("("+aStrMon+",0)")  : ("(0,"+aStrMon+")");

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

