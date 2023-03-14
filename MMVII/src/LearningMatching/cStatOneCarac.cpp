
#include "LearnDM.h"
#include "include/MMVII_Tpl_Images.h"
#include "include/MMVII_2Include_Serial_Tpl.h"

namespace MMVII
{

/* ************************************** */
/*                                        */
/*     cComputeSeparDist                  */
/*                                        */
/* ************************************** */

cComputeSeparDist::cComputeSeparDist() :
    mSomSep  (0.0),
    mSomP    (0.0)
{
}

void cComputeSeparDist::AddPops(double aPopA,double aPopB)
{
   double aS = aPopA+aPopB;
   if (aS)
   {
       mSomSep += (aPopA*aPopB) / aS;
       mSomP   += aS;
   }
}

double  cComputeSeparDist::Sep() const
{
    MMVII_INTERNAL_ASSERT_NotNul(mSomP);
    return 4.0 * (mSomSep / mSomP);  // =1 when PA=PB for all
}

template <class Type> double ComputeSep(const Type * aD1,const Type * aD2,int aNb)
{
   cComputeSeparDist aCSD;
   for (int aK=0 ; aK<aNb ; aK++)
      aCSD.AddPops(aD1[aK],aD2[aK]);
   return aCSD.Sep();
}

template <class Type,int Dim> double ComputeSep(const cDataTypedIm<Type,Dim> &aI1,const cDataTypedIm<Type,Dim> & aI2)
{
   aI2.AssertSameArea(aI2);

   return ComputeSep(aI1.RawDataLin(),aI2.RawDataLin(),aI1.NbElem());
    
}

#define INSTANTIATE_SEP_IMND(TYPE,DIM)\
template  double ComputeSep(const cDataTypedIm<TYPE,DIM>&,const cDataTypedIm<TYPE,DIM>&);


INSTANTIATE_SEP_IMND(tINT8,1)




/* ************************************** */
/*                                        */
/*           cStatOneVecCarac             */
/*                                        */
/* ************************************** */
cStatOneVecCarac::tHisto  & cStatOneVecCarac::Hist(int aNum)
{
   MMVII_INTERNAL_ASSERT_tiny((aNum>=0)&&(aNum<TheNbH),"Bad num for cStatOneVecCarac:Hist");
   return mHist[aNum];
}
const cStatOneVecCarac::tHisto  & cStatOneVecCarac::Hist(int aNum) const
{
   MMVII_INTERNAL_ASSERT_tiny((aNum>=0)&&(aNum<TheNbH),"Bad num for cStatOneVecCarac:Hist");
   return mHist[aNum];
}

const cStatOneVecCarac::tHisto  & cStatOneVecCarac::HistSom(int aFlag) const
{
   int aSzH = Hist(0).H().Sz();
   mHistSom = tHisto(aSzH);

   for (int aKF=0 ; aKF<TheNbH ; aKF++)
   {
       if (aFlag & (1<<aKF))
       {
           const cDataIm1D<tINT4>&  aH = Hist(aKF).H();
           for (int aV=0 ; aV<aSzH ; aV++)
           {
              mHistSom.AddV(aV,aH.GetV(aV));
           }
       }
   }
   mHistSom.MakeCumul();
   return mHistSom;
}

void cStatOneVecCarac::PackForSave()
{
   mImCr01.DIm().Resize(cPt2di(1,1));
   mImCr02.DIm().Resize(cPt2di(1,1));
}

cStatOneVecCarac::cStatOneVecCarac(const cPt2di & aSzCr)  :
   mImCr01  (aSzCr,nullptr,eModeInitImage::eMIA_Null),
   mImCr02  (aSzCr,nullptr,eModeInitImage::eMIA_Null)
{
/*
{
static int aCpt = 0;
aCpt++;
double aSz = (TheDynSave*TheDynSave) *((double) aCpt) * sizeof(tINT4) *2;
StdOut() << "SSZZZZ " << aSz << " ; Cpt " << aCpt << "\n";
}
*/
   for (int aKH=0 ; aKH<TheNbH ; aKH++)
       mHist[aKH] = tHisto(TheDyn4Save);
}

double  cStatOneVecCarac::Separ(int aN1,int aN2) const
{
   return ComputeSep(Hist(aN1).H(),Hist(aN2).H());
}

void cStatOneVecCarac::Inspect(const cStatOneVecCarac & aS2)
{
    StdOut() << "IIII "
              << " D0=" << Hist(0).H().LInfDist(aS2.Hist(0).H())
              << " D1=" << Hist(1).H().LInfDist(aS2.Hist(1).H())
              << " D2=" << Hist(2).H().LInfDist(aS2.Hist(2).H())
              << "\n";
    StdOut() << "AAAAA "
              << " " << &Hist(0).H()  << " " << &aS2.Hist(0).H() << ";"
             << "\n";
getchar();
}

cDataIm2D<tINT4> & cStatOneVecCarac::ImCr(bool isClose) { return isClose ? mImCr01.DIm() : mImCr02.DIm() ; }
const cDataIm2D<tINT4> & cStatOneVecCarac::ImCr(bool isClose) const { return isClose ? mImCr01.DIm() : mImCr02.DIm() ; }


double   cStatOneVecCarac::FiabCr(bool isClose) const
{
    const cDataIm2D<tINT4> & aIm = ImCr(isClose);
    double OK=0.0;
    double NotOK=0.0;
    for (const auto & aP : aIm)
    {
        int aVal = aIm.GetV(aP);
        if (aP.x() < aP.y())
        {
            OK += aVal;
        }
        else if (aP.x() == aP.y())
        {
            OK += aVal*0.5;
            NotOK += aVal*0.5;
        }
        else
        {
            NotOK += aVal;
        }
    }
    return NotOK / (NotOK+OK);
}

void  cStatOneVecCarac::SaveCr(int aDeZoom,bool isClose,const std::string & aName)
{
    cIm2D<tREAL8> aIm = Convert((tREAL8*)nullptr,ImCr(isClose));
    aIm = aIm.GaussDeZoom(aDeZoom,5);
    aIm.DIm().ToFile(aName);
}


void  cStatOneVecCarac::SaveHisto(int aSzVisu,const std::string & aName)
{
    // Compute max val of 3 histo to set the scaling
    int aVMax=0;
    for (int aKH=0 ; aKH<TheNbH ; aKH++)
    {
         const cDataIm1D<tINT4>&  aH = Hist(aKH).H();
         UpdateMax(aVMax,aH.MaxVal());
    }
    double aScale = aSzVisu / double(aVMax);
    std::vector<cIm2D<tU_INT1> > aVIm;
    for (int aKH=0 ; aKH<TheNbH ; aKH++)  // Parse 3 histogramme
    {
         cIm2D<tU_INT1>  aIm(cPt2di(TheDyn4Visu,aSzVisu));
         aVIm.push_back(aIm);
         cDataIm2D<tU_INT1>& aDIm = aIm.DIm();
         const cDataIm1D<tINT4>&  aH = Hist(aKH).H();
         for (int aX=0 ; aX<TheDyn4Visu ; aX++)   // Parse X , absice of histo
         {
             tREAL8 aAvg = aH.AvgInterv(cVecCaracMatch::FromVisu(aX),cVecCaracMatch::FromVisu(aX+1));
             int aLim = round_ni(aAvg*aScale); // Size of bar proportional to value
             for (int aY=0 ; aY<aSzVisu ; aY++)
             {
                  bool  inH = (aSzVisu-aY-1) <= aLim;  // Invert Y to have standard visu (Y-up)
                  aDIm.SetV(cPt2di(aX,aY), inH ? 255 : 0);
             }
         }
    }
    aVIm.at(0).DIm().ToFile(aName,aVIm.at(1).DIm(),aVIm.at(2).DIm());
}

void cStatOneVecCarac::AddData(const cAuxAr2007 & anAux)
{
   for (int aKH=0 ; aKH<TheNbH ; aKH++)
      MMVII::AddData(cAuxAr2007("H"+ToStr(aKH),anAux),mHist[aKH]);
   MMVII::AddData(cAuxAr2007("Cr01",anAux),mImCr01.DIm());
   MMVII::AddData(cAuxAr2007("Cr02",anAux),mImCr02.DIm());
}

void AddData(const cAuxAr2007 & anAux,cStatOneVecCarac& aS1VC)
{
    aS1VC.AddData(anAux);
}

void  cStatOneVecCarac::MakeCumul()
{
   for (int aKH=0 ; aKH<TheNbH ; aKH++)
       mHist[aKH].MakeCumul();
}



/* ************************************** */
/*                                        */
/*         cStatAllVecCarac               */
/*                                        */
/* ************************************** */

cStatAllVecCarac::cStatAllVecCarac(bool WithCr) :
   mWithCr (WithCr),
   mSzCr   (WithCr ? cPt2di(TheDyn4Visu,TheDyn4Visu):cPt2di(1,1))
{

   for (int aK=0 ; aK<TheNbVals ; aK++)
      mStats.push_back(cStatOneVecCarac(mSzCr));
}

void cStatAllVecCarac::AddOneFile(int aNum,const cFileVecCaracMatch & aFVC)
{
    for (const auto & aVCM : aFVC.VVCM())
    {
        for (int aK=0; aK<int(eModeCaracMatch::eNbVals) ; aK++)
        {
              int aVal = aVCM.Value((eModeCaracMatch)aK);
              mStats[aK].Add(aNum,aVal);
        }
    }
}

void cStatAllVecCarac::AddCr(const cFileVecCaracMatch & aFVCM1,const cFileVecCaracMatch & aFVCM2,bool isClose)
{
    MMVII_INTERNAL_ASSERT_strong(mWithCr,"No WithCr in cStatAllVecCarac::AddCr");
    const std::vector<cVecCaracMatch> & aVec1 =  aFVCM1.VVCM();
    const std::vector<cVecCaracMatch> & aVec2 =  aFVCM2.VVCM();
    MMVII_INTERNAL_ASSERT_User(aVec1.size()==aVec2.size(),eTyUEr::eUnClassedError,"Incompatible files in AddCr");

    
    for (int aKLab=0; aKLab<int(eModeCaracMatch::eNbVals) ; aKLab++)
    {
          eModeCaracMatch aLabel = (eModeCaracMatch)aKLab;
          cDataIm2D<tINT4> &  aImCr = mStats.at(aKLab).ImCr(isClose);
          for (int aKVal=0 ; aKVal<int(aVec1.size()) ; aKVal++)
          {
               int aVal1 = aVec1.at(aKVal).Value(aLabel);
               int aVal2 = aVec2.at(aKVal).Value(aLabel);
               aImCr.AddVal(cVecCaracMatch::ToVisu(cPt2di(aVal1,aVal2)),1);
          }
    }
}

void cStatAllVecCarac::ShowSepar(const std::string & aPat,cMultipleOfs & aOfs)
{
    for (const auto & aLab :  SubOfPat<eModeCaracMatch>(aPat,true))
    {
        aOfs << "Car:" << E2Str(aLab) ;
        const cStatOneVecCarac & aVC = mStats.at(int(aLab));
        aOfs
                 << " " << aVC.Separ(0,0)
                 << " " << aVC.Separ(0,1) 
                 << " " << aVC.Separ(0,2) 
               ;
       if (mWithCr)
       {
           aOfs << " CR1: " << aVC.FiabCr(true) << " CR2: " <<  aVC.FiabCr(false);
       }
       aOfs << "\n";
       // aOfs << "WITHCHR " << mWithCr << "\n";
    }
}

void cStatAllVecCarac::SaveCr(int aDeZoom,const std::string &aDir)
{
   CreateDirectories(aDir,true);
   for (int aKLab=0; aKLab<int(eModeCaracMatch::eNbVals) ; aKLab++)
   {
       eModeCaracMatch aLabel = (eModeCaracMatch)aKLab;
       for (bool isClose : {true,false})
       {
           std::string aName = "DMCr_" + E2Str(aLabel) + std::string(isClose ? "Close" : "Std") + ".tif";
           mStats.at(aKLab).SaveCr(aDeZoom,isClose,aDir+aName);
       }
   }
}

void cStatAllVecCarac::SaveHisto(int aSz,const std::string &aDir)
{
   CreateDirectories(aDir,true);
   for (int aKLab=0; aKLab<int(eModeCaracMatch::eNbVals) ; aKLab++)
   {
       eModeCaracMatch aLabel = (eModeCaracMatch)aKLab;
       std::string aName = "Histo_" + E2Str(aLabel) +  ".tif";
       mStats.at(aKLab).SaveHisto(aSz,aDir+aName);
   }
}

void cStatAllVecCarac::PackForSave()
{
    mWithCr = false;
    mSzCr = cPt2di(1,1);
    for (auto & aStat: mStats)
    {
        aStat.PackForSave();
    }
}

void cStatAllVecCarac::MakeCumul()
{
    for (auto & aStat: mStats)
    {
        aStat.MakeCumul();
    }
}

void  cStatAllVecCarac::AddData(const cAuxAr2007 & anAux)
{
   MMVII::AddData(cAuxAr2007("WithCr",anAux),mWithCr);
   MMVII::AddData(cAuxAr2007("SzCr",anAux),mSzCr);
   MMVII::AddData(cAuxAr2007("Stats",anAux),mStats);
}

void AddData(const cAuxAr2007 & anAux,cStatAllVecCarac& aVAll)
{
    aVAll.AddData(anAux);
}

const cStatOneVecCarac & cStatAllVecCarac::OneStat(eModeCaracMatch aLabel) const
{
   return mStats.at((int) aLabel);
}




};
