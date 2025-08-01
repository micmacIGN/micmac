#include "MMVII_PCSens.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_util_tpl.h"
#include "../BundleAdjustment/BundleAdjustment.h"

/**
   \file Measures.cpp

   \brief classes 4 handling measurement in image
*/


namespace MMVII
{

/* ********************************************* */
/*                                               */
/*             cOneLineAntiParal                 */
/*                                               */
/* ********************************************* */

cOneLineAntiParal::cOneLineAntiParal() :
    mSeg (cPt2dr(0,0),cPt2dr(1,0)),  // Diff else error (cstr check not identic)
    mAngDif     (-1),
    mWidth      (-1),
    mCumul      (-1),
    mRadHom     (-1),
    mSigmaLine  (-1)
{
}

void AddData(const cAuxAr2007 & anAux,cOneLineAntiParal & anEx)
{
      AddData(cAuxAr2007("Image",anAux),anEx.mNameIm);
      AddData(cAuxAr2007("P1",anAux),anEx.mSeg.P1());
      AddData(cAuxAr2007("P2",anAux),anEx.mSeg.P2());
      AddData(cAuxAr2007("ParalAng",anAux),anEx.mAngDif);
      AddData(cAuxAr2007("Width",anAux),anEx.mWidth);
      AddData(cAuxAr2007("Cumul",anAux),anEx.mCumul);
      AddData(cAuxAr2007("SigmaL",anAux),anEx.mSigmaLine);
}

void AddData(const cAuxAr2007 & anAux,cLinesAntiParal1Im & anEx)
{
      AddData(cAuxAr2007("Calib",anAux),anEx.mDirCalib);
      AddData(cAuxAr2007("Lines",anAux),anEx.mLines);
}



/* ********************************************* */
/*                                               */
/*             cHomogCpleIm                      */
/*                                               */
/* ********************************************* */

cHomogCpleIm::cHomogCpleIm(const cPt2dr & aP1,const cPt2dr & aP2) :
   mP1 (aP1),
   mP2 (aP2)
{
}

cHomogCpleIm::cHomogCpleIm() :
    mP1(),
    mP2()
{
}

void cHomogCpleIm::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("x1",anAux),mP1.x());
     MMVII::AddData(cAuxAr2007("y1",anAux),mP1.y());
     MMVII::AddData(cAuxAr2007("x2",anAux),mP2.x());
     MMVII::AddData(cAuxAr2007("y2",anAux),mP2.y());
}

void AddData(const  cAuxAr2007 & anAux,cHomogCpleIm & aCple)  {aCple.AddData(anAux);}






/* ************************************** */
/*                                        */
/*         cSetHomogCpleIm                */
/*                                        */
/* ************************************** */

cSetHomogCpleIm::cSetHomogCpleIm(size_t aReserve) 
{
	mSetH.reserve(aReserve);
}

const std::vector<cHomogCpleIm> & cSetHomogCpleIm::SetH() const {return mSetH;}
std::vector<cHomogCpleIm> & cSetHomogCpleIm::SetH() {return mSetH;}

void cSetHomogCpleIm::Add(const cHomogCpleIm & aCple)
{
	mSetH.push_back(aCple);
}

void cSetHomogCpleIm::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::AddData(anAux,mSetH);
}
void AddData(const  cAuxAr2007 & anAux,cSetHomogCpleIm & aSet)
{
    aSet.AddData(anAux);
}


void cSetHomogCpleIm::Clear()
{
	mSetH.clear();
}

void cSetHomogCpleIm::ToFile(const std::string & aName) const
{
    PushPrecTxtSerial(3); // avoid "ridiculous" number of digit for tie-points
    SaveInFile(mSetH,aName);
    PopPrecTxtSerial();   // restore previous number of digit
}

void  cSetHomogCpleIm::InitFromFile(const std::string & aName) 
{
	Clear();
	ReadFromFile(mSetH,aName);
}

cSetHomogCpleIm cSetHomogCpleIm::FromFile(const std::string & aName)
{
    cSetHomogCpleIm aRes;
    aRes.InitFromFile(aName);

    return aRes;
}


/* ********************************************* */
/*                                               */
/*             cMultipleImPt                     */
/*                                               */
/* ********************************************* */

cMultipleImPt::cMultipleImPt(int aNumPt) :
    mNumPt  (aNumPt)
{
}

void cMultipleImPt::Add(const cMesIm1Pt & aMes, int aNumIm, MMVII::cMes2DDirInfo *aMesDirInfo)
{
    mVMeasures.push_back(aMes.mPt);
    mVImages.push_back(aNumIm);
    mVMesDirInfo.push_back(aMesDirInfo);
}

const cPt2dr * cMultipleImPt::PtOfIm(int aIndIm) const
{
     for (size_t aK=0; aK<mVImages.size() ; aK++)
        if (mVImages[aK] == aIndIm)
           return &(mVMeasures[aK]);
     return nullptr;
}

const std::vector<cPt2dr> & cMultipleImPt::VMeasures() const  {return mVMeasures;}
const std::vector<int>    & cMultipleImPt::VImages()   const  {return mVImages  ;}

int cMultipleImPt::NumPt() const {return mNumPt;}


/* ********************************************* */
/*                                               */
/*             cSetMesImGCP                      */
/*                                               */
/* ********************************************* */


cSetMesGndPt::cSetMesGndPt() :
	mPhaseGCPFinished (false)
{
}

void cSetMesGndPt::Add1GCP(const cMes1Gnd3D & aMes)
{
     MMVII_INTERNAL_ASSERT_medium(!mPhaseGCPFinished,"cSetMesImGCP cannot add GCP after image");
     mMesGCP3D.push_back(aMes);
     m2MapPtInt.Add(aMes.mNamePt,false,"GCP-3D : " + aMes.mNamePt);
}

void cSetMesGndPt::AddMes3D(const cSetMesGnd3D &  aSet, cMes3DDirInfo *aMesDirInfo)
{
    for (const auto & aMes  : aSet.Measures())
    {
        Add1GCP(aMes);
        mMesGCP3D.back().mMesDirInfo = aMesDirInfo;
    }
}

const cSetMesPtOf1Im  & cSetMesGndPt::MesImInitOfName(const std::string & aNameIm) const
{
	return mMesImInit.at(m2MapImInt.Obj2I(aNameIm));
}

const cMes1Gnd3D &  cSetMesGndPt::MesGCPOfName(const std::string & aNamePt) const
{
    return mMesGCP3D.at(m2MapPtInt.Obj2IWithMsg(aNamePt,"GCP : " + aNamePt));
}

cMes1Gnd3D &  cSetMesGndPt::MesGCPOfName(const std::string & aNamePt)
{
    return mMesGCP3D.at(m2MapPtInt.Obj2I(aNamePt));
}


cSetMesGnd3D cSetMesGndPt::AllMesGCP() const
{
   cSetMesGnd3D aRes;
   for (const auto &  aGCP : mMesGCP3D)
      aRes.AddMeasure3D(aGCP);

   return aRes;
}



const cMes1Gnd3D &  cSetMesGndPt::MesGCPOfNum(int aNum) const
{
    return mMesGCP3D.at(aNum);
}

const cMes1Gnd3D & cSetMesGndPt::MesGCPOfMulIm(const cMultipleImPt & aMIP) const
{
   return MesGCPOfNum(aMIP.NumPt());
}



bool  cSetMesGndPt::NameIsGCP(const std::string & aNamePt) const
{
  return m2MapPtInt.Obj2I(aNamePt,true) >= 0;
}


cPt3dr  cSetMesGndPt::BundleInter(const cMultipleImPt & aMPT) const
{
     std::vector<tSeg3dr>  aVSeg;
     for (size_t aKI=0 ; aKI<aMPT.VMeasures().size() ; aKI++)
     {
         cSensorImage * aSIm = mVSens.at(aMPT.VImages().at(aKI));
         MMVII_INTERNAL_ASSERT_tiny(aSIm,"No sensor in  cSetMesImGCP::BundleInter");
         aVSeg.push_back(aSIm->Image2Bundle(aMPT. VMeasures().at(aKI)));
     }

     return BundleInters(aVSeg);
}



void cSetMesGndPt::AddMes2D(const cSetMesPtOf1Im & aSetMesIm, MMVII::cMes2DDirInfo *aMesDirInfo, cSensorImage* aSens, eLevelCheck aOnNonExistGCP)
{
    //  Are we beginning  the  image measurement phase
    {
        if (! mPhaseGCPFinished)
        {
            mMesImOfPt.reserve(mMesGCP3D.size());

            for (size_t aKp=0 ;  aKp<mMesGCP3D.size() ; aKp++)
                mMesImOfPt.push_back( cMultipleImPt(aKp));
        }
        mPhaseGCPFinished = true;
    }

    int aNumIm = m2MapImInt.Obj2I(aSetMesIm.NameIm(), true);
    if (aNumIm<0)
    {
        aNumIm = m2MapImInt.Add(aSetMesIm.NameIm(), true);
        mMesImInit.push_back(cSetMesPtOf1Im(aSetMesIm.NameIm()));
        mVSens.push_back(aSens);
    } else {
        MMVII_INTERNAL_ASSERT_tiny(aNumIm<int(mMesImInit.size()),"Incoherence in cSetMesImGCP::AddMes2D");
        MMVII_INTERNAL_ASSERT_tiny(mVSens.at(aNumIm) == aSens,"Variable sensor in cSetMesImGCP::AddMes2D");
    }

    for (auto & aMes : aSetMesIm.Measures())
    {
        int aNumPt = m2MapPtInt.Obj2I(aMes.mNamePt,true);
        if (aNumPt>=0)
        {
            mMesImOfPt.at(aNumPt).Add(aMes,aNumIm,aMesDirInfo);
            mMesImInit.at(aNumIm).AddMeasure(aMes,aMesDirInfo);
        }
        else
        {
            if (! starts_with(aMes.mNamePt,MMVII_NONE))
            {
               ErrorWarnNone
               (
                  aOnNonExistGCP,
                  "Measure Im w/o Ground, first occur Im=" + aSetMesIm.NameIm() + " Pt="  + aMes.mNamePt
               );
            }
        }
    }
}

const std::vector<cMes1Gnd3D> &        cSetMesGndPt::MesGCP()    const  {return mMesGCP3D; }
const std::vector<cMultipleImPt> &   cSetMesGndPt::MesImOfPt() const  {return mMesImOfPt;  }
const std::vector<cSensorImage*> &   cSetMesGndPt::VSens()     const  {return mVSens;}
const std::vector<cSetMesPtOf1Im> &  cSetMesGndPt::MesImInit() const  {return mMesImInit;}

std::vector<cMes1Gnd3D> &        cSetMesGndPt::MesGCP()   {return mMesGCP3D; }

void cSetMesGndPt::ExtractMes1Im(cSet2D3D&  aS23,const std::string &aNameIm,bool SVP) const
{
    aS23.Clear();


    int aNumIm = m2MapImInt.Obj2I(aNameIm,SVP);

    if (aNumIm<0) return;

    for (size_t aKp=0 ;  aKp<mMesGCP3D.size() ; aKp++)
    {
         const cPt2dr * aP2 = mMesImOfPt[aKp].PtOfIm(aNumIm);
	 if (aP2)
            aS23.AddPair(*aP2,mMesGCP3D[aKp].mPt);
    }
}

void cSetMesGndPt::AsserGCPFinished() const
{
  if (! mPhaseGCPFinished)
     MMVII_UnclasseUsEr("cSetMesImGCP : use with no image file");
}

int cSetMesGndPt::GetNbImMesForPoint(const std::string & aGCPName, bool SVP) const
{
    int aKGCP = m2MapPtInt.Obj2I(aGCPName,SVP);
    if (aKGCP<0)
        return 0;
    else
        return mMesImOfPt[aKGCP].VImages().size();
}

cSetMesGndPt *  cSetMesGndPt::FilterNonEmptyMeasure(int aNbMeasureMin) const
{
   AsserGCPFinished();

  cSetMesGndPt * aRes = new cSetMesGndPt;

  for (size_t aKGCP=0 ; aKGCP<mMesGCP3D.size() ; aKGCP++)
  {

       if ( int(mMesImOfPt[aKGCP].VImages().size()) >= aNbMeasureMin)
       {
             aRes->Add1GCP(mMesGCP3D[aKGCP]);
       }
   }

  for (size_t aKIm=0 ; aKIm<mMesImInit.size() ; aKIm++)
  {
     aRes->AddMes2D(mMesImInit.at(aKIm),nullptr,mVSens.at(aKIm)); //aMesDirInfo=nullptr means keep original MesDirInfo
  }

   return aRes;
}

tREAL8 cSetMesGndPt::AvgSqResidual() const
{
     cWeightAv<tREAL8>  aWA;

     for (size_t aKPt=0 ; aKPt<mMesGCP3D.size() ; aKPt++)
     {
         const cPt3dr & aPGr = mMesGCP3D.at(aKPt).mPt;
         const  cMultipleImPt & aMMIm = mMesImOfPt.at(aKPt);
         size_t aNbMes = aMMIm.VMeasures().size();
         for (size_t aKMes=0 ; aKMes<aNbMes ; aKMes++)
         {
             const cPt2dr & aPIm =  aMMIm.VMeasures().at(aKMes);
	     int aIndIm = aMMIm.VImages().at(aKMes);
	     cSensorImage * aSens = mVSens.at(aIndIm);

	     tREAL8 aD2 = SqN2(aPIm-aSens->Ground2Image(aPGr));

	     aWA.Add(1.0,aD2);
         }
     }

     if (ValidInvertibleFloatValue(aWA.SW()))
        return std::sqrt(aWA.Average());
     else
         return NAN;
}

cSetMesGnd3D  cSetMesGndPt::ExtractSetGCP(const std::string & aName) const
{
    cSetMesGnd3D aRes(aName);
    for (const auto &  aMesGCP : mMesGCP3D)
        aRes.AddMeasure3D(aMesGCP);

    return aRes;
}

/* ********************************************* */
/*                                               */
/*             cMesIm1Pt                         */
/*                                               */
/* ********************************************* */
        
cMesIm1Pt::cMesIm1Pt(const cPt2dr & aPt, const std::string & aNamePt, tREAL4 aS2) :
     mPt      (aPt),
     mNamePt  (aNamePt),
     mSigma2   {aS2,0,aS2},
     mMesDirInfo  (nullptr)
{
}

cMesIm1Pt::cMesIm1Pt() :
    cMesIm1Pt(cPt2dr(0,0),"???",-1)
{
}

void AddData(const  cAuxAr2007 & anAux,cMesIm1Pt & aMes)
{
   MMVII::AddData(cAuxAr2007("Name",anAux),aMes.mNamePt);
   MMVII::AddData(cAuxAr2007("Pt",anAux),aMes.mPt);
   MMVII::AddData(cAuxAr2007("Sigma2",anAux),aMes.mSigma2);
}

/* ********************************************* */
/*                                               */
/*             cSetMesPtOf1Im                    */
/*                                               */
/* ********************************************* */

cSetMesPtOf1Im::cSetMesPtOf1Im(const std::string & aNameIm) :
    mNameIm  (aNameIm)
{
}

cSetMesPtOf1Im::cSetMesPtOf1Im() :
    cSetMesPtOf1Im("???")
{
}


cSetMesPtOf1Im  cSetMesPtOf1Im::FromFile(const std::string & aNameFile)
{
    cSetMesPtOf1Im aRes;
    ReadFromFile(aRes,aNameFile);

    return aRes;
}

const std::string &              cSetMesPtOf1Im::NameIm()   const {return mNameIm;}
const std::vector<cMesIm1Pt> &   cSetMesPtOf1Im::Measures() const {return mMeasures;}
std::vector<cMesIm1Pt> &   cSetMesPtOf1Im::Measures() {return mMeasures;}

void cSetMesPtOf1Im::SetNameIm(const std::string & aNameIm)
{
    mNameIm = aNameIm;
}

void cSetMesPtOf1Im::AddMeasure(const cMesIm1Pt & aMeasure, cMes2DDirInfo *aMesDirInfo)
{
     mMeasures.push_back(aMeasure);
     if (aMesDirInfo)
        mMeasures.back().mMesDirInfo = aMesDirInfo;
}

void cSetMesPtOf1Im::AddMeasureIfNew(const cMesIm1Pt & aNewM,tREAL8 aTolDupl)
{
    for (const auto & aM : mMeasures)
    {
         if (aM.mNamePt == aNewM.mNamePt)
         {
            if (Norm2(aM.mPt- aNewM.mPt) <= aTolDupl)
            {
                // point are just identic up to tolerance, nothing to do
                return;
            }
            else
            {
                StdOut() << "PTS=" << aM.mPt << " " << aNewM.mPt << "\n";
                MMVII_UnclasseUsEr("Duplicate measure added for point : " + aM.mNamePt + " in Image " +mNameIm);
            }
         }
    }
    // Ok point doesnt exist, just add
    AddMeasure(aNewM);
}


void cSetMesPtOf1Im::AddSetMeasure(const cSetMesPtOf1Im & aSet,bool SuprNone,bool OkDupl)
{
    MMVII_INTERNAL_ASSERT_tiny(mNameIm==aSet.mNameIm,"Mix different images in AddSetMeasure " + mNameIm + "!=" + aSet.mNameIm);

    for (const auto & aMes : aSet.mMeasures)
    {
         if (NameHasMeasure(aMes.mNamePt))
         {
             if (! OkDupl)
             {
                 MMVII_INTERNAL_ERROR("Non autorize duplicate name for Im=" + mNameIm + " Pt=" +aMes.mNamePt);
             }
         }
         else
         {
             if ((!SuprNone) || (!starts_with(aMes.mNamePt,MMVII_NONE)) )
                mMeasures.push_back(aMes);
         }
    }
}



cMesIm1Pt *  cSetMesPtOf1Im::NearestMeasure(const cPt2dr & aPt) 
{
   return WhitchMinVect
          (
               mMeasures,
               [aPt](const auto & aMes) {return SqN2(aPt-aMes.mPt);}
          );
}



void cSetMesPtOf1Im::AddData(const  cAuxAr2007 & anAuxParam)
{
    cAuxAr2007 anAux("SetMesIm",anAuxParam);

    MMVII::AddData(cAuxAr2007("NameIm",anAux),mNameIm);
    MMVII::AddData(cAuxAr2007("Measures",anAux),mMeasures);
}

void AddData(const  cAuxAr2007 & anAux,cSetMesPtOf1Im & aGCPMI)
{
    aGCPMI.AddData(anAux);
}

void cSetMesPtOf1Im::ToFile(const std::string & aName) const
{
    SaveInFile(*this,aName);
}

void cSetMesPtOf1Im::SortMes()
{
    std::sort(mMeasures.begin(), mMeasures.end(),
              [](const cMesIm1Pt &a, const cMesIm1Pt &b)
                {
                    return a.mNamePt < b.mNamePt;
                }
    );
}

const std::string  cSetMesPtOf1Im::ThePrefixFiles = "MesIm-";

std::string cSetMesPtOf1Im::StdNameFileOfIm(const std::string & aName)
{
	return ThePrefixFiles +aName+ "."+ cMMVII_Appli::CurrentAppli().TaggedNameDefSerial();
}

std::string cSetMesPtOf1Im::StdNameFile() const
{
    return StdNameFileOfIm(mNameIm);
}

cMesIm1Pt *  cSetMesPtOf1Im::PrivateMeasuresOfName(const std::string & aNamePt,bool SVP) const
{
   const auto & anIt = find_if(mMeasures.begin(),mMeasures.end(),[aNamePt](const auto& aM){return aM.mNamePt==aNamePt;});

   if (anIt != mMeasures.end())
   {
       return  const_cast<cMesIm1Pt *>(&(*anIt));
   }

   if (! SVP)
   {
      MMVII_INTERNAL_ERROR("PrivateMeasuresOfName for "+ aNamePt);
   }

   return nullptr;
}

const cMesIm1Pt& cSetMesPtOf1Im::MeasuresOfName(const std::string & aN)const {return *(PrivateMeasuresOfName(aN,false));}
cMesIm1Pt& cSetMesPtOf1Im::MeasuresOfName(const std::string & aN){return *(PrivateMeasuresOfName(aN,false));}
bool cSetMesPtOf1Im::NameHasMeasure(const std::string & aN) const {return PrivateMeasuresOfName(aN,true)!=nullptr;}


/* ********************************************* */
/*                                               */
/*             cMes1GCP                          */
/*                                               */
/* ********************************************* */

cMes1Gnd3D::cMes1Gnd3D(const cPt3dr & aPt, const std::string & aNamePt, tREAL4 aSigma,
                   const std::string &aAdditionalInfo) :
    mPt       (aPt),
    mNamePt   (aNamePt),
    mAdditionalInfo(aAdditionalInfo),
    mMesDirInfo(nullptr),
    mOptSigma2(std::nullopt)
{
    if (aSigma>=0.)
        SetSigma2(aSigma);
}

cMes1Gnd3D::cMes1Gnd3D() :
    cMes1Gnd3D (cPt3dr::Dummy(),"??")
{
}

void cMes1Gnd3D::SetSigma2(const cPt3dr & aSigma)
{
     mOptSigma2 = {0.,0.,0.,0.,0.,0.};
     (*mOptSigma2)[IndXX] = (float) Square(aSigma.x());
     (*mOptSigma2)[IndYY] = (float) Square(aSigma.y());
     (*mOptSigma2)[IndZZ] = (float) Square(aSigma.z());
}

void cMes1Gnd3D::SetSigma2(tREAL8 aSigma)
{
     SetSigma2(cPt3dr::PCste(aSigma));
}

const cArray<tREAL4,6> & cMes1Gnd3D::Sigma2()  const { return  (*mOptSigma2); }

bool cMes1Gnd3D::Sigma2IsInit() const {return mOptSigma2.has_value();}


void cMes1Gnd3D::AddData(const  cAuxAr2007 & anAux)
{
   MMVII::AddData(cAuxAr2007("Name",anAux),mNamePt);
   MMVII::AddData(cAuxAr2007("Pt",anAux),mPt);
   MMVII::AddData(cAuxAr2007("AdditionalInfo",anAux),mAdditionalInfo);
   AddOptData(anAux, "Sigma2", mOptSigma2);
}

void AddData(const  cAuxAr2007 & anAux,cMes1Gnd3D & aMes)
{
	aMes.AddData(anAux);
}

void cMes1Gnd3D::ChangeCoord(const cDataMapping<tREAL8,3,3>& aMapping)
{
	// StdOut() << "PPPPP " << mPt << aMapping.Value(mPt) << aMapping.Inverse(mPt)<< "\n";
    mPt = aMapping.Value(mPt);
}

cPt3dr cMes1Gnd3D::SigmasXYZ() const
{
    if (mOptSigma2)
    {
        return {sqrt((*mOptSigma2)[IndXX]), sqrt((*mOptSigma2)[IndXX]), sqrt((*mOptSigma2)[IndXX]) };
    } else {
        return cPt3dr::Dummy();
    }
}

/* ********************************************* */
/*                                               */
/*             cSetMesGCP                        */
/*                                               */
/* ********************************************* */


const std::string  cSetMesGnd3D::ThePrefixFiles = "MesGCP";

cSetMesGnd3D::cSetMesGnd3D(const std::string &aNameSet) :
    mNameSet(aNameSet)
{
}

cSetMesGnd3D::cSetMesGnd3D() :
    cSetMesGnd3D("???")
{
}

std::list<std::string>  cSetMesGnd3D::ListOfNames() const
{
    std::list<std::string> aRes;
    for (const auto & aGCP : mMeasures)
        aRes.push_back(aGCP.mNamePt);

   return aRes;
}



void cSetMesGnd3D::ChangeCoord(const cDataMapping<tREAL8,3,3>& aMapping)
{
     for (auto & aGCP : mMeasures)
          aGCP.ChangeCoord(aMapping);
}

void cSetMesGnd3D::AddMeasure3D(const cMes1Gnd3D & aMeasure)
{
     mMeasures.push_back(aMeasure);
}

void cSetMesGnd3D::AddData(const  cAuxAr2007 & anAuxParam)
{
    cAuxAr2007 anAux("SetGCP",anAuxParam);

    MMVII::AddData(cAuxAr2007("NameSet",anAux),mNameSet);
    MMVII::AddData(cAuxAr2007("Measures",anAux),mMeasures);
}

void AddData(const  cAuxAr2007 & anAux,cSetMesGnd3D & aSet)
{
     aSet.AddData(anAux);
}

cSetMesGnd3D  cSetMesGnd3D::FromFile(const std::string & aNameFile)
{
    cSetMesGnd3D aRes;
    ReadFromFile(aRes,aNameFile);

    return aRes;
}

void cSetMesGnd3D::ToFile(const std::string & aNameFile)
{
    // we do this because by side effect it check efficiently the absence of duplicata
    {
        cSetMesGndPt aMesIG;
        aMesIG.AddMes3D(*this);
    }
     SaveInFile(*this,aNameFile);
}



std::string cSetMesGnd3D::StdNameFileOfSet(const std::string & aName)
{
     return ThePrefixFiles+"-"+ aName+ "."+GlobTaggedNameDefSerial();
}
std::string cSetMesGnd3D::StdNameFile() const {return StdNameFileOfSet(mNameSet);}

void cSetMesGnd3D::ToFile(const std::string & aNameFile) const
{
    SaveInFile(*this,aNameFile);
}


const std::vector<cMes1Gnd3D> &   cSetMesGnd3D::Measures() const {return mMeasures;}

cSetMesGnd3D  cSetMesGnd3D::Filter(const std::string &aFilter, const std::string &aFiltrAdditionalInfo) const
{
    cSetMesGnd3D  aRes(mNameSet);

    tNameSelector  aSelectName =  AllocRegex(aFilter);
    tNameSelector  aSelectInfo =  AllocRegex(aFiltrAdditionalInfo);

    for (const auto & aPt : mMeasures)
    {
        if ( aSelectName.Match(aPt.mNamePt) && aSelectInfo.Match(aPt.mAdditionalInfo) )
            aRes.AddMeasure3D(aPt);
    }

    return aRes;
}

std::map<std::string, MMVII::cSetMesGnd3D> cSetMesGnd3D::SplitPerOutDir(const std::string &aDefaultOutName) const
{
    std::map<std::string, MMVII::cSetMesGnd3D> output;
    for (const auto & aPt : mMeasures)
    {
        if (aDefaultOutName.empty())
            MMVII_INTERNAL_ASSERT_strong(aPt.mMesDirInfo, "cSetMesGnd3D::SplitPerOutDir(): point without OutDir: "+aPt.mNamePt);
        if (!aPt.mMesDirInfo)
            output[aDefaultOutName].AddMeasure3D(aPt);
        else
            output[aPt.mMesDirInfo->mDirNameOut].AddMeasure3D(aPt);
    }
    for (auto & [dir, setmes] : output)
    {
        setmes.mNameSet = mNameSet;
    }
    return output;
}

const cMes1Gnd3D * cSetMesGnd3D::GetAdrMeasureOfNamePt(const std::string& aName,bool SVP) const
{
    for (const auto & aMeas : mMeasures)
        if (aMeas.mNamePt==aName)
           return  & aMeas;
    MMVII_INTERNAL_ASSERT_strong(SVP,"Cannot found GCP of name : " +aName);

   return nullptr;
}

const cMes1Gnd3D & cSetMesGnd3D::GetMeasureOfNamePt(const std::string& aName) const
{
    return * GetAdrMeasureOfNamePt(aName);
}




/**********************************************/
/*                                            */
/*           cPair2D3D                        */
/*                                            */
/**********************************************/

 cPair2D3D::cPair2D3D(const cPt2dr & aP2,const cPt3dr & aP3) :
    mP2  (aP2),
    mP3  (aP3)
{
}

/**********************************************/
/*                                            */
/*           cWeightedPair2D3D                */
/*                                            */
/**********************************************/

 cWeightedPair2D3D::cWeightedPair2D3D(const cPair2D3D & aPair,double  aW) :
    cPair2D3D (aPair),
    mWeight   (aW)
{
}

cWeightedPair2D3D::cWeightedPair2D3D(const cPt2dr& aPIm,const cPt3dr& aPGround,double aWeight) :
    cWeightedPair2D3D(cPair2D3D(aPIm,aPGround),aWeight)
{
}

/**********************************************/
/*                                            */
/*           cSet2D3D                         */
/*                                            */
/**********************************************/

void cSet2D3D::AddPair(const tPair & aP23)
{
     mPairs.push_back(aP23);
}

void cSet2D3D::AddPair(const cPt2dr& aPIm,const cPt3dr& aPGround,double aWeight)
{
    AddPair(tPair(aPIm,aPGround,aWeight));
}


const cSet2D3D::tCont2D3D &  cSet2D3D::Pairs() const { return mPairs;}

const cWeightedPair2D3D &  cSet2D3D::KthPair(int aK) const {return mPairs.at(aK);}
size_t cSet2D3D::NbPair() const {return mPairs.size();}


void  cSet2D3D::Clear()
{
	mPairs.clear();
}

cWeightedPair2D3D  cSet2D3D::Centroid() const
{
    cPt2dr aSP2(0,0);
    cPt3dr aSP3(0,0,0);
    double aSW=0;
    for (const auto & aPair : mPairs)
    {
	    aSP2 += aPair.mP2 * aPair.mWeight;
	    aSP3 += aPair.mP3 * aPair.mWeight;
	    aSW += aPair.mWeight;
    }
    return cWeightedPair2D3D(aSP2/aSW,aSP3/aSW,aSW);
}

void cSet2D3D::Substract(const cPair2D3D& aSub)
{
    for (auto  & aPair : mPairs)
    {
        aPair.mP2 =  aPair.mP2-aSub.mP2;
        aPair.mP3 =  aPair.mP3-aSub.mP3;
    }
}

std::vector<cPt3dr> cSet2D3D::VP3() const
{
    std::vector<cPt3dr> aRes;
    for (auto  & aPair : mPairs)
        aRes.push_back(aPair.mP3);

    return aRes;
}

/**********************************************/
/*                                            */
/*           cFilterMesIm                     */
/*                                            */
/**********************************************/

cFilterMesIm::cFilterMesIm(cPhotogrammetricProject & aPhProj,const std::string & aNameIm)  :
    mPhProj      (aPhProj),
    mMesIm       (aNameIm),
    mFinished    (false)
{
    mPhProj.LoadGCP3D(mImGCP);  // init new GCP/IM with GCP
}

void cFilterMesIm::AddInOrOut(const cPt2dr & aPtIm,const std::string & aNamePt,bool isIn)
{
// StdOut() <<   " **AddInOrOutAddInOrOut " << isIn << std::endl;
     MMVII_INTERNAL_ASSERT_medium(!mFinished,"cFilterMesIm::AddInOut while fnished");
     cMesIm1Pt aMes(aPtIm,aNamePt,1.0 );
     if (isIn)
     {
        mMesIm.AddMeasure(cMesIm1Pt(aPtIm,aNamePt,1.0 ));
     }
     else
     {
        mSupr.push_back(aNamePt);
     }
}

void cFilterMesIm::SetFinished()
{
    if (! mFinished)
       mImGCP.AddMes2D(mMesIm);

    mFinished = true;
}
const cSetMesGndPt &   cFilterMesIm::SetMesImGCP()
{
     SetFinished();
     return mImGCP;
}

void cFilterMesIm::Save()
{
     MMVII_INTERNAL_ASSERT_medium(mFinished,"cFilterMesIm::Sve while not fnished");
     mPhProj.SaveMeasureIm(mMesIm);

     mPhProj.SaveAndFilterAttrEll(mMesIm,mSupr);
}

/* **************************************** */
/*                                          */
/*               cTiePMul                   */
/*                                          */
/* **************************************** */

cTiePMul::cTiePMul(cPt2dr  aPt,int anIndex) :
   mPt    (aPt) ,
   mId    (anIndex)
{
}

cTiePMul::cTiePMul() :
     cTiePMul(cPt2dr::Dummy(),-1)
{
}

void AddData(const cAuxAr2007 & anAux,cTiePMul & aPMul)
{
    AddData(cAuxAr2007("Pt",anAux),aPMul.mPt);
    AddData(cAuxAr2007("Id",anAux),aPMul.mId);
}

/* **************************************** */
/*                                          */
/*               cVecTiePMul                */
/*                                          */
/* **************************************** */

cVecTiePMul::cVecTiePMul(const std::string & anIm) :
   mNameIm (anIm)
{
}


}; // MMVII






