#include "MMVII_PCSens.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_util_tpl.h"


/**
   \file Measures.cpp

   \brief classes 4 handling measurement in image
*/


namespace MMVII
{

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

void cMultipleImPt::Add(const cMesIm1Pt & aMes,int aNumIm,bool WithSigma)
{
    MMVII_INTERNAL_ASSERT_tiny(!WithSigma,"4 now dont handle sigma in cMultipleImPt");
    mVMeasures.push_back(aMes.mPt);
    mVImages.push_back(aNumIm);
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


cSetMesImGCP::cSetMesImGCP() :
	mPhaseGCPFinished (false)
{
}

void cSetMesImGCP::Add1GCP(const cMes1GCP & aMes)
{
     MMVII_INTERNAL_ASSERT_medium(!mPhaseGCPFinished,"cSetMesImGCP cannot add GCP after image");
     mMesGCP.push_back(aMes);
     m2MapPtInt.Add(aMes.mNamePt,false,"GCP-3D : " + aMes.mNamePt);
}

void cSetMesImGCP::AddMes3D(const cSetMesGCP &  aSet)
{
     for (const auto & aMes  : aSet.Measures())
	 Add1GCP(aMes);
}

const cSetMesPtOf1Im  & cSetMesImGCP::MesImInitOfName(const std::string & aNameIm) const
{
	return mMesImInit.at(m2MapImInt.Obj2I(aNameIm));
}

const cMes1GCP &  cSetMesImGCP::MesGCPOfName(const std::string & aNamePt) const
{
    return mMesGCP.at(m2MapPtInt.Obj2IWithMsg(aNamePt,"GCP : " + aNamePt));
}

cMes1GCP &  cSetMesImGCP::MesGCPOfName(const std::string & aNamePt)   
{
    return mMesGCP.at(m2MapPtInt.Obj2I(aNamePt));
}




const cMes1GCP &  cSetMesImGCP::MesGCPOfNum(int aNum) const
{
    return mMesGCP.at(aNum);
}

const cMes1GCP & cSetMesImGCP::MesGCPOfMulIm(const cMultipleImPt & aMIP) const
{
   return MesGCPOfNum(aMIP.NumPt());
}



bool  cSetMesImGCP::NameIsGCP(const std::string & aNamePt) const
{
  return m2MapPtInt.Obj2I(aNamePt,true) >= 0;
}


cPt3dr  cSetMesImGCP::BundleInter(const cMultipleImPt & aMPT) const
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



void cSetMesImGCP::AddMes2D(const cSetMesPtOf1Im & aSetMesIm,cSensorImage* aSens,eLevelCheck aOnNonExistGCP)
{
    //  Are we beginning  the  image measurement phase
    {
        if (! mPhaseGCPFinished)
        {
            mMesImOfPt.reserve(mMesGCP.size());

	    for (size_t aKp=0 ;  aKp<mMesGCP.size() ; aKp++)
                mMesImOfPt.push_back( cMultipleImPt(aKp));
        }
        mPhaseGCPFinished = true;
    }

    int aNumIm = m2MapImInt.Add(aSetMesIm.NameIm());
    if (aNumIm==(int)mMesImInit.size())
    {
       mMesImInit.push_back(cSetMesPtOf1Im(aSetMesIm.NameIm()));
       mVSens.push_back(aSens);
    }
    else
    {
        MMVII_INTERNAL_ASSERT_tiny(aNumIm<int(mMesImInit.size()),"Incoherence in cSetMesImGCP::AddMes2D");
        MMVII_INTERNAL_ASSERT_tiny(mVSens.at(aNumIm) == aSens,"Variable sensor in cSetMesImGCP::AddMes2D");
    }

    for (const auto & aMes : aSetMesIm.Measures())
    {
        int aNumPt = m2MapPtInt.Obj2I(aMes.mNamePt,true);
	if (aNumPt>=0)
	{
	    mMesImOfPt.at(aNumPt).Add(aMes,aNumIm,false);
            mMesImInit.at(aNumIm).AddMeasure(aMes);
	}
	else
	{
             ErrorWarnNone(aOnNonExistGCP,"Measure Im w/o Ground, first occur Im=" + aSetMesIm.NameIm() + " Pt="  + aMes.mNamePt);
	}
    }
}

const std::vector<cMes1GCP> &        cSetMesImGCP::MesGCP()    const  {return mMesGCP; }
const std::vector<cMultipleImPt> &   cSetMesImGCP::MesImOfPt() const  {return mMesImOfPt;  }
const std::vector<cSensorImage*> &   cSetMesImGCP::VSens()     const  {return mVSens;}
const std::vector<cSetMesPtOf1Im> &  cSetMesImGCP::MesImInit() const  {return mMesImInit;}

std::vector<cMes1GCP> &        cSetMesImGCP::MesGCP()   {return mMesGCP; }

void cSetMesImGCP::ExtractMes1Im(cSet2D3D&  aS23,const std::string &aNameIm,bool SVP) const
{
    aS23.Clear();


    int aNumIm = m2MapImInt.Obj2I(aNameIm,SVP);

    if (aNumIm<0) return;

    for (size_t aKp=0 ;  aKp<mMesGCP.size() ; aKp++)
    {
         const cPt2dr * aP2 = mMesImOfPt[aKp].PtOfIm(aNumIm);
	 if (aP2)
            aS23.AddPair(*aP2,mMesGCP[aKp].mPt);
    }
}

cSetMesImGCP *  cSetMesImGCP::FilterNonEmptyMeasure(int aNbMeasureMin) const
{
  cSetMesImGCP * aRes = new cSetMesImGCP;

  for (size_t aKGCP=0 ; aKGCP<mMesGCP.size() ; aKGCP++)
  {
       if ( int(mMesImOfPt[aKGCP].VImages().size()) >= aNbMeasureMin)
       {
             aRes->Add1GCP(mMesGCP[aKGCP]);
       }
   }

  for (size_t aKIm=0 ; aKIm<mMesImInit.size() ; aKIm++)
  {
     aRes->AddMes2D(mMesImInit.at(aKIm),mVSens.at(aKIm));
  }

   return aRes;
}

tREAL8 cSetMesImGCP::AvgSqResidual() const
{
     cWeightAv<tREAL8>  aWA;

     for (size_t aKPt=0 ; aKPt<mMesGCP.size() ; aKPt++)
     {
         const cPt3dr & aPGr = mMesGCP.at(aKPt).mPt;
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

     return std::sqrt(aWA.Average());
}

cSetMesGCP  cSetMesImGCP::ExtractSetGCP(const std::string & aName) const
{
    cSetMesGCP aRes(aName);
    for (const auto &  aMesGCP : mMesGCP)
        aRes.AddMeasure(aMesGCP);

    return aRes;
}


/* ********************************************* */
/*                                               */
/*             cMesIm1Pt                         */
/*                                               */
/* ********************************************* */
        
cMesIm1Pt::cMesIm1Pt(const cPt2dr & aPt,const std::string & aNamePt,tREAL4 aS2) :
     mPt      (aPt),
     mNamePt  (aNamePt),
     mSigma2   {aS2,0,aS2}
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
   AddTabData(cAuxAr2007("Sigma2",anAux),aMes.mSigma2,3);
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

void cSetMesPtOf1Im::AddMeasure(const cMesIm1Pt & aMeasure)
{
     mMeasures.push_back(aMeasure);
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

cMes1GCP::cMes1GCP(const cPt3dr & aPt,const std::string & aNamePt,tREAL4 aSigma) :
    mPt       (aPt),
    mNamePt   (aNamePt),
    mOptSigma2   { {0,0,0,0,0,0} }
{
    (*mOptSigma2)[IndXX] = aSigma;
    (*mOptSigma2)[IndYY] = aSigma;
    (*mOptSigma2)[IndZZ] = aSigma;
}

cMes1GCP::cMes1GCP() :
    cMes1GCP (cPt3dr(0,0,0),"??",-1)
{
}

void AddData(const  cAuxAr2007 & anAux,cMes1GCP & aMes)
{
   MMVII::AddData(cAuxAr2007("Name",anAux),aMes.mNamePt);
   MMVII::AddData(cAuxAr2007("Pt",anAux),aMes.mPt);
   AddOptTabData(anAux,"Sigma2",aMes.mOptSigma2);
}

/* ********************************************* */
/*                                               */
/*             cSetMesGCP                        */
/*                                               */
/* ********************************************* */


const std::string  cSetMesGCP::ThePrefixFiles = "MesGCP";

cSetMesGCP::cSetMesGCP(const std::string &aNameSet) :
    mNameSet(aNameSet)
{
}

cSetMesGCP::cSetMesGCP() :
    cSetMesGCP("???")
{
}

void cSetMesGCP::AddMeasure(const cMes1GCP & aMeasure)
{
     mMeasures.push_back(aMeasure);
}

void cSetMesGCP::AddData(const  cAuxAr2007 & anAuxParam)
{
    cAuxAr2007 anAux("SetGCP",anAuxParam);

    MMVII::AddData(cAuxAr2007("NameSet",anAux),mNameSet);
    MMVII::AddData(cAuxAr2007("Measures",anAux),mMeasures);
}

void AddData(const  cAuxAr2007 & anAux,cSetMesGCP & aSet)
{
     aSet.AddData(anAux);
}

cSetMesGCP  cSetMesGCP::FromFile(const std::string & aNameFile)
{
    cSetMesGCP aRes;
    ReadFromFile(aRes,aNameFile);

    return aRes;
}

void cSetMesGCP::ToFile(const std::string & aNameFile)
{
    // we do this because by side effect it check efficiently the absence of duplicata
    {
        cSetMesImGCP aMesIG;
        aMesIG.AddMes3D(*this);
    }
     SaveInFile(*this,aNameFile);
}



std::string cSetMesGCP::StdNameFileOfSet(const std::string & aName) 
{
     return ThePrefixFiles+"-"+ aName+ "."+GlobTaggedNameDefSerial();
}
std::string cSetMesGCP::StdNameFile() const {return StdNameFileOfSet(mNameSet);}

void cSetMesGCP::ToFile(const std::string & aNameFile) const
{
    SaveInFile(*this,aNameFile);
}


const std::vector<cMes1GCP> &   cSetMesGCP::Measures() const {return mMeasures;}

cSetMesGCP  cSetMesGCP::Filter(const std::string &aFilter) const
{
    cSetMesGCP  aRes(mNameSet);

    tNameSelector  aSelect =  AllocRegex(aFilter);

    for (const auto & aPt : mMeasures)
    {
         if (aSelect.Match(aPt.mNamePt))
            aRes.AddMeasure(aPt);
    }

    return aRes;
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
    mPhProj.LoadGCP(mImGCP);  // init new GCP/IM with GCP
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
const cSetMesImGCP &   cFilterMesIm::SetMesImGCP()
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






