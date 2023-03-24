#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"


/**
   \file Measures.cpp

   \brief classes 4 handling measurement in image
*/


namespace MMVII
{


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

/* ********************************************* */
/*                                               */
/*             cSetMesImGCP                      */
/*                                               */
/* ********************************************* */


cSetMesImGCP::cSetMesImGCP() :
	mPhaseGCPFinished (false)
{
}

void cSetMesImGCP::AddMes3D(const cSetMesGCP &  aSet)
{
     MMVII_INTERNAL_ASSERT_medium(!mPhaseGCPFinished,"cSetMesImGCP cannot add GCP after image");
     for (const auto & aMes  : aSet.Measures())
     {
         mMesGCP.push_back(aMes);
	 m2MapPtInt.Add(aMes.mNamePt);
     }
}

void cSetMesImGCP::AddMes2D(const cSetMesPtOf1Im & aSetMesIm)
{
    //  Are we beginning  the  image measurement phase
    {
        if (! mPhaseGCPFinished)
        {
            mMesIm.reserve(mMesGCP.size());

	    for (size_t aKp=0 ;  aKp<mMesGCP.size() ; aKp++)
                mMesIm.push_back( cMultipleImPt(aKp));
        }
        mPhaseGCPFinished = true;
    }

    int aNumIm = m2MapImInt.Add(aSetMesIm.NameIm());
    for (const auto & aMes : aSetMesIm.Measures())
    {
        int aNumPt = m2MapPtInt.Obj2I(aMes.mNamePt);
	mMesIm.at(aNumPt).Add(aMes,aNumIm,false);
    }
}


void cSetMesImGCP::ExtractMes1Im(cSet2D3D&  aS23,const std::string &aNameIm)
{
    aS23.Clear();

    int aNumIm = m2MapImInt.Obj2I(aNameIm);

    for (size_t aKp=0 ;  aKp<mMesGCP.size() ; aKp++)
    {
         const cPt2dr * aP2 = mMesIm[aKp].PtOfIm(aNumIm);
	 if (aP2)
            aS23.AddPair(*aP2,mMesGCP[aKp].mPt);
    }
}

/* ********************************************* */
/*                                               */
/*             cMesIm1Pt                         */
/*                                               */
/* ********************************************* */
        
cMesIm1Pt::cMesIm1Pt(const cPt2dr & aPt,const std::string & aNamePt,tREAL4 aS) :
     mPt      (aPt),
     mNamePt  (aNamePt),
     mSigma2   {aS,0,aS}
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

void cSetMesPtOf1Im::AddMeasure(const cMesIm1Pt & aMeasure)
{
     mMeasures.push_back(aMeasure);
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
	return ThePrefixFiles +aName+ "."+PostF_XmlFiles;
}

std::string cSetMesPtOf1Im::StdNameFile() const
{
    return StdNameFileOfIm(mNameIm);
}




/* ********************************************* */
/*                                               */
/*             cMes1GCP                          */
/*                                               */
/* ********************************************* */

cMes1GCP::cMes1GCP(const cPt3dr & aPt,const std::string & aNamePt,tREAL4 aSigma) :
    mPt       (aPt),
    mNamePt   (aNamePt),
    mSigma2   {aSigma,0,0,aSigma,0,aSigma}
{
}

cMes1GCP::cMes1GCP() :
    cMes1GCP (cPt3dr(0,0,0),"??",-1)
{
}

void AddData(const  cAuxAr2007 & anAux,cMes1GCP & aMes)
{
   MMVII::AddData(cAuxAr2007("Name",anAux),aMes.mNamePt);
   MMVII::AddData(cAuxAr2007("Pt",anAux),aMes.mPt);
   AddTabData(cAuxAr2007("Sigma2",anAux),aMes.mSigma2,6);
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



std::string cSetMesGCP::StdNameFileOfSet(const std::string & aName) 
{
     return ThePrefixFiles+"-"+ aName+ "."+PostF_XmlFiles;
}
std::string cSetMesGCP::StdNameFile() const {return StdNameFileOfSet(mNameSet);}

void cSetMesGCP::ToFile(const std::string & aNameFile) const
{
    SaveInFile(*this,aNameFile);
}


const std::vector<cMes1GCP> &   cSetMesGCP::Measures() const {return mMeasures;}



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



}; // MMVII






