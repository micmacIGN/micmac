#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"


/**
   \file SensorBases.cpp

   \brief base classes used in all sensors
*/


namespace MMVII
{

template <class Type>  class cBijectiveMapI2O
{
    public :

        void Add(const Type & ,bool OkExist=false);

	Type *   I2Obj(const int) ;
	int      Obj2I(const Type & anOb);

    private :
        std::vector<Type>    mI2Obj;
        std::map<Type,int>   mObj2I;
};

template <class Type> 
   void cBijectiveMapI2O<Type>::Add(const Type & anObj,bool OkExist)
{
    if (mObj2I.find(anObj) != mObj2I.end())
    {
       MMVII_INTERNAL_ASSERT_tiny(OkExist,"cBijectiveMapI2O multiple add");
       return;
    }

    mObj2I[anObj] = mI2Obj.size();
    mI2Obj.push_back(anObj);
}

template <class Type> 
   Type * cBijectiveMapI2O<Type>::I2Obj(int anInd) 
{
   if ( (anInd<0) || (anInd>=int(mObj2I.size())) )
      return nullptr;

   return & mI2Obj.at(anInd);
}

template <class Type> 
   int  cBijectiveMapI2O<Type>::Obj2I(const Type & anObj)
{
    const auto & anIt = mObj2I.find(anObj) ;

    if (anIt== mObj2I.end())  return -1;

    return anIt->second;
}


template class cBijectiveMapI2O<std::string>;



class cSetMesImGCP
{
    public :
    private :

	    cBijectiveMapI2O<std::string>  m2MapPtInt;
};

/* ********************************************* */
/*                                               */
/*             cMesIm1Pt                         */
/*                                               */
/* ********************************************* */
        
cMesIm1Pt::cMesIm1Pt(const cPt2dr & aPt,const std::string & aNamePt,tREAL8 aS) :
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

std::string cSetMesPtOf1Im::StdNameFileOfIm(const std::string & aName)
{
	return "MesIm-"+aName+ "."+PostF_XmlFiles;
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

cMes1GCP::cMes1GCP(const cPt3dr & aPt,const std::string & aNamePt,tREAL8 aSigma) :
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


cSetMesGCP::cSetMesGCP(const std::string &aNameSet) :
    mNameSet(aNameSet)
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

std::string cSetMesGCP::StdNameFileOfSet(const std::string & aName) 
{
     return "MesGCP-"+aName+ "."+PostF_XmlFiles;
}
std::string cSetMesGCP::StdNameFile() const {return StdNameFileOfSet(mNameSet);}

void cSetMesGCP::ToFile(const std::string & aNameFile) const
{
    SaveInFile(*this,aNameFile);
}


/*

          void ToFile(const std::string & aNameFile) const;
          static std::string StdNameFileOfSet(const std::string &);
*/

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




/* ******************************************************* */
/*                                                         */
/*                    cPixelDomain                         */
/*                    cDataPixelDomain                     */
/*                                                         */
/* ******************************************************* */

      //  ============ cDataPixelDomain  ================

cDataPixelDomain::cDataPixelDomain(const cPt2di &aSz) :
     mSz  (aSz)
{
}

const cPt2di & cDataPixelDomain::Sz() const {return mSz;}

void cDataPixelDomain::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("NbPix",anAux),mSz);
}

      //  ============ cPixelDomain  ================

cPixelDomain::cPixelDomain(cDataPixelDomain * aDPD) :
     cDataBoundedSet<tREAL8,2>(cBox2dr(cPt2dr(0,0),ToR(aDPD->Sz()))),
     mDPD  (aDPD)
{
}

const cPt2di & cPixelDomain::Sz() const {return mDPD->Sz();}

/* ******************************************************* */
/*                                                         */
/*                   cSensorImage                          */
/*                                                         */
/* ******************************************************* */

cSensorImage::cSensorImage(const std::string & aNameImage) :
     mNameImage (aNameImage)
{
}

void cSensorImage::SetNameImage(const std::string & aNameImage)
{
     mNameImage = aNameImage;
}

const std::string & cSensorImage::NameImage() const {return mNameImage;}

double  cSensorImage::SqResidual(const cPair2D3D & aPair) const
{
     return SqN2(aPair.mP2-Ground2Image(aPair.mP3));
}

double  cSensorImage::AvgResidual(const cSet2D3D & aSet) const
{
     double aSum = 0;
     for (const auto & aPair : aSet.Pairs() )
     {
         aSum +=  SqResidual(aPair);
     }
     return std::sqrt(aSum/aSet.Pairs().size());
}


std::string cSensorImage::PrefixName() { return "Ori"; }


std::string  cSensorImage::NameOri_From_PrefixAndImage(const std::string & aPrefix,const std::string & aNameImage)
{ 
    return PrefixName() + "-" + aPrefix + "-" + aNameImage + ".xml"; 
}
std::string cSensorImage::NameOriStd() const { return  NameOri_From_PrefixAndImage(V_PrefixName(),mNameImage);}


cPt3dr cSensorImage::ImageAndDepth2Ground(const cPt2dr & aP2,const double & aDepth) const 
{
    return ImageAndDepth2Ground(cPt3dr(aP2.x(),aP2.y(),aDepth));
}


cSet2D3D  cSensorImage::SyntheticsCorresp3D2D (int aNbByDim,std::vector<double> & aVecDepth) const
{
    cSet2D3D aResult;

    std::vector<cPt2dr>  aVPts =  PtsSampledOnSensor(aNbByDim);

    for (const auto & aPIm : aVPts)
    {
        for (const auto & aDepth : aVecDepth)
        {
	     aResult.AddPair(aPIm,ImageAndDepth2Ground(aPIm,aDepth));
	}
    }

    return aResult;
}
         ///  call variant with vector, depth regularly spaced
         // cSet2D3D  SyntheticsCorresp3D2D (int aNbByDim,int aNbDepts,double aD0,double aD1) const;


/* ******************************************************* */
/*                                                         */
/*                   cSIMap_Ground2ImageAndProf            */
/*                                                         */
/* ******************************************************* */

cSIMap_Ground2ImageAndProf::cSIMap_Ground2ImageAndProf(cSensorImage * aSens)  :
    mSI  (aSens)
{
}

cPt3dr cSIMap_Ground2ImageAndProf::Value(const cPt3dr & aPt) const
{
	return mSI->Ground2ImageAndDepth(aPt);
}

cPt3dr cSIMap_Ground2ImageAndProf::Inverse(const cPt3dr & aPt) const
{
	return mSI->ImageAndDepth2Ground(aPt);
}

/* ******************************************************* */
/*                                                         */
/*                   cSetVisibility                        */
/*                                                         */
/* ******************************************************* */

cSetVisibility::cSetVisibility(cSensorImage * aSens,double aBorder) :
            cDataBoundedSet<tREAL8,3> (cBox3dr::BigBox()),
            mSens                     (aSens),
	    mBorder                   (aBorder)
{}

tREAL8 cSetVisibility::Insideness(const tPt & aP) const 
{
    return mSens->Visibility(aP) - mBorder;
}



}; // MMVII






