
#include "MMVII_Sensor.h"


/**
   \file SensorBases.cpp

   \brief base classes used in all sensors
*/


namespace MMVII
{

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
/*           cSet2D3D                         */
/*                                            */
/**********************************************/

void cSet2D3D::AddPair(const cPair2D3D & aP23)
{
     mPairs.push_back(aP23);
}

const cSet2D3D::tCont2D3D &  cSet2D3D::Pairs() const { return mPairs;}

void  cSet2D3D::Clear()
{
	mPairs.clear();
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






