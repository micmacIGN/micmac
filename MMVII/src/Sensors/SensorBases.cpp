#include "include/MMVII_all.h"


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
/*                                                         */
/* ******************************************************* */

cPixelDomain::~cPixelDomain()
{
}

cPixelDomain::cPixelDomain(const cPt2di &aSz) :
     cDataBoundedSet<tREAL8,2>(cBox2dr(cPt2dr(0,0),ToR(aSz))),
     mSz  (aSz)
{
}

cPixelDomain *  cPixelDomain::Dup_PS () const
{
    return new cPixelDomain(mSz);
}

const cPt2di & cPixelDomain::Sz() const {return mSz;}


/* ******************************************************* */
/*                                                         */
/*                   cSensorImage                          */
/*                                                         */
/* ******************************************************* */

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




}; // MMVII

