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

/*
struct cSet2D3D
{
     public :
         typedef std::vector<cPair2D3D>   tCont2D3D;

         void AddPair(const cPair2D3D &);
         const tCont2D3D &  Pairs() const;
         void  clear() ;

     private :
        tCont2D3D  mPairs;
};
*/



}; // MMVII

