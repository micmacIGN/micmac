#ifndef  _MMVII_MAPPINGS_H_
#define  _MMVII_MAPPINGS_H_

namespace MMVII
{

/** \file MMVII_Mappings.h
    \brief contain interface class for continuous mapping

   Most probably this will evolve a lot, with several reengenering 
  phases. 

*/


/// Class that represent a continous mapping R^k -> R^n

template <const int DimIn,const int DimOut> class cMapping : public cMemCheck
{
    public :

      virtual  cPtxd<double,DimOut> Direct(const cPtxd<double,DimIn> &) const = 0;
};



};

#endif  //  _MMVII_MAPPINGS_H_
