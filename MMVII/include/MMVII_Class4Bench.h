#ifndef  _MMVII_Class4Dench_H_
#define  _MMVII_Class4Dench_H_
#include "MMVII_memory.h"
#include "MMVII_Ptxd.h"

namespace MMVII
{

/** \file MMVII_Class4Bench.h 
    \brief Declare class used in bench, but who require some template instantiation
*/



/// class to illustrate basic serialization

class cTestSerial0
{
     public :
        cTestSerial0();
        bool operator ==   (const cTestSerial0 & aT0) const ;
        cPt2dr mP1;
        cPt2dr mP2;
};

//  To serialize cTestSerial0, just indicate that it is made of mP1 and mP2

void AddData(const cAuxAr2007 & anAux, cTestSerial0 &    aTS0);

///  a more complex class to illustrate serializaion
/**
    This class illustrate that there is no problem to use
  recursively the  serializain: once AddData has been defined
  in cTestSerial0 it can be used in AddData
*/

class cTestSerial1 : public cMemCheck
{
     public :
        cTestSerial1() ;

        bool operator ==   (const cTestSerial1 & aT1) const ;
        cTestSerial0            mTS0;
        std::string             mS;
        cPt2dr                  mP3;
        std::list<int>          mLI;
        std::vector<double>      mVD;
        std::optional<cPt2dr>   mO1;
        std::optional<cPt2dr>   mO2;
};
void AddData(const cAuxAr2007 & anAux, cTestSerial1 &    aTS1) ;


};

#endif  //  _MMVII_Class4Dench_H_
