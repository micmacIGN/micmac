#ifndef  _MMVII_RANDOM_H_
#define  _MMVII_RANDOM_H_

#include "MMVII_memory.h"
#include <random>

namespace MMVII
{


/** \file MMVII_Random.h
    \brief Pseudo-random generator

*/

class cRandGenerator : public cMemCheck
{
   public :
       virtual double Unif_0_1() = 0;
       virtual int    Unif_N(int aN) = 0;
       virtual void   setSeed(size_t aSeed) = 0;
       virtual size_t next() = 0;
       static cRandGenerator * TheOne();
       static void Close();
       static void Open();
       virtual ~cRandGenerator() {};
    private :
       static cRandGenerator * msTheOne;
};

};

#endif  //  _MMVII_RANDOM_H_
