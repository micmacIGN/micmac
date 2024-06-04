#ifndef  _MMVII_MorphoMat_H_
#define  _MMVII_MorphoMat_H_

#include "MMVII_Image2D.h"


namespace MMVII
{

/** \file "MMVII_ImageMorphoMath.h"
    \brief Declaration functionnalities for morpho-mat & graph like processing
*/

/* *********************************************** */
/*                                                 */
/*         Graph & Flag of bits                    */
/*                                                 */
/* *********************************************** */

/**  Class for representing the connected component of 8-neigboorhoud, using Freeman code */

class cCCOfOrdNeigh
{
     public :
        bool mIs1;  // is it a component with 1
        int  mFirstBit;  // First neighboor (include)
        int  mLastBit;   // last neighboor  (exclude)
};


///  class for storing a subset of neighboors as flag of bit on Freman neighbourhood


class cFlag8Neigh
{
     public :

        cFlag8Neigh() : mFlag(0) {}
        cFlag8Neigh(tU_INT1 aFlag) : mFlag(aFlag) {}

        /// If bit to 1 , safe because work with bit<0 or bit>=8 (a bit slower ...)
        inline bool  SafeIsIn(size_t aBit) const { return (mFlag& (1<<mod(aBit,8))) != 0; }

        inline void  AddNeigh(size_t aBit)           { mFlag |= (1<<aBit); }

        /**   return the description of 4-connected component of neigbourhoud
	 * WARN : case empty of full (flag 0/255) generate no components, which can be good or not, 
	 * eventually will add the other option
	 */
        inline const std::vector<cCCOfOrdNeigh> &  ConComp() const
        {
             static std::vector<std::vector<cCCOfOrdNeigh> > aVCC;
             static bool First=true;
             if (First)
             {
                First = false;
                ComputeCCOfeNeigh(aVCC);
             }
             return aVCC.at(mFlag);
        }

        /// compute the number of connected component of a given neighbourhood flag
        inline int NbConComp() const { return ConComp().size(); }

     private :
        tU_INT1  mFlag;

        static void  ComputeCCOfeNeigh(std::vector<std::vector<cCCOfOrdNeigh>> & aVCC);
};

/** Compute, as a flag of bit, the set of Fremaan-8 neighboor that are > over a pixel,
 * to have a strict order, for value==, the comparison is done on Y then X */
template <class Type>  cFlag8Neigh   FlagSup8Neigh(const cDataIm2D<Type> & aDIm,const cPt2di & aPt);


/**  Given a point, that is  a "topologicall saddle" (i.e multiple connected component of points >) ,
 * carecterize the "importance" of saddle */
template <class Type>  tREAL8   CriterionTopoSadle(const cDataIm2D<Type> & aDIm,const cPt2di & aPixC);


/**  Put in vector "VPts",  the connected component of a seed , of point having the same colour in Image,
 * update  with "NewMarq" */
 
void ConnectedComponent
     (
         std::vector<cPt2di> & aVPts,  // Vector of results
         cDataIm2D<tU_INT1>  & aDIm,   // Image marquing point of CC to compute
         const std::vector<cPt2di> & aNeighbourhood,  // generally 4 or 8 neighbourhood
         const std::vector<cPt2di>& aSeed,         // seeds of the connected component
         int aMarqInit=1,             // will extract CC of point having this colour
         int aNewMarq=0,              // will marq points reached with this colour
	 bool Restore=false           // if true restore initial marq at end
     );

/**  Idem, current and simplified case where the seed is a single point */
void ConnectedComponent
     (
         std::vector<cPt2di> & , cDataIm2D<tU_INT1>  & , const std::vector<cPt2di> & aNeighbourhood, 
         const cPt2di& aSeed,         // seed of the connected component
         int aMarqInit=1,  int aNewMarq=0,  bool Restore=false 
     );

};





#endif  //   _MMVII_MorphoMat_H_
