#include "MMVII_HeuristikOpt.h"
#include "MMVII_Mappings.h"

namespace MMVII
{


/**  Given a dim of point, and a Dist1Max corresponding to a neighboor (for ex : 1 for V4, 2 for V8 with Dim=2 ...),
 *    let P1 in a neighboord, we want to have the indexe of point P2 such that P1+P2 is not in a neighboorhoud,
 *
 *    It is used in "combinatory" optim, once we have the best point P1 in the neighboor, we want to explorate the points P2,
 *    neighboor of P1 that ,    have not been already explored at previus step (i.e were not in the neighboorhoud).
 *
 *   For ex, with Dim=2, DistMax= 2 , neigboorhood is something like (not sure numbering)
 *
 *          3  2  1
 *          4  x  0
 *          5  6  7
 *
 *       Res[0] = (1,0,7)
 *       Res[1] = (0,1,2,3,7)
 *   Also conventionnaly we add at the end a special entry with all neighboors
 *
 *      Res[8] = (0,1,2 ... 7)
 *
 */


template <const int Dim> const tVLInt &  VLNeighOptim(int aDist1Max)
{
    static std::map<int,tVLInt> TheMap;  // Buffer of result

    tVLInt & aVInt = TheMap[aDist1Max];
    if (aVInt.empty()) // if was not alerady computed
    {
       // tVLInt & aVInt = TheMap[aDist1Max];
       const auto & aVNeigh = AllocNeighbourhood<Dim>(aDist1Max);
       aVInt.resize(aVNeigh.size()+1);

       for (size_t aK1=0 ; aK1<aVNeigh.size() ; aK1++) // parse all neighboor
       {
           const cPtxd<int,Dim> & aP1 = aVNeigh.at(aK1);
           for (size_t aK2=0 ; aK2<aVNeigh.size() ; aK2++)
           {
                cPtxd<int,Dim>  aV12 =  aP1 + aVNeigh.at(aK2);
                //   P in neigboors  such  : NormInf(P) <=1 && Norm1(P) <= aDist1Max
                if ( (NormInf(aV12) >1) ||  (Norm1(aV12)>aDist1Max)  )
                   aVInt.at(aK1).push_back(aK2);
           }
           aVInt.at(aVNeigh.size()).push_back(aK1); // at the end we push all the  neighbours
       }
    }

    return aVInt;
}

/*  ******************************************* */
/*                                              */
/*              cOptimByStep                    */
/*                                              */
/*  ******************************************* */



template <const int Dim> tREAL8 cOptimByStep<Dim>::Value(const tPtR & aPt) const
{
     return mSign * mFunc.Value(aPt).x();
}

template <const int Dim> cOptimByStep<Dim>::cOptimByStep(const tMap & aMap,bool IsMin,tREAL8 aMaxDInfInit,int aDist1Max) :
    mFunc         (aMap),
    mSign         (IsMin ? 1.0 : -1.0),
    mWMin         (),
    mMaxDInfInit  (aMaxDInfInit),
    mDist1Max     (aDist1Max),
    mINeigh       (AllocNeighbourhood<Dim>(aDist1Max)),
    mIndNextN     (VLNeighOptim<Dim>(aDist1Max))
{
      VLNeighOptim<Dim>(aDist1Max);
}

template <const int Dim>
   std::pair<tREAL8,cPtxd<tREAL8,Dim>>  cOptimByStep<Dim>::Optim(const tPtR & aP0 ,tREAL8 aStepInit,tREAL8 aStepLim,tREAL8 aMul)
{
     mPt0 = aP0;
     mWMin = cWhichMin<tPtR,tREAL8>(aP0,Value(aP0));

     for (tREAL8 aStep = aStepInit ; aStep>= aStepLim ;  aStep *= aMul)
     {
         if (! DoOneStep(aStep) )
            return CurValue();
     }

     return CurValue();
}

template <const int Dim>   std::pair<tREAL8,cPtxd<tREAL8,Dim>> cOptimByStep<Dim>::CurValue() const
{
     return std::pair<tREAL8,tPtR>(mWMin.ValExtre()*mSign,mWMin.IndexExtre());
}
template <const int Dim> bool cOptimByStep<Dim>::DoOneStep(tREAL8 aStep)
{
    int aLastN = mINeigh.size(); // initially set to all neighboors
    for (;;)  // whil we can ameliorate
    {
        tPtR   aPOpt = mWMin.IndexExtre(); // current value of optimal point
        int aNewLast = -1; // by default no new index
        for (const auto & aKN : mIndNextN.at(aLastN))  // parse all indexes not already explored
        {
            tPtR aNewP = aPOpt + ToR(mINeigh.at(aKN)) * aStep;  // new point to test
            bool aNewBest =  mWMin.Add(aNewP,Value(aNewP));     // update optimum
            if (aNewBest) // if some update was made
            {
               aNewLast = aKN;  // the new best last indexe
               // precaution usefull with image processing
               if (NormInf(mPt0-mWMin.IndexExtre()) > mMaxDInfInit)
                  return false;
            }
        }
        //  if no update, finish w/o problem
        if (aNewLast<0)
           return true;
        aLastN = aNewLast;
    }
}


template class cOptimByStep<1>;
template class cOptimByStep<2>;
template class cOptimByStep<3>;



};

