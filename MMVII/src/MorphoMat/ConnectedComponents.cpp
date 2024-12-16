#include "MMVII_ImageMorphoMath.h"
namespace MMVII
{

void ConnectedComponent
     (
         std::vector<cPt2di> & aVPts,
         cDataIm2D<tU_INT1>  & aDIm,
         const std::vector<cPt2di> & aNeighbourhood,
         const  std::vector<cPt2di> & aVecSeed,
         int aMarqInit,
         int aNewMarq,
         bool Restore
     )
{
    aVPts.clear();
    for (const auto & aPSeed : aVecSeed)
    {
        if (aDIm.GetV(aPSeed) == aMarqInit)
        {
           aDIm.SetV(aPSeed,aNewMarq);
           aVPts.push_back(aPSeed);
        }
    }
    size_t aIndBottom = 0;

    while (aIndBottom!=aVPts.size())
    {
          cPt2di aP0 = aVPts[aIndBottom];
          for (const auto & aDelta : aNeighbourhood)
          {
              cPt2di aNeigh = aP0 + aDelta;
              if (aDIm.GetV(aNeigh)==aMarqInit)
              {
                  aDIm.SetV(aNeigh,aNewMarq);
                  aVPts.push_back(aNeigh);
              }
          }
          aIndBottom++;
    }

    if (Restore)
    {
       for (const auto & aPix : aVPts)
           aDIm.SetV(aPix,aMarqInit);
    }
}

void ConnectedComponent
     (
         std::vector<cPt2di> & aVPts,
         cDataIm2D<tU_INT1>  & aDIm,
         const std::vector<cPt2di> & aNeighbourhood,
         const cPt2di& aPtSeed,
         int aMarqInit,
         int aNewMarq,
         bool Restore
     )
{
    std::vector<cPt2di> aVSeed {aPtSeed};
    ConnectedComponent(aVPts,aDIm,aNeighbourhood,aVSeed,aMarqInit,aNewMarq,Restore);
}

void MakeImageDist(cIm2D<tU_INT1> aImIn,const std::string & aNameChamfer)
{
     // Apparently Distance image are not used for now, will write it if needed
     MMVII_INTERNAL_ERROR("No MakeImageDist");
     // const Chamfer & aChamf = Chamfer::ChamferFromName(aNameChamfer);
     // aChamf.im_dist(cMMV1_Conv<tU_INT1>::ImToMMV1(aImIn.DIm())) ;
}
/*
*/


};
