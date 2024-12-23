#include <cMMVII_Appli.h>
#include "MMVII_Geom3D.h"
#include "MMVII_Mappings.h"

#include <memory>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/io/LasReader.hpp>
#include <pdal/io/LasHeader.hpp>
#include <pdal/Options.hpp>

#include <pdal/Reader.hpp>
#include <pdal/Writer.hpp>
#include <pdal/Streamable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/util/ProgramArgs.hpp>

namespace MMVII {

  /* ********************************************************** */
  /*                                                            */
  /*                 cTriangulation3D                           */
  /*                                                            */
  /* ********************************************************** */


 using  namespace pdal;

  struct ClassificationTags
  {
    const int8_t Unclassified=1;
    const int8_t Ground=2;
    const int8_t Low_Vegetation=3;
    const int8_t Medium_Vegetation=4;
    const int8_t High_Vegetation=5;
    const int8_t Building=6;
    const int8_t Water=9;
    const std::string DSMMarker="dsm_marker";
    const std::string DTMMarker="dtm_marker";
  };

  enum class eLabelIm_MASQ : tU_INT1
  {
     eFree,     // Mode MicMac V1
     eReached,  // Mode filled
     eNbVals
  };


template <class Type>  cTriangulation3DLas<Type>::cTriangulation3DLas(const std::string & aName):
        cTriangulation<Type,3>(std::vector<tPt>())
{
    if (UCaseEqual(LastPostfix(aName),"las") || UCaseEqual(LastPostfix(aName),"laz") )
    {
       LasInit(aName);
    }
    else
    {
       MMVII_UserError(eTyUEr::eBadPostfix,"Unknown postfix in cTriangulation3D");
    }
}

template <class Type> unsigned int cTriangulation3DLas<Type>::NbPts()
  {
    return this->mVPts.size();
  }

template <class Type> const char cTriangulation3DLas<Type>::ProjStr()
  {
    return *mProjStr;
  }

template <class Type> bool cTriangulation3DLas<Type>::HasTime()
  {
    return mHasTime;
  }

template <class Type> bool cTriangulation3DLas<Type>::HasColor()
  {
    return mHasColor;
  }



template <class Type> void cTriangulation3DLas<Type>::SamplePts(const bool & targetted,const Type & aStep)
  {
    /// < Sample points either by targetted or by random sampling
    mVSelectedIds.clear();
   if (targetted)
     {
       // sample points in a grid

       /*
        * |----- |  -----  | -------|
        * |----- |  -----  | -------|
        * |----- |  -----  | -------|
        */
       // Random Ordering of points
       this->mVPts=RandomOrder(this->mVPts);

       // Empty grid of points to sample with size bbox/aStep
       cDataTypedIm<tU_INT1,2> aD_Grid(cPt2di(0,0),
                                     cPt2di(Pt_round_down(mDelimitBox.Sz()/aStep))
                                     );
       aD_Grid.InitCste(0);

       // fill grid
       size_t it=0;
       int allCellsFilled=aD_Grid.NbElem();
       while(it<this->mVPts.size() || allCellsFilled)
         {
            tPt aPt=this->mVPts[it];
            cPt2di aPix=cPt2di(aPt.x()/aStep,aPt.y()/aStep);
            if (aD_Grid.VI_GetV(aPix)==tU_INT1(eLabelIm_MASQ::eReached))
              {
                continue;
              }
            else
              {
                aD_Grid.VI_SetV(aPix,1);
                mVSelectedIds.push_back(it);
                allCellsFilled-=1;
              }
            it+=1;
         }
     }
   else
     {
     }
}

template <class Type> void cTriangulation3DLas<Type>::LasInit(const std::string & aNameFile)
{
  pdal::Option las_opt("filename", aNameFile);
  pdal::Options las_opts;
  las_opts.add(las_opt);
  pdal::PointTable table;
  pdal::LasReader las_reader;
  las_reader.setOptions(las_opts);
  las_reader.prepare(table);
  pdal::PointViewSet point_view_set = las_reader.execute(table);
  pdal::PointViewPtr point_view = *point_view_set.begin();
  //pdal::Dimension::IdList dims = point_view->dims();
  pdal::LasHeader las_header = las_reader.header();

  std::cout<<"POINT VIEW SIZE "<<point_view->size()<<std::endl;

  {
    /* Point cloud properties */

    mNbPoints= las_header.pointCount();
    /** @brief spatial reference
     */
    mProjStr = table.spatialReference().getWKT().c_str();

    mHasTime = las_header.hasTime();
    mHasColor = las_header.hasColor();
    //pdal::Dimension::IdList dims = point_view->dims()
    //bounding box
    mDelimitBox=cTplBox<Type,2>(cPtxd<Type,2>(las_header.minX(),las_header.minY()),
                        cPtxd<Type,2>(las_header.maxX(),las_header.maxY()));
  }
  // Read Points and classification


  //std::cout<<" GET ALL DIMENSIONS "<<table.layout()->dims()<<std::endl;
  auto aDsmMarkerDim = table.layout()->findProprietaryDim(ClassificationTags().DSMMarker);
  bool HasDsmMarker=table.layout()->hasDim(aDsmMarkerDim);

  if (HasDsmMarker) // read points tagged as useful for DSM generation and not on trees
    {
      for (pdal::PointId idx = 0; idx < point_view->size(); ++idx)
      {
         using namespace pdal::Dimension;
         auto IsForDsm=point_view->getFieldAs<int>(aDsmMarkerDim,idx);
         auto Classif=point_view->getFieldAs<int>(Id::Classification,idx);
         bool IsVeg=((Classif==ClassificationTags().Low_Vegetation) ||
                     (Classif==ClassificationTags().Medium_Vegetation) ||
                     (Classif==ClassificationTags().High_Vegetation)
                     );

         if (IsForDsm && !IsVeg)
           {
             tPt aP(point_view->getFieldAs<tREAL8>(Id::X, idx),
                    point_view->getFieldAs<tREAL8>(Id::Y, idx),
                    point_view->getFieldAs<tREAL8>(Id::Z, idx));
             this->mVPts.push_back(aP);
           }
      }
    }
  else  // assume point cloud is classified -> if there is not a tag dsm marker get points in GROUND, BUILDINGS
    {
        using namespace pdal::Dimension;

      for (pdal::PointId idx = 0; idx < point_view->size(); ++idx)
      {
        auto Classif=point_view->getFieldAs<int>(Id::Classification, idx);
        bool IsBuilding=(Classif==ClassificationTags().Building);
        bool IsGround=(Classif==ClassificationTags().Ground);
        bool IsUnclassified=(Classif==ClassificationTags().Unclassified);

        if ( IsBuilding || IsGround || IsUnclassified)
          {
            tPt aP(point_view->getFieldAs<tREAL8>(Id::X, idx),
                   point_view->getFieldAs<tREAL8>(Id::Y, idx),
                   point_view->getFieldAs<tREAL8>(Id::Z, idx));
            this->mVPts.push_back(aP);
          }
       }
    }

  // Read faces
  /*{
      std::vector<std::vector<size_t>> aVFace =   aLazF.getFaceIndices<size_t>();
      for (const auto & aFace : aVFace)
      {
          MMVII_INTERNAL_ASSERT_tiny(aFace.size()==3,"Bad face");
          this->AddFace(cPt3di(aFace[0],aFace[1],aFace[2]));
      }
  }*/

}



/* ========================== */
/*     INSTANTIATION          */
/* ========================== */

#define INSTANTIATE_TRI3DLAS(TYPE)\
template class cTriangulation3DLas<TYPE>;

INSTANTIATE_TRI3DLAS(tREAL4)
INSTANTIATE_TRI3DLAS(tREAL8)
INSTANTIATE_TRI3DLAS(tREAL16)

};
