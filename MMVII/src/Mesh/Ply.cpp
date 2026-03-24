#include "cMMVII_Appli.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Mappings.h"
#include "MMVII_2Include_Tiling.h"
#include "MMVII_PCSens.h"
#include "../PoseEstim/VisPoseAndStructure.h"

#include <omp.h>

#if MMVII_USE_PDAL
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
#endif

#include "ogrsf_frmts.h"
#include "MMVII_AimeTieP.h"
//#include "V1VII.h"
#include "MMVII_PtCorrel.h"
#include "MMVII_Stringifier.h"
#include <thread>


#include "happly.h"

#define WITH_MMV1_FUNCTION  false


namespace MMVII
{
/* A header-only implementation of the .ply file format.
 * https://github.com/nmwsharp/happly
 * By Nicholas Sharp - nsharp@cs.cmu.edu
 *
 * Version 2, July 20, 2019
 */

/*
MIT License

Copyright (c) 2018 Nick Sharp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


// clang-format off
/*

 === Changelog ===

  Significant changes to the file recorded here.

  - Version 5 (Aug 22, 2020)      Minor: skip blank lines before properties in ASCII files
  - Version 4 (Sep 11, 2019)      Change internal list format to be flat. Other small perf fixes and cleanup.
  - Version 3 (Aug 1, 2019)       Add support for big endian and obj_info
  - Version 2 (July 20, 2019)     Catch exceptions by const reference.
  - Version 1 (undated)           Initial version. Unnamed changes before version numbering.

*/
// clang-format on



#if MMVII_USE_PDAL
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
   const int8_t FictiveWaterBridge=66;
   const std::string DSMMarker="dsm_marker";
   const std::string DTMMarker="dtm_marker";
 };
#endif
 enum class eLabelIm_MASQ : tU_INT1
 {
    eFree,     // Mode MicMac V1
    eReached,  // Mode filled
    eNbVals
 };
/* ********************************************************** */
/*                                                            */
/*                 cTriangulation3D                           */
/*                                                            */
/* ********************************************************** */


using  namespace happly;

template <class Type> void cTriangulation3D<Type>::PlyWrite (const std::string & aNameFile,bool isBinary) const
{
  try
  {
   PLYData aPlyOut;
   //  convert Pts to array
   std::vector<std::array<double, 3>> aPlyPts;
   for (const auto & aPts : this->mVPts)
   {
       std::array<double,3> anArray;
       for (int aK=0 ; aK<3 ; aK++)
           anArray[aK] = aPts[aK];
       aPlyPts.push_back(anArray);
   }
   aPlyOut.addVertexPositions(aPlyPts);

   //  convert faces to vect
   std::vector<std::vector<size_t>> aPlyFaces;
   for (const auto & aFace : this->mVFaces)
   {
       std::vector<size_t> aVect;
       // aVect.push_back(3);
       for (int aK=0 ; aK<3 ; aK++)
           aVect.push_back(aFace[aK]);
       aPlyFaces.push_back(aVect);
   }
   aPlyOut.addFaceIndices(aPlyFaces);

   // Write data

   aPlyOut.write(aNameFile,  (isBinary?happly::DataFormat::Binary:happly::DataFormat::ASCII));
  }
  catch (const std::runtime_error &e)
  {
      MMVII_UserError(eTyUEr::eReadFile, std::string("Error writing PLY file \"") + aNameFile + "\": " + e.what());
  }
}

template <class Type> void cTriangulation3D<Type>::PlyWriteSelected (const std::string & aNameFile,
                                                                     std::list<std::vector<int> > & Patches,
                                                                     bool isBinary) const
{
  PLYData aPlyOut;
  //  convert Pts to array
  std::vector<std::array<double, 3>> aPlyPts;
    for (auto & aPatchId: Patches)
      {
        for (auto & anId:  aPatchId)
          {
            auto aPt=this->mVPts[anId];
            std::array<double,3> anArray;
            for (int aK=0 ; aK<3 ; aK++)
                anArray[aK] = aPt[aK];
            aPlyPts.push_back(anArray);
          }

      }

  aPlyOut.addVertexPositions(aPlyPts);
  // Write data
  aPlyOut.write(aNameFile,  (isBinary?happly::DataFormat::Binary:happly::DataFormat::ASCII));
}

template <class Type> void cTriangulation3D<Type>::PlyInit(const std::string & aNameFile)
{
 try
 {
  PLYData  aPlyF(aNameFile,false);
  auto aElementsNames = aPlyF.getElementNames();
  // Read points
    {
      std::vector<std::array<double, 3>> aVecPts = aPlyF.getVertexPositions() ;
      for (const auto & aPos : aVecPts)
      {
	  tPt aP(aPos.at(0),aPos.at(1),aPos.at(2));
	  this->mVPts.push_back(aP);
      }
    }
  // Read faces ("face" part is not mandatory in ply)
  if ( std::find(aElementsNames.begin(), aElementsNames.end(), "face")!= aElementsNames.end())
    {
      std::vector<std::vector<size_t>> aVFace =   aPlyF.getFaceIndices<size_t>();
      for (const auto & aFace : aVFace)
      {
      MMVII_INTERNAL_ASSERT_tiny(aFace.size()==3,"Bad face");
      this->AddFace(cPt3di(aFace[0],aFace[1],aFace[2]));
      }
    }
 }
 catch (const std::runtime_error &e)
 {
    MMVII_UserError(eTyUEr::eReadFile, std::string("Error reading PLY file \"") + aNameFile + "\": " + e.what());
 }
}

#if MMVII_USE_PDAL
 template <class Type> void cTriangulation3D<Type>::LasInit(const std::string & aNameFile, bool SelectForRegistration)
 {
   //StdOut()<<"START READING LAZ "<<std::endl;
   pdal::Option las_opt("filename", aNameFile);
   pdal::Options las_opts;
   las_opts.add(las_opt);
   pdal::PointTable table;
   pdal::LasReader las_reader;
   las_reader.setOptions(las_opts);
   las_reader.prepare(table);
   pdal::PointViewSet point_view_set = las_reader.execute(table);
   pdal::PointViewPtr point_view = *point_view_set.begin();
   pdal::LasHeader las_header = las_reader.header();
   std::cout<<"POINT VIEW SIZE "<<point_view->size()<<std::endl;
   auto aDsmMarkerDim = table.layout()->findProprietaryDim(ClassificationTags().DSMMarker);
   std::cout<<"DSM MARKER "<<pdal::Dimension::description(aDsmMarkerDim)<<std::endl;
      #pragma omp parallel
       {
           //omp_set_num_threads(16);
           using namespace pdal::Dimension;
           std::vector <tPt> aVPts_pp;
           #pragma omp for
           for (pdal::PointId idx = 0; idx < point_view->size(); ++idx)
           {
               if (SelectForRegistration)
               {
                    auto Classif=point_view->getFieldAs<int>(Id::Classification, idx);
                    bool IsUnclassified=(Classif==ClassificationTags().Unclassified);
                    bool IsFictive  = (Classif == 66);
                    bool IsWater=(Classif==ClassificationTags().Water);
                    bool IsVeg=(Classif==ClassificationTags().Low_Vegetation) ||
                               (Classif==ClassificationTags().Medium_Vegetation) ||
                               (Classif==ClassificationTags().High_Vegetation) ;


                if (! (IsWater || IsVeg || IsUnclassified || IsFictive)  )
                   {
                       tPt aP(point_view->getFieldAs<tREAL8>(Id::X, idx),
                              point_view->getFieldAs<tREAL8>(Id::Y, idx),
                              point_view->getFieldAs<tREAL8>(Id::Z, idx));
                       aVPts_pp.push_back(aP);
                   }
               }

               else
               {
                   tPt aP(point_view->getFieldAs<tREAL8>(Id::X, idx),
                          point_view->getFieldAs<tREAL8>(Id::Y, idx),
                          point_view->getFieldAs<tREAL8>(Id::Z, idx));
                   aVPts_pp.push_back(aP);
               }
           }
           #pragma omp critical
           this->mVPts.insert(this->mVPts.end(),
                              aVPts_pp.begin(),
                              aVPts_pp.end());
        }

       StdOut()<<"selecting points while reading las "<<std::endl;

 }
#endif

template <class Type> void cTriangulation3D<Type>::Bench()
{
/*
	 There is a problem with testing on ply based on triangulation equality

	 Will require more investigation,  for now test is MMVII_INTERNAL_ASSERT_Unresolved
*/
    std::string aDirI = cMMVII_Appli::CurrentAppli().InputDirTestMMVII() + "Ply/" ;

    tTriangulation3D  aTriTxt3D(aDirI+"MeshTxt.ply");
    tTriangulation3D  aTriBin3D(aDirI+"MeshBin.ply");

    MMVII_INTERNAL_ASSERT_Unresolved(aTriTxt3D.HeuristikAlmostEqual(aTriBin3D,1e-5,1e-5),"cTriangulation3D::Bench");
    //  With -1 on Tol Face : OK
    MMVII_INTERNAL_ASSERT_bench(aTriTxt3D.HeuristikAlmostEqual(aTriBin3D,1e-5,-1),"cTriangulation3D::Bench");
    //           bool  AlmostEqual (const cTriangulation<Dim> &,double)  const;

    // StdOut() << "SzP= " << aTriTxt3D.mVPts.size() << " " << aTriBin3D.mVPts.size() << std::endl;

    std::string aDirTmp = cMMVII_Appli::CurrentAppli().TmpDirTestMMVII();
    aTriBin3D.WriteFile(aDirTmp+"MeshBin.ply",true);
    aTriTxt3D.WriteFile(aDirTmp+"MeshTxt.ply",false);

    tTriangulation3D  aNewTriTxt3D(aDirTmp+"MeshTxt.ply");
    tTriangulation3D  aNewTriBin3D(aDirTmp+"MeshBin.ply");
    //  With -1 on Tol Face : OK
    MMVII_INTERNAL_ASSERT_bench(aTriBin3D.HeuristikAlmostEqual(aNewTriBin3D,1e-5,1e-5),"cTriangulation3D::Bench");
    MMVII_INTERNAL_ASSERT_bench(aTriTxt3D.HeuristikAlmostEqual(aNewTriTxt3D,1e-5,1e-5),"cTriangulation3D::Bench");

#if (WITH_MMV1_FUNCTION)
    cDataBoundedSet<tREAL8,3> * aMasq=  MMV1_Masq(aNewTriBin3D.BoxEngl().ToR(),aDirI+"AperiCloud_Basc_selectionInfo.xml");
    aNewTriBin3D.Filter(*aMasq);
    aNewTriBin3D.WriteFile(aDirTmp+"MeshFilteredBin.ply",true);

    //  BREAK_POINT("BENCHTRI");
    delete aMasq;
#endif
}

void BenchPly(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Ply")) return;

    cTriangulation3D<tREAL8>::Bench();
    cTriangulation3D<tREAL16>::Bench();

    aParam.EndBench();
}


template <class Type>  cTriangulation3D<Type>::cTriangulation3D(const tVPt& aVP,const tVFace & aVF) :
	cTriangulation<Type,3>(aVP,aVF)
{
}

template <class Type>  cTriangulation3D<Type>::cTriangulation3D(const std::string & aName, bool SelectPointsByClass):
        cTriangulation<Type,3>(std::vector<tPt>())
{
    if (UCaseEqual(LastPostfix(aName),"ply"))
    {
       PlyInit(aName);
    }
#if MMVII_USE_PDAL
    else if (UCaseEqual(LastPostfix(aName),"laz"))
    {
       LasInit(aName,SelectPointsByClass);
    }
#endif
    else
    {
       MMVII_UserError(eTyUEr::eBadPostfix,"Unknown postfix in cTriangulation3D");
    }
}

template <class Type> void cTriangulation3D<Type>::WriteFile(const std::string & aName,bool isBinary) const
{
    if (UCaseEqual(LastPostfix(aName),"ply"))
    {
       PlyWrite(aName,isBinary);
    } 
    else
    {
       MMVII_UserError(eTyUEr::eBadPostfix,"Unknown postfix in cTriangulation3D");
    }
}

template <class Type> class cDevBiFace
{
   public :
      cDevBiFace(const cTriangle<Type,2> & aT1, const cTriangle<Type,2> & aT2);
      cTriangle<Type,2> mT1;
      cTriangle<Type,2> mT2;
};

template <class Type> cTriangle<Type,2>   cTriangulation3D<Type>::TriDevlpt(int aKF,int aNumSom) const
{
     return cIsometry3D<Type>::ToPlaneZ0(aNumSom,this->KthTri(aKF));
}


template <class Type> cDevBiFaceMesh<Type> cTriangulation3D<Type>::DoDevBiFace(int aKF1,int aKS1) const 
{
     const  tFace & aFace1 = this->mVFaces[aKF1];
     int aIS1 = aFace1[aKS1];
     int aIS2 = aFace1[(aKS1+1)%3];
     cEdgeDual *  aE12 = this->mDualGr.GetEdgeOfSoms(aIS1,aIS2);
     MMVII_INTERNAL_ASSERT_tiny(aE12!=nullptr,"Bad edge in CheckOri3D");
     int aKF2 = aE12->GetOtherFace(aKF1,true);

     if (aKF2<0)
	return cDevBiFaceMesh<Type> ();

     cTriangle<Type,2> aT1 = TriDevlpt(aKF1,aKS1);  // cIsometry3D<Type>::ToPlaneZ0(aKS1,this->KthTri(aKF1));



     tFace aFace2 = this->mVFaces[aKF2];
     int aKS2 = IndOfSomInFace(aFace2,aIS2);
     cTriangle<Type,2> aT2 = cIsometry3D<Type>::ToPlaneZ0(aKS2,this->KthTri(aKF2),false);

     if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny)
     {
            Type aDif =  Norm2(this->KthPts(aIS1)-this->KthPts(aIS2)) -  Norm2(aT1.Pt(0)-aT1.Pt(1));
	    Fake4ReleaseUseIt(aDif);
            MMVII_INTERNAL_ASSERT_tiny(std::abs(aDif)<1e-8,"Bad ToPlaneZ0");

            MMVII_INTERNAL_ASSERT_tiny(NormInf(aT1.Pt(0) )<1e-8,"Bad ToPlaneZ0");
            MMVII_INTERNAL_ASSERT_tiny(NormInf(aT2.Pt(0) )<1e-8,"Bad ToPlaneZ0");
            MMVII_INTERNAL_ASSERT_tiny(NormInf(aT1.Pt(1)-aT2.Pt(1) )<1e-8,"Bad ToPlaneZ0");

     }
     return cDevBiFaceMesh<Type> (aT1,aT2);
}



template <class Type> void cTriangulation3D<Type>::CheckOri3D()
{
     this->MakeTopo();


     int aNbBadOri = 0;
     int aNbEInt = 0;
     int aNbEExt = 0;
     for (size_t aKF1=0 ; aKF1<this->NbFace() ; aKF1++)
     {
          for (int aKS1=0 ; aKS1<3 ; aKS1++)
	  {
              cDevBiFaceMesh<Type>  aDBF = DoDevBiFace(aKF1,aKS1);
              if (aDBF.Ok())
	      {
                  if (! aDBF.WellOriented())
                     aNbBadOri++;
                  aNbEInt++;
	      }
	      else
	      {
                  aNbEExt++;
	      }
	  }
     }
     StdOut() << " * NbEdgeInt=" << aNbEInt  << " NbEdgeExt=" << aNbEExt  <<  std::endl;
     StdOut() << " Non Orientable  Edges :  " << aNbBadOri <<  " on " << (aNbEInt+aNbEExt) << "\n" << std::endl;
}

template <class Type> void cTriangulation3D<Type>::CheckOri2D()
{
     int aNbOriP = 0;
     int aNbOriM = 0;
     for (size_t aKF=0 ; aKF<this->NbFace() ; aKF++)
     {
          cTriangle<Type,3> aT = this->KthTri(aKF);
	  cPtxd<Type,3> aV = aT.KVect(0) ^aT.KVect(1);

          if (aV.z()>0)
             aNbOriP++;
          else
             aNbOriM++;
     }

     int aNbBadOri = std::min(aNbOriP,aNbOriM);
     StdOut() << " 2D-Bad Orientation " << aNbBadOri << " on " << this->NbFace() <<  "\n" << std::endl;
}

template <class Type> cBox2dr  cTriangulation3D<Type>::Box2D() const
{
    // create the bounding box of all points
    cTplBoxOfPts<tREAL8,2> aBoxObj;  // Box of object 
    for (size_t aKP=0 ; aKP<this->NbPts() ; aKP++)
    {
        aBoxObj.Add(ToR(Proj(this->KthPts(aKP))));
    }
    // create the "compiled" box from the dynamix
    return aBoxObj.CurBox();
}

template <class Type> 
   void cTriangulation3D<Type>::MakePatches
        (
             std::list<std::vector<int> > & aLPatches,
             tREAL8 aDistNeigh,
             tREAL8 aDistReject,
             int    aSzMin
         ) const
{
    cBox2dr aBox = Box2D();

    // indexation of all points
    cTiling<cTil2DTri3D<Type> >  aTileAll(aBox,true,this->NbPts()/20,this);
    for (size_t aKP=0 ; aKP<this->NbPts() ; aKP++)
    {
        aTileAll.Add(cTil2DTri3D<Type>(aKP));
    }

    // int aCpt=0;
    // indexation of all points selecte as center of patches
    cTiling<cTil2DTri3D<Type> >  aTileSelect(aBox,true,this->NbPts()/20,this);

    // parse all points
    for (size_t aKP=0 ; aKP<this->NbPts() ; aKP+=1)
    {
        cPt2dr aPt  = ToR(Proj(this->KthPts(aKP)));
        // if the points is not close to an existing center of patch : create a new patch
        if (aTileSelect.GetObjAtDist(aPt,aDistReject).empty())
        {
           //  Add it in the tiling of select 
           aTileSelect.Add(cTil2DTri3D<Type>(aKP));
           // extract all the point close enough to the center
           auto aLIptr = aTileAll.GetObjAtDist(aPt,aDistNeigh);
           std::vector<int> aPatch; // the patch itself = index of points
           aPatch.push_back(aKP);  // add the center at begining
           for (const auto aPtrI : aLIptr)
           {
               if (aPtrI->Ind() !=aKP) // dont add the center twice
               {
                  aPatch.push_back(aPtrI->Ind());
               }
           }
           // some requirement on minimal size
           if ((int)aPatch.size() > aSzMin)
           {
              // aCpt += aPatch.size();
              aLPatches.push_back(aPatch);
           }
        }
    }
}

template <class Type> void cTriangulation3D<Type>::SamplePts(
                                        const bool & targetted,
                                        const tREAL8 & aStep
                                                                )
 {
   /// < Sample points either by targetted or by random sampling

    this->mVSelectedIds.clear();
   cBox2dr aBox = Box2D();
    bool defined =!targetted;
  if (targetted)
        {
      // sample points in a grid

      /*
       * |----- |  -----  | -------|
       * |----- |  -----  | -------|
       * |----- |  -----  | -------|
       */
      // Random Ordering of points
      //this->mVPts=RandomOrder(this->mVPts);

      // Empty grid of points to sample with size bbox/aStep
      cDataTypedIm<tU_INT1,2> aD_Grid(cPt2di(0,0),
                                    cPt2di(Pt_round_down(aBox.Sz()/aStep))
                                    );
      aD_Grid.InitCste(0);

      StdOut()<<"Grid Size "<<aD_Grid.Sz()<<std::endl;
      // fill grid
      size_t it=0;
      int AreallCellsFilled=aD_Grid.NbElem();
      while((it<this->NbPts()) && AreallCellsFilled)
        {
           cPt2dr aPt  = ToR(Proj(this->KthPts(it)));
           //std::cout<<aPt<<std::endl;
           cPt2di aPix=cPt2di((aPt.x()-aBox.P0().x())/aStep,
                              (aPt.y()-aBox.P0().y())/aStep);
           //std::cout<<aPix<<std::endl;
           if(aD_Grid.Inside(aPix))
             {
               if (aD_Grid.VI_GetV(aPix)==tU_INT1(eLabelIm_MASQ::eFree))
                 {
                   //std::cout<<"  "<<" cell "<<allCellsFilled<<" c"<<eLabelIm_MASQ::eFree<<std::endl;
                   aD_Grid.VI_SetV(aPix,1);
                   this->mVSelectedIds.push_back(it);
                   AreallCellsFilled--;
                 }
             }
           it++;
        }
    }
  else if (defined)
    {

        std::vector<cPt2dr> aVecPatchCenters;
        // READ GeoJson File containing centers of patches to facilitate patch sampling

        std::string NameGeoJson="/home/MAChebbi/Documents/RecalageLidarImage_MicMac_PDAL/LiDAR/centers_more2.geojson";
        GDALAllRegister();

        GDALDataset *poDS = static_cast<GDALDataset*>(
        GDALOpenEx( NameGeoJson.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL ));
        if( poDS == NULL )
        {
            printf( "Open failed.\n" );
            exit( 1 );
        }

        OGRLayer  *poLayer = poDS->GetLayerByName( "centers_more2" );
        poLayer->ResetReading();
        OGRFeature *poFeature;
        while( (poFeature = poLayer->GetNextFeature()) != NULL )
        {
            OGRGeometry *poGeometry = poFeature->GetGeometryRef();
            if( poGeometry != NULL
                && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint )
            {
                OGRPoint *poPoint = (OGRPoint *) poGeometry;
                cPt2dr aPt(poPoint->getX(), poPoint->getY());
                aVecPatchCenters.push_back(aPt);
                printf( "%.3f,%.3f\n", poPoint->getX(), poPoint->getY() );
            }
            else
            {
                printf( "no point geometry\n" );
            }
            OGRFeature::DestroyFeature( poFeature );
        }
        GDALClose( poDS );

        // Determine Nearest LidarPoint By Tree Search

        cBox2dr aBox = Box2D();

        // indexation of all points
        cTiling<cTil2DTri3D<Type> >  aTileAll(aBox,true,this->NbPts()/20,this);
        for (size_t aKP=0 ; aKP<this->NbPts() ; aKP++)
        {
            aTileAll.Add(cTil2DTri3D<Type>(aKP));
        }

        // find nearest center and save index into mSelectedIds
        for(const auto aPt: aVecPatchCenters)
        {
            auto aLIptr = aTileAll.GetObjAtDist(aPt,1.0);
            for (const auto aPtr : aLIptr)
            {
                this->mVSelectedIds.push_back(aPtr->Ind());
                break;
            }
        }
    }
    else
    {
        // random sampling and patch selection based on planarity features and visibility criteria

        // target: distributed + sufficient + planar
        //1. distributed

        // Determine Nearest LidarPoint By Tree Search

        /*cBox2dr aBox = Box2D();

        // indexation of all points
        cTiling<cTil2DTri3D<Type> >  aTileAll(aBox,true,this->NbPts()/20,this);
        for (size_t aKP=0 ; aKP<this->NbPts() ; aKP++)
        {
            aTileAll.Add(cTil2DTri3D<Type>(aKP));
        }

        // indexation of all points selecte as center of patches
        cTiling<cTil2DTri3D<Type> >  aTileSelect(aBox,true,this->NbPts()/20,this);


        // find nearest center and save index into mSelectedIds
        for (size_t aKP=0 ; aKP<this->NbPts() ; aKP+=1)
        {
            cPt2dr aPt  = ToR(Proj(this->KthPts(aKP)));
            // if the points is not close to an existing center of patch : create a new patch
            if (aTileSelect.GetObjAtDist(aPt,aDistReject).empty())
            {

            }

        }*/
    }

}


template <class Type>
bool cTriangulation3D<Type>::IsGoodPatchNadir(const std::vector<cPt3dr>& aVPts,
                                         const std::vector<cSensorCamPC*> & aCameras,
                                         const std::vector<cDataGenUnTypedIm<2>*> & mVIms,
                                         tREAL8 AC_RHO,
                                         tREAL8 VAR_RHO,
                                         int    aSzMin,
                                         tREAL8 aThreshold,
                                         tREAL8 aSzW,
                                         tINT1 aScale,
                                         bool addAutoCorrel)
{

    // multiply by 2^aScale to find scale image coordinates
    //tREAL8 TT_SEUIL_CutAutoCorrel_REJECTION = 0.65;       // Seuil d'acceptation rapide par auto correl entiere
    //tREAL8 TT_SEUIL_CutAutoCorrel_REEL_REJECTION = 0.75;  // Seuil d'acceptation rapide par auto correl reelle
    //tREAL8 TT_SEUIL_AutoCorrel_ACCEPT = 0.85 ;            // Seuil d'elimination par auto-correlation



    ///< Finds "good" patches based on geometry criteria and radiometric resemblance between patches
    ///
    /// Use Patch planarity as filter
    /// check for good saliency between orthos --> Autocorrelation in a neighborhood
    ///
    /// Patch is planar
    bool isPlanar=IsPlanarityIdxPdal(aVPts,25.0,6.0);

    // compute per image orthos and evaluate auto-correlation of central pixel compared to neighbors

    std::vector<bool> aSetVisibs;

    //tREAL8 aMinVisibAngle = 1e8;
    tREAL8 aMinDistance2Center = 1e12;

    int aMinAngleInd=0;

    for (size_t aKIm=0 ; aKIm<aCameras.size() ; aKIm++)
    {
        const cSensorCamPC * aCam = aCameras[aKIm]; // extract cam

        std::vector<tREAL8> aPatchDepths;

        if (aCam->IsVisible(aVPts.at(0)))
        {
            // Visbility of patch center in images
            for  ( const auto & aPt: aVPts)
            {
                if (aCam->IsVisible(aPt))
                {
                    aPatchDepths.push_back(aCam->Pose().Inverse(aPt).z());
                }
            }
            tREAL8 aDepthMin = *min_element(aPatchDepths.begin(),aPatchDepths.end());
            tREAL8 aDepthMax = *max_element(aPatchDepths.begin(),aPatchDepths.end());
            // Compute visibility criterion
            tREAL8 aVisibility=exp(-pow((aPatchDepths.at(0)-aDepthMin),2)/pow((aDepthMax-aDepthMin+1e-8),2));

            if (
                (aVisibility > aThreshold)
                &&
                ((int)aPatchDepths.size()>aSzMin)
                )
            {
                aSetVisibs.push_back(true);
            }
            else
                aSetVisibs.push_back(false);

            // search for nadir image
            //tREAL8 aAngleVisib= aCam->DegreeVisibility(aVPts.at(0));

            cPt2dr aPIm = aCam->Ground2Image(aVPts.at(0));
            cPt2dr aPCenter = cPt2dr(aCam->SzPix().x()/2, aCam->SzPix().y()/2);
            tREAL8 Distance2Center = SqN2(aPCenter-aPIm);

            if (Distance2Center<aMinDistance2Center)
            {
                aMinDistance2Center= Distance2Center;
                aMinAngleInd= aKIm;
            }
        }
    }

    // now compute  auto correl, need to scale point to take into account scaled cameras

    // most nadir image
    bool isNotAutoCorr=false;
     addAutoCorrel= false ;
    if (addAutoCorrel)
    {
         StdOut()<<"compute autocorelle "<<std::endl;
        const cSensorCamPC * aCam = aCameras[aMinAngleInd]; // extract cam
        const auto & aDIm = mVIms[aMinAngleInd]; // extract image


        if (aCam->IsVisible(aVPts.at(0)))
        {
            /// Autocorrel for most nadir image
            cPt2dr aPIm= MulCByC(aCam->Ground2Image(aVPts[0]),
                                  cPt2dr(pow (2,aScale),pow (2,aScale)));

            if (WindInside4BL(*aDIm,ToI(aPIm),Pt_round_up(cPt2dr(AC_RHO+aSzW+1,AC_RHO+aSzW+1))))
            {
                cCutAutoCorrelDir<tU_INT1> aCACD(*aDIm,cPt2di(0,0),AC_RHO,2,aSzW);

                //StdOut()<<getchar()<<std::endl;
                /*isNotAutoCorr =  !(aCACD.AutoCorrel((ToI(aPIm)),
                                                 TT_SEUIL_CutAutoCorrel_REJECTION,
                                                 TT_SEUIL_CutAutoCorrel_REEL_REJECTION,
                                                 TT_SEUIL_AutoCorrel_ACCEPT)
                                );*/

                isNotAutoCorr = !(aCACD.AutoCensusQ(ToI(aPIm),0.6,0.55));

                /*tREAL8 aStdDev= CubGaussWeightStandardDev(aDIm,ToI(aPIm),VAR_RHO);
                if (aStdDev<=0)
                    isNotAutoCorr=false;*/


                // add a center variability criterion
                if (0) //isNotAutoCorr && isPlanar)
                {
                    std:: string filename= "CORR_"+
                                           aCam->NameImage()+
                                           ToStr(aMinAngleInd)+"_"+
                                           ToStr(ToI(aPIm).x())+"_"+
                                           ToStr(ToI(aPIm).y())+"_"+
                                           ToStr(aCACD.mCorOut)+"_"+
                                           ToStr(aCACD.mNumOut)+".tif";
                    aCACD.writeCorrelImage(AC_RHO,filename);
                }
            }


            /*
            ///< Correl the reprojected lidar patch given its size and not a small defined window size
            ///
            ///
            cTplBoxOfPts<tREAL8,2> aBoxObj;
            std::vector<cPt2dr> aProjPatchInIm;
            // add index of central patch
            aProjPatchInIm.push_back(cPt2dr(0,0));
            aBoxObj.Add(aPIm);
            bool AllPatchInImage = true;
            for (size_t aK=1; aK< aVPts.size(); aK++)
            {
                if (!aCam->IsVisible(aVPts[aK]))
                    {
                        AllPatchInImage=false;
                        break;
                    }
                cPt2dr aP2 = MulCByC(aCam->Ground2Image(aVPts[aK]),
                                         cPt2dr(pow (2,aScale),pow (2,aScale)));

                // store points locations with respect to center
                aProjPatchInIm.push_back(aP2-aPIm);

                aBoxObj.Add(aP2);
            }

            //StdOut()<<"ABox obJ "<<aBoxObj.P0()<<"  "<<aBoxObj.P1()<<std::endl;

            //StdOut()<<"Is all patch in Image "<<AllPatchInImage<<std::endl;

            if (AllPatchInImage)
            {
                cBox2dr aBoxOfPatch = aBoxObj.CurBox();

                if (WindInside4BL(aDIm,
                                  aPIm,
                                  Pt_round_up(cPt2dr(AC_RHO+aBoxOfPatch.Sz().x()+1,
                                                     AC_RHO+aBoxOfPatch.Sz().y()+1))
                                  )
                    )
                {
                    // compute correl given the whole patch
                    cCutAutoCorrelDir<tU_INT1> aCACD(aDIm,cPt2di(0,0),AC_RHO,1,aProjPatchInIm);

                    isNotAutoCorr =  !(aCACD.AutoCorrelNonRegularPatch((ToI(aPIm)),
                                                     TT_SEUIL_CutAutoCorrel_REJECTION,
                                                     TT_SEUIL_CutAutoCorrel_REEL_REJECTION,
                                                     TT_SEUIL_AutoCorrel_ACCEPT)
                                    );



                    tREAL8 aStdDev= CubGaussWeightStandardDev(aDIm,ToI(aPIm),VAR_RHO);
                    if (aStdDev<=0)
                        isNotAutoCorr=false;

                    // add a center variability criterion
                    if (isNotAutoCorr && isPlanar)
                    {
                        std:: string filename= "CORR_"+
                                               aCam->NameImage()+
                                               ToStr(aMinAngleInd)+"_"+
                                               ToStr(ToI(aPIm).x())+"_"+
                                               ToStr(ToI(aPIm).y())+"_"+
                                               ToStr(aCACD.mCorOut)+".tif";
                        aCACD.writeCorrelImage(AC_RHO,filename);
                    }
                }
            }*/


        }

    }

    if ( aSetVisibs.empty())
        return false;


    int MinVisibTimes=0;
    for (size_t itc=0; itc<aSetVisibs.size(); itc++)
    {
        if (aSetVisibs[itc])
            MinVisibTimes++;
    }

    if ((MinVisibTimes>=2) &&
        isPlanar         &&
        (addAutoCorrel ?  isNotAutoCorr : true ) &&
        aSetVisibs[aMinAngleInd])
        return true;
    else
        return false;
}


template <class Type>
 void cTriangulation3D<Type>::MakePatchesTargetted
      (
           std::list<std::vector<int> > & aLPatches,
           tREAL8 aDistNeigh,
           tREAL8 aDistReject,
           int    aSzMin,
           const std::vector<cSensorCamPC * > & aCameras,
           const std::vector<cDataGenUnTypedIm<2>*> & mVIms,
           tREAL8 aThreshold,
           const std::vector<std::vector<cSensorCamPC * >> & mVSCams,
           int  aScale
       )
{
    cBox2dr aBox = Box2D();

    // indexation of all points
    cTiling<cTil2DTri3D<Type>>  aTileAll(aBox,true,this->NbPts()/20,this);
    #pragma omp parallel for
    for (size_t aKP=0 ; aKP<this->NbPts() ; aKP++)
    {
        aTileAll.Add(cTil2DTri3D<Type>(aKP));
    }
    //StdOut()<<"Tile2D3D filled "<<std::endl;
    // indexation of all points selected as center of patches
    cTiling<cTil2DTri3D<Type> >  aTileSelect(aBox,true,this->NbPts()/20,this);
    // parse all  points
    /*#pragma omp declare reduction (merge :std::vector<std::vector<int> > : \
        omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp parallel for reduction(merge: aLPatches)*/
    for (size_t aKP=0 ; aKP<this->NbPts() ; aKP+=1)
      {
          cPt2dr aPt  = ToR(Proj(this->KthPts(aKP)));
          // if the points is not close to an existing center of patch : create a new patch
          if (aTileSelect.GetObjAtDist(aPt,aDistReject).empty())
          {
             //  Add it in the tilisng of select
             aTileSelect.Add(cTil2DTri3D<Type>(aKP));
             // extract all the point close enough to the center
             //StdOut()<<"get elemm "<<std::endl;
             auto aLIptr = aTileAll.GetObjAtDist(aPt,aDistNeigh);
             std::vector<int> aPatch; // the patch itself = index of points
             aPatch.push_back(aKP);  // add the center at begining
             for (const auto aPtrI : aLIptr)
             {
                 if (aPtrI->Ind() !=aKP) // dont add the center twice
                 {
                    aPatch.push_back(aPtrI->Ind());
                 }
             }

             std::vector<cPt3dr> aVP;
             for (const auto anInd : aPatch)
                 aVP.push_back(ToR(this->KthPts(anInd)));


             if ( (int) aPatch.size()>aSzMin)
             {
                 bool isGoodPatch = false;
                 if (aScale)
                 {
                         std::vector<cSensorCamPC * > atScaleCams;
                         for (auto aSCam : mVSCams)
                             {
                                atScaleCams.push_back(aSCam[aScale-1]);
                             }

                        isGoodPatch =IsGoodPatchNadir(aVP,
                                                  atScaleCams,
                                                  mVIms,
                                                  5.0,
                                                  7.0,
                                                  aSzMin,
                                                  aThreshold,
                                                  3,
                                                  aScale);
                  }
                 else
                    {
                         isGoodPatch =IsGoodPatchNadir(aVP,
                                                        aCameras,
                                                        mVIms,
                                                        5.0,
                                                        7.0,
                                                        aSzMin,
                                                        aThreshold,
                                                        3,
                                                        false);
                    }
                if (isGoodPatch)
                     aLPatches.push_back(aPatch);
             }

          }
      }
}


/* ********************************************************** */
/*                                                            */
/*                                                            */
/*                                                            */
/* ********************************************************** */

template <class Type> cDevBiFaceMesh<Type>::cDevBiFaceMesh() :
    mOk (false),
    mWellOriented (false),
    mT1 (tPt(0,0),tPt(0,0),tPt(0,0)),
    mT2 (tPt(0,0),tPt(0,0),tPt(0,0))
{
}

template <class Type> cDevBiFaceMesh<Type>::cDevBiFaceMesh(const cTriangle<Type,2> & aT1, const cTriangle<Type,2> & aT2) :
    mOk (true),
    mWellOriented (false),
    mT1 (aT1),
    mT2 (aT2)
{
    Type aY1 = aT1.Pt(2).y() ;
    Type aY2 = aT2.Pt(2).y() ;

    mWellOriented = ((aY1>=0) != (aY2>0)) ;
}

template <class Type> void cDevBiFaceMesh<Type>::AssertOk() const
{
      MMVII_INTERNAL_ASSERT_tiny(mOk,"Un-init cDevBiFaceMesh");
}


template <class Type> const cTriangle<Type,2> &  cDevBiFaceMesh<Type>::T1() const {AssertOk(); return mT1;}
template <class Type> const cTriangle<Type,2> &  cDevBiFaceMesh<Type>::T2() const {AssertOk(); return mT2;}
template <class Type> bool  cDevBiFaceMesh<Type>::Ok() const { return mOk;}
template <class Type> bool  cDevBiFaceMesh<Type>::WellOriented() const {AssertOk(); return mWellOriented;}

/*
      cDevBiFaceMesh(const cTriangle<Type,2> & aT1, const cTriangle<Type,2> & aT2);
      cDevBiFaceMesh();
      bool Ok() const;
      const cTriangle<Type,2> & T1() const;
      const cTriangle<Type,2> & T2() const;
      */



/* ========================== */
/*     INSTANTIATION          */
/* ========================== */

#define INSTANTIATE_TRI3D(TYPE)\
template class cTriangulation3D<TYPE>;\
template class cDevBiFaceMesh<TYPE>;


INSTANTIATE_TRI3D(tREAL4)
INSTANTIATE_TRI3D(tREAL8)
INSTANTIATE_TRI3D(tREAL16)


/* ********************************************************** */
/*                                                            */
/*                   cAppli_VisuPoseStr3D                     */
/*                                                            */
/* ********************************************************** */

void cAppli_VisuPoseStr3D::WritePly(cComputeMergeMulTieP * & aTPts, const std::vector<cSensorImage *>& aVSens)
{
    PLYData aPlyOut;

    //  convert Pts to array
    std::vector<std::array<double, 3>> aPlyPts;

    // add 3d points
    size_t aNum3DPt=0;
    if (aTPts)
    {
        for (auto aAllConfigs : aTPts->Pts())
        {
            const auto & aConfig = aAllConfigs.first;
            auto & aVals = aAllConfigs.second;

            size_t aNbIm = aConfig.size();
            size_t aNbPts = aVals.mVIdPts.size();


            for (size_t aKPts=0; aKPts<aNbPts; aKPts++)
            {
                const cPt3dr & aP3D = aVals.mVPGround.at(aKPts);


                for (size_t aKIm=0; aKIm<aNbIm; aKIm++)
                {
                    size_t aKImSorted = aConfig.at(aKIm);

                    const cPt2dr aPIm = aVals.mVPIm.at(aKPts*aNbIm+aKIm);
                    cSensorImage* aCam = aVSens.at(aKImSorted);

                    if (aCam->IsVisibleOnImFrame(aPIm) && aCam->IsVisible(aP3D))
                    {

                        double aResidual = Norm2(aPIm - aCam->Ground2Image(aP3D));

                        if (aResidual<mErrProjMax)
                        {
                            std::array<double,3> anArray;
                            for (int aK=0 ; aK<3 ; aK++)
                                anArray[aK] = aP3D[aK];
                            aPlyPts.push_back(anArray);

                            aNum3DPt++;
                        }

                    }
                }
            }
        }
    }

    // add camera centers
    size_t aNumImPlane=0;
    for (auto aCam : aVSens)
    {
        cPt3dr aCenter = aCam->PseudoCenterOfProj();

        std::array<double,3> anArray;
        for (int aK=0 ; aK<3 ; aK++)
            anArray[aK] = aCenter[aK];
        aPlyPts.push_back(anArray);

        aNumImPlane++;
    }

    // add points in image plane
    int aSteps = 50;
    for (auto aSens : aVSens)
    {
        cSensorCamPC *  aCamPC = aSens->GetSensorCamPC();

        cPt2di aSz = aSens->Sz();
        double aFPix = aCamPC->InternalCalib()->F();
        double aF = CalculateFDepth(aSz,aFPix);

        cPt2dr aImStepSz(aSz[0]/aSteps,aSz[1]/aSteps);

        for (int aS=0; aS<=aSteps; aS++)
        {
            std::vector<cPt3dr> aImVPts;
            double aDX = aImStepSz[0]*aS-1;
            double aDY = aImStepSz[1]*aS-1;

            aImVPts.push_back(aSens->ImageAndDepth2Ground( cPt3dr(0,aDY,aF) ));
            aImVPts.push_back(aSens->ImageAndDepth2Ground( cPt3dr(aDX,0,aF) ));
            aImVPts.push_back(aSens->ImageAndDepth2Ground( cPt3dr(aSz[0],aDY,aF) ));
            aImVPts.push_back(aSens->ImageAndDepth2Ground( cPt3dr(aDX,aSz[1],aF) ));


            for (auto aP : aImVPts)
            {
                std::array<double,3> anArray;
                for (int aK=0 ; aK<3 ; aK++)
                    anArray[aK] = aP[aK];
                aPlyPts.push_back(anArray);

                aNumImPlane++;
            }
        }
    }

    // add points on the frustum
    for (auto aSens : aVSens)
    {
        cSensorCamPC *  aCamPC = aSens->GetSensorCamPC();

        cPt2di aSz = aSens->Sz();
        double aFPix = aCamPC->InternalCalib()->F();
        double aF = CalculateFDepth(aSz,aFPix);

        double aFStepSz = aF/aSteps;


        for (int aS=0; aS<=aSteps; aS++)
        {
            std::vector<cPt3dr> aImVPts;
            double aDepth = aFStepSz*aS;
            aImVPts.push_back(aSens->ImageAndDepth2Ground( cPt3dr(0,0,aDepth) ));
            aImVPts.push_back(aSens->ImageAndDepth2Ground( cPt3dr(0,aSz[1],aDepth) ));
            aImVPts.push_back(aSens->ImageAndDepth2Ground( cPt3dr(aSz[0],0,aDepth) ));
            aImVPts.push_back(aSens->ImageAndDepth2Ground( cPt3dr(aSz[0],aSz[1],aDepth) ));


            for (auto aP : aImVPts)
            {
                std::array<double,3> anArray;
                for (int aK=0 ; aK<3 ; aK++)
                    anArray[aK] = aP[aK];
                aPlyPts.push_back(anArray);
            }
        }
    }

    aPlyOut.addVertexPositions(aPlyPts);

    // assign colors
    std::vector<std::array<double, 3>> colors;
    for (size_t aK=0; aK<aPlyPts.size(); aK++)
    {
        std::array<double,3> anArray;
        if (aK<aNum3DPt)
        {
            // todo: add colors from images
            for (int aI=0 ; aI<3 ; aI++)
                anArray[aI] = 1.0;
        }
        else if (aK<(aNum3DPt+aNumImPlane))
        {
            anArray[0] = 1.0;
            anArray[1] = 0;
            anArray[2] = 0;
        }
        else
        {
            anArray[0] = 0;
            anArray[1] = 1.0;
            anArray[2] = 0;
        }
        colors.push_back(anArray);

    }
    aPlyOut.addVertexColors(colors);

    // write ply
    aPlyOut.write(mOutfile,(mBinary?happly::DataFormat::Binary:happly::DataFormat::ASCII));
}

double cAppli_VisuPoseStr3D::CalculateFDepth(const cPt2di& aSz, const double& aF)
{
    double aDiag = std::sqrt(std::pow(aSz[0],2)+std::pow(aSz[1],2));
    double aRatioDiagF = aDiag/aF ;

    return aRatioDiagF*mCamScale;
}

};
