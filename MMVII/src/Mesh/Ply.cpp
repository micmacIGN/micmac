#include "cMMVII_Appli.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Mappings.h"
#include "MMVII_2Include_Tiling.h"

#include "happly.h"

#define WITH_MMV1_FUNCTION  false

namespace MMVII
{

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

template <class Type>  cTriangulation3D<Type>::cTriangulation3D(const std::string & aName):
	cTriangulation<Type,3>(std::vector<tPt>())
{
    if (UCaseEqual(LastPostfix(aName),"ply"))
    {
       PlyInit(aName);
    } 
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
    for (size_t aKP=0 ; aKP<this->NbPts() ; aKP++)
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

};
