#include "include/MMVII_all.h"

namespace MMVII
{

/* ********************************************** */
/*                                                */
/*           cTriangle                            */
/*                                                */
/* ********************************************** */

template <class Type,const int Dim>  
	 cTriangle<Type,Dim>::cTriangle(const tPt & aP0,const tPt & aP1,const tPt & aP2)
{
   mPts[0] = aP0;
   mPts[1] = aP1;
   mPts[2] = aP2;
}

template <class Type,const int Dim>  
    int cTriangle<Type,Dim>::IndexLongestSeg() const
{
    cWhitchMax<int,typename tPt::tBigNum> aWMax(0,SqN2(KVect(0)));
    for (int aK=1 ; aK<3 ; aK++)
        aWMax.Add(aK,SqN2(KVect(1)));

    return aWMax.IndexExtre();
}


template <class Type,const int Dim> const cPtxd<Type,Dim>& cTriangle<Type,Dim>::Pt(int aK) const
{
     MMVII_INTERNAL_ASSERT_tiny((aK>=0) && (aK<3),"cTriangle2D::Pt");
     return mPts[aK];
}

template <class Type,const int Dim> const cPtxd<Type,Dim>& cTriangle<Type,Dim>::PtCirc(int aK) const
{
     return mPts[mod(aK,3)];
}

template <class Type,const int Dim>  cPtxd<Type,Dim> cTriangle<Type,Dim>::KVect(int aK) const
{
     MMVII_INTERNAL_ASSERT_tiny((aK>=0) && (aK<3),"cTriangle2D::Pt");
     return mPts[(aK+1)%3] - mPts[aK];
}

template <class Type,const int Dim> cPtxd<Type,Dim> cTriangle<Type,Dim>::FromCoordBarry(const cPtxd<Type,3> & aP) const
{
     Type aSP = aP.x()+aP.y()+aP.z();
     MMVII_INTERNAL_ASSERT_tiny(aSP!=0,"Sum weith null in barry");

     return (aP.x()*mPts[0]+aP.y()*mPts[1]+aP.z()*mPts[2]) / aSP;
}

template <class Type,const int Dim> cPtxd<Type,Dim> cTriangle<Type,Dim>::Barry() const
{
	return (mPts[0]+mPts[1]+mPts[2]) / Type(3.0);
}

template <class Type,const int Dim> cTplBox<Type,Dim> cTriangle<Type,Dim>::BoxEngl() const
{
     cTplBoxOfPts<Type,Dim> aTBox;
     for (int aK=0 ; aK<3 ; aK++)
         aTBox.Add(mPts[aK]);
     return aTBox.CurBox();
}

template <class Type,const int Dim> cTplBox<int,Dim> cTriangle<Type,Dim>::BoxPixEngl() const
{
    cTplBox<Type,Dim> aBoxE = BoxEngl();
    return cTplBox<int,Dim> (Pt_round_down(aBoxE.P0()),Pt_round_up(aBoxE.P1()));
}

template <class Type,const int Dim> Type cTriangle<Type,Dim>::Regularity() const
{
    tPt aV01 = mPts[1]-mPts[0];
    tPt aV02 = mPts[2]-mPts[0];
    tPt aV12 = mPts[2]-mPts[1];
    Type aSomSqN2 = SqN2(aV01) + SqN2(aV02) + SqN2(aV12);
    if (aSomSqN2==0) return 0;
    return AbsSurfParalogram(aV01,aV02) / aSomSqN2;
}

template <class Type,const int Dim> Type cTriangle<Type,Dim>::Area() const
{
    return AbsSurfParalogram(mPts[1]-mPts[0],mPts[2]-mPts[0]) / 2.0;
}


template <class Type,const int Dim> cTriangle<Type,Dim> cTriangle<Type,Dim>::TriSwapPt(int aK0) const
{
   return cTriangle<Type,Dim>(mPts[(aK0+1)%3],mPts[(aK0)%3],mPts[(aK0+2)%3]);
}


/** Artefact to have a partial specialization */
template <class Type,const int Dim>  cPtxd<Type,Dim> GlobCenterInscribedCircle(const cPtxd<Type,Dim> * aPts)
{
   cLeasSqtAA<Type>  aSys(Dim);

   for (int aKp=0 ; aKp<3 ; aKp++)
   {
       const cPtxd<Type,Dim> & aP0 = aPts[aKp];
       const cPtxd<Type,Dim> & aP1 = aPts[(aKp+1)%3];
       cPtxd<Type,Dim> aMil = (aP0+aP1)/Type(2.0);
       cPtxd<Type,Dim> aV01 = aP1-aP0;

       aSys.AddObservation(Type(1.0),aV01.ToVect(),Scal(aMil,aV01));
   }

   // For Dim=3 we have computed inter of 3 plane mediator that co-intersect in a line
   // we need to add also a equation inside the plane
   MMVII_INTERNAL_ASSERT_tiny(Dim==2,"CenterInscribedCircle to Finish for DIm!=2");

   return cPtxd<Type,Dim>::FromVect(aSys.Solve());
}
template <const int Dim>  cPtxd<int,Dim> GlobCenterInscribedCircle(const cPtxd<int,Dim> *)
{
   MMVII_INTERNAL_ERROR("No CenterInscribedCircle for integer type");
   return  cPtxd<int,Dim>::PCste(2);
}

template <class Type,const int Dim>  cPtxd<Type,Dim> cTriangle<Type,Dim>::CenterInscribedCircle() const
{
    return GlobCenterInscribedCircle(mPts);
	/*
   cLeasSqtAA<Type>  aSys(Dim);

   for (int aKp=0 ; aKp<3 ; aKp++)
   {
       const tPt & aP0 = mPts[aKp];
       const tPt & aP1 = mPts[(aKp+1)%3];
       tPt aMil = (aP0+aP1)/Type(2.0);
       tPt aV01 = aP1-aP0;

       aSys.AddObservation(Type(1.0),aV01.ToVect(),Scal(aMil,aV01));
   }

   // For Dim=3 we have computed inter of 3 plane mediator that co-intersect in a line
   // we need to add also a equation inside the plane
   MMVII_INTERNAL_ASSERT_tiny(Dim==2,"CenterInscribedCircle to Finish for DIm!=2");

   return tPt::FromVect(aSys.Solve());
   */
}

/*
template <const int Dim>  cPtxd<int,Dim> cTriangle<int,Dim>::CenterInscribedCircle() const
{
      return  cPtxd<int,Dim>();
}
*/


template<class Type> cPtxd<Type,3> NormalUnit(const cTriangle<Type,3> & aTri)
{
	//  V01 ^ V12 = V01 ^ (V10 + V02) = V01 ^ V02
	return VUnit(aTri.KVect(0) ^ aTri.KVect(1));
}

template <class Type,const int Dim>  
         cTriangle<Type,Dim>  cTriangle<Type,Dim>::RandomTri(const Type & aSz,const Type & aRegulMin )
{
	cTriangle<Type,Dim> aRes(tPt::PRandC()*aSz,tPt::PRandC()*aSz,tPt::PRandC()*aSz);

	if (aRes.Regularity() > aRegulMin) return aRes;

	return RandomTri(aSz,aRegulMin);
}

/* *********************************************************** */
/*                                                             */
/*                 cEdgeDual                                   */
/*                                                             */
/* *********************************************************** */


cEdgeDual::cEdgeDual() :
     mS  {NO_INIT,NO_INIT},
     mF  {NO_INIT,NO_INIT}
{
}

cEdgeDual::cEdgeDual(int aS1,int aS2,int aF1) :
     mS  {aS1,aS2},
     mF  {aF1,NO_INIT}
{
}

void cEdgeDual::SetFace2(int aF2) 
{
     MMVII_INTERNAL_ASSERT_tiny((mF[0]!=NO_INIT)&&(mF[1]==NO_INIT),"SetOtherFace incohe");
     mF[1] = aF2;
}



/* *********************************************************** */
/*                                                             */
/*                cGraphDual                                   */
/*                                                             */
/* *********************************************************** */

cGraphDual::cGraphDual()
{
}

void cGraphDual::Init(int aNbSom, const std::vector<tFace>& aVFace)
{
    mSomNeigh = std::vector<tListAdj>  (aNbSom);
    mFaceNeigh = std::vector<tListAdj> (aVFace.size());
    mReserve.clear();

    for (int aKF=0 ; aKF<int(aVFace.size()) ; aKF++)
        AddTri(aKF,aVFace[aKF]);
}

void  cGraphDual::AddEdge(int aFace,int aS1,int aS2)
{
    // s1->s2 and  s2->s1 are the same physicall edge, one exists iff the other exists
    cEdgeDual * anE12 = GetEdgeOfSoms(aS1,aS2);
    if ( anE12==nullptr)
    {
         MMVII_INTERNAL_ASSERT_tiny(GetEdgeOfSoms(aS2,aS1)==nullptr,"Sym Check in cGraphDual::AddEdge");
         mReserve.push_back(cEdgeDual(aS1,aS2,aFace));
	 anE12 = &mReserve.back();
         mSomNeigh.at(aS1).push_back(anE12);
         mSomNeigh.at(aS2).push_back(anE12);
         mFaceNeigh.at(aFace).push_back(anE12);
    }
    else
    {
         MMVII_INTERNAL_ASSERT_tiny(GetEdgeOfSoms(aS2,aS1)==anE12,"Sym Check in cGraphDual::AddEdge");
	 anE12->SetFace2(aFace);
         mFaceNeigh.at(aFace).push_back(anE12);
    }
}

void  cGraphDual::AddTri(int aNumFace,const cPt3di & aTri)
{
      for (int aK=0 ; aK<3 ; aK++)
          AddEdge(aNumFace,aTri[aK],aTri[(aK+1)%3]);
}

cEdgeDual *  cGraphDual::GetEdgeOfSoms(int aS1,int aS2)
{
     for (const auto & aPtrE : mSomNeigh.at(aS1))
     {
        // it exist iff s2 is one of both submit (the other being s1)
	if (aPtrE->GetOtherSom(aS2,true)== aS1)
	   return aPtrE;
     }
     return nullptr;
}

void  cGraphDual::GetSomsNeighOfSom(std::vector<int> & aRes,int aS1) const
{
   aRes.clear();
   for (const auto & aPtrE : mSomNeigh.at(aS1))
      aRes.push_back(aPtrE->GetOtherSom(aS1,false));
}

void  cGraphDual::GetFacesNeighOfFace(std::vector<int> & aRes,int aF1) const
{
   aRes.clear();
   for (const auto & aPtrE : mFaceNeigh.at(aF1))
   {
      int aNF = aPtrE->GetOtherFace(aF1,true);
      if (aNF!= cEdgeDual::NO_INIT)
         aRes.push_back(aNF);
   }
}



/* *********************************************************** */
/*                                                             */
/*                cTriangulation                               */
/*                                                             */
/* *********************************************************** */



template <class Type,const int Dim> cTriangulation<Type,Dim>::cTriangulation(const tVPt& aVPts,const tVFace& aVFace)  :
    mVPts  (aVPts)
{
    for (const auto & aFace : aVFace)
        AddFace(aFace);
}
template <class Type,const int Dim> void cTriangulation<Type,Dim>::ResetTopo()
{
    mVFaces.clear();
}

template <class Type,const int Dim> bool cTriangulation<Type,Dim>::ValidFace(const tFace & aFace) const
{
   for (int aK=0; aK<3 ; aK++)
   {
       if ((aFace[aK]<0) || (aFace[aK] >=int(mVPts.size())) )
       {
          return false;
       }
   }
   return true;

}

template <class Type,const int Dim> const cGraphDual &  cTriangulation<Type,Dim>::DualGr() const {return mDualGr;}



template <class Type,const int Dim> void cTriangulation<Type,Dim>::MakeTopo()
{
   mDualGr.Init(mVPts.size(),mVFaces);
}


template <class Type,const int Dim> void cTriangulation<Type,Dim>::AddFace(const tFace & aFace)
{
    MMVII_INTERNAL_ASSERT_tiny(ValidFace(aFace),"Bad face");
    mVFaces.push_back(aFace);
}

template <class Type,const int Dim> size_t cTriangulation<Type,Dim>::NbFace() const { return mVFaces.size(); }
template <class Type,const int Dim> size_t cTriangulation<Type,Dim>::NbPts() const { return mVPts.size(); }




template <class Type,const int Dim> const  std::vector<cPt3di> & cTriangulation<Type,Dim>::VFaces() const
{
   return mVFaces;
}
template <class Type,const int Dim> const cPt3di & cTriangulation<Type,Dim>::KthFace(size_t aK) const
{
   return mVFaces.at(aK);
}

template <class Type,const int Dim> const cPtxd<Type,Dim> & cTriangulation<Type,Dim>::KthPts(size_t aK) const
{
   return mVPts.at(aK);
}
template <class Type,const int Dim> const std::vector<cPtxd<Type,Dim> > & cTriangulation<Type,Dim>::VPts() const
{
   return mVPts;
}
  // const std::vector<tPt> &   VPts() const;  ///< Standard Accessor

template <class Type,const int Dim> cTriangle<Type,Dim>    cTriangulation<Type,Dim>::KthTri(int aK) const
{
    const cPt3di &  aIndT = KthFace(aK);
    return cTriangle<Type,Dim>(mVPts.at(aIndT.x()),mVPts.at(aIndT.y()),mVPts.at(aIndT.z()));
}

template <class Type,const int Dim> cTplBox<Type,Dim>    
         cTriangulation<Type,Dim>::BoxEngl(Type aFactMargin) const
{
    cTplBox<Type,Dim> aBox =  cTplBox<Type,Dim>::FromVect(mVPts);

    if (aFactMargin!=0)
       aBox = aBox.ScaleCentered(1+aFactMargin) ;
    return aBox;
}

template <class Type,const int Dim> Type HeuristikDistTri (const cTriangle<Type,Dim> & aT1,const cTriangle<Type,Dim> & aT2)
{
    Type aDMin = 1e20;

    // Must be equal up to an offset (same orientation) Test all possible offset
    for (int aOffset=0 ; aOffset<3 ; aOffset++)
    {
         Type aSomD=0 ;
	 for (int aK1=0 ; aK1<3 ;aK1++)
             aSomD += Norm1(   aT1.Pt(aK1)-aT2.Pt((aK1+aOffset)%3)   );
	 UpdateMin(aDMin,aSomD);
    }
    return aDMin;
}


template <class Type,const int Dim> 
         bool cTriangulation<Type,Dim>::HeuristikAlmostInclude 
	       (const cTriangulation<Type,Dim> &aT2,Type aTolP,Type aTolF)  const
{
     for (const auto & aP1 : mVPts)
     {
          Type aDMin =aTolP+1000;
          for (int aK2=0 ; (aK2<int(aT2.mVPts.size())) && (aDMin>aTolP) ; aK2++)
	  {
              UpdateMin(aDMin,(Type)Norm2(aP1-aT2.mVPts[aK2]));
	  }
	  if (aDMin>aTolP) 
             return false;
     }

     if (aTolF>0)
     {
        for (int aK1=0 ; aK1<int(mVFaces.size()) ; aK1+=10)
        {
             Type aDMin =aTolF+1000;
             for (int aK2=0 ; (aK2<int(aT2.mVFaces.size())) && (aDMin>aTolF) ; aK2++)
	     {
                 UpdateMin(aDMin,HeuristikDistTri(KthTri(aK1),aT2.KthTri(aK2)));
	     }
	     if (aDMin>aTolF) 
	     {
                return false;
	     }
        }
     }

     return true;
}

template <class Type,const int Dim> 
         bool cTriangulation<Type,Dim>::HeuristikAlmostEqual 
	      (const cTriangulation<Type,Dim> &aT2,Type aTolP,Type aTolF)  const
{
    if  (mVPts.size() != aT2.mVPts.size()) 
        return false;

    if  (mVFaces.size() != aT2.mVFaces.size()) 
        return false;

    return HeuristikAlmostInclude (aT2,aTolP,aTolF) && aT2.HeuristikAlmostInclude(*this,aTolP,aTolF);
}

template <class Type,const int Dim> 
    void cTriangulation<Type,Dim>::Filter
                        (const cDataBoundedSet<tREAL8,Dim> & aSet,int aNbVertixThres) 
{
    std::vector<tFace> aVFaces;
    std::vector<tPt>   aVPts;

    // -1- compute points in the set
    std::vector<bool>  aVPtsInSet; // are points inside the set 
    std::vector<bool>  aVPtsInTri;  // do point belongs to one  of the selected triangle
    for (const auto & aPts : mVPts)
    {
        aVPtsInSet.push_back(aSet.Inside(ToR(aPts))); // Cast to real8, dont want instantiate all mappin
	aVPtsInTri.push_back(false);
    }

    // -2- compute faces selected and points belonging at least to one of it
    for (const auto & aFace : mVFaces)
    {
        int aNbIn = aVPtsInSet.at(aFace.x()) + aVPtsInSet.at(aFace.y()) + aVPtsInSet.at(aFace.z());
	if (aNbIn>=aNbVertixThres)
	{
             aVFaces.push_back(aFace);
             for (int aK=0 ; aK<3 ; aK++)  // if we maintain face, maintain all vertices
                aVPtsInTri.at(aFace[aK]) = true;
	}
    }
    
    // -3- subset of selcted points and new numerotation
    std::vector<int>  aVNewNums;
    for (int aKP=0 ; aKP<int(mVPts.size()) ; aKP++)
    {
        if (aVPtsInTri.at(aKP))
	{
            aVNewNums.push_back(aVPts.size());
	    aVPts.push_back(mVPts.at(aKP));
	}
	else
	{
            aVNewNums.push_back(-1e9);
	}
    }

    // -4- update the numeration of selected faces
    for (auto & aFace : aVFaces)
    {
         for (int aK=0 ; aK<3 ; aK++)  // if we maintain face, maintain all vertices
             aFace[aK] = aVNewNums.at(aFace[aK]);
    }

    mVFaces = aVFaces;
    mVPts = aVPts;
}

template <class Type,const int Dim> typename cTriangulation<Type,Dim>::tPt  cTriangulation<Type,Dim>::PAvg() const
{
    cWeightAv<Type,tPt> aWA;

    for (int aKF=0 ; aKF<int(mVFaces.size()) ; aKF++)
    {
       tTri  aTri = KthTri(aKF);
       aWA.Add(aTri.Area(),aTri.Barry());
    }
    return aWA.Average();
}

template <class Type,const int Dim> int  cTriangulation<Type,Dim>::IndexClosestFace(const tPt& aPClose) const
{
    cWhitchMin<int,double> aWMin;

    for (int aKF=0 ; aKF<int(mVFaces.size()) ; aKF++)
    {
       aWMin.Add(aKF,Norm2(aPClose- KthTri(aKF).Barry()));
    }
    return aWMin.IndexExtre();
}

template <class Type,const int Dim> int cTriangulation<Type,Dim>::IndexCenterFace() const
{
	return IndexClosestFace(PAvg());
}


#if (0)
#endif

/* ========================== */
/*       INSTANTIATION        */
/* ========================== */

template class cTriangle<int,2>;
#define  INSTANTIATE_TRI_DIM(TYPE,DIM)\
template class cTriangulation<TYPE,DIM>;\
template class cTriangle<TYPE,DIM>;

#define  INSTANTIATE_TRI(TYPE)\
template cPtxd<TYPE,3> NormalUnit(const cTriangle<TYPE,3> & aTri);\
INSTANTIATE_TRI_DIM(TYPE,2)\
INSTANTIATE_TRI_DIM(TYPE,3)\
//
// INSTANTIATE_TRI_DIM(TYPE,3)


INSTANTIATE_TRI(tREAL4)
INSTANTIATE_TRI(tREAL8)
INSTANTIATE_TRI(tREAL16)

/*
template class cTriangulation<tREAL8,2>;
template class cTriangulation<tREAL8,3>;

template class cTriangle<tREAL8,2>;
template class cTriangle<tREAL8,3>;
template class cTriangle<tREAL16,2>;
template class cTriangle<tREAL16,3>;
*/

};
