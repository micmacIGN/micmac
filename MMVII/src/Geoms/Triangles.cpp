#include "include/MMVII_all.h"

namespace MMVII
{

/* ********************************************** */
/*                                                */
/*           cTriangle                            */
/*                                                */
/* ********************************************** */

template <const int Dim>  cTriangle<Dim>::cTriangle(const tPt & aP0,const tPt & aP1,const tPt & aP2)
{
   mPts[0] = aP0;
   mPts[1] = aP1;
   mPts[2] = aP2;
}


template <const int Dim> const cPtxd<double,Dim>& cTriangle<Dim>::Pt(int aK) const
{
     MMVII_INTERNAL_ASSERT_tiny((aK>=0) && (aK<3),"cTriangle2D::Pt");
     return mPts[aK];
}

template <const int Dim> cPtxd<double,Dim> cTriangle<Dim>::FromCoordBarry(const cPt3dr & aP) const
{
     double aSP = aP.x()+aP.y()+aP.z();
     MMVII_INTERNAL_ASSERT_tiny(aSP!=0,"Sum weith null in barry");

     return (aP.x()*mPts[0]+aP.y()*mPts[1]+aP.z()*mPts[2]) / aSP;
}

template <const int Dim> cTplBox<double,Dim> cTriangle<Dim>::BoxEngl() const
{
     cTplBoxOfPts<tREAL8,Dim> aTBox;
     for (int aK=0 ; aK<3 ; aK++)
         aTBox.Add(mPts[aK]);
     return aTBox.CurBox();
}

template <const int Dim> cTplBox<int,Dim> cTriangle<Dim>::BoxPixEngl() const
{
    cTplBox<tREAL8,Dim> aBoxE = BoxEngl();
    return cTplBox<int,Dim> (Pt_round_down(aBoxE.P0()),Pt_round_up(aBoxE.P1()));
}

template <const int Dim> double cTriangle<Dim>::Regularity() const
{
    tPt aV01 = mPts[1]-mPts[0];
    tPt aV02 = mPts[2]-mPts[0];
    tPt aV12 = mPts[2]-mPts[1];
    double aSomSqN2 = SqN2(aV01) + SqN2(aV02) + SqN2(aV12);
    if (aSomSqN2==0) return 0;
    return AbsSurfParalogram(aV01,aV02) / aSomSqN2;
}


template <const int Dim>  cPtxd<double,Dim> cTriangle<Dim>::CenterInscribedCircle() const
{
   cLeasSqtAA<double>  aSys(Dim);

   for (int aKp=0 ; aKp<3 ; aKp++)
   {
       const tPt & aP0 = mPts[aKp];
       const tPt & aP1 = mPts[(aKp+1)%3];
       tPt aMil = (aP0+aP1)/2.0;
       tPt aV01 = aP1-aP0;

       aSys.AddObservation(1.0,aV01.ToVect(),Scal(aMil,aV01));
   }

   // For Dim=3 we have computed inter of 3 plane mediator that co-intersect in a line
   // we need to add also a equation inside the plane
   MMVII_INTERNAL_ASSERT_tiny(Dim==2,"CenterInscribedCircle to Finish for DIm!=2");

   return tPt::FromVect(aSys.Solve());
}

/* *********************************************************** */
/*                                                             */
/*                cTriangulation                               */
/*                                                             */
/* *********************************************************** */

template <const int Dim> cTriangulation<Dim>::cTriangulation(const std::vector<tPt>& aVPts)  :
    mVPts  (aVPts)
{
}
template <const int Dim> void cTriangulation<Dim>::ResetTopo()
{
    mVFaces.clear();
}

template <const int Dim> bool cTriangulation<Dim>::ValidFace(const tFace & aFace) const
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

template <const int Dim> void cTriangulation<Dim>::AddFace(const tFace & aFace)
{
    MMVII_INTERNAL_ASSERT_tiny(ValidFace(aFace),"Bad face");
    mVFaces.push_back(aFace);
}

template <const int Dim> int cTriangulation<Dim>::NbTri() const
{
   return mVFaces.size();
}

template <const int Dim> const cPt3di & cTriangulation<Dim>::KthFace(int aK) const
{
   return mVFaces.at(aK);
}

template <const int Dim> cTriangle<Dim>    cTriangulation<Dim>::KthTri(int aK) const
{
    const cPt3di &  aIndT = KthFace(aK);
    return cTriangle<Dim>(mVPts.at(aIndT.x()),mVPts.at(aIndT.y()),mVPts.at(aIndT.z()));
}

template <const int Dim> cTplBox<tREAL8,Dim>    cTriangulation<Dim>::BoxEngl(double aFactMargin) const
{
    cTplBox<tREAL8,Dim> aBox =  cTplBox<tREAL8,Dim>::FromVect(mVPts);

    if (aFactMargin!=0)
       aBox = aBox.ScaleCentered(1+aFactMargin) ;
    return aBox;
}

template <const int Dim> double HeuristikDistTri (const cTriangle<Dim> & aT1,const cTriangle<Dim> & aT2)
{
    double aDMin = 1e20;

    // Must be equal up to an offset (same orientation) Test all possible offset
    for (int aOffset=0 ; aOffset<3 ; aOffset++)
    {
         double aSomD=0 ;
	 for (int aK1=0 ; aK1<3 ;aK1++)
             aSomD += Norm1(   aT1.Pt(aK1)-aT2.Pt((aK1+aOffset)%3)   );
	 UpdateMin(aDMin,aSomD);
    }
    return aDMin;
}



template <const int Dim> bool cTriangulation<Dim>::HeuristikAlmostInclude (const cTriangulation<Dim> &aT2,double aTolP,double aTolF)  const
{
     for (const auto & aP1 : mVPts)
     {
          double aDMin =aTolP+1000;
          for (int aK2=0 ; (aK2<int(aT2.mVPts.size())) && (aDMin>aTolP) ; aK2++)
	  {
              UpdateMin(aDMin,Norm2(aP1-aT2.mVPts[aK2]));
	  }
	  if (aDMin>aTolP) 
             return false;
     }

     if (aTolF>0)
     {
        for (int aK1=0 ; aK1<int(mVFaces.size()) ; aK1+=10)
        {
             double aDMin =aTolF+1000;
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

template <const int Dim> bool cTriangulation<Dim>::HeuristikAlmostEqual (const cTriangulation<Dim> &aT2,double aTolP,double aTolF)  const
{
    if  (mVPts.size() != aT2.mVPts.size()) 
        return false;

    if  (mVFaces.size() != aT2.mVFaces.size()) 
        return false;

    return HeuristikAlmostInclude (aT2,aTolP,aTolF) && aT2.HeuristikAlmostInclude(*this,aTolP,aTolF);
}

template <const int Dim> 
    void cTriangulation<Dim>::Filter
                        (const cDataBoundedSet<tREAL8,Dim> & aSet,int aNbVertixThres) 
{
    cTriangulation<Dim> aRes;
    std::vector<tFace> aVFaces;
    std::vector<tPt>   aVPts;

    // -1- compute points in the set
    std::vector<bool>  aVPtsInSet; // are points inside the set 
    std::vector<bool>  aVPtsInTri;  // do point belongs to one  of the selected triangle
    for (const auto & aPts : mVPts)
    {
        aVPtsInSet.push_back(aSet.Inside(aPts));
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

/* ========================== */
/*       INSTANTIATION        */
/* ========================== */

template class cTriangulation<2>;
template class cTriangulation<3>;

template class cTriangle<2>;
template class cTriangle<3>;

};
