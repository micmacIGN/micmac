#include "MMVII_PCSens.h"
// #include "MMVII_BundleAdj.h"


/* We have the image formula w/o distorsion:

Wasserstein 
knowing obly the min and not the vector , how usefull ?

p 26 =< N2 log N  not so much ?
p 30 => how many 


*/

namespace MMVII
{

class cHomogCpleIm
{
      public :
           cHomogCpleIm(const cPt2dr &,const cPt2dr &);
           cPt2dr  mP1;
           cPt2dr  mP2;
};

class cSetHomogCpleIm
{
      public :
        std::vector<cHomogCpleIm>  mSetH;
};

class cHomogCpleDir
{
      public :
           cHomogCpleDir(const cPt3dr & aP1,const cPt3dr & aP2);
           void SetVectMatEss(cDenseVect<tREAL8> &,tREAL8 & aRHS) const;
           cPt3dr  mP1;
           cPt3dr  mP2;
};


class cSetHomogCpleDir
{
      public :
        std::vector<cHomogCpleDir>  mSetD;
	cSetHomogCpleDir(const cSetHomogCpleIm &,const cPerspCamIntrCalib &,const cPerspCamIntrCalib &);
};


/*
 *              (a b c) (x2)
 *              (d e f) (y2)
 *   (x1 y1 z1) (g h i) (z2)
 */
class cMatEssential
{
    public :
        cMatEssential(const cSetHomogCpleDir &,cLinearOverCstrSys<tREAL8> & aSys);
    private :
        cDenseMatrix<tREAL8> mMat;
};

/* ************************************** */
/*                                        */
/*            cHomogCpleDir               */
/*                                        */
/* ************************************** */

cHomogCpleDir::cHomogCpleDir(const cPt3dr & aP1,const cPt3dr & aP2) :
   mP1  (aP1),
   mP2  (aP2)
{
}

void cHomogCpleDir::SetVectMatEss(cDenseVect<tREAL8> &aVect,tREAL8 & aRHS) const
{
         aVect(0) = mP1.x() *  mP2.x();
         aVect(1) = mP1.x() *  mP2.y();
         aVect(2) = mP1.x() *  mP2.z();

         aVect(3) = mP1.y() *  mP2.x();
         aVect(4) = mP1.y() *  mP2.y();
         aVect(5) = mP1.y() *  mP2.z();

         aVect(6) = mP1.z() *  mP2.x();
         aVect(7) = mP1.z() *  mP2.y();
         aRHS    = -mP1.z() *  mP2.z();
}

/* ************************************** */
/*                                        */
/*         cSetHomogCpleDir               */
/*                                        */
/* ************************************** */

cSetHomogCpleDir::cSetHomogCpleDir
(
     const cSetHomogCpleIm &    aSetH,
     const cPerspCamIntrCalib & aCal1,
     const cPerspCamIntrCalib & aCal2
) 
{
     for (const auto & aCplH : aSetH.mSetH)
     {
         cPt3dr aP1 =  aCal1.DirBundle(aCplH.mP1);
         cPt3dr aP2 =  aCal2.DirBundle(aCplH.mP2);
	 mSetD.push_back(cHomogCpleDir(aP1,aP2));
     }
}

/* ************************************** */
/*                                        */
/*         cSetHomogCpleDir               */
/*                                        */
/* ************************************** */

cMatEssential::cMatEssential(const cSetHomogCpleDir & aSetD,cLinearOverCstrSys<tREAL8> & aSys) :
    mMat  (cPt2di(3,3))
{
     aSys.Reset();
     cDenseVect<tREAL8> aVect(8);
     tREAL8 aRHS;
     for (const auto & aCple : aSetD.mSetD)
     {
         aCple.SetVectMatEss(aVect,aRHS);
	 aSys.AddObservation(1.0,aVect,aRHS);
     }
     cDenseVect<tREAL8> aSol = aSys.Solve();

     SetLine(0,mMat,cPt3dr(aSol(0),aSol(1),aSol(2)));
     SetLine(1,mMat,cPt3dr(aSol(3),aSol(4),aSol(5)));
     SetLine(2,mMat,cPt3dr(aSol(6),aSol(7),    1.0));
}


}; // MMVII




