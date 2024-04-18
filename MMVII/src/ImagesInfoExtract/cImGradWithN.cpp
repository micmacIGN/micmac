#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Geom2D.h"

#include "MMVII_ExtractLines.h"
#include "MMVII_TplGradImFilter.h"
#include "MMVII_util_tpl.h"

namespace MMVII
{


cTabulateGrad::cTabulateGrad(int aVMax) :
   mVMin     (-aVMax),
   mVMax     (aVMax),
   mNbDirTabAng (0),
   mP0       (-mVMax,-mVMax),
   mP1       (1+mVMax,1+mVMax),
   mTabRho   (mP0,mP1),
   mDataRho  (&mTabRho.DIm()),
   mTabTeta  (mP0,mP1),
   mDataTeta (&mTabTeta.DIm()),
   mImIndAng (cPt2di(1,1)),
   mDataIIA  (nullptr)
{
   for (const auto & aPix : *mDataRho)
   {
        cPt2dr aPol = ToPolar(ToR(aPix),0.0);
        mDataRho->SetV(aPix,aPol.x());
        mDataTeta->SetV(aPix,aPol.y());
   }
}

int cTabulateGrad::VMax() const {return mVMax;}

int cTabulateGrad::Teta2Index(tREAL8 Teta) const
{
    return  mod (  round_ni((mNbDirTabAng*Teta)/(2*M_PI)) , mNbDirTabAng);
}

tREAL8 cTabulateGrad::Index2Teta(int aInd) const
{
    return  (mod(aInd,mNbDirTabAng) * 2*M_PI) / mNbDirTabAng;
}

void cTabulateGrad::TabulateTabAng(int aNbDir)
{
     if (mNbDirTabAng==aNbDir)
        return;

     mNbDirTabAng= aNbDir;

     mImIndAng =  cIm2D<tInAng> (mP0,mP1);
     mDataIIA  = & mImIndAng.DIm();

     for (const auto & aPix : *mDataRho)
     {
         mDataIIA->SetV(aPix,Teta2Index(mDataTeta->GetV(aPix)));
     }
}

void cTabulateGrad::TabulateNeighMaxLocGrad(int aNbDir,tREAL8 aRho0,tREAL8 aRho1)
{
    TabulateTabAng(aNbDir);
    tREAL8 aTetaMax = (M_PI/4.0) + 0.001; // opening angle, here PI/4 with a small margin
    tREAL8 aSinMax= std::sin(aTetaMax);

    if ((int) mTabNeighMaxLocGrad.size() != aNbDir)
    {
        mTabNeighMaxLocGrad.clear();
        mTabIndMLGRho0.clear();
        for (int aKTeta=0 ; aKTeta<mNbDirTabAng ; aKTeta++)
        {
             cPt2dr aDirTeta = FromPolar(1.0,Index2Teta(aKTeta));
	     std::vector<cPt2di> aNeighbourhood;

             for (const auto aPixN : cRect2::BoxWindow(round_up(aRho1)))
             {
                 tREAL8 aNorm = Norm2(aPixN);
                 if ((aNorm>0) && (aNorm<=aRho1))
                 {
                     cPt2dr   aDirLoc = VUnit(ToR(aPixN)) / aDirTeta;
                     if ((aDirLoc.x() >0) && (std::abs(aDirLoc.y())  < aSinMax))
                     {
                         aNeighbourhood.push_back(aPixN);
                     }
                 }
             }
	     SortOnCriteria(aNeighbourhood,[](const cPt2di & aPt) {return SqN2(aPt);});
	     size_t aIndRho0 = 0;
	     while (  (aIndRho0<aNeighbourhood.size())  && (Norm2(aNeighbourhood.at(aIndRho0)) < aRho0)   )
                   aIndRho0++;

	     mTabNeighMaxLocGrad.push_back(aNeighbourhood);
	     mTabIndMLGRho0.push_back(aIndRho0);
        }
    }
}

std::pair<const std::vector<cPt2di>*,int>  cTabulateGrad::TabNeighMaxLocGrad(const cPt2di & aPGrad) const
{
    int anInd = mDataIIA->GetV(aPGrad);

    return std::pair<const std::vector<cPt2di>*,int> ( &mTabNeighMaxLocGrad.at(anInd)  ,  mTabIndMLGRho0.at(anInd) ) ;
}



/* ************************************************************************ */
/*                                                                          */
/*                       cImGradWithN                                       */
/*                                                                          */
/* ************************************************************************ */


template <class Type>   
  cImGradWithN<Type>::cImGradWithN(const cPt2di & aSz) :
     cImGrad<Type>  (aSz),
     mNormG         (aSz),
     mDataNG        (mNormG.DIm())
{
}

template <class Type>    void cImGradWithN<Type>::SetDeriche(cDataIm2D<Type> & aDIm,Type aAlphaDeriche)
{
    mNormG.DIm().AssertSameArea(aDIm);
    ComputeDericheAndNorm(*this,aDIm,aAlphaDeriche);
}

template<class Type> void cImGradWithN<Type>::SetQuickSobel(cDataIm2D<Type> & aDIm,cTabulateGrad & aTab,int aDiv)
{
   aTab.ComputeSobel(*this->mDGx,*this->mDGy,aDIm,aDiv);
   aTab.ComputeNorm(mDataNG, *this->mDGx,*this->mDGy);
}


/*   The test is  more complicated than traditionnal local maxima :
 *
 *     - 1- On a given edge if we take all the 8-neighboors, one the neighbors in the edge will
 *         have higher value, that why we take the neigbours that are +- in the direction of grad
 *         See "NeighborsForMaxLoc()"
 *
 *     - 2- with thin line, the oposite contour may have value that may be higer than the contour itself,
 *      that's why we consider only the point which are orientd in the same direction (test "Scal > 0"),
 *      note this work for a dark or light line on a average back-ground
 */
template<class Type> bool  cImGradWithN<Type>::IsMaxLocDirGrad(const cPt2di& aPix,const std::vector<cPt2di> & aVP,tREAL8 aRatioXY) const
{
    //  [1] Compute unitary vector
    tREAL8 aN = mDataNG.GetV(aPix);
    if (aN==0) return false;  // avoid div by 0
    cPt2dr aDirGrad = ToR(this->Grad(aPix)) * (1.0/ aN);

    //[2]   A Basic test to reject point on integer neighbourhood in the two direction
    {
        cPt2di aIDirGrad = ToI(aDirGrad);
        for (int aSign : {-1,1})
        {
             cPt2di  aNeigh =  aPix + aIDirGrad * aSign;
             if ( (mDataNG.DefGetV(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->Grad(aNeigh)))>0) )
             {
                return false;
             }
        }
    }

    // [3]

    cPt2dr aConjDG =  conj(aDirGrad);
    for (const auto & aDeltaNeigh : aVP)
    {
        // cPt2dr aDirLoc = ToR(aDeltaNeigh) / aDirGrad;
        cPt2dr aDirLoc = ToR(aDeltaNeigh) * aConjDG;
        if (std::abs(aDirLoc.x()) >= std::abs(aDirLoc.y()*aRatioXY))
	{
            cPt2di aNeigh = aPix + aDeltaNeigh;

            if ( (mDataNG.DefGetV(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->Grad(aNeigh))) >0))
               return false;
	}
        /*cPt2dr aNeigh = ToR(aPix) + ToR(aDeltaNeigh) * aDirGrad;  
        if ( (mDataNG.DefGetVBL(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->GradBL(aNeigh))) >0))
           return false;
        */
    }

    return true;
}

bool PtIsSup00(const cPt2di & aP)
{
     if (aP.x()>0)  return true;
     if (aP.x()<0)  return false;
     return aP.y() > 0;
}


template<class Type> bool  cImGradWithN<Type>::TabIsMaxLocDirGrad(const cPt2di& aPix,const cTabulateGrad & aTabul,bool isWhite) const
{
    int aSignGlob = isWhite ? -1 : 1;
    Type  aNormG = mDataNG.GetV(aPix);
    cPt2di aGrad(this->mDGx->GetV(aPix),this->mDGy->GetV(aPix));
    auto [aPtrVecNeigh,aIndR0] = aTabul.TabNeighMaxLocGrad(aGrad);

    for (int anInd=0 ; anInd<aIndR0 ; anInd++)
    {
        for (auto aSign : {-1,1})
	{
             cPt2di aNeigh = aPtrVecNeigh->at(anInd) * aSign ;
             Type aNormNeigh =  mDataNG.GetV(aPix+aNeigh);

             if (aNormG<aNormNeigh) 
                return false;
             if ((aNormG==aNormNeigh) && PtIsSup00(aNeigh))
                return false;
	}
    }

    for (size_t anInd=aIndR0 ; anInd<aPtrVecNeigh->size() ; anInd++)
    {
        cPt2di aNeigh = aPtrVecNeigh->at(anInd) * aSignGlob;
        Type aNormNeigh =  mDataNG.GetV(aPix+aNeigh);

        if (aNormG<aNormNeigh) 
           return false;
        if ((aNormG==aNormNeigh) && PtIsSup00(aNeigh))
           return false;
    }

    for (size_t anInd=aIndR0 ; anInd<aPtrVecNeigh->size() ; anInd++)
    {
        cPt2di aNeigh = aPtrVecNeigh->at(anInd) * (-aSignGlob);
        cPt2di aGradNeigh(this->mDGx->GetV(aPix+aNeigh),this->mDGy->GetV(aPix+aNeigh));
	if (Scal(aGrad,aGradNeigh) > 0)
        {
            Type aNormNeigh =  mDataNG.GetV(aPix+aNeigh);

            if (aNormG<aNormNeigh) 
               return false;
            if ((aNormG==aNormNeigh) && PtIsSup00(aNeigh))
               return false;
	}
    }

    return true;
}
/*
*/

//  return simply all the point except (0,0)
template<class Type> std::vector<cPt2di>   cImGradWithN<Type>::NeighborsForMaxLoc(tREAL8 aRay) 
{ 
     return SortedVectOfRadius(0.5,aRay); 
}

/*   Make one iteration of position computation :
 *
 *     -1- compute the value of gradient norm befor and after the point in gradient direction
 *     -2- extract the maximal by parabol fiting
 */

template<class Type> cPt2dr   cImGradWithN<Type>::OneRefinePos(const cPt2dr & aP1) const
{
     if (! mDataNG.InsideBL(aP1)) 
         return aP1;

     cPt2dr aGr = ToR(this->GradBL(aP1));
     tREAL8 aNorm = Norm2(aGr);

     if (aNorm==0) 
        return aP1;
     aGr = aGr / aNorm;

     // extract point "before"
     cPt2dr aP0 = aP1- aGr;
     if (! mDataNG.InsideBL(aP0)) 
         return aP1;

     // extract point "after"
     cPt2dr aP2 = aP1+ aGr;
     if (! mDataNG.InsideBL(aP2)) 
         return aP1;

     // extract abscisse of maxim of parabol
     tREAL8 aAbs = StableInterpoleExtr(mDataNG.GetVBL(aP0),mDataNG.GetVBL(aP1),mDataNG.GetVBL(aP2));

     return aP1 + aGr * aAbs;
}

//  Very basic just max two iterations
template<class Type> cPt2dr   cImGradWithN<Type>::RefinePos(const cPt2dr & aP1) const
{
    return OneRefinePos(OneRefinePos(aP1));
}



          /* ************************************************ */

template<class Type> void ComputeDericheAndNorm(cImGradWithN<Type> & aResGrad,const cDataIm2D<Type> & aImIn,double aAlpha) 
{
         ComputeDeriche(aResGrad,aImIn,aAlpha);
         auto & aDN =  aResGrad.NormG().DIm();
         for (const auto &  aPix : aDN)
         {
               aDN.SetV(aPix,Norm2(aResGrad.Grad(aPix)));
         }
}


template class cImGradWithN<tREAL4>;
template void ComputeDericheAndNorm(cImGradWithN<tREAL4> & aResGrad,const cDataIm2D<tREAL4> & aImIn,double aAlpha) ;

};

