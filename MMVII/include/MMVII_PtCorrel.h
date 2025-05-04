#include "MMVII_Tpl_Images.h"
#include "MMVII_Geom2D.h"


namespace MMVII
{
/*************************************************************/
/*                                                           */
/*           Auto Correlation directionnelle                 */
/*                                                           */
/*************************************************************/

template <class Type> class  cAutoCorrelDir
{
public :

    cAutoCorrelDir(cDataIm2D<Type> & anIm,const cPt2di & aP0,double aRho,int aSzW) :
        mDIm   (&anIm),
        mP0    (aP0),
        mRho   (aRho),
        mSzW   (aSzW)
    {
        //mDIm->Resize(anIm.P0(),anIm.P1());
        //mDIm->DupIn(anIm);
    }

    cAutoCorrelDir(const cDataIm2D<Type> & anIm,const cPt2di & aP0,double aRho,int aSzW) :
        mDIm   (&(const_cast<cDataIm2D<Type>&>(anIm))),
        mP0    (aP0),
        mRho   (aRho),
        mSzW   (aSzW)
    {
    }

    cPt2dr DoIt()
    {
        double aStep0 = 1/mRho;
        int aNb = MMVII::round_up(M_PI/aStep0);
        cPt2dr aRes0 = DoItOneStep(0.0,aNb,aStep0);
        cPt2dr aRes1 = DoItOneStep(aRes0.x(),3,aStep0/4.0);
        cPt2dr aRes2 = DoItOneStep(aRes1.x(),2,aStep0/10.0);
        return aRes2;
    }

    void ResetIm(cDataIm2D<Type> & aTIm) {mDIm->DupIn(aTIm);}


protected :

    cPt2dr  DoItOneStep(double aTeta0,int aNb,double aStep)
    {
        double aScMax = -1e10;
        double aTetaMax = 0;
        for (int aK=-aNb; aK<aNb ; aK++)
        {
            double aTeta =  aTeta0 + aK * aStep;
            double aVal =  CorrelTeta(aTeta) ;
            if (aVal >aScMax)
            {
                aScMax = aVal;
                aTetaMax = aTeta;
            }
        }
        return cPt2dr(aTetaMax,aScMax);
    }


    double  RCorrelOneOffset(const cPt2di & aP0,const cPt2dr & anOffset,int aSzW)
    {
        cMatIner2Var<double> aMat;
        for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
        {
            for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
            {
                cPt2di aP1 = aP0 + cPt2di(aDx,aDy);
                aMat.Add(mDIm->GetV(aP1),
                         mDIm->GetVBL(ToR(aP1)+anOffset));
            }
        }
        return aMat.Correl();
    }

    double  ICorrelOneOffset(const cPt2di & aP0,const cPt2di & anOffset,int aSzW)
    {
        cMatIner2Var<double> aMat;
        for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
        {
            for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
            {
                cPt2di aP1 = aP0 + cPt2di(aDx,aDy);
                aMat.Add(mDIm->GetV(aP1),mDIm->GetV(aP1+anOffset));
            }
        }
        return aMat.Correl();
    }

    double  CorrelTeta(double aTeta)
    {
        return RCorrelOneOffset(mP0,
                                FromPolar(mRho,aTeta),
                                mSzW);
    }


    cDataIm2D<Type>  *mDIm;
    cPt2di   mP0;
    double  mRho;
    int     mSzW;
};


template <class Type> class cCutAutoCorrelDir : public cAutoCorrelDir<Type>
{
public :
    int mNumOut;
    double  mCorOut;

    cCutAutoCorrelDir(cDataIm2D<Type> & anIm,const cPt2di & aP0,double aRho,int aSzW ) :
        cAutoCorrelDir<Type> (anIm,aP0,aRho,aSzW),
        mVPt (SortedVectOfRadius(0.99,aRho)),
        mNbPts                 (mVPt.size())
        {
        }

        cCutAutoCorrelDir(const cDataIm2D<Type> & anIm,const cPt2di & aP0,double aRho,int aSzW ) :
            cAutoCorrelDir<Type> (anIm,aP0,aRho,aSzW),
            mVPt (SortedVectOfRadius(0.99,aRho)),
            mNbPts                 (mVPt.size())
        {
        }

    void ResetIm(cDataIm2D<Type> & anIm)
        {
            cAutoCorrelDir<Type>::ResetIm(anIm);
        }

    bool  AutoCorrel(const cPt2di & aP0,double aRejetInt,double aRejetReel,double aSeuilAccept,cPt2dr * aPtrRes=0)
    {

        this->mP0 = aP0;
        double aCorrMax = -2;
        int    aKMax = -1;
        //std::cout<<"Debug "<<mVPt.size()<<std::endl;
        for (int aK=0 ; aK<mNbPts ; aK++)
        {
            double aCor = this->ICorrelOneOffset(this->mP0,mVPt[aK],this->mSzW);
            //std::cout << "CCcccI " << aCor << " " << this->mTIm.sz() << "\n";
            if (aCor > aSeuilAccept)
            {
                mCorOut = aCor;
                mNumOut = 0;
                return true;
            }
            if (aCor > aCorrMax)
            {
                aCorrMax = aCor;
                aKMax = aK;
            }
        }
        MMVII_INTERNAL_ASSERT_strong(aKMax!=-1,"AutoCorrel no K" );
        if (aCorrMax < aRejetInt)
        {
            mCorOut = aCorrMax;
            mNumOut = 1;
            return false;
        }

        cPt2dr aRhoTeta = ToPolar<tREAL8>(ToR(mVPt[aKMax]),0.0);
        //std::cout<<" aRhoTeta  Max "<<aRhoTeta<<std::endl;

        double aStep0 = 1/this->mRho;
        //  Pt2dr aRes1 =  this->DoItOneStep(aRhoTeta.y,aStep0*0.5,2);  BUG CORRIGE VERIF AVEC GIANG
        cPt2dr aRes1 =  this->DoItOneStep(aRhoTeta.y(),2,aStep0*0.5);

        if (aRes1.y()>aSeuilAccept)
        {
            mNumOut = 2;
            mCorOut = aRes1.y();
            return true;
        }
        if (aRes1.y()<aRejetReel)
        {
            mNumOut = 3;
            mCorOut = aRes1.y();
            return false;
        }

        // Pt2dr aRes2 =  this->DoItOneStep(aRes1.x,aStep0*0.2,2); BUG CORRIGE VERIF AVEC GIANG
        cPt2dr aRes2 =  this->DoItOneStep(aRes1.x(),2,aStep0*0.2);

        if (aPtrRes)
            *aPtrRes = aRes2;

        mNumOut = 4;
        mCorOut = aRes2.y();
        return aRes2.y() > aSeuilAccept;
    }

private :
    std::vector<cPt2di> mVPt;
    int mNbPts;
};



/* ========================== */
/*     INSTANTIATION          */
/* ========================== */


#define INSTANTIATE_CUT_CORREL(TYPE)\
template class cCutAutoCorrelDir<TYPE>;

#define INSTANTIATE_CORREL(TYPE)\
template class cAutoCorrelDir<TYPE>;

INSTANTIATE_CORREL(tU_INT1);
INSTANTIATE_CORREL(tU_INT2);
INSTANTIATE_CORREL(tREAL4);

INSTANTIATE_CUT_CORREL(tU_INT1);
INSTANTIATE_CUT_CORREL(tU_INT2);
INSTANTIATE_CUT_CORREL(tREAL4);

}
