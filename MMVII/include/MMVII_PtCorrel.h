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

    cAutoCorrelDir(cDataIm2D<Type> & anIm, const cPt2di & aP0, double aRho, std::vector<cPt2dr> aIndW):
        mDIm (&anIm),
        mP0 (aP0),
        mRho(aRho),
        mIndW (aIndW)
    {
    }

    cAutoCorrelDir(const cDataIm2D<Type> & anIm, const cPt2di & aP0, double aRho, std::vector<cPt2dr> aIndW):
        mDIm (&(const_cast<cDataIm2D<Type>&>(anIm))),
        mP0 (aP0),
        mRho(aRho),
        mIndW (aIndW)
    {
    }

    /*cPt2dr DoIt()
    {
        double aStep0 = 1/mRho;
        int aNb = MMVII::round_up(M_PI/aStep0);
        cPt2dr aRes0 = DoItOneStep(0.0,aNb,aStep0);
        cPt2dr aRes1 = DoItOneStep(aRes0.x(),3,aStep0/4.0);
        cPt2dr aRes2 = DoItOneStep(aRes1.x(),2,aStep0/10.0);
        return aRes2;
    }*/

    void ResetIm(cDataIm2D<Type> & aTIm) {mDIm->DupIn(aTIm);}


protected :

    cPt2dr  DoItOneStep(double aRho0, double aTeta0,int aNb,double aStep, bool RegularPatch=true)
    {
        double aScMax = -1e10;
        double aTetaMax = 0;
        for (int aK=-aNb; aK<aNb ; aK++)
        {
            double aTeta =  aTeta0 + aK * aStep;
            double aVal =  CorrelTeta(aRho0,aTeta,RegularPatch) ;
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

    double  RCorrelOneOffset(const cPt2dr & aP0,const cPt2dr & anOffset,std::vector<cPt2dr> aIndW)
    {
        cMatIner2Var<double> aMat;
        for (const auto aP: aIndW)
        {
            cPt2dr aP1 = aP0 + aP;
            aMat.Add(mDIm->GetVBL(aP1),mDIm->GetVBL(aP1+anOffset));
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

    double   ICensusOneOffset(const cPt2di & aP0,const cPt2di & anOffset,int aSzW)
    {
        double aVC1 = mDIm->GetV(aP0);
        double aVC2 = mDIm->GetV(aP0+anOffset);

        double aSomDif = 0.0;
        double aNb     = 0.0;
        for (const auto & aDP : cRect2::BoxWindow(aSzW))
        {
            double aV1 = mDIm->GetV(aP0+aDP);
            double aV2 = mDIm->GetV(aP0+aDP+anOffset);
            double aR1 = NormalisedRatioPos(aV1,aVC1);
            double aR2 = NormalisedRatioPos(aV2,aVC2);

            aSomDif += std::abs(aR1-aR2);
            aNb++;
        }
        double aRes = std::min(1.0,aSomDif / aNb);
        aRes = pow(aRes,0.25);

        return aRes;
    }



    double  CorrelTeta(double aRho, double aTeta, bool isRegularPatch)
    {
        if (isRegularPatch)
            return RCorrelOneOffset(mP0,
                                FromPolar(aRho,aTeta),
                                mSzW);
        else
            return RCorrelOneOffset(ToR(mP0),
                                    FromPolar(aRho,aTeta),
                                    mIndW);
    }


    cDataIm2D<Type>  *mDIm;
    cPt2di   mP0;
    double  mRho;
    int     mSzW;
    std::vector<cPt2dr> mIndW;
};


template <class Type> class cCutAutoCorrelDir : public cAutoCorrelDir<Type>
{
public :
    int mNumOut;
    double  mCorOut;

    cCutAutoCorrelDir(cDataIm2D<Type> & anIm,const cPt2di & aP0,double aRho,int aBuf,int aSzW ) :
        cAutoCorrelDir<Type> (anIm,aP0,aRho,aSzW),
        //mVPt(SortedAngleFlux2StdCont(mVPt,circle(Pt2dr(0,0),aRho))),
        mVPt (SortedVectOfRadius(0.0,aRho,false)),
        mNbPts                 (mVPt.size())
        {
            //SortedAngleFlux2StdCont(mVPt,circle(Pt2dr(0,0),aRho));
            //mNbPts = mVPt.size();
        }

        cCutAutoCorrelDir(const cDataIm2D<Type> & anIm,const cPt2di & aP0,double aRho,int aBuf, int aSzW ) :
            cAutoCorrelDir<Type> (anIm,aP0,aRho,aSzW),
            mVPt (SortedVectOfRadius(0.0,aRho,false)),
            mNbPts                 (mVPt.size())
        {
            //SortedAngleFlux2StdCont(mVPt,circle(Pt2dr(0,0),aRho));
            //mNbPts = mVPt.size();
        }

        cCutAutoCorrelDir(cDataIm2D<Type> & anIm,const cPt2di & aP0,double aRho,int aBuf,std::vector<cPt2dr> aIndW  ) :
            cAutoCorrelDir<Type> (anIm,aP0,aRho,aIndW),
            mVPt (SortedVectOfRadius(0.0,aRho,false)),
            mNbPts                 (mVPt.size())
        {
        }

        cCutAutoCorrelDir(const cDataIm2D<Type> & anIm,const cPt2di & aP0,double aRho,int aBuf, std::vector<cPt2dr> aIndW ) :
            cAutoCorrelDir<Type> (anIm,aP0,aRho,aIndW),
            mVPt (SortedVectOfRadius(0.0,aRho,false)),
            mNbPts                 (mVPt.size())
        {
        }


    void ResetIm(cDataIm2D<Type> & anIm)
        {
            cAutoCorrelDir<Type>::ResetIm(anIm);
        }


        void GetVPts()
        {
            for (const auto aPt: mVPt)
            {
                StdOut()<<"aPt X: "<<aPt.x()<<" Y: "<<aPt.y()<<std::endl;
            }
        }

        void writeImage(int aRho, std::string filename)
        {
            cDataIm2D<tU_INT1> aDIm (cPt2di(0,0),cPt2di(2*aRho,2*aRho));
            cPt2di aP2;
            for (const auto aPt: mVPt)
            {
                aP2 =cPt2di(aPt.x()+aRho,
                             aPt.y()+aRho);
                //StdOut()<<aP2<<std::endl;

                if (aDIm.Inside(aP2))
                    aDIm.SetV(aP2,1);
            }

        // save image
            aDIm.ToFile(filename);
        }


        void writeCorrelImage (int aRho, std::string filename)
        {
            cDataIm2D<tREAL4> aDIm (cPt2di(0,0),cPt2di(2*aRho,2*aRho),nullptr,eModeInitImage::eMIA_Null);
            cPt2di aP2;
            for (int aK=0 ; aK<mNbPts ; aK++)
            {
                double aCor = this->ICorrelOneOffset(this->mP0,mVPt[aK],this->mSzW);
                //cPt2di aPIm= this->mP0 +mVPt[aK];
                aP2 =   cPt2di(mVPt[aK].x()+aRho,
                            mVPt[aK].y()+aRho)  ;
                if (aDIm.Inside(aP2))
                    aDIm.SetV(aP2,aCor);
            }
            aDIm.ToFile(filename);
        }


        /*   compute mi between two arrays
      static double mutual_information(cv::Mat ref, cv::Mat flt)
      {
         cv::Mat joint_histogram(256, 256, CV_64FC1, cv::Scalar(0));

         for (int i=0; i<ref.cols; ++i) {
            for (int j=0; j<ref.rows; ++j) {
               int ref_intensity = ref.at<uchar>(j,i);
               int flt_intensity = flt.at<uchar>(j,i);
               joint_histogram.at<double>(ref_intensity, flt_intensity) = joint_histogram.at<double>(ref_intensity, flt_intensity)+1;
               double v = joint_histogram.at<double>(ref_intensity, flt_intensity);
            }
         }



         for (int i=0; i<256; ++i) {
            for (int j=0; j<256; ++j) {
               joint_histogram.at<double>(j, i) = joint_histogram.at<double>(j, i)/(1.0*ref.rows*ref.cols);
               double v = joint_histogram.at<double>(j, i);
            }
         }

         cv::Size ksize(7, 7);
         cv::GaussianBlur(joint_histogram, joint_histogram, ksize, 7, 7);


         double entropy = 0.0;
         for (int i=0; i<256; ++i) {
            for (int j=0; j<256; ++j) {
               double v = joint_histogram.at<double>(j, i);
               if (v > 0.000000000000001) {
                  entropy += v*log(v)/log(2);
               }
            }
         }
         entropy *= -1;

         //    std::cout << entropy << "###";



         std::vector<double> hist_ref(256, 0.0);
         for (int i=0; i<joint_histogram.rows; ++i) {
            for (int j=0; j<joint_histogram.cols; ++j) {
               hist_ref[i] += joint_histogram.at<double>(i, j);
            }
         }

         cv::Size ksize2(5,0);
         //  cv::GaussianBlur(hist_ref, hist_ref, ksize2, 5);


         std::vector<double> hist_flt(256, 0.0);
         for (int i=0; i<joint_histogram.cols; ++i) {
            for (int j=0; j<joint_histogram.rows; ++j) {
               hist_flt[i] += joint_histogram.at<double>(j, i);
            }
         }

         //   cv::GaussianBlur(hist_flt, hist_flt, ksize2, 5);



         double entropy_ref = 0.0;
         for (int i=0; i<256; ++i) {
            if (hist_ref[i] > 0.000000000001) {
               entropy_ref += hist_ref[i] * log(hist_ref[i])/log(2);
            }
         }
         entropy_ref *= -1;
         //std::cout << entropy_ref << "~~ ";

         double entropy_flt = 0.0;
         for (int i=0; i<256; ++i) {
            if (hist_flt[i] > 0.000000000001) {
               entropy_flt += hist_flt[i] * log(hist_flt[i])/log(2);
            }
         }
         entropy_flt *= -1;
         // std::cout << entropy_flt << "++ ";

         double mutual_information = entropy_ref + entropy_flt - entropy;
         return mutual_information;
      }*/




    bool AutoCensusQ (const cPt2di & aP0,
                         double aRejetThreshold,
                         double aRejetCens)
    {

        this-> mP0 = aP0;

        double aCensusMax = 0 ;
        //int aKMax =-1 ;

        for (int aK=0; aK<mNbPts; aK++)
        {
            double aCensVal = 1.0 - this->ICensusOneOffset(this->mP0,mVPt[aK],this->mSzW);

            if (aCensVal>aRejetThreshold)
            {
                mCorOut = aCensVal;
                mNumOut = 0;
                return true;
            }

            if (aCensVal>aCensusMax)
            {
                aCensusMax = aCensVal;
                //aKMax= aK;
            }
        }

        //
        if (aCensusMax<aRejetCens)
        {
            mCorOut= aCensusMax;
            mNumOut= 1;
            return false;
        }
        return true;
    }


    bool  AutoCorrel(const cPt2di & aP0,
                        double aRejetInt,
                        double aRejetReel,
                        double aSeuilAccept,
                        cPt2dr * aPtrRes=0)
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

        /*cPt2dr aRhoTeta = ToPolar<tREAL8>(ToR(mVPt[aKMax]),0.0);
        //std::cout<<" aRhoTeta  Max "<<aRhoTeta<<std::endl;

        double aStep0 = 1/(aRhoTeta.x()+1e-8);
        //  Pt2dr aRes1 =  this->DoItOneStep(aRhoTeta.y,aStep0*0.5,2);  BUG CORRIGE VERIF AVEC GIANG
        cPt2dr aRes1 =  this->DoItOneStep(aRhoTeta.x(),aRhoTeta.y(),2,aStep0*0.5);

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

        cPt2dr aRes2 =  this->DoItOneStep(aRhoTeta.x(), aRes1.x(),2,aStep0*0.2);

        if (aPtrRes)
            *aPtrRes = aRes2;

        mNumOut = 4;
        mCorOut = aRes2.y();
        return aRes2.y() > aSeuilAccept;*/

        return true;
    }



    bool  AutoCorrelNonRegularPatch(const cPt2di & aP0,
                                   double aRejetInt, // 0.4
                                   double aRejetReel, //0.5
                                   double aSeuilAccept // 0.6
                                   )
    {

        this->mP0 = aP0;
        double aCorrMax = -2;
        int    aKMax = -1;
        //std::cout<<"Debug "<<mVPt.size()<<std::endl;
        for (int aK=0 ; aK<mNbPts ; aK++)
        {
            double aCor = this->RCorrelOneOffset(ToR(this->mP0),ToR(mVPt[aK]),this->mIndW);
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
        cPt2dr aRes1 =  this->DoItOneStep(aRhoTeta.y(),2,aStep0,false);

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

        /*cPt2dr aRes2 =  this->DoItOneStep(aRes1.x(),2,aStep0*0.2,false);

        mNumOut = 4;
        mCorOut = aRes2.y();
        return aRes2.y() > aRejetReel;*/


        return true;
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
