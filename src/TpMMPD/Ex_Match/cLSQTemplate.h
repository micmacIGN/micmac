#ifndef _cLSQTemplate_
#define _cLSQTemplate_

#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"


typedef Im2D<double,double>     tIm2DM; 	// define a short name for Im2D double image
typedef TIm2D<double,double>    tTIm2DM;

class cParamLSQMatch
{
    public:
        bool mDisp;
        double mStepCorrel;
        double mStepLSQ;
        int mStepPxl;
        int mNbIter;
};

class cImgMatch
{
	public:
        cImgMatch(string aName, cInterfChantierNameManipulateur * mICNM);
        bool GetImget (Pt2dr aP, Pt2dr aSzW,  Pt2dr aRab = Pt2dr(0,0));

        cInterfChantierNameManipulateur *ICNM() {return mICNM;}
        Pt2dr &  SzIm() {return mSzIm;}
        tIm2DM  & Im2D() {return mIm2D;}
        tTIm2DM & TIm2D() {return mTIm2D;}
        tIm2DM  & CurImgetIm2D() {return mCurImgetIm2D;}
        tTIm2DM & CurImgetTIm2D() {return mCurImgetTIm2D;}
        void Load();
        Pt2dr & CurPt(){return mCurPt;}

	private:
        string  mName;
        cInterfChantierNameManipulateur *mICNM;
        Tiff_Im mTif;
		Pt2dr   mSzIm;
        tIm2DM  mIm2D;              // target image
        tTIm2DM mTIm2D;             // target image
        tIm2DM  mCurImgetIm2D;      // current imaget
        tTIm2DM mCurImgetTIm2D;     // current imaget
        Pt2dr mCurPt;               // store current matching point on mImg
};

class cLSQMatch
{
	public:
        cLSQMatch(cImgMatch * aTemplate, cImgMatch * aImg);	// match an Template "aTemplate" with an Image "aImg"
        cParamLSQMatch & Param() {return mParam;}
        bool DoMatchbyLSQ();
        bool DoMatchbyCorel();
        bool MatchbyLSQ(Pt2dr aPt1,
                            const tIm2DM & aImg1,
                            const tIm2DM & aImg2,
                            Pt2dr aPt2,
                            Pt2di aSzW,
                            double aStep
                        , Im1D_REAL8 &aSol);
        cInterfChantierNameManipulateur * ICNM() {return mICNM;}
        tIm2DM & ImRes() {return mImRes;}
        void update(double CurErr, Pt2dr aPt);
        double & MinErr() {return mMinErr;}
        Pt2dr & PtMinErr() {return mPtMinErr;}
        cInterpolateurIm2D<double>  * Interpol(){return mInterpol;}
	private:
		Pt2dr mPM;	// Point matched
		cImgMatch * mTemplate;
		cImgMatch * mImg;		
        cInterfChantierNameManipulateur * mICNM;
        cInterpolateurIm2D<double>  * mInterpol;
        double mcurrErr;
        tIm2DM  mImRes;              // Image de residue

        Video_Win * mWTemplate;
        Video_Win * mWTarget;
        cParamLSQMatch mParam;

        double mMinErr;
        Pt2dr mPtMinErr;
};

double Tst_Correl1Win
                             (
                                const tIm2DM & Im1,
                                const Pt2di & aP1,
                                const tIm2DM & Im2,
                                const Pt2di & aP2,
                                const int   aSzW,
                                const int   aStep
                             );

cResulRechCorrel      Tst_Correl
                      (
                             const tIm2DM & Im1,
                             const Pt2di & aP1,
                             const tIm2DM & Im2,
                             const Pt2di & aP2,
                             const int   aSzW,
                             const int   aStep,
                             const int   aSzRech,
                             tIm2DM & ImScore
                      );

#endif
