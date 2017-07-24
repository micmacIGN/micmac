#ifndef _cLSQTemplate_
#define _cLSQTemplate_

#include "StdAfx.h"

typedef Im2D<double,double>     tIm2DM; 	// define a short name for Im2D double image
typedef TIm2D<double,double>    tTIm2DM;

class cImgMatch
{
	public:
        cImgMatch(string aName, cInterfChantierNameManipulateur * mICNM);
        bool GetImget (Pt2dr aP, Pt2dr aSzW);

        cInterfChantierNameManipulateur *ICNM() {return mICNM;}
        Pt2dr &  SzIm() {return mSzIm;}
        tIm2DM  & Im2D() {return mIm2D;}
        tTIm2DM & TIm2D() {return mTIm2D;}
        tIm2DM  & CurImgetIm2D() {return mCurImgetIm2D;}
        tTIm2DM & CurImgetTIm2D() {return mCurImgetTIm2D;}
	private:
        string  mName;
        cInterfChantierNameManipulateur *mICNM;
        Tiff_Im mTif;
		Pt2dr   mSzIm;
        tIm2DM  mIm2D;              // target image
        tTIm2DM mTIm2D;             // target image
        tIm2DM  mCurImgetIm2D;      // current imaget
        tTIm2DM mCurImgetTIm2D;     // current imaget
};

class cLSQMatch
{
	public:
        cLSQMatch(cImgMatch * aTemplate, cImgMatch * aImg);	// match an Template "aTemplate" with an Image "aImg"
        bool DoMatch();
        cInterfChantierNameManipulateur * ICNM() {return mICNM;}
	private:
		Pt2dr mPM;	// Point matched
		cImgMatch * mTemplate;
		cImgMatch * mImg;		
        cInterfChantierNameManipulateur * mICNM;
        cInterpolateurIm2D<double>  * mInterpol;
        double mcurrErr;
        tIm2DM  mImRes;              // Image de residue
};

#endif
