#ifndef _cLSQTemplate_
#define _cLSQTemplate_

#include "StdAfx.h"

typedef Im2D<double,double>     tIm2DM; 	// define a short name for Im2D double image
typedef TIm2D<double,double>    tTIm2DM;

class cImgMatch
{
	public:
        cImgMatch(string aName, cInterfChantierNameManipulateur * mICNM);
	private:
        string  mName;
        cInterfChantierNameManipulateur *mICNM;
        Tiff_Im mTif;
		Pt2dr   mSzIm;
		tIm2DM  mIm2D;
		tTIm2DM mTIm2D;

};

class cLSQTemplate
{
	public:
		cLSQTemplate(cImgMatch * aTemplate, cImgMatch * aImg);	// match an Template "aTemplate" with an Image "aImg"
		bool DoMatch(Pt2dr aP);
        cInterfChantierNameManipulateur * ICNM() {return mICNM;}
	private:
		Pt2dr mPM;	// Point matched
		cImgMatch * mTemplate;
		cImgMatch * mImg;		
        cInterfChantierNameManipulateur * mICNM;

};

#endif
