/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/


/*
   Sur le long terme, modifier :

     - la possiblite d'avoir des fonctions multiple (Dim N)
     - associer a un fctr, un cSetEqFormelles
*/

#include "StdAfx.h"


/************************************************************/
/*                                                          */
/*                   cEqEllipseImage                        */
/*                                                          */
/************************************************************/

cEqEllipseImage::cVarRay::cVarRay(INT aNum) :
    mFRay   (0.0,std::string("Diam") + ToString(aNum)),
    mAdrVar (0)
{
}


std::vector<cEqEllipseImage::cVarRay>  cEqEllipseImage::VarsRays()
{
     std::vector<cEqEllipseImage::cVarRay> aRes;
     for (INT aK=0 ; aK< mMire.NbDiam() ; aK++)
        aRes.push_back(cVarRay(aK));

     return aRes;
}

Fonc_Num    cEqEllipseImage::EqResidu()
{
    Fonc_Num fSurf = 0;
    INT aSign = 1;

    for (INT aK= 0 ; aK < mMire.NbDiam() ; aK++)
    {
        // REAL F =  mMire.KthDiam(0) / mMire.KthDiam(aK);
        Fonc_Num F = mVarsRays[0].mFRay / mVarsRays[aK].mFRay;
        fSurf = fSurf + FN_SurfIER(fCentre,fA*F,fB*F,fC*F,fP0,fP1) * aSign;
        aSign *= -1;
    }
    fSurf = fSurf / Square(2*fLarg);
        
    return fNoir * fSurf + fBlanc * (1-fSurf) - fGray;

}

void cEqEllipseImage::GenCode()
{
     cElCompileFN::DoEverything
     (
         std::string("CodeGenere")+ELISE_CAR_DIR+"photogram"+ELISE_CAR_DIR,
         mNameType,
         fNumEqResidu,
         mLInterv
     );

}


cEqEllipseImage::cEqEllipseImage
(
     cSetEqFormelles & anEq,
     const cMirePolygonEtal & aMire,
     Pt2dr aCentre,
     REAL  anA,
     REAL  aB,
     REAL  aC,
     REAL  aLarg,
     REAL  aBlanc,
     REAL  aNoir,
     bool  Code2Gen
) :
  cElemEqFormelle (anEq,false),
  mMire     (aMire),
  mNameType ("cEqCibleEllipse" + ToString(aMire.NbDiam())),
  mEq       (anEq),
  mAlloc    (mEq.Alloc()),
  mI0       (mAlloc.CurInc()),
  mIndCentre (mAlloc.CurInc()),
  mCentre   (aCentre),
  fCentre   (mAlloc.NewPt2("Ellipse",mCentre)),
  mIndA     (mAlloc.CurInc()),
  mA        (anA),
  fA        (mAlloc.NewF("Ellipse","A",&mA)),
  mIndB     (mAlloc.CurInc()),
  mB        (aB),
  fB        (mAlloc.NewF("Ellipse","B",&mB)),
  mIndC     (mAlloc.CurInc()),
  mC        (aC),
  fC        (mAlloc.NewF("Ellipse","C",&mC)),
  mIndLarg  (mAlloc.CurInc()),
  mLarg     (aLarg),
  fLarg     (mAlloc.NewF("Ellipse","Larg",&mLarg)),
  fPLarg    (fLarg,fLarg),
  mNPX      ("XPix"),
  mNPY      ("YPix"),
  fCPIX     (cVarSpec(0,mNPX),cVarSpec(0,mNPY)),
  fP0       (fCPIX-fPLarg),
  fP1       (fCPIX+fPLarg),
  mIndBlanc (mAlloc.CurInc()),
  mBlanc    (aBlanc),
  fBlanc    (mAlloc.NewF("Ellipse","Blanc",&mBlanc)),
  mIndNoir  (mAlloc.CurInc()),
  mNoir     (aNoir),
  fNoir     (mAlloc.NewF("Ellipse","Noir",&mNoir)),
  mNGR      ("Gray"),
  fGray     (cVarSpec(0,mNGR)),
  mVarsRays    (VarsRays()),
  fNumEqResidu (EqResidu())
{
    CloseEEF();
    IncInterv().SetName("ElIma");
    mLInterv.AddInterv(IncInterv());

    mFctrEqResidu = cElCompiledFonc::AllocFromName(mNameType);
    if ((mFctrEqResidu == 0) || (Code2Gen))
    {
	    GenCode();
	    if (Code2Gen)
               return;
	     mFctrEqResidu = cElCompiledFonc::DynamicAlloc(mLInterv,fNumEqResidu);
	    // ELISE_ASSERT(false,"No AllocFromName cEqCibleEllipse");
	     for (INT aK= 0 ; aK< 20 ; aK++)
                 cout << "FONCTEUR DYN FOR " << mNameType << "\n";
    }
    mFctrEqResidu->SetMappingCur(mLInterv,&anEq);
    pAdrX = mFctrEqResidu->RequireAdrVarLocFromString(mNPX);
    pAdrY = mFctrEqResidu->RequireAdrVarLocFromString(mNPY);
    pGray = mFctrEqResidu->RequireAdrVarLocFromString(mNGR);

    for (INT aK=0 ; aK< mMire.NbDiam() ; aK++)
    {
	 mVarsRays[aK].mAdrVar = mFctrEqResidu->RequireAdrVarLocFromString
		                 (
			               mVarsRays[aK].mFRay.Name()
				  );
	 *(mVarsRays[aK].mAdrVar) = mMire.KthDiam(aK);
    }
    mSet.AddFonct(mFctrEqResidu);
}

REAL cEqEllipseImage::AddEq(REAL aX,REAL aY,REAL aGray,REAL aPds,bool WithD2)
{
     *pAdrX = aX;
     *pAdrY = aY;
     *pGray = aGray;
     return mSet.AddEqFonctToSys(mFctrEqResidu,aPds,WithD2);
}

REAL cEqEllipseImage::GraySynt(REAL aX,REAL aY)
{
     *pAdrX = aX;
     *pAdrY = aY;
     *pGray = 0;
     return mSet.ResiduSigne(mFctrEqResidu);
}


cMultiContEQF cEqEllipseImage::ContrFigeABC()
{
   cMultiContEQF aRes;
   AddFoncRappInit(aRes,mIndA-mI0,mIndA-mI0+3,cContrainteEQF::theContrStricte);
   return aRes;
}

cMultiContEQF cEqEllipseImage::ContrFigeLarg()
{
   cMultiContEQF aRes;
   AddFoncRappInit(aRes,mIndLarg-mI0,mIndLarg-mI0+1,cContrainteEQF::theContrStricte);
   return aRes;
}


const cMirePolygonEtal & cEqEllipseImage::Mire() const
{
      return mMire;
}


bool cEqEllipseImage::OkCurEllispe() const
{
    REAL Bid1,Bid2,Bid3;
    return EllipseEq2ParamPhys(Bid1,Bid2,Bid3,mA,mB,mC);
}


Pt2dr cEqEllipseImage::CurCentre() const {return mCentre;}
REAL  cEqEllipseImage::CurA() const      {return mA;}
REAL  cEqEllipseImage::CurB() const      {return mB;}
REAL  cEqEllipseImage::CurC() const      {return mC;}
REAL  cEqEllipseImage::CurLarg() const   {return mLarg;}
REAL  cEqEllipseImage::CurBlanc() const  {return mBlanc;}
REAL  cEqEllipseImage::CurNoir() const   {return mNoir;}

void cEqEllipseImage::SetCentre(Pt2dr aPC) 
{
     SetValAndInit(aPC.x,mIndCentre);
     SetValAndInit(aPC.y,mIndCentre+1);
}
void cEqEllipseImage::SetA(REAL anA)       { SetValAndInit(anA,mIndA);}
void cEqEllipseImage::SetB(REAL aB)        { SetValAndInit( aB,mIndB);}
void cEqEllipseImage::SetC(REAL aC)        { SetValAndInit( aC,mIndC);}
void cEqEllipseImage::SetLarg(REAL aL)     { SetValAndInit( aL,mIndLarg);}
void cEqEllipseImage::SetBlanc(REAL aB)    { SetValAndInit( aB,mIndBlanc);}
void cEqEllipseImage::SetNoir(REAL aN)     { SetValAndInit( aN,mIndNoir);}

Box2dr  cEqEllipseImage::BoxCurEllipse(REAL aFact) const
{
    return BoxEllipse(mCentre,mA/aFact,mB/aFact,mC/aFact);
}

REAL cEqEllipseImage::SurfIER(Pt2dr aP,REAL aFact,REAL aLarg) const
{
	if (aLarg<=0)
           aLarg = mLarg;
	Pt2dr PL(aLarg,aLarg);
	return  ::SurfIER
		(
		       mCentre,
		       mA/aFact,mB/aFact,mC/aFact,
		       aP-PL,aP+PL
		);

}






/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
