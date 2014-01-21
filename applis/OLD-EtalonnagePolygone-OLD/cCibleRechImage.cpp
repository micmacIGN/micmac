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

/************************************************************/
/*                                                          */
/*                      cPointeCible                        */
/*                                                          */
/************************************************************/
#include "all_etal.h"

Video_Win * cCibleRechImage::AllocW()
{
   if (mZoom <=0)
      return 0;

   Video_Win * aRes = Video_Win::PtrWStd(mSzIm*mZoom);
   aRes = aRes->PtrChc(Pt2dr(0,0),Pt2dr(mZoom,mZoom),true);

   return aRes;
}


cCibleRechImage::cCibleRechImage
(
     cEtalonnage & anEtal,
     INT   aSz,
     INT   aZoom
)   :
    mEtal     (anEtal),
    pCam      (0),
    mSzIm     (aSz,aSz),
    mZoom     (aZoom),
    mIm       (mSzIm.x,mSzIm.y),
    mTIm      (mIm),
    mImSynth  (mSzIm.x,mSzIm.y),
    mTImSynth (mImSynth),
    mImInside (mSzIm.x,mSzIm.y),
    mCentreImSynt (Pt2dr(mSzIm)/2.0),
    mImPds    (mSzIm.x,mSzIm.y),
    pWIm      (AllocW()),
    pWSynth   (AllocW()),
    pWFFT     (AllocW()),
    pWGlob    (0),

    mSetM7    (),
    mSetM6    (),
    mSetM5    (),
    mSetS3    (),
    mSetS2    (),
    mSetSR5   (),
    mSetMT0   (),
    mSetMN6   (),

    pSet      (0),
    mDefLarg  (anEtal.Param().DefLarg()),

    pEqElImM7 (mSetM7.NewEqElIm(cMirePolygonEtal::IGNMire7())),
    pEqElImM6 (mSetM6.NewEqElIm(cMirePolygonEtal::MtdMire9())),
    pEqElImM5 (mSetM5.NewEqElIm(cMirePolygonEtal::IGNMire5())),

    pEqElImS3 (mSetS3.NewEqElIm(cMirePolygonEtal::SofianeMire3())),
    pEqElImS2 (mSetS2.NewEqElIm(cMirePolygonEtal::SofianeMire2())),
    pEqElImSR5 (mSetSR5.NewEqElIm(cMirePolygonEtal::SofianeMireR5())),
    pEqElImMT0 (mSetMT0.NewEqElIm(cMirePolygonEtal::MT0())),
    pEqElImN6  (mSetMN6.NewEqElIm(cMirePolygonEtal::IgnMireN6())),

    pEqElIm   (0)
{
    if (pWFFT)
    {
        Pt2dr aSz = Pt2dr(mSzIm) * double(mZoom);
        REAL aZG = aSz.RatioMin(anEtal.SzIm());
	pWGlob = pWFFT->PtrChc(Pt2dr(0,0),Pt2dr(aZG,aZG));
	pWFFT =  pWFFT->PtrChc(Pt2dr(0,0),Pt2dr(aZoom,aZoom)/2.0);
    }
   mSetM7.SetClosed();
   mSetM6.SetClosed();
   mSetM5.SetClosed();
   mSetS2.SetClosed();
   mSetS3.SetClosed();
   mSetSR5.SetClosed();
   mSetMT0.SetClosed();
   mSetMN6.SetClosed();
}

void cCibleRechImage::ShowCible(Video_Win *pW,INT aCoul)
{
     if (! pW)
        return;
     const cMirePolygonEtal &  aMire = pEqElIm->Mire();
     for (INT aK=0; aK<aMire.NbDiam() ; aK++)
     {
         REAL F = aMire.KthDiam(aK) / aMire.KthDiam(0);
	 if (aK==0)
         {
             Box2dr aBox = pEqElIm->BoxCurEllipse();
             pW->draw_rect(aBox._p0,aBox._p1, pW->pdisc()(P8COL::yellow));
         }
	 pW->draw_ellipse_loc
         (
               pEqElIm->CurCentre(),
	       pEqElIm->CurA() / F,
	       pEqElIm->CurB() / F,
	       pEqElIm->CurC() / F,
	       pW->pdisc()(aCoul)
         );
     }
}

const cParamEtal & cCibleRechImage::ParamEtal() const
{
	return mEtal.Param();
}


REAL cCibleRechImage::OneItereRaffinement
     (
         REAL FPts,
         bool LargLibre,
         bool ABCLibre
     )
{
     // pSet->GSSR_Reset(true);

     if (! LargLibre) 
        pSet->AddContrainte(pEqElIm->ContrFigeLarg(),true);
     if (! ABCLibre)
        pSet->AddContrainte(pEqElIm->ContrFigeABC(),true);


      pSet->SetPhaseEquation();

      Box2dr aBox =  pEqElIm->BoxCurEllipse(FPts);
      REAL  Rab  =  pEqElIm->CurLarg() ;
      INT X0 = ElMax(0       , round_down(aBox._p0.x-Rab));
      INT Y0 = ElMax(0       , round_down(aBox._p0.y-Rab));
      INT X1 = ElMin(mSzIm.x , round_up(aBox._p1.x+Rab+1));
      INT Y1 = ElMin(mSzIm.y , round_up(aBox._p1.y+Rab+1));

      RMat_Inertie aMat;
      for (INT aX = X0 ; aX<X1 ; aX++)
      {
          for (INT aY = Y0 ; aY<Y1 ; aY++)
          {
              Pt2di aP(aX,aY);
              if (mTIm.inside(aP)) 
              {
                  REAL S = pEqElIm->SurfIER(Pt2dr(aP),FPts,0.5);
		  if (S > 0.1)
		  {

		       REAL aGr1 = mIm.data()[aY][aX];
		       REAL aGr2 = pEqElIm->GraySynt(aX,aY);
		       aMat.add_pt_en_place(aGr1,aGr2,S);
		       pEqElIm->AddEq(aP.x,aP.y,aGr1,S);
		  }
	      }
	  }
      }

     bool OK;

     pSet->SolveResetUpdate(-1,&OK);
     if (!OK) return -2;
     ELISE_ASSERT(OK,"Solve pb detected in cCibleRechImage::OneItereRaffinement");


     if (aMat.s() ==0)
         return -2;
     REAL aRes =  aMat.correlation();
     return aRes;
}

void cCibleRechImage::RechercheImage
     (
         cHypDetectCible  & anHyp
     )
{
    const cCibleCalib & aCC = *(anHyp.Cible().CC());
    REAL aDConfCentre = anHyp.Set().DistConfusionCentre();
    REAL aDConfShape  = anHyp.Set().DistConfusionShape();

    cout << "NUM C " << anHyp.Cible().Ind() << "\n";
    // CHARGEMENT DE L'IMAGE
    pCam = & anHyp.Cam();
    mDecIm = round_ni(anHyp.Centr0() - Pt2dr(mSzIm)/2.0);

    ELISE_COPY
    (
          mIm.all_pts(),
	  trans(pCam->Tiff().in(-1),mDecIm),
	  mIm.out()
    );
    ELISE_COPY
    (
          select(mIm.all_pts(),mIm.in()<0),
	  255* 15 * frandr(),
	  mIm.out()
    );


    // VISU EVENTUELLE
    if (pWIm)
    {
         pWGlob->draw_circle_abs(anHyp.Centr0(),2.0,pWGlob->pdisc()(P8COL::red));
         INT Rank = 11;
         Pt2di PR(Rank,Rank);

	 Symb_FNum F (
	               rect_rank(mIm.in_proj(),Box2di(-PR,PR),256*16) 
	             * (255.0 / (ElSquare(1+2*Rank)))
	             );
         ELISE_COPY
         (
             mIm.all_pts(),
	     Virgule(F,F,F),
	     pWIm->orgb()
         );
    }

    REAL FPtsCorrel = 1.0;
    REAL FPtsRaff   = 1.0;

    // SELECTION DU BON ENSEMBLE D'EQUATIONS (M5 ou M7)
    const cMirePolygonEtal & aMire = anHyp.Cible().Mire();



    if (&aMire  == & cMirePolygonEtal::MtdMire9())
    {
        pSet = & mSetM6;
        pEqElIm = pEqElImM6;
	FPtsCorrel = 1.0;
	FPtsRaff   = 0.8333;
    }
    else if (&aMire  == & cMirePolygonEtal::IGNMire7())
    {
        pSet = & mSetM7;
        pEqElIm = pEqElImM7;
	FPtsCorrel = 1.0;
	FPtsRaff   = 0.8333;
    }
    else if (& aMire == & cMirePolygonEtal::IGNMire5())
    {
        pSet = & mSetM5;
        pEqElIm = pEqElImM5;
	FPtsCorrel = 1.4;
	FPtsRaff   = 1.4;
    }
    else if (& aMire == & cMirePolygonEtal::SofianeMire2())
    {
        pSet = & mSetS2;
        pEqElIm = pEqElImS2;
	FPtsCorrel = 1.5;
	FPtsRaff   = 1.5;
    }
    else if (& aMire == & cMirePolygonEtal::SofianeMire3())
    {
        pSet = & mSetS3;
        pEqElIm = pEqElImS3;
	FPtsCorrel = 2.0;
	FPtsRaff   = 1.5;
    }
    else if (& aMire == & cMirePolygonEtal::SofianeMireR5())
    {
        pSet = & mSetSR5;
        pEqElIm = pEqElImSR5;
	FPtsCorrel = 1.5;
	FPtsRaff   = 1.5;
    }
    else if (& aMire == & cMirePolygonEtal::MT0())
    {
        pSet = & mSetMT0;
        pEqElIm = pEqElImMT0;
	FPtsCorrel = 1.5;
	FPtsRaff   = 1.5;
    }
    else if (& aMire == & cMirePolygonEtal::IgnMireN6())
    {
        pSet = & mSetMN6;
        pEqElIm = pEqElImN6;
	FPtsCorrel = 1.2;
	FPtsRaff   = 1.1;

std::cout << "BL " <<  pEqElIm->CurBlanc()  << " N " << pEqElIm->CurNoir() << "\n";
    }
    else
    {
       ELISE_ASSERT(false,"Unknown mire in cCibleRechImage::RechercheImage");
    }


    FPtsCorrel = aCC.FacteurElargRechCorrel().ValWithDef(FPtsCorrel);
    FPtsRaff = aCC.FacteurElargRechRaffine().ValWithDef(FPtsRaff);

    // CALCUL d'UNE IMAGE d'ELLIPSE DE SYNTHESE

    pEqElIm->SetCentre(mCentreImSynt);
    pEqElIm->SetA(anHyp.A0());
    pEqElIm->SetB(anHyp.B0());
    pEqElIm->SetC(anHyp.C0());
    pEqElIm->SetLarg(mDefLarg);
// std::cout << "---- NEG ----   " << aMire.IsNegatif() << "\n";
    if (aMire.IsNegatif())
    {
       pEqElIm->SetBlanc(0);
       pEqElIm->SetNoir(255);
    }
    else
    {
       pEqElIm->SetBlanc(255);
       pEqElIm->SetNoir(0);
    }

    REAL aStot = 0;
    mPtsCible.clear();
    mImPds.raz();
    ELISE_COPY(mImSynth.all_pts(),255,mImSynth.out());

    Box2dr aBox = pEqElIm->BoxCurEllipse(FPtsCorrel);
    INT  Rab  =  1 ;
    INT X0 = ElMax(0       , round_down(aBox._p0.x-Rab));
    INT Y0 = ElMax(0       , round_down(aBox._p0.y-Rab));
    INT X1 = ElMin(mSzIm.x , round_up(aBox._p1.x+Rab+1));
    INT Y1 = ElMin(mSzIm.y , round_up(aBox._p1.y+Rab+1));

    cout << "FPtsCorrel = " << FPtsCorrel << "\n";
    cout << "FPtsRaff   = " << FPtsRaff   << "\n";


/*
bool  Test=true;
double aLarg=1;
if (Test)
{
    aLarg = 2.5;
    FPtsCorrel = 1.5;
    FPtsRaff = 1.5;
    pEqElIm->SetLarg(aLarg);
    A FAIR RAHOUTER LARGINT+FPTCORRL + FPTRS Raffine en param
}
*/

    for (INT aX = X0 ; aX<X1 ; aX++)
    {
        for (INT aY = Y0 ; aY<Y1 ; aY++)
        {
             Pt2dr aP(aX,aY);
             REAL S = pEqElIm->SurfIER(aP,FPtsCorrel,0.5) ;
             aStot += S;
             mImPds.data()[aY][aX] = S > 0.5 ;
             if (S > 0.5)
                mPtsCible.push_back(Pt2di(aX,aY));
              mImSynth.data()[aY][aX] = round_ni(pEqElIm->GraySynt(aX,aY));
        }
    }




    // Visu de l'image de synthese
    if (pWIm)
    {
           ELISE_COPY
           (
	        mImSynth.all_pts(),
		Virgule(mImSynth.in(),255*mImPds.in(),mImSynth.in()),
		pWSynth->orgb()
           );
std::cout << "cCibleRechImage.cpp :" << pWIm << "\n"; getchar();
    }


    // Calcul du centre initial par Max de correlation
    
        // Calcul de l'image de Correlation
    Im2D_REAL8   aCor = ElFFTPonderedCorrelNCPadded
                        (
                            mIm.in(),
                            mImSynth.in(),
                            mSzIm,
                            1,
                            mImPds.in(),
                            1e-5,
                            aStot * 0.9
                        );

    Pt2d<Fonc_Num> aPF =  FN_DecFFT2DecIm(aCor);
        // Visu  de l'image de Correlation
    if (pWIm)
    {
        ELISE_COPY
        (
           aCor.all_pts(),
           Max(0,Min(255,128.0*(aCor.in()+1))),
           pWFFT->ogray()
        );


	ELISE_COPY
	(
	     select(aCor.all_pts(), sqrt(Square(aPF.x)+Square(aPF.y)) > aDConfCentre),
	     P8COL::red,
             pWFFT->odisc()
	);
    }
   
        // Calcul du Max de correlation
    Pt2di aDecMax;
    ELISE_COPY
    (
         select
	 (
	     aCor.all_pts(),
             sqrt(Square(aPF.x)+Square(aPF.y)) < aDConfCentre
	 ),
	 aCor.in(),
	 aDecMax.WhichMax()
    );
    aDecMax = DecFFT2DecIm(aCor,aDecMax);

    Pt2dr aPMax = mCentreImSynt + Pt2dr(aDecMax);
    if  (anHyp.PosFromPointe())
    {

    // std::cout << "------########################\n";
    // std::cout << "------########################\n";
    // std::cout << "------########################\n";
    // std::cout << aDecMax << (anHyp.Centr0()-(mDecIm+ mCentreImSynt)) <<"\n";
    // std::cout <<  aPMax << (anHyp.Centr0()-mDecIm) << "\n";
       aPMax = anHyp.Centr0()- Pt2dr(mDecIm);
    }

    pEqElIm->SetCentre(aPMax);

    if (pWIm)
    {
	pWIm->draw_circle_abs(aPMax,2.0,pWIm->pdisc()(P8COL::blue));
    }


    // Calcul du blanc et noir initiaux
    REAL Noir  =   255;
    REAL Blanc =   0;
    {
	RMat_Inertie aMat;

	for 
	(
              std::vector<Pt2di>::iterator  itP = mPtsCible.begin();
	      itP != mPtsCible.end();
	      itP++
	)
	{
           Pt2di aPS =*itP;
           Pt2di aPI = aPS+ aDecMax;
           if (( mTIm.inside(aPI)) &&  (mTImSynth.inside(aPS)))
           {
                aMat.add_pt_en_place
               (
		   REAL (mIm.data()[aPI.y][aPI.x]),
                   REAL (mImSynth.data()[aPS.y][aPS.x])
               );
           }
	}

	if (aMat.s() ==0)
           return;
	aMat = aMat.normalize();
	REAL aRatio = sqrt(aMat.s11()/aMat.s22());
	Noir  =   aMat.s1() + (0 - aMat.s2()) * aRatio;
	Blanc =   aMat.s1() + (255 - aMat.s2()) * aRatio;
        pEqElIm->SetBlanc(Blanc);
        pEqElIm->SetNoir(Noir);


       if (pWIm)
       {
           Fonc_Num f = (mIm.in()-Noir) / (Blanc-Noir);
           ELISE_COPY
           (
	         mIm.all_pts(),
	         Max(0,Min(255, 32+196*f)),
	         pWIm->ogray() | pWSynth->ogray()
           );
       }
    }
    ShowCible(pWSynth,P8COL::magenta);

    if (ParamEtal().MakeImagesCibles())
    {
        std::string aDir = ParamEtal().Directory() + "TMP/C"
		           + ToString(anHyp.Cible().Ind());
	
	REAL aFact = 200 / ElMax(1.0,Blanc);
	if (aFact > 0.2 ) 
            aFact = 0.2;
	Tiff_Im::Create8BFromFonc
        (
            aDir+"Image.tif",
	    mIm.sz(), 
            Max(0,Min(255,mIm.in()*aFact))
        );
	Tiff_Im::Create8BFromFonc
        (
            aDir+"ImBin.tif",
	    mIm.sz(), 
	    255 * (mIm.in() > ((Blanc+Noir)/2.0))
        );

	Pt2di aP0,aP1;
	ELISE_COPY
        (
	    select(mImPds.all_pts(), mImPds.in()>0.5),
	    Virgule(FX,FY),
	        Virgule(VMin(aP0.x),VMin(aP0.y))
	    |   Virgule(VMax(aP1.x),VMax(aP1.y))

        );

	ELISE_COPY
        (
	    mImPds.all_pts(),
	    mImPds.in()>0.5,
            pWIm->odisc()
        );
	pWIm->draw_rect(Pt2dr(aP0),Pt2dr(aP1),pWIm->pdisc()(P8COL::red));

	Tiff_Im::Create8BFromFonc
        (
            aDir+"Cible.tif",
	    (aP1-aP0+Pt2di(3,3)),
            trans((64+(mImSynth.in(0)*0.75)) * mImPds.in(0),aP0-Pt2di(1,1))
        );


    }

    REAL Correl = -1;
    REAL DShape = 0;

    for (INT aK=0 ; (aK<5) && (DShape<aDConfShape) && (Correl>=-1) ; aK++)
    {
        Correl = OneItereRaffinement(FPtsRaff,true,true);
        DShape = SimilariteEllipse
                  (
                      pEqElIm->CurA(),pEqElIm->CurB(),pEqElIm->CurC(),
                      anHyp.A0(),anHyp.B0(),anHyp.C0()
                  );

    }

    ShowCible(pWIm,P8COL::red);

    Pt2dr aC = pEqElIm->CurCentre() + Pt2dr(mDecIm);
    REAL DCentre = euclid(aC,anHyp.Centr0());


    bool Ok =    (DCentre < aDConfCentre)
	      && (DShape  < aDConfShape)
	      && (Correl > ParamEtal().SeuilCorrel());

    cout << "DEC IM = " << mDecIm <<  " PRES = " << aC << "\n";
    cout  << "[Ok = " << Ok << "]"
	 << " LARG = "   <<  pEqElIm->CurLarg()
	 << " Correl = " << Correl
	 << " Dist C = " << DCentre
	 << " Dist Forme " << DShape
         << "\n";
    anHyp.SetResult(aC,pEqElIm->CurLarg(),Ok,Correl,DCentre,DShape);
    if (pWIm)
       getchar();
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
