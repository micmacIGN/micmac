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

#include "general/all.h"
#include "private/all.h"


void Banniere_Piphom()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     Programme d'              *\n";
   std::cout <<  " *     Inspection des            *\n";
   std::cout <<  " *     Point                     *\n";
   std::cout <<  " *     HOMologues                *\n";
   std::cout <<  " *********************************\n";

}

typedef enum 
{
     eModeP1,
     eModePN,
     eModePax
} eModeSubW;

class cSubWPif
{
    public :
        cSubWPif
	(
	    Video_Win           aWPos,
	    bool                enDessous,
	    const std::string&  aFile,
	    Pt2di               aSz,
	    double              aDZ,
	    eModeSubW           aMode,
	    ElPackHomologue  *  aPack
           
       );
        Video_Win W() {return  mW;}
	void SetDec(Pt2dr );
    private :
        Pt2dr mDec;
        Video_Win   mW;
	Tiff_Im     mTif;
	double      mZoom;
	eModeSubW   mMode;
	ElPackHomologue * mPack;
        
};


void cSubWPif::SetDec(Pt2dr  aP0)
{
    const ElCplePtsHomologues  * aCpl = mPack->Cple_Nearest(aP0,true);
    aP0 =  (mMode == eModePN) ? aCpl->P2() : aCpl->P1();

    mDec = aP0  - (mW.sz() / (2*mZoom));
    mW = mW.chc(mDec,Pt2dr(mZoom,mZoom));

    double aVMin,aVMax;

    if (false) // (mMode== eModePax)
    {
        ELISE_COPY (mW.all_pts(), mTif.in(0), mW.ocirc());
    }
    else
    {
        ELISE_COPY(mW.all_pts(),Rconv(mTif.in_proj()), VMin(aVMin)|VMax(aVMax));

        if ( aVMin == aVMax) aVMax++;

        ELISE_COPY
        (
            mW.all_pts(),
	    Max(0,Min(255,(mTif.in(0)-aVMin) * (255.0/(aVMax-aVMin)))),
	    mW.ogray()
       );
   }


    
    if ((mMode == eModeP1) || (mMode == eModePN))
    {
        int aCpt=0;
        for 
        (
           ElPackHomologue::const_iterator itP=mPack->begin();
           itP != mPack->end();
           itP++
        )
        {
	   aCpt++;
	   Pt2dr aP = (mMode == eModeP1) ? itP->P1() : itP->P2() ;
	   if (euclid(aP,aP0) < 300)
              mW.draw_circle_abs(aP,2.5,mW.pdisc()(aCpt%8));
        }
    }

    mW.draw_circle_abs(aP0,4.0,mW.pdisc()(P8COL::red));
}


cSubWPif::cSubWPif
(
      Video_Win           aWPos,
      bool                enDessous,
      const std::string&  aFile,
      Pt2di               aSz,
      double              aZoom,
      eModeSubW           aMode,
      ElPackHomologue *   aPack
)  :
    mW   (aWPos,enDessous ?Video_Win::eBasG : Video_Win::eDroiteH,aSz),
    mTif (Tiff_Im::StdConv(aFile)),
    mZoom (aZoom),
    mMode (aMode),
    mPack (aPack)
{
   ELISE_COPY(mW.all_pts(),((FX+2*FY)*10)%256,mW.ogray());
}


class cAppliPiphom
{
    public :
         cAppliPiphom(int argc,char ** argv);

	 void NoOp(){}
	 void ExeClik(Clik aCl);
    private :
      void AddSubw
           (
                  const std::string&  aFile,
                  eModeSubW,
                  ElPackHomologue *   aPack
          );

        void ShowEmprH();
      
        std::vector<cSubWPif*> mSubW;
        Pt2di                  mSzSubW;
	double                 mZoom;

        Pt2di                  mSzWP;
        std::string            mDir;
        std::string            mNH;
        std::string            mN1;
        std::string            mN2;
        std::string            mNPx;
	std::vector<string>    mNameSup;

	Tiff_Im *           mT1;
	Video_Win*          mW;
        EliseStdImageInteractor * mESII;
	ElPackHomologue           mPack0;
};

void cAppliPiphom::AddSubw
     (
                  const std::string&  aFile,
                  eModeSubW           aMode,
                  ElPackHomologue *   aPack
     )
{
 
    Video_Win aWPos = mSubW.empty() ? *mW : mSubW.back()->W();
    bool enDessous  = (mSubW.size()==0) || (mSubW.size()==4);

    double aZoom = (aMode==eModePax) ? mZoom/4.0 : mZoom ;

    cSubWPif * aSub = new cSubWPif
                      (
		          aWPos, enDessous, aFile,
                          mSzSubW, aZoom, aMode, aPack
                      );
    mSubW.push_back(aSub);
}

void cAppliPiphom::ShowEmprH()
{
    for 
    (
       ElPackHomologue::const_iterator itP=mPack0.begin();
       itP != mPack0.end();
       itP++
    )
    {
       mW->draw_circle_abs(mESII->U2W(itP->P1()),1.0,mW->pdisc()(P8COL::green));
    }
}




void cAppliPiphom::ExeClik(Clik aCl)
{
    ShowEmprH();
    std::cout << mESII->W2U(aCl._pt)  << aCl._b << "\n";
    if (aCl._w == *mW)
    {
       const ElCplePtsHomologues  * aCpl = mPack0.Cple_Nearest(mESII->W2U(aCl._pt),true);
       if (aCpl==0)
       {
           ELISE_ASSERT(false,"Pack Vide \n");
       }

       for (int aK=0 ; aK< int(mSubW.size()) ; aK++)
           mSubW[aK]->SetDec(aCpl->P1());
    }
}

cAppliPiphom::cAppliPiphom(int argc,char ** argv)
{
    mSzWP   = Pt2di(700,500);
    mSzSubW = Pt2di(250,250);
    mZoom = 2.0;
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAM(mDir) 
                      << EAM(mNH) 
                      << EAM(mN1) 
                      << EAM(mN2) ,
           LArgMain() << EAM(mNameSup,"Ims",true)
	              << EAM(mSzWP,"SzW",true)
	              << EAM(mNPx,"Px",true)
	              << EAM(mZoom,"Zoom",true)
    );

    std::string aNTif1 = mDir+mN1;

    mT1  = new Tiff_Im(Tiff_Im::StdConv(aNTif1));
    mW =  Video_Win::PtrWStd(mSzWP);

    VideoWin_Visu_ElImScr * aVVE = new VideoWin_Visu_ElImScr(*mW,*mT1);
    ElPyramScroller * aPyr = ElImScroller::StdPyramide(*aVVE,aNTif1);
    mESII = new EliseStdImageInteractor(*mW,*aPyr,2,5,4);

    mPack0 =  ElPackHomologue::FromFile(mDir+mNH);

    AddSubw(aNTif1,eModeP1,&mPack0);
    AddSubw(mDir+mN2,eModePN,&mPack0);
    if (mNPx != "")
        AddSubw(mDir+mNPx,eModePax,&mPack0);

    while (1)
    {
         ExeClik(mESII->clik_press())  ;
    }
}


int main(int argc,char ** argv)
{
    cAppliPiphom aAP(argc,argv);
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
