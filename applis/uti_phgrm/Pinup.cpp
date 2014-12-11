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
#include "XML_GEN/all.h"



void Banniere_Pinup()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     P-rogramme d'             *\n";
   std::cout <<  " *     I-nspection des           *\n";
   std::cout <<  " *     NU-ages de                *\n";
   std::cout <<  " *     P-oints                   *\n";
   std::cout <<  " *********************************\n";

}

typedef enum 
{
     eModeP1,
     eModePN,
     eModePax
} eModeSubW;


/****************************************************/
/*                                                  */
/*          HEADER                                  */
/*                                                  */
/****************************************************/

class cSubWPin;
class cAppliPinup;

class cSubWPin
{
    public :
        cSubWPin
	(
	    cAppliPinup &       anAppli,
	    Video_Win           aWPos,
	    bool                enDessous,
	    const std::string&  aFile,
	    Pt2di               aSz,
	    double              aDZ
           
       );
        Video_Win W() {return  mW;}
	void SetDec(Pt3dr );
    private :
        cAppliPinup & mAppli;
        Pt2dr         mDec;
        Video_Win     mW;
	Tiff_Im       mTif;
	std::string   mNameOri;
	Ori3D_Std     mOri;
	double        mZoom;
};

class cAppliPinup
{
    public :
         cAppliPinup(int argc,char ** argv);

	 void NoOp(){}
	 void ExeClik(Clik aCl);

	 const std::string & KeyOri() {return mKeyOri;}
	 cInterfChantierNameManipulateur * ICNM() {return mICNM;}
	 const std::string & Dir() {return mDir;}
    private :
      void AddSubw(const std::string&  aFile);

        void ShowEmprH();
      
        std::vector<cSubWPin*> mSubW;
        Pt2di                  mSzSubW;
	double                 mZoom;

        Pt2di                  mSzWP;
        std::string            mDir;
        std::string            mNNuage;
        std::string            mN1;
        std::string            mN2;
        std::string            mNPx;
	std::vector<string>    mNameSup;
	std::string            mKeyOri;
	cInterfChantierNameManipulateur * mICNM;

	Tiff_Im *           mT1;
	Video_Win*          mW;
        EliseStdImageInteractor * mESII;

        Pt2di        mSzIm;
	Im2D_REAL4    mImP;
	Im2D_REAL4    mImX;
	Im2D_REAL4    mImY;
	Im2D_REAL4    mImZ;
};


/****************************************************/
/*                                                  */
/*          cSubWPin                                */
/*                                                  */
/****************************************************/

void cSubWPin::SetDec(Pt3dr  aPTer)
{

   Pt2dr aP0 = mOri.to_photo(aPTer);
   std::cout << "P0 = " << aP0 << "\n";

    mDec = aP0  - (mW.sz() / (2*mZoom));
    mW = mW.chc(mDec,Pt2dr(mZoom,mZoom));

    double aVMin,aVMax;

    ELISE_COPY(mW.all_pts(),Rconv(mTif.in_proj()), VMin(aVMin)|VMax(aVMax));

    if ( aVMin == aVMax) aVMax++;

    ELISE_COPY
    (
            mW.all_pts(),
	    Max(0,Min(255,(mTif.in(0)-aVMin) * (255.0/(aVMax-aVMin)))),
	    mW.ogray()
    );

    
    mW.draw_circle_abs(aP0,4.0,mW.pdisc()(P8COL::red));
}


cSubWPin::cSubWPin
(
      cAppliPinup &       anAppli,
      Video_Win           aWPos,
      bool                enDessous,
      const std::string&  aFile,
      Pt2di               aSz,
      double              aZoom
)  :
    mAppli      (anAppli),
    mW          (aWPos,enDessous ?Video_Win::eBasG : Video_Win::eDroiteH,aSz),
    mTif        (Tiff_Im::StdConv(anAppli.Dir()+aFile)),
    mNameOri    (mAppli.Dir()+mAppli.ICNM()->Assoc1To1(mAppli.KeyOri(),aFile,true)),
    mOri        (mNameOri.c_str()),
    mZoom       (aZoom)
{
   std::cout << aFile << " " << mNameOri << "\n";
   ELISE_COPY(mW.all_pts(),((FX+2*FY)*10)%256,mW.ogray());
}

/****************************************************/
/*                                                  */
/*          cAppliPinup                             */
/*                                                  */
/****************************************************/

void cAppliPinup::AddSubw(const std::string&  aFile)
{
 
    Video_Win aWPos = mSubW.empty() ? *mW : mSubW.back()->W();
    bool enDessous  = (mSubW.size()==0) || (mSubW.size()==4);

    double aZoom =  mZoom ;

    cSubWPin * aSub = new cSubWPin
                      (
		          *this,aWPos, enDessous, aFile,
                          mSzSubW, aZoom
                      );
    mSubW.push_back(aSub);
}

void cAppliPinup::ShowEmprH()
{
    std::cout << "DO NOTHING \n";
}




void cAppliPinup::ExeClik(Clik aCl)
{
    ShowEmprH();
    std::cout << mESII->W2U(aCl._pt)  << aCl._b << "\n";
    if (aCl._w == *mW)
    {

       Pt2di aP0 = mESII->W2U(aCl._pt);

       if (
                 (aP0.x>=0)
              && (aP0.y>=0)
              && (aP0.x<mSzIm.x)
              && (aP0.y<mSzIm.y)
              && (mImP.data()[aP0.y][aP0.x] > 0)
          )
       {
           Pt3dr aPTer
	         (
		      mImX.data()[aP0.y][aP0.x],
		      mImY.data()[aP0.y][aP0.x],
		      mImZ.data()[aP0.y][aP0.x]
		 );

           std::cout << aPTer << "\n";

           for (int aK=0 ; aK< int(mSubW.size()) ; aK++)
               mSubW[aK]->SetDec(aPTer);
       }
    }
}


cAppliPinup::cAppliPinup(int argc,char ** argv):
   mImP  (1,1),
   mImX  (1,1),
   mImY  (1,1),
   mImZ  (1,1)
{
    mSzWP   = Pt2di(700,500);
    mSzSubW = Pt2di(250,250);
    mZoom = 2.0;
    mKeyOri = "Orient-From-Porto";
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAM(mDir) 
                      << EAM(mNNuage) 
                      << EAM(mN1) 
                      << EAM(mN2) ,
           LArgMain() << EAM(mNameSup,"Ims",true)
	              << EAM(mSzWP,"SzW",true)
	              << EAM(mNPx,"Px",true)
	              << EAM(mKeyOri,"Kor",true)
    );

    Tiff_Im aFileN = Tiff_Im::StdConv(mDir+mNNuage);

    mSzIm  = aFileN.sz();
    std::cout << "SzIm = " << mSzIm << "\n";
    mImP = Im2D_REAL4(mSzIm.x,mSzIm.y);
    mImX = Im2D_REAL4(mSzIm.x,mSzIm.y);
    mImY = Im2D_REAL4(mSzIm.x,mSzIm.y);
    mImZ = Im2D_REAL4(mSzIm.x,mSzIm.y);

    ELISE_COPY
    (
       aFileN.all_pts(),
       aFileN.in(),
       Virgule
       (
           mImP.out(),
           mImX.out(),
           mImY.out(),
           mImZ.out()
       )
    );

    std::string aNTif1 = mDir+mN1;

    mT1  = new Tiff_Im(Tiff_Im::StdConv(aNTif1));
    mW =  Video_Win::PtrWStd(mSzWP);

    VideoWin_Visu_ElImScr * aVVE = new VideoWin_Visu_ElImScr(*mW,*mT1);
    ElPyramScroller * aPyr = ElImScroller::StdPyramide(*aVVE,aNTif1);
    mESII = new EliseStdImageInteractor(*mW,*aPyr,2,5,4);

    cTplValGesInit<std::string> aNoStr;
    mICNM =  cInterfChantierNameManipulateur::StdAlloc(mDir,aNoStr);


    AddSubw(mN1);
    AddSubw(mN2);

    while (1)
    {
         ExeClik(mESII->clik_press())  ;
    }
}



/****************************************************/
/*                                                  */
/*            main                                  */
/*                                                  */
/****************************************************/

int main(int argc,char ** argv)
{
    cAppliPinup aAP(argc,argv);
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
