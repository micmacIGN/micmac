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

#ifndef _VINO_H_
#define _VINO_H_

#include "StdAfx.h"

#if (ELISE_X11)

#define TheNbMaxChan  10


std::string StrNbChifSignNotSimple(double aVal,int aNbCh);
std::string StrNbChifSign(double aVal,int aNbCh);
std::string SimplString(std::string aStr);


void CorrectRect(Pt2di &  aP0,Pt2di &  aP1,const Pt2di & aSz);
void FillStat(cXml_StatVino & aStat,Flux_Pts aFlux,Fonc_Num aFonc);




typedef enum
{
   eModeGrapZoomVino,
   eModeGrapTranslateVino,
   eModeGrapAscX,
   eModeGrapAscY,
   eModeGrapShowRadiom,
   eModeVinoPopUp
}  eModeGrapAppli_Vino;


class cPopUpMenuMessage : public PopUpMenuTransp
{
   public :
      cPopUpMenuMessage(Video_Win aW,Pt2di aSz) ;
      void ShowMessage(const std::string & aName, Pt2di aP,Pt3di aCoul);
      void Hide();

};



class cAppli_Vino : public cXml_EnvVino,
                    public Grab_Untill_Realeased ,
                    public cElScrCalcNameSsResol
{
     public :
        cAppli_Vino(int,char **);
        void PostInitVirtual();
        void  Boucle();
        cXml_EnvVino & EnvXml() {return static_cast<cXml_EnvVino &> (*this);}


     private :
        void  MenuPopUp();
        void InitMenu();
        void ShowOneVal();
        void ShowOneVal(Pt2dr aP);
        void EffaceVal();
        bool OkPt(const Pt2di & aPt);
        void End();


        cXml_StatVino  StatRect(Pt2di &  aP0,Pt2di &  P1);

        std::string NamePyramImage(int aZoom);
        std::string  CalculName(const std::string & aName, INT InvScale); // cElScrCalcNameSsResol

        void  GUR_query_pointer(Clik,bool);
        void ExeClikGeom(Clik);
        void ZoomMolette();
        void ShowAsc();
        Pt2dr ToCoordAsc(const Pt2dr & aP);

        std::string               mNameXmlOut;
        std::string               mNameXmlIn;
        std::string               mDir;
        std::string               mNameIm;
        Tiff_Im  *                mTiffIm;
        std::string               mNameTiffIm;
        Pt2di                     mTifSz;
        bool                      mCoul;
        int                       mNbChan;
        double                    mNbPix;
        double                    mRatioFul;
        Pt2dr                     mRatioFulXY;

        Pt2di                     mSzIncr;
        Video_Win *               mWAscH;
        Video_Win *               mW;
        Video_Win *               mWAscV;
        Video_Display *           mDisp;
        std::string               mTitle;
        Visu_ElImScr *            mVVE;
        ElImScroller *            mScr;
        std::vector<INT>          mVEch;
        Pt2dr                     mP0Click;
        bool                      mInitP0StrVal;
        Pt2di                     mP0StrVal;
        Pt2di                     mP1StrVal;
        double                    mScale0;
        Pt2dr                     mTr0;
        int                       mBut0;
        bool                      mCtrl0;
        bool                      mShift0;
        eModeGrapAppli_Vino       mModeGrab;

        double                    mNbPixMinFile;
        double                    mSzEl;


         // Menus contextuels

          Pt2di                   mSzCase;
          GridPopUpMenuTransp*    mPopUpBase;
          CaseGPUMT *             mCaseExit;

          GridPopUpMenuTransp*    mPopUpCur;

};

#endif



#endif // _VINO_H_

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
aooter-MicMac-eLiSe-25/06/2007*/
