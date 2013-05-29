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
#include <algorithm>
#include "im_tpl/image.h"
#include "im_special/hough.h"



static std::string TheGlobName = "/media/MYPASSPORT/Documents/Cours/ENSG-TI-IT2/Interro-2010/Filtre/";

double aSeuilBas = 2;
double aSeuilHaut = 5;

class cTestFiltre
{
    public :

        cTestFiltre(const std::string & aPost) :
            mPost (aPost),
            mName (CalcNameResult("")),
            mFile (Tiff_Im::StdConv(mName)),
            mSz   (mFile.sz()),
            mIm   (mSz.x,mSz.y)
        {
             std::cout << "NAME " << mName << "\n";
             ELISE_COPY(mFile.all_pts(),mFile.in(),mIm.out());
        }

        std::string CalcNameResult(const std::string& aStep)
        {
            return  TheGlobName+(aStep==""? "" : "F_") +mPost+ aStep + ".tif";
        }

        void DoAll();

        std::string  mPost;
        std::string  mName;
        Tiff_Im      mFile;
        Pt2di        mSz;
        Im2D_REAL8   mIm;
};


void cTestFiltre::DoAll()
{
    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Der_X_1"),
          mSz,
          Max(0,Min(255,128+6*deriche(mIm.in_proj(),1).v0()))
    );
    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Der_Y_1"),
          mSz,
          Max(0,Min(255,128+6*deriche(mIm.in_proj(),1).v1()))
    );

    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Der_N_1"),
          mSz,
          Max(0,Min(255,15*polar(deriche(mIm.in_proj(),1),0).v0()))
    );

    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Der_N_2"),
          mSz,
          Max(0,Min(255,9*polar(deriche(mIm.in_proj(),3),0).v0()))
    );
    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Der_N_03"),
          mSz,
          Max(0,Min(255,30*polar(deriche(mIm.in_proj(),0.3),0).v0()))
    );





    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Dilat"),
          mSz,
          rect_max(mIm.in_proj(),2)
    );

    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Erod"),
          mSz,
          rect_min(mIm.in_proj(),2)
    );

    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Ouv"),
          mSz,
          rect_max(rect_min(mIm.in_proj(),2),2)
    );

    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Ferm"),
          mSz,
          rect_min(rect_max(mIm.in_proj(),2),2)
    );



//===============================

    int aNbV=12;

    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Moy"),
          mSz,
          rect_som(mIm.in(0),aNbV)/rect_som(mIm.inside(),aNbV)
    );

    int aFact = 3;
    int aNbVR=  aNbV /aFact;
    Fonc_Num aF1 = mIm.in(0);
    Fonc_Num aF2 = mIm.inside();
    for (int aK=0 ; aK<aFact*aFact ; aK++)
    {
          aF1 = rect_som(aF1,aNbVR) / ElSquare(1.0+2*aNbVR);
          aF2 = rect_som(aF2,aNbVR) / ElSquare(1.0+2*aNbVR);
    }
    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_MoyIt"),
          mSz,
          aF1/Max(1e-5,aF2)
    );

    Tiff_Im::Create8BFromFonc
    (
          CalcNameResult("_Median"),
          mSz,
          rect_median(mIm.in_proj(),3,256)
    );

}



void TestFiltre(const std::string & aName)
{
     cTestFiltre aTF(aName);
     aTF.DoAll();
}


int main(int argc,char ** argv)
{

     // TestReco("0",3.0,false); 
     // TestReco("00",3.0,true); 
     // TestReco("1",3.0,false); 
     //  TestReco("img_0252",3.0,false); 
     // TestReco("img_0258",3.0,false); 
     // TestReco("img_0254",3.0,false); 
     // TestFiltre("Taupe"); 
     // TestFiltre("Croix"); 
     TestFiltre("PartBanBreizh-1"); 

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
