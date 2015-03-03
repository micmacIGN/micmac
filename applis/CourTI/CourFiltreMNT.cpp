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



static std::string TheGlobName = "/media/MYPASSPORT/Documents/Cours/ENSG-TI-IT2/MNT/pleiades/";

double aSeuilBas = 2;
double aSeuilHaut = 5;

class cFiltreMNT
{
    public :

        cFiltreMNT(const std::string & aPost) :
            mPost (aPost),
            mName (CalcNameResult("")),
            mFile (Tiff_Im::StdConv(mName)),
            mSz   (mFile.sz()),
            mIm   (mSz.x,mSz.y),
            mMasq (mSz.x,mSz.y),
            mDTM  (mSz.x,mSz.y),
            mSurSol  (mSz.x,mSz.y),
            mImBati  (mSz.x,mSz.y)
        {
             std::cout << "NAME " << mName << "\n";
             Tiff_Im aFM = Tiff_Im::StdConv(CalcNameResult("_Masq"));
             ELISE_COPY(aFM.all_pts(),aFM.in(),mMasq.out());

             ELISE_COPY
             (
                 mFile.all_pts(),
                 round_ni(mFile.in()*mMasq.in()),
                 mIm.out() | VMax(mMax) | VMin(mMin)
             );
             ELISE_COPY(mIm.all_pts(),mIm.in()-mMin,mIm.out());
             mMax -= mMin;
             mMin = 0;


             
             // if (isSynt)
                  // ELISE_COPY(mIm.all_pts(),mIm.in() + 0.01*FX+0.0123*FY,mIm.out());
        }

        std::string CalcNameResult(const std::string& aStep)
        {
            return  TheGlobName+mPost+ aStep + ".tif";
        }


        void MNT(const std::string &,int aSzW);
        void Fitrage(const std::string &,int aSzW);


        std::string  mPost;
        std::string  mName;
        Tiff_Im      mFile;
        int          mMax;
        int          mMin;
        Pt2di        mSz;
        Im2D_INT2   mIm;
        Im2D_U_INT1 mMasq;
        Im2D_INT2   mDTM;
        Im2D_INT2   mSurSol;

        Im2D_U_INT1 mImBati;
};


void cFiltreMNT::MNT(const std::string & aPost,int aSzW)
{
   Fonc_Num aMNT = mMasq.in(0)*mIm.in(mMax) + (1- mMasq.in(0))*mMax;

   double aKth=0.15;
   int aNbW = ElSquare(1+2*aSzW);
   int aIK = round_ni(aNbW*aKth);

   ELISE_COPY
   (
          mIm.all_pts(),
          rect_kth
          (
              rect_kth(aMNT,aIK,aSzW,mMax+1),
              aNbW-aIK,aSzW,mMax+1
          ),
          mDTM.out()
     
   );

   Tiff_Im::CreateFromIm(mDTM,CalcNameResult(aPost));
}

void cFiltreMNT::Fitrage(const std::string & aName,int aSzW)
{
   //MNT(aName,aSzW);
   Tiff_Im aMNE = Tiff_Im::StdConv(CalcNameResult(aName));

   ELISE_COPY(aMNE.all_pts(),mIm.in()-aMNE.in(),mSurSol.out());

   Tiff_Im::CreateFromIm(mSurSol,CalcNameResult("SurS"+aName));

   ELISE_COPY(mImBati.all_pts(),255*(mSurSol.in()>15),mImBati.out());

   Tiff_Im::CreateFromIm(mImBati,CalcNameResult("Bati_"+aName));
}


void TestMNT(const std::string & aName,double aDyn,bool IsSynt)
{
     cFiltreMNT aRE(aName);

     aRE.Fitrage("90_15per",90);
     
}


int main(int argc,char ** argv)
{

     TestMNT("ZCarto-CHU-Rangueil-50",3.0,false); 

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
