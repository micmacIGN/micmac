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


//  bin/Filter $1 "Harris 1.5 2 Moy 1 3" Out=$2


#define DEF_OFSET -12349876


class cAppliFilter
{
    public :
       cAppliFilter(int argc,char ** argv);

        void AddFoncNum(Fonc_Num aFRB,Pt2di aP0,Fonc_Num aMasq,double aScale,Pt2di aSz);
        void StdAdd(std::string  ,Pt2dr aP0,std::string Masq,double aScale);
        void Write();
    private  :
        Pt2di       mSz;
        std::string mDir;
        std::string mNameRes;

        Im2D_U_INT1   mImR;
        Im2D_U_INT1   mImV;
        Im2D_U_INT1   mImB;

      
};

void cAppliFilter::Write()
{
   Tiff_Im::Create8BFromFonc
   (
      mDir+mNameRes,
      mImR.sz(),
      Virgule(mImR.in(),mImV.in(),mImB.in())
   );
}

cAppliFilter::cAppliFilter(int argc,char ** argv) :
   mSz (1000,1400),
   mDir ("/media/MYPASSPORT/Documents/BookMicMac/Visuel/"),
   mImR (mSz.x,mSz.y,0),
   mImV (mSz.x,mSz.y,0),
   mImB (mSz.x,mSz.y,0)
{
    std::string aFullNameIn;
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(mNameRes),
	LArgMain() << EAM(mSz,"Sz",true)
                    << EAM(mDir,"Dir",true)
    );	
}


void cAppliFilter::AddFoncNum
     (
          Fonc_Num aFRVB,
          Pt2di    aP0,
          Fonc_Num aFMasq,
          double   aScale,
          Pt2di    aSz
     )
{
    if (aFRVB.dimf_out()==1)
      aFRVB  = Virgule(aFRVB,aFRVB,aFRVB);

    aSz = round_ni(Pt2dr(aSz) /aScale);
    Pt2dr aPDil(1,1);
    Pt2dr aPScale(aScale,aScale);

    Im2D_U_INT1  aImR (aSz.x,aSz.y);
    Im2D_U_INT1  aImV (aSz.x,aSz.y);
    Im2D_U_INT1  aImB (aSz.x,aSz.y);
    Im2D_U_INT1  aImMasq (aSz.x,aSz.y);
    ELISE_COPY
    (
        aImR.all_pts(),
        Max(0,Min(255,round_ni(StdFoncChScale(Virgule(aFRVB,aFMasq),Pt2dr(0,0),aPScale,aPDil)))),
        Virgule(aImR.out(),aImV.out(),aImB.out(),aImMasq.out())
    );


    ELISE_COPY
    (
          select
          (
                rectangle(aP0,aP0+aSz),
                trans(aImMasq.in(0),-aP0)
          ),
          trans(Virgule(aImR.in(),aImV.in(),aImB.in()),-aP0),
          Virgule(mImR.oclip(),mImV.oclip(),mImB.oclip())
    );
}


void cAppliFilter::StdAdd
     (
           std::string  aNameRVB,
           Pt2dr aP0,
           std::string aNameMasq,
           double aScale
      )
{
    aNameRVB  = mDir + aNameRVB;
    aNameMasq = mDir + aNameMasq;

    Tiff_Im aFrvb = Tiff_Im::StdConv(aNameRVB);
    Tiff_Im aFM = Tiff_Im::StdConv(aNameMasq);

    AddFoncNum(aFrvb.in(0),aP0,aFM.in(0),aScale,aFrvb.sz());
}

          // Fonc_Num aFRVB, Pt2dr    aP0, Fonc_Num aFMasq, double   aScale, Pt2di    aSz


int main(int argc,char ** argv)
{
    cAppliFilter anAppli(argc,argv);

    anAppli.StdAdd
    (
       "MoulinSnap03.tif",
       Pt2di(0,800),
       "Gray-MoulinSnap03_Masq.tif",
        1.5
    );

    anAppli.StdAdd
    (
       "Sep-LambaleShade.tif",
       Pt2di(0,-100),
       "LambaleShade_Masq.tif",
        2.5
    );


    anAppli.StdAdd
    (
       "Small_MNE_1_25.tif",
       Pt2di(50,300),
       "Small_MNE_1_25_Masq.tif",
        1
    );

    anAppli.StdAdd
    (
       "Ajacio.tif",
       Pt2di(500,100),
       "Gray-Ajacio_Masq.tif",
        1.5
    );



    anAppli.StdAdd
    (
       "Ele.tif",
       Pt2di(-100,400),
       "Gray-Ele_Masq.tif",
        0.7
    );

    anAppli.StdAdd
    (
       "Sommeil-PlastilineSnap00.tif",
       Pt2di(-400,500),
       "Gray-Sommeil-PlastilineSnap00_Masq.tif",
        1.0
    );


    anAppli.Write();
    return 0;
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
