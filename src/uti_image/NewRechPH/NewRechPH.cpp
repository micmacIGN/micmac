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


#include "NewRechPH.h"

void  cAppli_NewRechPH::Clik()
{
   if (mW1) mW1->clik_in();
}

void cAppli_NewRechPH::AddScale(cOneScaleImRechPH * aI1,cOneScaleImRechPH *)
{
    mVI1.push_back(aI1);
}

cAppli_NewRechPH::cAppli_NewRechPH(int argc,char ** argv,bool ModeTest) :
    mPowS     (pow(2.0,1/5.0)),
    mNbS      (30),
    mW1       (0),
    mModeTest (ModeTest),
    mDistMinMax (3.0),
    mDoMin      (true),
    mDoMax      (true)
{
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mName, "Name Image",  eSAM_IsPatFile),
         LArgMain()   << EAM(mPowS,         "PowS",true,"Scale Pow")
                      << EAM(mNbS,       "NbS",true,"If true do debugging")
   );

   AddScale(cOneScaleImRechPH::FromFile(*this,mName,Pt2di(0,0),Pt2di(-1,-1)),0);

   mSzIm = mVI1.back()->Im().sz();
   if (mModeTest)
   {
      mW1 = Video_Win::PtrWStd(mSzIm);
      ELISE_COPY(mW1->all_pts(),mVI1.back()->Im().in(),mW1->ogray());
   }

   for (int aK=1 ; aK<mNbS ; aK++)
   {
        AddScale
        (
              cOneScaleImRechPH::FromScale(*this,*mVI1.back(),pow(mPowS,aK)),
              0
        );
        if (mW1)
        {
           ELISE_COPY(mW1->all_pts(),mVI1.back()->Im().in(),mW1->ogray());
        }

        if (aK==mNbS/2)
        {
           mVI1.back()->CalcPtsCarac();
        }
   }

   Clik();
}



int Test_NewRechPH(int argc,char ** argv)
{
   cAppli_NewRechPH anAppli(argc,argv,true);

   return EXIT_SUCCESS;

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
aooter-MicMac-eLiSe-25/06/2007*/
