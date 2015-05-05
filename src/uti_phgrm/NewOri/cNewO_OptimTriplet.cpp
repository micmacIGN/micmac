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



#include "NewOri.h"

class cAppliOptimTriplet;
class cPairOfTriplet;
class cImOfTriplet ;

class cImOfTriplet 
{
   public :
         cImOfTriplet(cAppliOptimTriplet &,const std::string &,const ElRotation3D & aR);
   private :
};


class cPairOfTriplet 
{
     public :
          cPairOfTriplet(cNewO_OneIm * aI1,cNewO_OneIm *aI2,cAppliOptimTriplet & anAppli);
     private :
          cAppliOptimTriplet & mAppli;
          cNewO_OneIm *        mIm1;
          cNewO_OneIm *        mIm2;
          std::vector<Pt2df>   mVP1;
          std::vector<Pt2df>   mVP2;
};


class cAppliOptimTriplet
{
      public :
          cAppliOptimTriplet(int argc,char ** argv);
          cNewO_NameManager * NM() {return mNM;}
      private :
          std::string mN1;
          std::string mN2;
          std::string mN3;
          std::string mNameOriCalib;
          std::string mDir;
          cNewO_NameManager * mNM;
          cNewO_OneIm * mIm1;
          cNewO_OneIm * mIm2;
          cNewO_OneIm * mIm3;
};

/**************************************************/
/*                                                */
/*            cPairOfTriplet                      */
/*                                                */
/**************************************************/

/*
cPairOfTriplet::cPairOfTriplet(cNewO_OneIm * aI1,cNewO_OneIm *aI2,cAppliOptimTriplet & anAppli):
{
}
*/


/**************************************************/
/*                                                */
/*            cAppliOptimTriplet                  */
/*                                                */
/**************************************************/



cAppliOptimTriplet::cAppliOptimTriplet(int argc,char ** argv)  :
    mDir ("./")
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mN1,"Image one")
                   << EAMC(mN2,"Image two")
                   << EAMC(mN3,"Image three"),
        LArgMain() << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration ")
                   << EAM(mDir,"OriCalib",true,"Orientation for calibration ")
   );

   mNM = new cNewO_NameManager(mDir,mNameOriCalib,"dat");

   mIm1 = new cNewO_OneIm(*mNM,mN1);
   mIm2 = new cNewO_OneIm(*mNM,mN2);
   mIm3 = new cNewO_OneIm(*mNM,mN3);

}



/*
int GenTriplet_main(int argc,char ** argv)
{
   cAppli_GenTriplet anAppli(argc,argv);
   anAppli.GenTriplet();
   return EXIT_SUCCESS;
}
*/


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
