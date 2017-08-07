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

class cOrHom_AttrSom;
class cOrHom_AttrASym;
class cOrHom_AttrArc;
class cAppli_GenTriplet;

typedef  ElSom<cOrHom_AttrSom,cOrHom_AttrArc>         tSomGT;
typedef  ElArc<cOrHom_AttrSom,cOrHom_AttrArc>         tArcGT;
typedef  ElSomIterator<cOrHom_AttrSom,cOrHom_AttrArc> tItSGT;
typedef  ElArcIterator<cOrHom_AttrSom,cOrHom_AttrArc> tItAGT;
typedef  ElGraphe<cOrHom_AttrSom,cOrHom_AttrArc>      tGrGT;
typedef  ElSubGraphe<cOrHom_AttrSom,cOrHom_AttrArc>   tSubGrGT;


/************************************************/
/*                                              */
/*         cOrHom_AttrSom                       */
/*                                              */
/************************************************/

class cOrHom_AttrSom
{
     public :
        cOrHom_AttrSom(const std::string & aName);
        cOrHom_AttrSom();
     private :
        std::string mName;
};

cOrHom_AttrSom::cOrHom_AttrSom(const std::string & aName) :
   mName (aName)
{
}

cOrHom_AttrSom::cOrHom_AttrSom() 
{
}


/************************************************/
/*                                              */
/*         cOrHom_AttrArc                       */
/*                                              */
/************************************************/

class cOrHom_AttrArc
{
};

class cAppli_Hom1Im : public cCommonMartiniAppli
{
    public :
        cAppli_Hom1Im(int argc,char ** argv,bool ModePrelim);
    private:
        tSomGT * AddAnIm(const std::string & aName);

        bool        mModePrelim;
        std::string mPat;
        std::string mNameC;
        cElemAppliSetFile mEASF;
        std::map<std::string,tSomGT *> mMapS;
        tGrGT                                  mGr;
};


tSomGT * cAppli_Hom1Im::AddAnIm(const std::string & aName)
{
   if (mMapS[aName] == 0)
   {
      mMapS[aName]  = &(mGr.new_som(cOrHom_AttrSom(aName)));
   }

   return mMapS[aName];
}

cAppli_Hom1Im::cAppli_Hom1Im(int argc,char ** argv,bool aModePrelim) :
   mModePrelim (aModePrelim)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mPat,"Central image"),
        LArgMain() << ArgCMA()
   );

   mEASF.Init(mPat);
   const cInterfChantierNameManipulateur::tSet * aVN = mEASF.SetIm();
   if (mModePrelim)
   {
       ELISE_ASSERT(aVN->size()==1,"Expect just one image in preliminary mode");
       mNameC = (*aVN)[0];
   }
    
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les mêmes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,    l'utilisation,    la modification et/ou au
developpement et  la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe 
manipuler et qui le reserve donc   des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a  charger  et  tester  l'adequation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a  l'utiliser et l'exploiter dans les mêmes conditions de securite.

Le fait que vous puissiez acceder a  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007*/


