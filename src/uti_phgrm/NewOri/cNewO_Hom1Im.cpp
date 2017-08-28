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
class cOrHom_AttrSymA;
class cAppli_HomOrIm ;

typedef  ElSom<cOrHom_AttrSom,cOrHom_AttrArc>         tSomHO;
typedef  ElArc<cOrHom_AttrSom,cOrHom_AttrArc>         tArcHO;
typedef  ElSomIterator<cOrHom_AttrSom,cOrHom_AttrArc> tItSHO;
typedef  ElArcIterator<cOrHom_AttrSom,cOrHom_AttrArc> tItAHO;
typedef  ElGraphe<cOrHom_AttrSom,cOrHom_AttrArc>      tGrHO;
typedef  ElSubGraphe<cOrHom_AttrSom,cOrHom_AttrArc>   tSubGrHO;


/************************************************/
/*                                              */
/*         cOrHom_AttrSom                       */
/*                                              */
/************************************************/

class cOrHom_AttrSom
{
     public :
        cOrHom_AttrSom(const std::string & aName,cAppli_HomOrIm & anAppli);
        cOrHom_AttrSom()  : mAppli (0) {};
        const std::string & Name() const {return mName;}
        cAppli_HomOrIm & Appli() {return *mAppli;}

     private :
        std::string mName;
        cAppli_HomOrIm * mAppli;
};

cOrHom_AttrSom::cOrHom_AttrSom(const std::string & aName,cAppli_HomOrIm & anAppli) :
   mName    (aName),
   mAppli   (&anAppli)
{
}
// cOrHom_AttrSom::cOrHom_AttrSom() { }



    // =============        cOrHom_AttrArc    =============

class cOrHom_AttrSymA
{
    public :
       cOrHom_AttrSymA();
};


class cOrHom_AttrArc
{
     public :
        cOrHom_AttrArc(tSomHO * aS1,tSomHO * aS2,const cXml_Ori2Im & aXO,cOrHom_AttrSymA *);  // Arc ds le sens S1 S2
        cOrHom_AttrArc(const cOrHom_AttrArc &,cOrHom_AttrSymA *);     // Arc reciproque
     private :
        cElHomographie        mHom12;
        std::vector<Pt2dr>    mVP1;
        cOrHom_AttrSymA &     mASym;
};




   // ==================  cAppli_HomOrIm  ===================

class cAppli_HomOrIm : public cCommonMartiniAppli
{
    public :
        cAppli_HomOrIm(int argc,char ** argv,bool ModePrelim);
        bool  ModePrelim () const {return mModePrelim;}
        cNewO_NameManager & NM () {return *mNM;}
    private:
        tSomHO * GetSom(const std::string & aName,bool Create);
        tArcHO * AddArc(tSomHO * aS1,tSomHO * aS2);

        bool        mModePrelim;
        std::string mPat;
        std::string mNameC;
        cElemAppliSetFile mEASF;
        cNewO_NameManager * mNM;
        std::map<std::string,tSomHO *> mMapS;
        std::vector<tSomHO *>          mVSoms;
        tGrHO                          mGr;
        tSomHO *                       mSomC;
};

/************************************************/
/*                                              */
/*         cOrHom_AttrSymA                      */
/*         cOrHom_AttrArc                       */
/*                                              */
/************************************************/

cOrHom_AttrSymA::cOrHom_AttrSymA()
{
}

// XmlHomogr
cOrHom_AttrArc::cOrHom_AttrArc(tSomHO * aS1,tSomHO * aS2,const cXml_Ori2Im & aXO,cOrHom_AttrSymA * anAS) :
   mHom12 (aXO.Geom().Val().HomWithR().Hom()),
   mASym  (*anAS)
{
   cGenGaus2D aGG(aXO.Geom().Val().Elips2().Val());
   aGG.GetDistribGaus(mVP1,2,2); //  2,2 => 5x5
}

cOrHom_AttrArc::cOrHom_AttrArc(const cOrHom_AttrArc & anA12,cOrHom_AttrSymA * anAS) :
   mHom12 (anA12.mHom12.Inverse()),
   mASym  (*anAS)
{
    for (int aKP=0 ; aKP<int(anA12.mVP1.size()) ; aKP++)
    {
         mVP1.push_back(anA12.mHom12(anA12.mVP1[aKP]));
    }
}

/************************************************/
/*                                              */
/*         cAppli_HomOrIm                       */
/*                                              */
/************************************************/

tArcHO * cAppli_HomOrIm::AddArc(tSomHO * aS1,tSomHO * aS2)
{
   if (aS1->attr().Name() >  aS2->attr().Name())
      ElSwap(aS1,aS2);

   std::string aName = aS1->attr().Appli().NM().NameXmlOri2Im(aS1->attr().Name(),aS2->attr().Name(),true);
   cXml_Ori2Im   aXO =   StdGetFromSI(aName,Xml_Ori2Im);
   if ((! aXO.Geom().IsInit()) || (!aXO.Geom().Val().Elips2().IsInit()))
      return 0;

   // std::cout << "NAME: " << aName  << " => "  << aXO.Geom().Val().HomWithR().ResiduHom()  << "\n";
 
   tArcHO * anArc = mGr.arc_s1s2(*aS1,*aS2);
   if (anArc)
      return anArc;

   cOrHom_AttrSymA * anASym = new cOrHom_AttrSymA;
  
   cOrHom_AttrArc anA12(aS1,aS2,aXO,anASym);
   cOrHom_AttrArc anA21(anA12,anASym);

   return &(mGr.add_arc(*aS1,*aS2,anA12,anA21));
}




tSomHO * cAppli_HomOrIm::GetSom(const std::string & aName,bool Create)
{
   tSomHO * & aRes = mMapS[aName];
   if (aRes == 0)
   {
      if (Create)
      {
         aRes  = &(mGr.new_som(cOrHom_AttrSom(aName,*this)));
         mVSoms.push_back(aRes);
      }
      else
         return 0;
   }

   return aRes;
}

cAppli_HomOrIm::cAppli_HomOrIm(int argc,char ** argv,bool aModePrelim) :
   mModePrelim (aModePrelim),
   mSomC       (0)
{

   // Lecture standard  des arguments
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mPat,"Central image"),
        LArgMain() << ArgCMA()
   );
   mEASF.Init(mPat);
   mNM = cCommonMartiniAppli::NM(mEASF.mDir);
   const cInterfChantierNameManipulateur::tSet * aVN = mEASF.SetIm();

   if (mModePrelim)
   {
       ELISE_ASSERT(aVN->size()==1,"Expect just one image in preliminary mode");
       mNameC = (*aVN)[0];
   }

   //  Creation des sommets du  graphe 
   for (int aK=0 ; aK<int(aVN->size()) ; aK++)
   {
        const std::string & aName = (*aVN)[aK];
        tSomHO * aSom = GetSom(aName,true);
        if (aName==mNameC)
        {
            mSomC = aSom;
        }
        std::list<std::string> aLV = mNM->ListeImOrientedWith2Way(aName);

        // Cas Prelim, on rajoute tous les voisins du noyau
        // for (auto  itL=aLV.begin(); itL!=aLV.end() ; itL++)
        for (std::list<std::string>::const_iterator itL=aLV.begin(); itL!=aLV.end() ; itL++)
        {
             GetSom(*itL,true);
        }
   }


   //  Creation des arc du  graphe 
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
        tSomHO* aS1 = mVSoms[aK];
        std::list<std::string> aLV = mNM->ListeImOrientedWith2Way(aS1->attr().Name());
        for (std::list<std::string>::const_iterator itL=aLV.begin(); itL!=aLV.end() ; itL++)
        {
            tSomHO* aS2 = GetSom(*itL,false);
            if (aS2)
            {
               AddArc(aS1,aS2);
            }
        }
   }

}


int CPP_HomOr1Im(int argc,char ** argv)
{
   cAppli_HomOrIm anAppli(argc,argv,true);

   return EXIT_SUCCESS;
}

// class cAppli_HomOrIm : public cCommonMartiniAppli


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


