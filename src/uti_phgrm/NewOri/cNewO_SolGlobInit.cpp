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

class cNOSolIn_AttrSom;
class cNOSolIn_AttrASym;
class cNOSolIn_AttrArc;
class cAppli_NewSolGolInit;
class cNOSolIn_Triplet;

typedef  ElSom<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tSomNSI;
typedef  ElArc<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tArcNSI;
typedef  ElSomIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItSNSI;
typedef  ElArcIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItANSI;
typedef  ElGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>      tGrNSI;
typedef  ElSubGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>   tSubGrNSI;



class cNOSolIn_AttrSom
{
     public :
         cNOSolIn_AttrSom(const std::string & aName,cAppli_NewSolGolInit & anAppli);
         cNOSolIn_AttrSom() {}
         std::vector<cNOSolIn_Triplet *> & V3() {return mV3;}
         cNewO_OneIm * Im() {return mIm;}
     private :
         std::string                      mName;
         cAppli_NewSolGolInit *           mAppli;
         cNewO_OneIm *                    mIm;
         std::vector<cNOSolIn_Triplet *>  mV3;

};



class cNOSolIn_AttrASym
{
     public :
         void AddTriplet(cNOSolIn_Triplet * aTrip);
     private :
         std::vector<cNOSolIn_Triplet *> mV3;
           
};
class cNOSolIn_AttrArc
{
     public :
           cNOSolIn_AttrArc(cNOSolIn_AttrASym *);
           cNOSolIn_AttrASym * ASym() {return mASym;}
     private :
           cNOSolIn_AttrASym * mASym;
};


class cNOSolIn_Triplet
{
      public :
          cNOSolIn_Triplet(tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit &);
          void SetArc(int aK,tArcNSI *);
      private :
          tSomNSI *     mSoms[3];
          tArcNSI *     mArcs[3];
          ElRotation3D  mR2on1;
          ElRotation3D  mR3on1;
};



class cAppli_NewSolGolInit
{
    public :
        cAppli_NewSolGolInit(int , char **);
        cNewO_NameManager & NM() {return *mNM;}

    private :
        void                 CreateArc(tSomNSI *,tSomNSI *,cNOSolIn_Triplet *,int aK);
        std::string          mFullPat;
        std::string          mOriCalib;
        cElemAppliSetFile    mEASF;
        cNewO_NameManager  * mNM;
        bool                 mQuick;
        bool                 mTest;
 
        tGrNSI               mGr;
        std::map<std::string,tSomNSI *> mMapS;
};

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrSom                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_AttrSom::cNOSolIn_AttrSom(const std::string & aName,cAppli_NewSolGolInit & anAppli) :
   mName  (aName),
   mAppli (&anAppli),
   mIm    (new cNewO_OneIm(mAppli->NM(),mName))
{
}

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_Triplet                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_Triplet::cNOSolIn_Triplet(tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit & aTrip) :
    mR2on1 (Xml2El(aTrip.Ori2On1())),
    mR3on1 (Xml2El(aTrip.Ori3On1()))
{
   mSoms[0] = aS1;
   mSoms[1] = aS2;
   mSoms[2] = aS3;
}

void cNOSolIn_Triplet::SetArc(int aK,tArcNSI * anArc)
{
   mArcs[aK] = anArc;
}

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrASym                                       */
/*                                                                         */
/***************************************************************************/


void  cNOSolIn_AttrASym::AddTriplet(cNOSolIn_Triplet * aTrip)
{
    mV3.push_back(aTrip); 
}

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrArc                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_AttrArc::cNOSolIn_AttrArc(cNOSolIn_AttrASym * anASym) :
   mASym (anASym)
{
}

/***************************************************************************/
/*                                                                         */
/*                 cAppli_NewSolGolInit                                    */
/*                                                                         */
/***************************************************************************/

void   cAppli_NewSolGolInit::CreateArc(tSomNSI * aS1,tSomNSI * aS2,cNOSolIn_Triplet * aTripl,int aK)
{
     tArcNSI * anArc = mGr.arc_s1s2(*aS1,*aS2);
     if (anArc==0)
     {
         cNOSolIn_AttrASym * anAttrSym = new cNOSolIn_AttrASym;
         cNOSolIn_AttrArc anAttr12(anAttrSym);
         cNOSolIn_AttrArc anAttr21(anAttrSym);
         anArc = &(mGr.add_arc(*aS1,*aS2,anAttr12,anAttr21));
     }
     anArc->attr().ASym()->AddTriplet(aTripl);
     aTripl->SetArc(aK,anArc);

     // return anArc;
}


cAppli_NewSolGolInit::cAppli_NewSolGolInit(int argc, char ** argv) :
    mQuick (true),
    mTest  (true)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mFullPat,"Pattern"),
        LArgMain() << EAM(mOriCalib,"OriCalib",true,"Orientation for calibration ", eSAM_IsExistDirOri)
                   << EAM(mQuick,"Quick",true,"Quick version",eSAM_IsBool)
                   << EAM(mTest,"Test",true,"Test for tuning",eSAM_IsBool)
   );


   mEASF.Init(mFullPat);
   mNM = new cNewO_NameManager(mQuick,mEASF.mDir,mOriCalib,"dat");
   const cInterfChantierNameManipulateur::tSet * aVIm = mEASF.SetIm();

   for (int aKIm=0 ; aKIm <int(aVIm->size()) ; aKIm++)
   {
       const std::string & aName = (*aVIm)[aKIm];
       tSomNSI & aSom = mGr.new_som(cNOSolIn_AttrSom(aName,*this));
       mMapS[aName] = & aSom;
   }


    cXml_TopoTriplet aXml3 =  StdGetFromSI(mNM->NameTopoTriplet(true),Xml_TopoTriplet);

    for
    (
         std::list<cXml_OneTriplet>::const_iterator it3=aXml3.Triplets().begin() ;
         it3 !=aXml3.Triplets().end() ;
         it3++
    )
    {
            tSomNSI * aS1 = mMapS[it3->Name1()];
            tSomNSI * aS2 = mMapS[it3->Name2()];
            tSomNSI * aS3 = mMapS[it3->Name3()];
            if (aS1 && aS2 && aS3)
            {


                 std::string  aN3 = mNM->NameOriGenTriplet
                                    (
                                        mQuick,
                                        !mTest,
                                        aS1->attr().Im(),
                                        aS2->attr().Im(),
                                        aS3->attr().Im()
                                    );
                 cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aN3,Xml_Ori3ImInit);


                 cNOSolIn_Triplet * aTriplet = new cNOSolIn_Triplet(aS1,aS2,aS3,aXml3Ori);

                 aS1->attr().V3().push_back(aTriplet);
                 aS2->attr().V3().push_back(aTriplet);
                 aS3->attr().V3().push_back(aTriplet);
                 CreateArc(aS1,aS2,aTriplet,0);
                 CreateArc(aS2,aS3,aTriplet,1);
                 CreateArc(aS3,aS1,aTriplet,2);
            }
    }
}


int CPP_NewSolGolInit_main(int argc, char ** argv)
{
    cAppli_NewSolGolInit anAppli(argc,argv);
    return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
