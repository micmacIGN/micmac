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
         cImOfTriplet(cAppliOptimTriplet &,cNewO_OneIm *,const ElRotation3D & aR);
         cNewO_OneIm * Im() {return mIm;}
         const ElRotation3D & Rot() {return mRot;}
         cAppliOptimTriplet &Appli() {return mAppli;}
         std::vector<Pt2df> &  VPOf3() {return mVPOf3;}  // Points triples
   private :
         cAppliOptimTriplet & mAppli;
         cNewO_OneIm * mIm;
         ElRotation3D  mRot;
         std::vector<Pt2df>    mVPOf3;
};


class cPairOfTriplet 
{
     public :
          cPairOfTriplet(cImOfTriplet * aI1,cImOfTriplet *aI2);
          double ResiduMoy();
          const ElRotation3D & R1() const {return mIm1->Rot();}
          const ElRotation3D & R2() const {return mIm2->Rot();}
     private :
          cAppliOptimTriplet & mAppli;
          cImOfTriplet *        mIm1;
          cImOfTriplet *        mIm2;
          std::vector<Pt2df>    mVP1;
          std::vector<Pt2df>    mVP2;
};


class cAppliOptimTriplet
{
      public :
          cAppliOptimTriplet(int argc,char ** argv);
          cNewO_NameManager * NM() {return mNM;}
          double ResiduTriplet();
      private :
          std::string mNameOriCalib;
          std::string mDir;
          cNewO_NameManager * mNM;
          cImOfTriplet *   mIm1;
          cImOfTriplet *   mIm2;
          cImOfTriplet *   mIm3;
          cPairOfTriplet * mP12;
          cPairOfTriplet * mP13;
          cPairOfTriplet * mP23;
};

/**************************************************/
/*                                                */
/*            cImOfTriplet                        */
/*                                                */
/**************************************************/

cImOfTriplet::cImOfTriplet(cAppliOptimTriplet & anAppli,cNewO_OneIm * anIm,const ElRotation3D & aR) :
    mAppli (anAppli),
    mIm    (anIm),
    mRot   (aR)
{
}
/*
*/


/**************************************************/
/*                                                */
/*            cPairOfTriplet                      */
/*                                                */
/**************************************************/

cPairOfTriplet::cPairOfTriplet(cImOfTriplet * aI1,cImOfTriplet *aI2) :
    mAppli (aI1->Appli()),
    mIm1   (aI1),
    mIm2   (aI2)
{
   mAppli.NM()->LoadHomFloats(mIm1->Im(),mIm2->Im(),&mVP1,&mVP2);

    std::cout << "cPairOfTriplet " << mVP1.size() << " " << mVP2.size() << "\n";
}


double cPairOfTriplet::ResiduMoy()
{
    std::vector<double> aVRes;
    for (int aK=0 ; aK<int(mVP1.size()) ; aK++)
    {
        std::vector<Pt3dr> aW1;
        std::vector<Pt3dr> aW2;
        AddSegOfRot(aW1,aW2,R1(),mVP1[aK]);
        AddSegOfRot(aW1,aW2,R2(),mVP2[aK]);
        bool OkI;
        Pt3dr aI = InterSeg(aW1,aW2,OkI);
        if (OkI)
        {
            double aRes1 = Residu(mIm1->Im(),R1(),aI,mVP1[aK]);
            double aRes2 = Residu(mIm2->Im(),R2(),aI,mVP2[aK]);

            aVRes.push_back((aRes1+aRes2)/2.0);
        }
    }
    return MedianeSup(aVRes);
}

/**************************************************/
/*                                                */
/*            cAppliOptimTriplet                  */
/*                                                */
/**************************************************/


double cAppliOptimTriplet::ResiduTriplet()
{
    std::vector<double> aVRes;
    for (int aK=0 ; aK<int(mIm1->VPOf3().size()) ; aK++)
    {
        std::vector<Pt3dr> aW1;
        std::vector<Pt3dr> aW2;
        AddSegOfRot(aW1,aW2,mIm1->Rot(),mIm1->VPOf3()[aK]);
        AddSegOfRot(aW1,aW2,mIm2->Rot(),mIm2->VPOf3()[aK]);
        AddSegOfRot(aW1,aW2,mIm3->Rot(),mIm3->VPOf3()[aK]);
        bool OkI;
        Pt3dr aI = InterSeg(aW1,aW2,OkI);
        if (OkI)
        {
            double aRes1 = Residu(mIm1->Im(),mIm1->Rot(),aI,mIm1->VPOf3()[aK]);
            double aRes2 = Residu(mIm2->Im(),mIm2->Rot(),aI,mIm2->VPOf3()[aK]);
            double aRes3 = Residu(mIm3->Im(),mIm3->Rot(),aI,mIm3->VPOf3()[aK]);
/*
            double aRes2 = Residu(mIm2->Im(),R2(),aI,mVP2[aK]);
*/

            aVRes.push_back((aRes1+aRes2+aRes3)/3.0);
        }
    }
    return MedianeSup(aVRes);
}

cAppliOptimTriplet::cAppliOptimTriplet(int argc,char ** argv)  :
    mDir ("./")
{
   std::string aN1,aN2,aN3;
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(aN1,"Image one")
                   << EAMC(aN2,"Image two")
                   << EAMC(aN3,"Image three"),
        LArgMain() << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration ")
                   << EAM(mDir,"Dir",true,"Directoru, Def=./ ")
   );
   cTplTriplet<std::string> a3S(aN1,aN2,aN3);

   mNM = new cNewO_NameManager(mDir,mNameOriCalib,"dat");

   cNewO_OneIm * aIm1 = new cNewO_OneIm(*mNM,a3S.mV0);
   cNewO_OneIm * aIm2 = new cNewO_OneIm(*mNM,a3S.mV1);
   cNewO_OneIm * aIm3 = new cNewO_OneIm(*mNM,a3S.mV2);

   std::string  aName3R = mNM->NameOriInitTriplet(true,aIm1,aIm2,aIm3);
   cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);

   mIm1 = new  cImOfTriplet(*this,aIm1,ElRotation3D::Id);
   mIm2 = new  cImOfTriplet(*this,aIm2,Xml2El(aXml3Ori.Ori2On1()));
   mIm3 = new  cImOfTriplet(*this,aIm3,Xml2El(aXml3Ori.Ori3On1()));

   mP12 = new cPairOfTriplet(mIm1,mIm2);
   mP13 = new cPairOfTriplet(mIm1,mIm3);
   mP23 = new cPairOfTriplet(mIm2,mIm3);

   mNM->LoadTriplet(mIm1->Im(),mIm2->Im(),mIm3->Im(),&mIm1->VPOf3(),&mIm2->VPOf3(),&mIm3->VPOf3());
   
   std::cout << "NB TRIPLE " << mIm2->VPOf3().size()  << " Resi3: " <<  ResiduTriplet() << "\n";


   if (1)
   {
      std::cout << "RESIDU/PAIRES " << mP12->ResiduMoy() << " " << mP13->ResiduMoy() << " " << mP23->ResiduMoy() << " " << "\n";
   }
}


/**************************************************/
/*                                                */
/*            ::                                  */
/*                                                */
/**************************************************/

int CPP_OptimTriplet_main(int argc,char ** argv)
{
   cAppliOptimTriplet anAppli(argc,argv);
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
Footer-MicMac-eLiSe-25/06/2007*/
