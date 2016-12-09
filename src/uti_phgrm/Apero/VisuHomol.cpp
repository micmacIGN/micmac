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
#include "Apero.h"

class cVisuResidHom
{
      public :
            cVisuResidHom(cBasicGeomCap3D *,cBasicGeomCap3D *,ElPackHomologue &,const std::string & aPrefOut);
      private :

            std::string NameFile(const std::string & aPost) {return mPrefOut + aPost;}
            void AddPair(const Pt2dr & aP1,const Pt2dr & aP2);
     
            cBasicGeomCap3D * mCam1;
            cBasicGeomCap3D * mCam2;
            ElPackHomologue   mPack;
            std::string       mPrefOut;
            double            mResol;
            Pt2di             mSzIm1;
            Pt2di             mSzResol;
            cPlyCloud         mPlyC;
};

void cVisuResidHom::AddPair(const Pt2dr & aP1,const Pt2dr & aP2)
{
    ElSeg3D aSeg1 = mCam1->Capteur2RayTer(aP1);
    ElSeg3D aSeg2 = mCam2->Capteur2RayTer(aP2);

    Pt3dr aPInter =  aSeg1.PseudoInter(aSeg2) ; 

    Pt3dr aPI2 = aPInter + aSeg2.TgNormee() * 1e-5;

    Pt2dr aQA = mCam1->Ter2Capteur(aPInter);
    Pt2dr aQB = mCam1->Ter2Capteur(aPI2);
    
    Pt2dr aDirEpi = vunit(aQB-aQA);

    Pt2dr aDif = (aP1- aQA) / aDirEpi;

    std::cout << "DIFFFF = " << aDif << "\n";

    mPlyC.AddSphere(Pt3di(255,0,0),Pt3dr(aP1.x,aP1.y,aDif.y*1000),5,3);
    
}

cVisuResidHom::cVisuResidHom
(
      cBasicGeomCap3D * aCam1,
      cBasicGeomCap3D * aCam2,
      ElPackHomologue & aPack,
      const std::string & aPrefOut
) :
   mCam1    (aCam1),
   mCam2    (aCam2),
   mPack    (aPack),
   mPrefOut (aPrefOut),
   mResol   (10.0),
   mSzIm1   (aCam1->SzBasicCapt3D()),
   mSzResol (round_up(Pt2dr(mSzIm1) / mResol))
{
     double aStep = 20;
     for (double aX =0 ; aX <mSzIm1.x ; aX+= aStep)
     {
         for (double aY =0 ; aY <mSzIm1.y ; aY+= aStep)
         {
              mPlyC.AddPt(Pt3di(128,128,128),Pt3dr(aX,aY,0));
         }
     }


     for (ElPackHomologue::iterator itP=mPack.begin() ; itP!=mPack.end() ; itP++)
     {
         AddPair(itP->P1(),itP->P2());
     }

     mPlyC.PutFile("Test.ply");
}


int VisuResiduHom(int argc,char ** argv)
{
    std::string aIm1,aIm2,Aero,aSetHom;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aIm1,"Name first image")
                    << EAMC(aIm2,"Name first image")
                    << EAMC(Aero,"Orientation", eSAM_IsExistDirOri),
        LArgMain()
                    << EAM(aSetHom,"SH",true,"Set Homologue")
    );

     std::string aDir,aLocIm1;
     SplitDirAndFile(aDir,aLocIm1,aIm1);

     StdCorrecNameOrient(Aero,aDir);
     StdCorrecNameHomol(aSetHom,aDir);

     cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

     cBasicGeomCap3D * aCam1 = aICNM->StdCamGenOfNames(Aero,aIm1);
     cBasicGeomCap3D * aCam2 = aICNM->StdCamGenOfNames(Aero,aIm2);

     std::string aNameH = aICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+aSetHom+"@dat",aIm1,aIm2,true);
      
     ElPackHomologue  aPack = ElPackHomologue::FromFile(aNameH);

     cVisuResidHom aVRH(aCam1,aCam2,aPack,"TestVisu/");

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
