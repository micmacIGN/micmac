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

#include "StdAfx.h"



/********************************************************************/
/*                                                                  */
/*         cExo_SimulTieP                                           */
/*                                                                  */
/********************************************************************/
class cAppliSimulTieP;
class cIma_TieP;


class cIma_TieP
{
    public:
     
        cIma_TieP(cAppliSimulTieP&,tSomAWSI &);
        void ProjP(const Pt3dr & aP);

        Pt2dr mCurP;
        bool mOkP;
        double mCurRR;


    // private :
        cAppliSimulTieP & mAppli;
        std::string       mNameIm;
        CamStenope *      mCam;
};

class cCmpPtrI
{
   public :
      bool operator () (cIma_TieP * aI1,cIma_TieP * aI2)
      {
          return aI1->mCurRR < aI2->mCurRR;
      }
};

typedef  std::pair<cIma_TieP *,cIma_TieP *> tPairIm;
typedef  std::map<tPairIm,ElPackHomologue> tMapH;


class cAppliSimulTieP : public cAppliWithSetImage
{
    public :

        cAppliSimulTieP(int argc, char** argv);
       
    //private :

        double             mTiePNoise;
        std::string         mNameMnt;
        cElNuage3DMaille *  mMNT;

        Pt2di               mSzMNT;
        std::vector<cIma_TieP *> mVIms;
        std::map<std::pair<cIma_TieP *,cIma_TieP *> ,ElPackHomologue> mMapH;
};
/*
*/





cAppliSimulTieP::cAppliSimulTieP(int argc, char** argv):
    cAppliWithSetImage (argc-1,argv+1,0),
    mTiePNoise  (2.0)
{
  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pattern)")
                    << EAMC(mOri,"Orientation")
                    << EAMC(mNameMnt,"Name of DSM"),
        LArgMain()  << EAM(mTiePNoise,"TPNoise",true,"Noise on Tie Points")
   );

  
   std::cout << "Nb Image " << mDicIm.size() << "]\n";

    for (int aKIm=0 ;aKIm<int(mVSoms.size()) ; aKIm++)
    {
        mVIms.push_back(new cIma_TieP(*this,*mVSoms[aKIm]));
    }
    mMNT = cElNuage3DMaille::FromFileIm(mDir+mNameMnt);

   std::cout << "Sz Geom " << mMNT->SzGeom() << "\n";

    mSzMNT =  mMNT->SzGeom();
    int aStep = 3;
    int aMultMax = 6;

    for (int anX0 = 0 ; anX0 <mSzMNT.x ; anX0+=aStep)
    {
       int anX1 = ElMin(anX0+aStep,mSzMNT.x);
       for (int anY0 = 0 ; anY0 <mSzMNT.x ; anY0 +=aStep)
       {
           int anY1 = ElMin(anY0+aStep,mSzMNT.y);
           Box2di aBox(Pt2di(anX0,anY0),Pt2di(anX1,anY1));
           Pt2di  aPRan = round_ni(aBox.RandomlyGenereInside());
           if (mMNT->IndexHasContenu(aPRan))
           {
               int aNbOk = 0;
               Pt3dr aPTer = mMNT->PtOfIndex(aPRan);

               std::vector<cIma_TieP *> aVSel;
	       for (int aKIm=0 ;aKIm<int(mVIms.size()) ; aKIm++)
               {
                    cIma_TieP & anI = *(mVIms[aKIm]);
                    anI.ProjP(aPTer);
                    if (anI.mOkP)
                    {
                       aNbOk++;
                       aVSel.push_back(&anI);
                       anI.mCurRR = NRrandom3();
                     }
               }
               if (int(aVSel.size()) >= 2)
               {
                  cCmpPtrI aCmp;
                  std::sort(aVSel.begin(),aVSel.end(),aCmp);
                  int aNbMul = ElMax(2,round_ni(aMultMax * ElSquare(NRrandom3())));
                  while (int(aVSel.size()) > aNbMul) aVSel.pop_back();
                  std::cout << "MULTIPLICITE " << aNbOk << " =>" << aVSel.size()<< "\n";
                  for (int aK1=0 ; aK1<int(aVSel.size()) ; aK1++)
                  {
                      for (int aK2=0 ; aK2<int(aVSel.size()) ; aK2++)
                      {
                         if ((aK1 != aK2) && (NRrandom3() < 0.75))
                         {
                             tPairIm aPair;
                             aPair.first = aVSel[aK1];
                             aPair.second = aVSel[aK2];
                             Pt2dr aP1 = aVSel[aK1]->mCurP;
                             Pt2dr aP2 = aVSel[aK2]->mCurP;
                             mMapH[aPair].Cple_Add(ElCplePtsHomologues(aP1,aP2));
                         }
                      }
                  }
               }
           }
       }
    }

    std::string aKey = "NKS-Assoc-CplIm2Hom@Simul@dat";

    for (tMapH::iterator itM=mMapH.begin(); itM!=mMapH.end() ; itM++)
    {
         cIma_TieP * aIm1 = itM->first.first;
         cIma_TieP * aIm2 = itM->first.second;
         std::string aNameH = mICNM->Assoc1To2(aKey,aIm1->mNameIm,aIm2->mNameIm,true);
          itM->second.StdPutInFile(aNameH);
          std::cout << aNameH << "\n";
    }
}


cIma_TieP::cIma_TieP(cAppliSimulTieP& anAppli,tSomAWSI & aSom) :
   mAppli (anAppli),
   mNameIm (aSom.attr().mIma->mNameIm),
   mCam   (aSom.attr().mIma->mCam)
{
}

void cIma_TieP::ProjP(const Pt3dr & aPTer)
{
  Pt2dr aNoise(NRrandC(),NRrandC());

  mCurP = mCam->R3toF2(aPTer) + aNoise * mAppli.mTiePNoise;
  mOkP = mCam->IsInZoneUtile(mCurP);

}

/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

int ExoSimulTieP_main(int argc, char** argv)
{
   cAppliSimulTieP anAppli(argc,argv);

   return EXIT_SUCCESS;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
