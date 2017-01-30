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
/*                      cExo_SimulTieP                              */
/*                                                                  */
/********************************************************************/
class cAppliSimulTieP;
class cIma_TieP;


class cIma_TieP
{
    public:
        cIma_TieP(cAppliSimulTieP&,tSomAWSI &);
        void ProjP(const Pt3dr & aP);
 // private :
        Pt2dr mCurP;
        bool mOkP;
        double mCurRR;  //current random rank
        cAppliSimulTieP & mAppli;
        std::string       mNameIm;
        CamStenope *      mCam;
};

class cCmpPtrI      // order images depending on their affected rank
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



/********************************************************************/
/*                                                                  */
/*                      cAppliSimulTieP                             */
/*                                                                  */
/********************************************************************/

cAppliSimulTieP::cAppliSimulTieP(int argc, char** argv):    // cAppliWithSetImage is used to initialize the images
    cAppliWithSetImage (argc-1,argv+1,0),		// it has to be used without the first argument (name of the command)
    mTiePNoise  (2.0)    // default value for the noise in tie points
{
  ElInitArgMain
  (     // initialisation of the arguments
        argc,argv,
        LArgMain()  << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)")        // EAMC = mandatory arguments
                    << EAMC(mOri,"Orientation")
                    << EAMC(mNameMnt,"Name of DSM"),
        LArgMain()  << EAM(mTiePNoise,"TPNoise",true,"Noise on Tie Points")    // EAM = optionnal argument
   );

   std::cout << "Nb Image " << mDicIm.size() << "]\n";

    for (int aKIm=0 ;aKIm<int(mVSoms.size()) ; aKIm++)       // mVSoms = image list
    {
        mVIms.push_back(new cIma_TieP(*this,*mVSoms[aKIm]));    // images loaded in a vector
    }
    mMNT = cElNuage3DMaille::FromFileIm(mEASF.mDir+mNameMnt); // loading the DSM

    std::cout << "Sz Geom " << mMNT->SzGeom() << "\n";       // DSM size

    mSzMNT =  mMNT->SzGeom();
    int aStep = 3;          // we will browse the DSM using a box to pick points. aStep defines the size of the box
    int aMultMax = 6;       // defines the maximum amount of image in which one tie point can be seen

    for (int anX0 = 0 ; anX0 <mSzMNT.x ; anX0+=aStep)
    {
       int anX1 = ElMin(anX0+aStep,mSzMNT.x);            // ElMin(a,b) to pick the lowest -> to avoid to pick point outside of the DSM
       for (int anY0 = 0 ; anY0 <mSzMNT.y ; anY0 +=aStep)       // [anX0,anY0] = lower left corner of the box
       {
           int anY1 = ElMin(anY0+aStep,mSzMNT.y);               // [anX1,anY1] = upper right corner of the box
           Box2di aBox(Pt2di(anX0,anY0),Pt2di(anX1,anY1));
           Pt2di  aPRan = round_ni(aBox.RandomlyGenereInside());        // randomly pick a point in the box
           if (mMNT->IndexHasContenu(aPRan))             // if there is a point in the DSM at that place
           {
               int aNbOk = 0;
               Pt3dr aPTer = mMNT->PtOfIndex(aPRan);    // get the 3d coordinate (ground geometry) of that point

               std::vector<cIma_TieP *> aVSel;
               for (int aKIm=0 ;aKIm<int(mVIms.size()) ; aKIm++)       // browse the image list
               {
                    cIma_TieP & anI = *(mVIms[aKIm]);       // load image
                    anI.ProjP(aPTer);             // project the point in the current image
                    if (anI.mOkP)                  // if the 3d point can be projected
                    {
                       aNbOk++;
                       aVSel.push_back(&anI);       // list of images in which the current point can be seen
                       anI.mCurRR = NRrandom3();            // assign a rank to the image (will be used further to randomly simulate hidden parts/undetections)
                     }
               }
               if (int(aVSel.size()) >= 2)  // if the point is visible in at least 2 images
               {
                  cCmpPtrI aCmp;
                  std::sort(aVSel.begin(),aVSel.end(),aCmp);    // order the list of images (in which the point is seen) by their rank (random order)
                  int aNbMul = ElMax(2,round_ni(aMultMax * ElSquare(NRrandom3())));
                  while (int(aVSel.size()) > aNbMul) aVSel.pop_back();      // if the point is seen in too many images, reduce the list
                  std::cout << "MULTIPLICITE " << aNbOk << " =>" << aVSel.size()<< "\n";
                  for (int aK1=0 ; aK1<int(aVSel.size()) ; aK1++)
                  {         // browse the list (reduced) of images in which the point is seen
                      for (int aK2=0 ; aK2<int(aVSel.size()) ; aK2++)
                      {
                         if ((aK1 != aK2) && (NRrandom3() < 0.75))      // for 2 different images, 3 times on 4, build the dictionnary of image names and point coordinates
                         {
                             tPairIm aPair;     //  image pair
                             aPair.first = aVSel[aK1];  // img1
                             aPair.second = aVSel[aK2]; // img2
                             Pt2dr aP1 = aVSel[aK1]->mCurP; // pt_img1
                             Pt2dr aP2 = aVSel[aK2]->mCurP; // pt_img2
                             mMapH[aPair].Cple_Add(ElCplePtsHomologues(aP1,aP2));       // save : im1 im2 pt_im1 pt_im2
                         }
                      }
                  }
               }
           }
       }
    }

    std::string aKey = "NKS-Assoc-CplIm2Hom@Simul@dat";     // association key, here results will be saved in a folder "HomolSimul", as .dat files

    for (tMapH::iterator itM=mMapH.begin(); itM!=mMapH.end() ; itM++)       // browse the dictionnary
    {
         cIma_TieP * aIm1 = itM->first.first;       // img1
         cIma_TieP * aIm2 = itM->first.second;      // img2
         std::string aNameH = mEASF.mICNM->Assoc1To2(aKey,aIm1->mNameIm,aIm2->mNameIm,true);      // name of the file to save ("HomolSimul/Pastis....dat")
         itM->second.StdPutInFile(aNameH);      // save pt_im1 & pt_im2 in that file
         std::cout << aNameH << "\n";
    }
}



/********************************************************************/
/*                                                                  */
/*                          cIma_TieP                               */
/*                                                                  */
/********************************************************************/

cIma_TieP::cIma_TieP(cAppliSimulTieP& anAppli,tSomAWSI & aSom) :
   mAppli (anAppli),
   mNameIm (aSom.attr().mIma->mNameIm),
   mCam    (aSom.attr().mIma->CamSNN())
{
}

void cIma_TieP::ProjP(const Pt3dr & aPTer)      // function used to project 3d points in the images, and to add noise in the coordinates
{
  Pt2dr aNoise(NRrandC(),NRrandC());        // creation of a white noise (centered on [-1;1])

  mCurP = mCam->R3toF2(aPTer) + aNoise * mAppli.mTiePNoise;     // projection of the point, noise added
  mOkP = mCam->IsInZoneUtile(mCurP);        // check if the point is in the image
}



/********************************************************************/
/*                                                                  */
/*                          Main                                    */
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

