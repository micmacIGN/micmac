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


template  class ElGraphe<NS_ParamApero::cPoseCam*,NS_ParamApero::cAttrArcPose>;
template  class ElTabDyn<ElSom<NS_ParamApero::cPoseCam*, NS_ParamApero::cAttrArcPose> >;


/*******************************************/
/*                                         */
/*          cAttrArcPose                   */
/*                                         */
/*******************************************/

cAttrArcPose::cAttrArcPose() :
   mPds (0.0),
   mNb  (0)
{
}

double & cAttrArcPose::Pds() { return mPds; }
double  cAttrArcPose::Pds() const { return mPds; }

int & cAttrArcPose::Nb() { return mNb; }
int  cAttrArcPose::Nb() const { return mNb; }

/*******************************************/
/*                                         */
/*          cAppliApero                    */
/*                                         */
/*******************************************/

bool BugBestCam = false;


void  cAppliApero::ConstructMST
      (
          const std::list<std::string> & aLNew,
          const cPoseCameraInc &   aPCI
      )
{
bool MST_DEBUG = false &&  MPD_MM();



   if (mNbRotPreInit==0)
   {
      ELISE_ASSERT(false,"Cannot built MST on empty base\n");
   }
   ELISE_ASSERT
   (
       aPCI.PoseFromLiaisons().IsInit(),
       "MST requires init by Liaisons"
   );
   const cPoseFromLiaisons & aPFL  = aPCI.PoseFromLiaisons().Val();
   std::string aIdBd = aPFL.LiaisonsInit().begin()->IdBD();


   const cMEP_SPEC_MST & aMST  = aPCI.MEP_SPEC_MST().Val();
   bool Show = aMST.Show().Val() ;
   int  aNbPtsMin  = aMST.MinNbPtsInit().Val();
   double anExpDist = aMST.ExpDist().Val();
   double anExpNb   = aMST.ExpNb().Val();
   if (Show)
   {
       std::cout << "MST  " << aLNew.size() << "\n";
   }
   if (MST_DEBUG)
   {
       for (const auto & aN : aLNew )
           std::cout <<  " ---  NNeew " << aN<< "\n";
   }


   bool OnInit = aMST.MontageOnInit().Val();


   //   aVCible  : ceux  qui doivent etre atteints

   std::vector<cPoseCam *> aVCible;

   // Initialisation de aVGerms + calcul de flags
   for (int aK=0 ; aK<int(mVecPose.size()) ; aK++)
   {
       if (mVecPose[aK]->PreInit())
       {
if (MST_DEBUG)
{
    std::cout << "IIIIIII " << mVecPose[aK]->Name() << "\n";
}
            ELISE_ASSERT
            (
                !BoolFind(aLNew,mVecPose[aK]->Name()),
                 "MST : New deja init "
            );
       }
   }
   for 
   (
       std::list<std::string>::const_iterator itS=aLNew.begin();
       itS != aLNew.end();
       itS++
   )
   {
       aVCible.push_back(PoseFromName(*itS));
   }

   int aNbC = (int)aVCible.size(); 

   cPoseCam * aLastRigidSeed;
   std::vector<cPoseCam *> aVRigid2Init;
   bool UseBloc = false;
   if (aMST.MSTBlockRigid().IsInit())
   {
        UseBloc = true;
        PreInitBloc(aMST.MSTBlockRigid().Val());
   }
if (MST_DEBUG)
{
   std::cout << "MM " << MPD_MM() << " BLoc=" << UseBloc << "\n";
   getchar();
}

   // A chaque iteration on va affecter un sommet
   for (int aTimes=0 ; aTimes<aNbC ; aTimes++)
   {
if (MST_DEBUG) std::cout << "aVRigid2Init SIIZE " << aVRigid2Init.size() << "\n";
       cPoseCam  * aBestCam = 0;
       int aNbRotPreInit = -1;
       std::vector<cPoseCam *>  aVBestC;

       if (aVRigid2Init.empty())
       {
           cObsLiaisonMultiple * aBestPack=0; GccUse(aBestPack);
           double           aPdsMax = -1e40;
           // Recherche du sommet a affecter
           for (int aKC=0 ; aKC<aNbC ; aKC++)
           {
               cPoseCam * aPcK = aVCible[aKC];

               if (! aPcK->PreInit())
               {
                  cObsLiaisonMultiple * anOLM = PackMulOfIndAndNale(aIdBd,aPcK->Name());
                  anOLM->CompilePose();

                  bool GotPMul;
                  double aPds = anOLM->StdQualityZone(ZuUseInInit(),OnInit,aNbPtsMin,anExpDist,anExpNb,GotPMul);

if (MST_DEBUG&& (aTimes==0))
{
   std::cout <<  "MSTPDs " << aPds  << " " << aPcK->Name() << " " <<  anOLM->NbRotPreInit()  << "\n";
}

                  if (aPds> 0)
                  {
                      if (aPds>aPdsMax)
                      {
                          aPdsMax = aPds;
                          aBestCam = aPcK;
                          aBestPack= anOLM;
                          aNbRotPreInit = anOLM->NbRotPreInit();
                      }
                  }
               }
           }
if (MST_DEBUG && aBestCam)
{
    std::cout << "aBestCamwwww " << aBestCam->Name() << " Time " << aTimes << "\n";
    if (aTimes==0) getchar();
}

           // On calcule les pere-mere
           if (aBestCam != 0)
           {
               std::vector<double> aVCost;
               cObsLiaisonMultiple * anOLM = PackMulOfIndAndNale(aIdBd,aBestCam->Name());
               aVBestC = anOLM->BestPoseInitStd
                     (
                           ZuUseInInit(),
                           OnInit,
                           aVCost,
                           //aNbPtsMin,
                           0,
                           anExpDist,
                           anExpNb
                     );
           }
           else  // Si pas de meilleure cam, pb de connexion => erreur
           {
                for (int aKC=0 ; aKC<aNbC ; aKC++)
                {
                   cPoseCam * aPcK = aVCible[aKC];
                   if (! aPcK->PreInit())
                   {
                      std::cout << "  === NON INIT : " << aPcK->Name() << "\n";
                   }
                }
                ELISE_ASSERT(false,"aBestCam==0");
           }
      }
      else
      {
          aNbRotPreInit = 1;
          aVBestC.push_back(aLastRigidSeed);
          aBestCam = aVRigid2Init.back();
      }



       if (aVRigid2Init.empty() && (int(aVBestC.size()) < ElMin(2,aNbRotPreInit)))
       {
          if (aBestCam !=0)
          {
              std::cout << "BEST CAM TESTED " << aBestCam->Name()  
                        << " NB Annc " << aVBestC.size() << "\n";
          }
          else
          {
              std::cout << " NO BEST CAM\n";
          }


          for (int aKC=0 ; aKC<aNbC ; aKC++)
          {
             cPoseCam * aSK = aVCible[aKC];
             if (! aSK->PreInit())
             {
                std::cout << "UN-Connected : " << aSK->Name() << "\n";
             }
          }
          if (aNbRotPreInit <  aMST.NbInitMinBeforeUnconnect().Val())
          {
             ELISE_ASSERT ( false, "Connection pb in MST");
          }
          aTimes= aNbC-1;  //Sortie de boucle 
       }
       else
       {

          for (int aKC=0 ; aKC<int(aVBestC.size()) ; aKC++)
          {
              if (aKC==0)
                 aBestCam->SetPoseInitMST1(aVBestC[aKC]);
              else
                 aBestCam->SetPoseInitMST2(aVBestC[aKC]);
              aBestCam->UpdateHeriteProf2Init(*(aVBestC[aKC]));
          }

      

          if (Show)
          {
             std::cout << aBestCam->Name() 
                    << "[" << aBestCam->Prof2Init() << "]"
                    << " ; Pere : " << aVBestC[0]->Name();
              if (aVBestC.size()>1)
                  std::cout << " ; Mere : " << aVBestC[1]->Name();
              std::cout << "\n";
          }
          aBestCam->SetNbPosOfInit(aNbRotPreInit);
          aBestCam->DoInitIfNow();
          if (aBestCam)
          {
              mProfMax = ElMax(mProfMax,aBestCam->Prof2Init());
          }
        }

 
        if (aVRigid2Init.empty())
        {
            if (UseBloc  && aBestCam && (aBestCam->GetSRI(true)==nullptr))
            {
                 cPreCompBloc * aPCB = aBestCam->GetPreCompBloc(true);
                 if (aPCB)
                 {
                      ElRotation3D  aR0 =  aBestCam->GetPreCB1Pose(false)->mRot;
                      for (auto & aPC : aPCB->mGrp)
                      {
                          if (aPC != aBestCam)
                          {
                              ElRotation3D  aL0 =  aPC->GetPreCB1Pose(false)->mRot;
                              aPC->SetSRI(new cStructRigidInit(aBestCam,aR0.inv() * aL0));
                              aVRigid2Init.push_back(aPC);
                          }
                      }
                 }
                 if (!aVRigid2Init.empty())
                 {
                     aLastRigidSeed = aBestCam;
                 }
            }
         }
         else
         {
            // Forcement pas vide si on est la
            aVRigid2Init.pop_back();
            if (aVRigid2Init.empty())
               aLastRigidSeed = 0;
         }
   }


if (MST_DEBUG) { std::cout << "ENDDDDDDDD MST \n"; getchar(); }
   // aSInit.pushlast(anAppli.PoseFromName(*itI)->Som());
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
