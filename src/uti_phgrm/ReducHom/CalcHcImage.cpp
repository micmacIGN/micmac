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
#include "ReducHom.h"





/*************************************************/
/*                                               */
/*                  cImagH                       */
/*                                               */
/*************************************************/

bool cImagH::ComputeLnkHom(cLink2Img & aLnk)
{
   // const ElPackHomologue aPack=ElPackHomologue::FromFile(mAppli.Dir()+aLnk.NameH());
   const ElPackHomologue & aPack=    aLnk.Pack() ; //   ElPackHomologue::FromFile(mAppli.Dir()+aLnk.NameH());
   int aNbPts = aPack.size();
   aLnk.NbPts() = aPack.size();

   if (aNbPts < mAppli.MinNbPtH())
      return false;

   bool Ok;
   double anEcart,aQual;
   cElHomographie  aHom12 = cElHomographie::RobustInit(anEcart,&aQual,aPack,Ok,NB_RANSAC_H,90.0,1000);
   // mLnk.push_back(cLink2Img(anI2,aNameH));

   if (!Ok)
     return false;

   aLnk.Hom12() = aHom12;
   aLnk.Qual() = aQual;

   mSomQual += ElMin(aQual,mAppli.SeuilQual()) * aNbPts;
   mSomNbPts += aNbPts;

   return true;
}

void cImagH::ComputeLnkHom()
{
    tSetLinks  aNewL;
    for ( tSetLinks::iterator itL = mLnks.begin(); itL != mLnks.end(); itL++)
    {
        if (ComputeLnkHom(*(itL->second)))
           aNewL[itL->first] = itL->second ;
        else
           delete itL->second;
    }
    mLnks = aNewL;
    aNewL.clear();

    if (mSomNbPts)
    {
       mSomQual /= mSomNbPts;
       double aSeuilQual = mSomQual*mAppli.RatioQualMoy();
       for (tSetLinks::iterator itL = mLnks.begin(); itL != mLnks.end(); itL++)
       {
           bool Ok = itL->second->Qual() < aSeuilQual;
           if (Ok)
           {
              aNewL[itL->first] = itL->second ;
           }
           else
           {
               delete itL->second;
           }
           if (Ok)
           {
              std::cout
                     << "IMS " << mName << " " << itL->second->Dest()->Name()
                     << " QUAL " << itL->second->Qual() << " NB " << itL->second->NbPts()
                     << (Ok ? " " : "  ******")
                     <<  "\n";
           }
       }
    }
    mLnks = aNewL;

    std::cout << mName << " QMOY " << (mSomNbPts ? mSomQual : 1e10) << "\n\n";
}


void cImagH::VoisinsNonMarques(const std::vector<cImagH*> & aIn,std::vector<cImagH*> & aVois,int aFlagN,int aFlagT )
{
   aVois.clear();

   for (int aKS=0 ; aKS<int(aIn.size()) ; aKS++)
   {
       cImagH * aIK1 = aIn[aKS];
       for (tSetLinks::iterator itL1 = aIK1->mLnks.begin(); itL1 != aIK1->mLnks.end(); itL1++)
       {
            cImagH * aImTest  = itL1->second->Dest();
            if ((! aImTest->Marqued(aFlagN)) && (! aImTest->Marqued(aFlagT)))
            {
                aImTest->SetMarqued(aFlagT);
                aVois.push_back(aImTest);
            }
        }
    }
    for (int aKT=0 ; aKT<int(aVois.size()) ; aKT++)
        aVois[aKT]->SetUnMarqued(aFlagT);
}

void cImagH::VoisinsMarques(std::vector<cLink2Img*> & aVois,int aFlagN)
{
    aVois.clear();
    for ( tSetLinks::iterator itL = mLnks.begin(); itL != mLnks.end(); itL ++)
    {
        cImagH * aI2  = itL->second->Dest();
        if (aI2->Marqued(aFlagN))
            aVois.push_back(itL->second);
     }
}




void  cAppliReduc::QuadrReestimFromVois(std::vector<cImagH*> & aVLocIm,int aFlag)
{

    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
         cImagH * anI =  mIms[aK];
         anI->HF()->SetModeCtrl(anI->Marqued(aFlag) ? cNameSpaceEqF::eHomLibre : cNameSpaceEqF::eHomFigee);
         // anI->HF()->SetModeCtrl( cNameSpaceEqF::eHomFigee);
    }
    aVLocIm[0]->HF()->SetModeCtrl(cNameSpaceEqF::eHomFigee);
    for (int aK=0 ; aK<int(aVLocIm.size()) ; aK++)
    {
         cImagH * anI = aVLocIm[aK];
         anI->HF()->ReinitHom(anI->Hi2t());
    }
         // anI->HF()->SetModeCtrl( cNameSpaceEqF::eHomFigee);
    

    for (int aFois = 0 ; aFois<5 ; aFois++)
    {
         for (int aK=0 ; aK<int(mIms.size()) ; aK++)
         {
              cImagH * anI =  mIms[aK];
              mSetEq.AddContrainte(anI->HF()->StdContraintes(),true);
         }
         mSetEq.SetPhaseEquation();

         double aSomR=0;
         double aSomP=0;
         for (int aK=0 ; aK<int(aVLocIm.size()) ; aK++)
         {
             cImagH * anI1 =  aVLocIm[aK];
             const tSetLinks & aLL = anI1->Lnks();
             for (tSetLinks::const_iterator itL = aLL.begin(); itL != aLL.end(); itL++)
             {
                 cImagH * anI2 = itL->second->Dest();
                 if (anI2->Marqued(aFlag)) 
                 {
                     cElHomographie aH12 = itL->second->Hom12();
                     const std::vector<Pt3dr> &  anEchP1 = itL->second->EchantP1();
                     cEqHomogFormelle * anEqF = itL->second->EqHF() ;
                     for (int aK=0 ; aK<int(anEchP1.size()) ; aK++)
                     {
                         const Pt3dr & aQ3 = anEchP1[aK];
                         double aPds = aQ3.z;
                         Pt2dr aP1 (aQ3.x,aQ3.y);
                         Pt2dr aP2  = aH12.Direct(aP1);
                         Pt2dr aResidu = anEqF->StdAddLiaisonP1P2(aP1,aP2,aPds,false);
                         aSomR+=square_euclid(aResidu) * aPds;
                         aSomP+= aPds;
                     }
                 }
             }
         }
         std::cout << "RES "  <<  sqrt(aSomR/aSomP)  <<  "\n";
         mSetEq.SolveResetUpdate();
    }
    std::cout << "\n";
    for (int aK=0 ; aK<int(aVLocIm.size()) ; aK++)
    {
        cImagH * anI =  aVLocIm[aK];
        anI->Hi2t() =  anI->HF()->HomCur();
    }
}

void cAppliReduc::TestMerge_CalcHcImage()
{
    cParamMerge  aParam;
    
    cAlgoMergingRec<cImagH,cAttrLnkIm,cParamMerge> anAlgo(mIms,aParam,0);

    const std::set<tNodIm *> & aRoot = anAlgo.Roots();


    for 
    (
           std::set<tNodIm *>::const_iterator itN=aRoot.begin();
           itN!=aRoot.end();
           itN++
    )
    {
              anAlgo.Show(*itN);
    }
}


/*
*/
/*
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
