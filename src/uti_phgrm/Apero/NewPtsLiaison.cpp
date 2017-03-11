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
#include "../TiepTri/MultTieP.h"


class cCam_NewBD
{
    public :
      cCam_NewBD(cGenPoseCam *);
      cGenPoseCam * mCam;
};


class cCompile_BDD_NewPtMul
{
    public :
         cCompile_BDD_NewPtMul (const cBDD_NewPtMul &,cSetTiePMul *);
         const cBDD_NewPtMul & CBN() const;
         cSetTiePMul *         SetPM() ;
    private :
         cBDD_NewPtMul               mCBN;
         cSetTiePMul *               mSetPM;
};


/**************************************************/
/*                                                */
/*                                                */
/*                                                */
/**************************************************/

cCam_NewBD::cCam_NewBD(cGenPoseCam * aCam) :
    mCam (aCam)
{
}


/**************************************************/
/*                                                */
/*              cCompile_BDD_NewPtMul             */
/*                                                */
/**************************************************/

cCompile_BDD_NewPtMul::cCompile_BDD_NewPtMul (const cBDD_NewPtMul & aCBN,cSetTiePMul * aSet) :
    mCBN   (aCBN),
    mSetPM (aSet)
{
}

const cBDD_NewPtMul & cCompile_BDD_NewPtMul::CBN() const
{
   return mCBN;
}

cSetTiePMul *   cCompile_BDD_NewPtMul::SetPM() 
{
   return mSetPM;
}

/**************************************************/
/*                                                */
/*                                                */
/*                                                */
/**************************************************/


void cAppliApero::InitNewBDL()
{
    for 
    (
         std::list<cBDD_NewPtMul>::const_iterator itBDN=mParam.BDD_NewPtMul().begin() ; 
         itBDN!=mParam.BDD_NewPtMul().end() ; 
         itBDN++
    )
    {
        InitNewBDL(*itBDN);
    }
}


void cAppliApero::InitNewBDL(const cBDD_NewPtMul & aBDN)
{
     if (mDicoNewBDL[aBDN.Id()] != 0)
     {
         std::cout << "For Id = " << aBDN.Id() << "\n";
         ELISE_ASSERT(false,"cAppliApero::InitNewBDL multiple use of id in BDD_NewPtMul");
     }
     const std::vector<std::string> *  aSTP= cSetTiePMul::StdSetName(mICNM,aBDN.SH(),false);
     if (aSTP->size()==0) return;

     static std::vector<std::string>  * aVNameFilter=0;
     if (aVNameFilter==0)
     {
         aVNameFilter = new std::vector<std::string>; 
         for (int aKP=0 ; aKP<int(mVecGenPose.size()) ; aKP++)
         {
              const std::string & aName = mVecGenPose[aKP]->Name();
              aVNameFilter->push_back(aName);
              // std::cout << "gggggGggggg  " << aName << "\n";
         }
     }

    cSetTiePMul * aSet = cSetTiePMul::FromFiles(*aSTP,aVNameFilter);

    for (int aKP=0 ; aKP<int(mVecGenPose.size()) ; aKP++)
    {
        cCelImTPM *  aCBN = aSet->CelFromName(mVecGenPose[aKP]->Name());
        if (aCBN)
        {
           aCBN->SetVoidData(new cCam_NewBD(mVecGenPose[aKP]) );
        }
    }
    

    cCompile_BDD_NewPtMul * aComp = new cCompile_BDD_NewPtMul(aBDN,aSet);

    mDicoNewBDL[aBDN.Id()] = aComp;
    
}

bool cAppliApero::CDNP_InavlideUse_StdLiaison(const std::string & aName)
{

   cCompile_BDD_NewPtMul * aCDN = CDNP_FromName(aName);

   return (aCDN!=0) && (aCDN->CBN().SupressStdHom() );
}

cCompile_BDD_NewPtMul * cAppliApero::CDNP_FromName(const std::string & aName)
{
    std::map<std::string,cCompile_BDD_NewPtMul *>::iterator anIt = mDicoNewBDL.find(aName);

    if (anIt != mDicoNewBDL.end()) return anIt->second;
    return 0;
}

void  cAppliApero::CDNP_Compense(cSetPMul1ConfigTPM* aConf,cSetTiePMul* aSet,const cObsLiaisons & anObsOl)
{
   const std::vector<int> & aVIdIm = aConf->VIdIm();
   int aNbIm = aConf->NbIm();
   int aNbPts = aConf->NbPts();
   std::vector<cGenPoseCam *> aVCam;
   std::vector<cGenPDVFormelle *> aVGPdvF;

   for (int aKIdIm = 0 ; aKIdIm<aNbIm ; aKIdIm++)
   {
      cCelImTPM * aCel = aSet->CelFromInt(aVIdIm[aKIdIm]);
      cCam_NewBD * aCamNBD = static_cast<cCam_NewBD *>(aCel->GetVoidData());
      cGenPoseCam * aCamGen = aCamNBD->mCam;

      aVCam.push_back(aCamGen);
      aVGPdvF.push_back(aCamGen->PDVF());

      // std::cout << "CAMMM NAME " << aCam->mCam->Name() << "\n";
   }

   std::cout << "AAAAAAAAaaa\n";
   mGlobManiP3TI->SubstInitWithArgs(aVGPdvF,0,true);
   std::cout << "Bbbbbbbbbb\n";
   
   // cManipPt3TerInc

   for (int aKp=0 ; aKp<aNbPts ; aKp++)
   {
       std::vector<Pt2dr>    aVPt;
       std::vector<ElSeg3D>  aVSeg;
       for (int aKIm=0 ; aKIm <aNbIm ; aKIm++)
       {
           Pt2dr aPt = aConf->Pt(aKp,aKIm);
           aVPt.push_back(aPt);
           ElSeg3D aSeg =  aVCam[aKIm]->GenCurCam()->Capteur2RayTer(aPt);
           aVSeg.push_back(aSeg);
       }
       bool Ok;
       Pt3dr aPInt = InterSeg(aVSeg,Ok);
       if (Ok)
       {
           double aDist=0;
           for (int aKIm=0 ; aKIm <aNbIm ; aKIm++)
           {
               Pt2dr aPt = aConf->Pt(aKp,aKIm);
               Pt2dr aQ  =  aVCam[aKIm]->GenCurCam()->Ter2Capteur(aPInt);
               aDist += euclid(aPt-aQ);
           }
           // std::cout << "D=" << aDist/aNbIm << "\n";
       }
       else
       {
            // std::cout << "------------------Not ok---------------------\n";
       }
   }


   // std::cout << "-----------------------------\n";
}


void cAppliApero::CDNP_Compense(const std::string & anId,const cObsLiaisons & anObsOl)
{


    // std::cout << "cAppliApero::CDNP_Compe " <<  mGlobManiP3TI << "\n"; getchar();

//  new cManipPt3TerInc(aVCF[0]->Set(),anEqS,aVCF)


     cCompile_BDD_NewPtMul * aCDN = CDNP_FromName(anId);

     if (aCDN==0)
        return; 
     
    cSetTiePMul *  aSetPM = aCDN->SetPM() ;
    const std::vector<cSetPMul1ConfigTPM *> &  aVPM = aSetPM->VPMul();

    std::cout << "cAppliApero::CDNP_Compens:NBCONFIG " <<  aVPM.size() << "\n";

    for (int aKConf=0 ; aKConf<int(aVPM.size()) ; aKConf++)
    {
        CDNP_Compense(aVPM[aKConf],aSetPM,anObsOl);
    }
    std::cout <<"FFFFFF ==== \n";
    getchar();
}

/*
void cAppliApero::ObsNewBDL(const cObsLiaisons & anObsL)
{
}

void cAppliApero::ObsNewBDL()
{
}

*/


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
