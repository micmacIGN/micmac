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

namespace NS_ParamApero
{

/***********************************************************/
/*                                                         */
/*                                                         */
/*                                                         */
/***********************************************************/


class cIBC_ImsOneTime
{
    public :
        cIBC_ImsOneTime(int aNbCam,int aNum,const std::string& aNameTime) ;
        void  AddPose(cPoseCam *, int aNum);
        cPoseCam * Pose(int aKP);

    private :

        std::vector<cPoseCam *> mCams;
        int                     mNum;
        std::string             mNameTime;
};


class cIBC_OneCam
{
      public :
          cIBC_OneCam(const std::string & ,int aNum);
          const int & Num() const;
      private :
          std::string mNameCam;
          int         mNum;
};



class cImplemBlockCam
{
    public :
         // static cImplemBlockCam * AllocNew(cAppliApero &,const cStructBlockCam,const std::string & anId);
         cImplemBlockCam(cAppliApero & anAppli,const cStructBlockCam,const std::string & anId );

         void EstimCurOri(const cEstimateOrientationInitBlockCamera &);
    private :

         cAppliApero &               mAppli;
         cStructBlockCam             mSBC;
         std::string                 mId;
         cRelEquivPose               mRelGrp;
         cRelEquivPose               mRelId;

         std::map<std::string,cIBC_OneCam *>   mName2Cam;
         std::vector<cIBC_OneCam *>            mNum2Cam;
         int                                   mNbCam;
         int                                   mNbTime;

         std::map<std::string,cIBC_ImsOneTime *> mName2ITime;
         std::vector<cIBC_ImsOneTime *>          mNum2ITime;
};

    // =================================
    //              cIBC_ImsOneTime
    // =================================

cIBC_ImsOneTime::cIBC_ImsOneTime(int aNb,int aNum,const std::string & aNameTime) :
       mCams     (aNb),
       mNum      (aNum),
       mNameTime (aNameTime)
{
}

void  cIBC_ImsOneTime::AddPose(cPoseCam * aPC, int aNum) 
{
    cPoseCam * aPC0 =  mCams.at(aNum);
    if (aPC0 != 0)
    {
         std::cout <<  "For cameras " << aPC->Name() <<  "  and  " << aPC0->Name() << "\n";
         ELISE_ASSERT(false,"Conflicting name from KeyIm2TimeCam ");
    }
    
    mCams[aNum] = aPC;
}

cPoseCam * cIBC_ImsOneTime::Pose(int aKP)
{
   return mCams.at(aKP);
}
    // =================================
    //              cIBC_OneCam 
    // =================================

cIBC_OneCam::cIBC_OneCam(const std::string & aNameCam ,int aNum) :
    mNameCam (aNameCam ),
    mNum     (aNum)
{
}

const int & cIBC_OneCam::Num() const {return mNum;}

    // =================================
    //       cImplemBlockCam
    // =================================

cImplemBlockCam::cImplemBlockCam(cAppliApero & anAppli,const cStructBlockCam aSBC,const std::string & anId) :
      mAppli (anAppli),
      mSBC   (aSBC),
      mId    (anId)
{
    const std::vector<cPoseCam*> & aVP = mAppli.VecAllPose();
   

    // On initialise les camera
    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
          cPoseCam * aPC = aVP[aKP];
          std::string aNamePose = aPC->Name();
          std::pair<std::string,std::string> aPair =   mAppli.ICNM()->Assoc2To1(mSBC.KeyIm2TimeCam(),aNamePose,true);
          std::string aNameCam = aPair.second;
          if (! DicBoolFind(mName2Cam,aNameCam))
          {

               cIBC_OneCam *  aCam = new cIBC_OneCam(aNameCam,mNum2Cam.size());
               mName2Cam[aNameCam] = aCam;
               mNum2Cam.push_back(aCam); 
          }
    }
    mNbCam  = mNum2Cam.size();

    
    // On regroupe les images prises au meme temps
    for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
    {
          cPoseCam * aPC = aVP[aKP];
          std::string aNamePose = aPC->Name();
          std::pair<std::string,std::string> aPair =   mAppli.ICNM()->Assoc2To1(mSBC.KeyIm2TimeCam(),aNamePose,true);
          std::string aNameTime = aPair.first;
          std::string aNameCam = aPair.second;
          
          cIBC_ImsOneTime * aIms =  mName2ITime[aNameTime];
          if (aIms==0)
          {
               aIms = new cIBC_ImsOneTime(mNbCam,mNum2ITime.size(),aNameTime);
               mName2ITime[aNameTime] = aIms;
               mNum2ITime.push_back(aIms);
          }
          cIBC_OneCam * aCam = mName2Cam[aNameCam];
          aIms->AddPose(aPC,aCam->Num());
    }
    mNbTime = mNum2ITime.size();
}

void cImplemBlockCam::EstimCurOri(const cEstimateOrientationInitBlockCamera &)
{
   for (int aKC=1 ; aKC<mNbCam ; aKC++)
   {
       for (int aKT=0 ; aKT<mNbTime ; aKT++)
       {
            cIBC_ImsOneTime *  aTime =  mNum2ITime[aKT];
            cPoseCam * aP0 = aTime->Pose(0);
            cPoseCam * aP1 = aTime->Pose(aKC);

            ElRotation3D  aR0toM = aP0->CurCam()->Orient().inv();
            ElRotation3D  aR1toM = aP1->CurCam()->Orient().inv();

            ElRotation3D aR1to0 = aR0toM.inv() * aR1toM;

            std::cout << "EstimCurOri " << aP0->Name() <<  " " << aP1->Name() << "\n";
            std::cout << "  " <<  aR1to0.ImAff(Pt3dr(0,0,0)) 
                              << " " << aR1to0.teta01() 
                              << " " << aR1to0.teta02() 
                              << " " << aR1to0.teta12() 
                              << "\n";
       }
   }
}

void cAppliApero::InitBlockCameras()
{
  for 
  (
        std::list<cBlockCamera>::const_iterator itB= mParam.BlockCamera().begin();
        itB!=mParam.BlockCamera().end();
        itB++
  )
  {
       std::string anId = itB->Id().ValWithDef(itB->NameFile());
       cStructBlockCam aSB = StdGetObjFromFile<cStructBlockCam>
                             (
                                 mICNM->Dir() + itB->NameFile(),
                                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                 "StructBlockCam",
                                 "StructBlockCam"
                             );
       cImplemBlockCam * aIBC = new cImplemBlockCam(*this,aSB,anId);
       mBlockCams[anId] = aIBC;
  }
}


cImplemBlockCam * cAppliApero::GetBlockCam(const std::string & anId)
{
   cImplemBlockCam* aRes = mBlockCams[anId];
   ELISE_ASSERT(aRes!=0,"cAppliApero::GetBlockCam");

   return aRes;
}

void  cAppliApero::EstimateOIBC(const cEstimateOrientationInitBlockCamera & anEOIB)
{ 
    cImplemBlockCam * aBlock = GetBlockCam(anEOIB.Id());
    aBlock->EstimCurOri(anEOIB);
}



};

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
