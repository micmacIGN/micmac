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

#include "general/all.h"
#include "private/all.h"
#include "Traj_Aj.h"

using namespace NS_AJ;

cAppli_Traj_AJ::cAppli_Traj_AJ(cResultSubstAndStdGetFile<cParam_Traj_AJ> aP2) :
     mParam   (*aP2.mObj),
     mICNM    (aP2.mICNM),
     mDC      (aP2.mDC)
{
    InitImages();
    InitLogs();
    InitAppuis();
    TxtExportProjImage();
}

// TEST COMMIT

     //  =============== EXPORT PROJ IMAGE =================

void cAppli_Traj_AJ::TxtExportProjImage()
{
   for 
   (
       std::list<cTrAJ2_ExportProjImage>::const_iterator itEPI=mParam.TrAJ2_ExportProjImage().begin();
       itEPI!=mParam.TrAJ2_ExportProjImage().end();
       itEPI++
   )
   {
      TxtExportProjImage(*itEPI);
   }
}

     //  =============== APPUIS =================

void cAppli_Traj_AJ::InitAppuis()
{
    for 
    (
        std::list<cTrAJ2_ConvertionAppuis>::const_iterator itI = mParam.TrAJ2_ConvertionAppuis().begin();
        itI != mParam.TrAJ2_ConvertionAppuis().end();
        itI++
    )
    {
       InitOneAppuis(*itI);
    }
}

void  cAppli_Traj_AJ::InitOneAppuis(const cTrAJ2_ConvertionAppuis &anAp)
{
   mDicApp[anAp.Id()] = new cTAj2_LayerAppuis(*this,anAp);
}


     //  =============== MATCH =================

void cAppli_Traj_AJ::DoMatch()
{
   for 
   (
        std::list<cTrAJ2_SectionMatch>::iterator itM=mParam.TrAJ2_SectionMatch().begin();
        itM !=mParam.TrAJ2_SectionMatch().end();
        itM++
   )
   {
       DoOneMatch(*itM);
   }
}

     //  =============== IMAGES =================

void cAppli_Traj_AJ::InitImages()
{
    for 
    (
        std::list<cTrAJ2_SectionImages>::const_iterator itI = mParam.TrAJ2_SectionImages().begin();
        itI != mParam.TrAJ2_SectionImages().end();
        itI++
    )
    {
       InitOneLayer(*itI);
    }
}

void cAppli_Traj_AJ::InitOneLayer(const cTrAJ2_SectionImages & aSI)
{
    cTAj2_OneLayerIm * aLay = new cTAj2_OneLayerIm(*this,aSI);
    mLayIms.push_back(aLay);
    mDicLIms[aSI.Id()] = aLay;

    const std::vector<std::string> *aV = mICNM->Get(aSI.KeySetIm());
    for (int aK=0 ; aK<int(aV->size()) ; aK++)
    {
         aLay->AddIm((*aV)[aK]);
    }
    aLay->Finish();
}

cTAj2_OneLayerIm * cAppli_Traj_AJ::ImLayerOfId(const std::string & anId)
{
    return   GetEntreeNonVide(mDicLIms,anId,"Layer Im of Id");
}

bool cAppli_Traj_AJ::TraceImage(const cTAj2_OneImage & anIm) const
{
   return     mParam.TraceImages().IsInit()
          &&  mParam.TraceImages().Val()->Match(anIm.Name());
}

bool cAppli_Traj_AJ::TraceLog(const cTAj2_OneLogIm & aLog) const
{
   return     mParam.TraceLogs().IsInit()
          &&  mParam.TraceLogs().Val()->Match(ToString(aLog.KLine()));
}

     //  =============== LOGS  =================

void cAppli_Traj_AJ::InitLogs(const cTrAJ2_SectionLog & aSL)
{
   cTAj2_LayerLogIm * aLog = new cTAj2_LayerLogIm(*this,aSL);
   mLayLogs.push_back(aLog),
   mDicLogs[aSL.Id()] = aLog;
}

void cAppli_Traj_AJ::InitLogs()
{
    for 
    (
        std::list<cTrAJ2_SectionLog>::const_iterator itI = mParam.TrAJ2_SectionLog().begin();
        itI != mParam.TrAJ2_SectionLog().end();
        itI++
    )
    {
       cAppli_Traj_AJ::InitLogs(*itI);
    }
}

cTAj2_LayerLogIm * cAppli_Traj_AJ::LogLayerOfId(const std::string & anId)
{
    return   GetEntreeNonVide(mDicLogs,anId,"Layer Log of Id");
}


     //  =============== GLOBAL  =================

void cAppli_Traj_AJ::DoAll()
{
    DoMatch();
}

const std::string& cAppli_Traj_AJ::DC()
{
   return mDC;
}

cInterfChantierNameManipulateur * cAppli_Traj_AJ::ICNM()
{
   return mICNM;
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
