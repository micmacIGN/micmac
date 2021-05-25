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



void cAppliApero::ExportMesuresFromCarteProf
     (
          const cExportMesuresFromCarteProf & anEM,
          const cCartes2Export &              aC,
          cElNuage3DMaille *                  aNuage,
          const ElPackHomologue &             aPackH,
          cGenPoseCam *                       aPose2Compl,
          const std::string &                 aNameCarte
     )
{
   if (aNameCarte == aPose2Compl->Name())
      return;
   if (  
           (! anEM.KeyAssocLiaisons12().IsInit())
        && (! anEM.KeyAssocLiaisons21().IsInit())
      )
  {
     return;
  }

  cPoseCam * aPoseCarte = PoseFromName(aNameCarte);
  const CamStenope * aCSC = aPoseCarte->CurCam();
  Pt2di aSzIm = aCSC->Sz();

  ElPackHomologue aNewPack;

  for 
  (
       ElPackHomologue::const_iterator itH=aPackH.begin();
       itH!=aPackH.end();
       itH++
  )
  {
      Pt2dr aI1 = aNuage->Plani2Index(itH->P1());
      if (aNuage->IndexHasContenuForInterpol(aI1))
      {
          Pt3dr aPTer = aNuage->PtOfIndexInterpol(aI1);
          Pt2dr aP1 = aCSC->R3toF2(aPTer);
          if ((aP1.x>0) && (aP1.y>0) && (aP1.x<aSzIm.x) && (aP1.y<aSzIm.y))
          {
              aNewPack.Cple_Add(ElCplePtsHomologues(aP1,itH->P2()));
          }
      }
  }

  if (anEM.KeyAssocLiaisons12().IsInit())
  {
     std::string aName =    mDC  
                          + mICNM->Assoc1To2
                            (
                                anEM.KeyAssocLiaisons12().Val(),
                                aNameCarte,
                                aPose2Compl->Name(),
                                true
                            );
     if (anEM.LiaisonModeAdd().Val())
        aNewPack.StdAddInFile(aName);
     else
        aNewPack.StdPutInFile(aName);
  }

  if (anEM.KeyAssocLiaisons21().IsInit())
  {
     std::string aName =    mDC  
                          + mICNM->Assoc1To2
                            (
                                anEM.KeyAssocLiaisons21().Val(),
                                aPose2Compl->Name(),
                                aNameCarte,
                                true
                            );
     aNewPack.SelfSwap();
     if (anEM.LiaisonModeAdd().Val())
        aNewPack.StdAddInFile(aName);
     else
        aNewPack.StdPutInFile(aName);
  }




}

void cAppliApero::ExportMesuresFromCarteProf
     (
          const cExportMesuresFromCarteProf & anEM,
          const cCartes2Export &              aC
     )
{
    for (std::list<std::string>::const_iterator itIm1= aC.Im1().begin() ; itIm1!=aC.Im1().end() ; itIm1++)
    {
       cElRegex anAutom(aC.FilterIm2().Val(),10);
       std::string aNameN = mDC+mICNM->StdCorrect(aC.Nuage(),*itIm1,true);
       cElNuage3DMaille * aNuage = cElNuage3DMaille::FromFileIm(aNameN);
       cObsLiaisonMultiple* aPackM = PackMulOfIndAndNale(anEM.IdBdLiaisonIn(),*itIm1);

       const std::vector<cOneElemLiaisonMultiple *>  & aVELM = aPackM->VPoses();
       for (int aKP=0 ; aKP<int(aVELM.size()) ; aKP++)
       {
          cGenPoseCam * aPose2Compl = aVELM[aKP]->GenPose();
          ElPackHomologue aPackH;
       
          if (anAutom.Match(aPose2Compl->Name())
              && aPackM->InitPack(aPackH,aPose2Compl->Name())
             )
          {
             for 
             (
                std::list<std::string>::const_iterator itS=aC.ImN().begin(); 
                itS!=aC.ImN().end(); 
                itS++
             )
             {
                if (*itS != *itIm1)
                {
                    ExportMesuresFromCarteProf (anEM,aC,aNuage,aPackH,aPose2Compl,*itS);
                }
             }
             if (anEM.KeyAssocAppuis().IsInit())
             {
                 cListeAppuis1Im aXmlApp; 
                 std::string aNameRes = mDC+mICNM->Assoc1To2(anEM.KeyAssocAppuis().Val(),aPose2Compl->Name(),*itIm1,true);
                 if (anEM.AppuisModeAdd().Val())
                 {
                     if (ELISE_fp::exist_file(aNameRes))
                     {
                        aXmlApp=StdGetObjFromFile<cListeAppuis1Im>
                             (
                                 aNameRes,
                                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                 "ListeAppuis1Im",
                                 "ListeAppuis1Im"
                             );
                     }
                 }
                 aXmlApp.NameImage().SetVal(aPose2Compl->Name());
                 for 
                 (
                      ElPackHomologue::const_iterator itH=aPackH.begin();
                      itH!=aPackH.end();
                      itH++
                 )
                 {
                    Pt2dr aI1 = aNuage->Plani2Index(itH->P1());
                    if (aNuage->IndexHasContenuForInterpol(aI1))
                    {
                       cMesureAppuis aMA;
                       aMA.Im() = itH->P2();
                       aMA.Ter() = aNuage->PtOfIndexInterpol(aI1);
                       aXmlApp.Mesures().push_back(aMA);
                    }
                 }
                 MakeFileXML<cListeAppuis1Im>(aXmlApp, aNameRes);

             }
          }
       }

       delete aNuage;
    }
}

void cAppliApero::ExportMesuresFromCarteProf(const cExportMesuresFromCarteProf & anEM)
{
    for
    (
         std::list<cCartes2Export>::const_iterator itC = anEM.Cartes2Export().begin();
         itC != anEM.Cartes2Export().end();
         itC++
    )
    {
        ExportMesuresFromCarteProf(anEM,*itC);
    }

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
