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

#include "Xeres.h"



/*********************************************************************************/
/*                                                                               */
/*               cAppliXeres                                                     */
/*                                                                               */
/*********************************************************************************/


void  cAppliXeres::CalculHomMatch(const std::string & anOri)
{
    for (int aKC=0 ; aKC<int(mVCam.size()) ; aKC++)
    {
        cXeres_Cam * aCam0 = mVCam[aKC];
        if (aCam0->HasIm())
        {
            cImSecOfMaster  anISOM;
            anISOM.Master() = aCam0->NameIm();
            anISOM.UsedPenal() = 0.0;
            cOneSolImageSec anOSIS;
            std::vector<cXeres_Cam *> aVV =  NeighVois(aCam0,1);
            anOSIS.Coverage() = aVV.size() * 0.3;
            anOSIS.Score() =  anOSIS.Coverage();

            for (int aKV=0 ; aKV<int(aVV.size()) ; aKV++)
            {
                cXeres_Cam * aCamV = aVV[aKV];
                anOSIS.Images().push_back(aCamV->NameIm());
            }
            anISOM.Sols().push_back(anOSIS);
            // "NKS-Assoc-ImSec"
            MakeFileXML
            (
               anISOM,
               mICNM->Assoc1To1("NKS-Assoc-ImSec@-"+anOri,aCam0->NameIm(),true)
            );
        }
    }
}
  
void  cAppliXeres::ExeTapioca(const std::string & aFile)
{
    std::string aCom =   MM3dBinFile_quotes("Tapioca")
                       + " File "
                       +  mDir + mNameCpleXml
                       +  " " + ToString(mSzTapioca);

    System(aCom);
}

std::string cAppliXeres::ExtractId(const std::string & aNameIm)
{
    static cElRegex TheAutom("([A-Z][0-9]{1,2})_.*",10);
    static std::string TheReplace = "$1";
    return MatchAndReplace(TheAutom,aNameIm,TheReplace);
}

std::string cAppliXeres::Make2CurSeq(const std::string & aNameIm)
{
     return ExtractId(aNameIm) + "_" + mSeq + ".jpg";
}

void cAppliXeres::CalculTiePoint(int aSz,int aNBHom,const std::string & aNameAdd)
{
    mSzTapioca = aSz;
    cSauvegardeNamedRel aXmlCples;
    
    for (int aKC=0 ; aKC<int(mVCam.size()) ; aKC++)
    {
        cXeres_Cam * aCam0 = mVCam[aKC];
        if (aCam0->HasIm())
        {
            std::vector<cXeres_Cam *> aVV =  NeighVois(aCam0,aNBHom);

            for (int aKV=0 ; aKV<int(aVV.size()) ; aKV++)
            {
                cXeres_Cam * aCamV = aVV[aKV];
                if (aCam0->NS().Name() < aCamV->NS().Name())
                {
                    aXmlCples.Cple().push_back(cCpleString(aCam0->NameIm(),aCamV->NameIm()));
                }
            }
        }
    }

    if (aNameAdd!="")
    {
        cSauvegardeNamedRel aXmlAdd = StdGetFromPCP(mDir+aNameAdd,SauvegardeNamedRel); 
        for (std::vector<cCpleString>::const_iterator itC=aXmlAdd.Cple().begin() ; itC!=aXmlAdd.Cple().end() ; itC++)
        {
            std::string aN1 =  Make2CurSeq(itC->N1());
            std::string aN2 =  Make2CurSeq(itC->N2());
            if (ELISE_fp::exist_file(mDir+aN1) && ELISE_fp::exist_file(mDir+aN2))
            {
               aXmlCples.Cple().push_back(cCpleString(aN1,aN2));        
            }
            else
            {
            }
        }
    }



    MakeFileXML(aXmlCples,mDir+mNameCpleXml);
    ExeTapioca(mNameCpleXml);

     std::string aKH = "NKS-Assoc-CplIm2Hom@@dat";

     std::vector<cCpleString> aVC = aXmlCples.Cple();

     for (int aKCpl=0 ; aKCpl<int(aVC.size()) ; aKCpl++)
     {
         const std::string & aN1 = aVC[aKCpl].N1() ;
         const std::string & aN2 = aVC[aKCpl].N2() ;

         std::string aNDir = mDir+ mICNM->Assoc1To2(aKH,aN1,aN2,true);
         if (ELISE_fp::exist_file(aNDir))
         {
             std::string aNInv = mDir+ mICNM->Assoc1To2(aKH,aN2,aN1,true);
             ElPackHomologue  aPackH = ElPackHomologue::FromFile(aNDir);
             aPackH.SelfSwap();
             aPackH.StdPutInFile(aNInv);
         }
     }

}


void cAppliXeres::FusionneHom(const std::vector<cAppliXeres *> aVAppli,const std::string & aPostOut)
{
     int aNbCam    = aVAppli[0]->mVCam.size();
     cInterfChantierNameManipulateur * aICNM =  aVAppli[0]->mICNM;
     std::string aKHIn = "NKS-Assoc-CplIm2Hom@@dat";
     std::string aKHOut = "NKS-Assoc-CplIm2Hom@"+ aPostOut + "@dat";
     std::string aDir =  aVAppli[0]->mDir;

     ElTimer aChrono;
     for (int aKC1=0 ; aKC1<aNbCam ; aKC1++)
     {
         std::cout << "RESTE " << (aNbCam-aKC1) << " Cam to do ; time " << aChrono.uval() << "\n";
         for (int aKC2=0 ; aKC2< aNbCam ; aKC2++)
         {
               ElPackHomologue aRes;
               std::set<Pt2dr> aS1;
               std::set<Pt2dr> aS2;

               for (int aKA=0 ; aKA<int(aVAppli.size()) ; aKA++)
               {
                   std::vector<cXeres_Cam *>    aVCam = aVAppli[aKA]->mVCam;
                   std::string aN1 = aVCam[aKC1]->NameIm();
                   std::string aN2 = aVCam[aKC2]->NameIm();
                   std::string aNameH = aDir + aICNM->Assoc1To2(aKHIn,aN1,aN2,true);
                   if (ELISE_fp::exist_file(aNameH))
                   {
                       ElPackHomologue  aPackH = ElPackHomologue::FromFile(aNameH);
                       for (ElPackHomologue::const_iterator itH=aPackH.begin() ; itH!=aPackH.end() ; itH++)
                       {
                            const Pt2dr & aP1 = itH->P1();
                            const Pt2dr & aP2 = itH->P2();
                            if ((!BoolFind(aS1,aP1)) && (!BoolFind(aS2,aP2)))
                            {
                                 aS1.insert(aP1);
                                 aS2.insert(aP2);
                                 aRes.Cple_Add(ElCplePtsHomologues(aP1,aP2));
                            }
                       }
                   }
               }
               if (aRes.size())
               {
                   std::vector<cXeres_Cam *>    aVCam = aVAppli[0]->mVCam;
                   std::string aN1 = aVCam[aKC1]->NameIm();
                   std::string aN2 = aVCam[aKC2]->NameIm();
                   std::string aNameH = aDir + aICNM->Assoc1To2(aKHOut,aN1,aN2,true);
                   aRes.StdPutInFile(aNameH);
               }
         }
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
aooter-MicMac-eLiSe-25/06/2007*/
