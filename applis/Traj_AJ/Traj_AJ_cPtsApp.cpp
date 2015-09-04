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

/**************************************************************/
/*                                                            */
/*                 cAppli_Traj_AJ::ExportProj                 */
/*                                                            */
/**************************************************************/


void cAppli_Traj_AJ::TxtExportProjImage(const cTrAJ2_ExportProjImage & anEPI)
{
   char aBuf[2000];
   FILE * aFP = 0;
   if (anEPI.KeyGenerateTxt().IsInit())
   {
       std::string aNF = mDC + anEPI.NameFileOut();
       aFP = FopenNN(aNF.c_str(),"w","cAppli_Traj_AJ::TxtExportProjImage");
   }

   std::list<std::string> aLIm = mICNM->StdGetListOfFile(anEPI.KeySetOrPatIm());
   cDicoAppuisFlottant aDAF = StdGetObjFromFile<cDicoAppuisFlottant>
                                       (
                                          mDC + anEPI.NameAppuis(),
                                          StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                          "DicoAppuisFlottant",
                                          "DicoAppuisFlottant"
                                       );


   for 
   (
        std::list<std::string>::const_iterator itIm=aLIm.begin();
        itIm!=aLIm.end();
        itIm++
   )
   {
        std::string aNameOr = mICNM->Assoc1To1(anEPI.KeyAssocIm2Or(),*itIm,true);
        CamStenope * aCamS = CamStenope::StdCamFromFile(false,mDC+aNameOr,mICNM);

        Box2dr aBox(Pt2dr(0,0),Pt2dr(aCamS->Sz()));
        for
        (
             std::list<cOneAppuisDAF>::const_iterator itAp=aDAF.OneAppuisDAF().begin();
             itAp!=aDAF.OneAppuisDAF().end();
             itAp++
        )
        {
            Pt2dr  aPIm = aCamS->R3toF2(itAp->Pt());
            Pt3dr  aPTer = itAp->Pt();
            if (aBox.inside(aPIm))
            {
               sprintf
               (
                   aBuf,
                   "%s@%s@%lf@%lf@%lf@%lf@%lf",
                   itIm->c_str(),
                   itAp->NamePt().c_str(),
                   aPIm.x,aPIm.y,
                   aPTer.x,aPTer.y,aPTer.z
               );
                // std::cout << *itIm << " " << itAp->NamePt() << " " << aPIm << "\n";
                 // std::cout << aBuf << "\n";
                if (aFP)
                {
                    std::string aStr = mICNM->Assoc1To1(anEPI.KeyGenerateTxt().Val(),aBuf,true);
                    std::cout << aStr << "\n";
                    fprintf(aFP,"%s\n",aStr.c_str());
                }
            }
        }
   }

   if (aFP) 
   {
      ElFclose(aFP);
   }
}


/**************************************************************/
/*                                                            */
/*               cTAj2_LayerAppuis                             */
/*                                                            */
/**************************************************************/

#define TBUF 1000


cTAj2_LayerAppuis::cTAj2_LayerAppuis
(
    cAppli_Traj_AJ & anAppli,
    const cTrAJ2_ConvertionAppuis &  aSAp
) :
   mAppli  (anAppli),
   mSAp    (aSAp),
   mCom    (0),
   mSIn    (cSysCoord::FromXML(aSAp.SystemeIn(),mAppli.DC().c_str())),
   mSOut   (cSysCoord::FromXML(aSAp.SystemeOut(),mAppli.DC().c_str()))
{
    if (mSAp.AutomComment().IsInit())
    {
        mCom = mSAp.AutomComment().Val();
    }

    for 
    (
       std::list<cTraJ2_FilesInputi_Appuis>::const_iterator itF =aSAp.TraJ2_FilesInputi_Appuis().begin();
       itF!=aSAp.TraJ2_FilesInputi_Appuis().end();
       itF++
    )
    {
        std::list<std::string> aL = mAppli.ICNM()->StdGetListOfFile(itF->KeySetOrPat());
        for 
        (
            std::list<std::string>::const_iterator itS=aL.begin();
            itS!=aL.end();
            itS++
        )
        {
             AddFile(*itF,*itS);
        }
    }

    std::string aNamT =  mSAp.OutMesTer().ValWithDef("");
    std::string aNamI =  mSAp.OutMesIm().ValWithDef("");

    if (aNamT!="")
    {
        cDicoAppuisFlottant aDAF;
        for
        (
            std::map<std::string,cOneAppuisDAF>::const_iterator it=mMapPtsAp.begin();
            it!=mMapPtsAp.end();
            it++
        )
        {
          aDAF.OneAppuisDAF().push_back(it->second);
        }
        MakeFileXML(aDAF,mAppli.DC()+aNamT,"Global");
    }
    if (aNamI!="")
    {
        cSetOfMesureAppuisFlottants  aSAF;
        for
        (
            std::map<std::string,cMesureAppuiFlottant1Im>::const_iterator it=mMapMesIm.begin();
            it!=mMapMesIm.end();
            it++
        )
        {
          aSAF.MesureAppuiFlottant1Im().push_back(it->second);
        }
        if (aNamI!=aNamT)
        {
           MakeFileXML(aSAF,mAppli.DC()+aNamI);
        }
        else
        {
           AddFileXML(aSAF,mAppli.DC()+aNamI);
        }
    }
}

void cTAj2_LayerAppuis::AddFile (const cTraJ2_FilesInputi_Appuis & aFIn,const std::string & aNameFile)
{
    cElRegex* aRegEx = aFIn.Autom();


    std::string aNF = mAppli.DC() + aNameFile;
    ELISE_fp  aFP (aNF.c_str(),ELISE_fp::READ);

    char * aBUF;
    int aKLine=0;
    while (( aBUF = aFP.std_fgets()))
    {
        if ((mCom==0) || (! mCom->Match(aBUF)))
        {
            if (!aRegEx->Match(aBUF))
            {
                std::cout << "AT LINE " << aKLine << "\n";
                std::cout << "LINE=["<<aBUF<<"]\n";
                ELISE_ASSERT(false,"NO MATCH");
            }
            std::string  aNamePt = aRegEx->KIemeExprPar(aFIn.KIdPt());
            if (aFIn.GetMesTer())
            {
                std::vector<double> aVC;
                aVC.push_back(aRegEx->VNumKIemeExprPar(mSAp.KxTer()));
                aVC.push_back(aRegEx->VNumKIemeExprPar(mSAp.KyTer()));
                aVC.push_back(aRegEx->VNumKIemeExprPar(mSAp.KzTer()));

                aVC = VecCorrecUnites(aVC,mSAp.UnitesCoord());
                Pt3dr aPTer = mSOut->FromSys2This(*mSIn,Pt3dr::FromTab(aVC));

                cOneAppuisDAF & aDAF = mMapPtsAp[aNamePt];

                double aIPlani = mSAp.ValIncertPlani().Val();
                {
                   int aKPl = mSAp.KIncertPlani().Val();
                   if (aKPl >=0) aIPlani = aRegEx->VNumKIemeExprPar(aKPl);
                }
                double aIAlti = mSAp.ValIncertAlti().Val();
                {
                   int aKAlt = mSAp.KIncertAlti().Val();
                   if (aKAlt >=0) aIAlti = aRegEx->VNumKIemeExprPar(aKAlt);
                }

                Pt3dr anIncert(aIPlani,aIPlani,aIAlti);

                if (aDAF.NamePt()=="")
                {
                   aDAF.NamePt() = aNamePt;
                   aDAF.Pt() = aPTer;
                   aDAF.Incertitude() = anIncert;
                   std::cout << aNamePt << "\n";
                }
                else
                {
                   Pt3dr aP2T = aDAF.Pt();
                   double aDist = euclid(aP2T-aPTer);
                   if (aDist>1e-8)
                   {
                      std::cout << "AT LINE " << aKLine  << "\n";
                      std::cout << "Name " << aNamePt << "P1 " << aDAF.Pt() << "P2 " << aPTer << " D=" << aDist << "\n";
                      ELISE_ASSERT(false,"Incoherence in PTer");
                   }
                   Pt3dr aInc2 = aDAF.Incertitude();
                   ELISE_ASSERT(euclid(aInc2-anIncert)<1e-8,"Incoherence in Incert PTer");
                }
            }
 
            if (aFIn.GetMesIm())
            {

                Pt2dr aPIm  (  aRegEx->VNumKIemeExprPar(mSAp.KIIm()),
                               aRegEx->VNumKIemeExprPar(mSAp.KJIm())
                            );
                aPIm = aPIm + Pt2dr(mSAp.OffsetIm().Val());


                std::string  aNameIm = mAppli.ICNM()->Assoc1To1(mSAp.KeyId2Im(),aRegEx->KIemeExprPar(mSAp.KIdIm()),true);
                // std::string  aNamePt = aRegEx->KIemeExprPar(mSAp.KIdPtMesure());

                cMesureAppuiFlottant1Im &  aMFI = mMapMesIm[aNameIm];
                aMFI.NameIm() = aNameIm;
                cOneMesureAF1I aMesIm;
                aMesIm.PtIm() = aPIm;
                aMesIm.NamePt() = aNamePt;
                aMFI.OneMesureAF1I().push_back(aMesIm);


            std::cout << aPIm  << "\n";

            }
            aKLine++;
       }
    }
    aFP.close();
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
