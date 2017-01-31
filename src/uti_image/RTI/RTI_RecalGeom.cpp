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

#include "RTI.h"


const cXml_RTI_Ombre & cAppli_RTI::Ombr() const
{
    ELISE_ASSERT(mParam.ParamOmbre().IsInit(),"cOneIm_RTI::Ombr");
    return mParam.ParamOmbre().Val();
}

CamStenope * cAppli_RTI::OriMaster()
{
   if (mOriMaster==0)
   {
      
      mOriMaster = mICNM->StdCamStenOfNames(mMasterIm->Name(),Ombr().OrientMaster()); // ,mMasterIm->Name());
   }
   return mOriMaster;
}

void cOneIm_RTI::DoPoseFromOmbre
    (
           const cDicoAppuisFlottant & aDAF,
           const cSetOfMesureAppuisFlottants & aMAFI
    )
{
    if (!mXml) return;

    if (! mXml->Export().IsInit())
    {
         mXml->Export().SetVal(cXml_RTI_ExportIm());
    }
    Pt3dr  aPMoy(0,0,0);
    for (std::list<std::string>::const_iterator itN=mXml->NameOmbre().begin();itN!=mXml->NameOmbre().end();itN++)
    {
        std::string aNamePt = *itN;
        const cOneAppuisDAF * aPAf = GetApOfName(aDAF,aNamePt);
        ELISE_ASSERT(aPAf!=0,"cOneIm_RTI::DoPoseFromOmbre no Appuis");
        
        std::vector<cOneMesureAF1I> aVM = GetMesureOfPtsIm(aMAFI,"Ombre-"+aNamePt,mName);
        ELISE_ASSERT(aVM.size()==1,"Incoh in GetMesureOfPtsIm");
        Pt3dr aPOmbr = mAppli.OriMaster()->ImEtZ2Terrain(aVM[0].PtIm(),0.0);
        Pt3dr aPObj = aPAf->Pt();

        double aZLum = mAppli.Ombr().DefAltiLum();
        Pt3dr aVect = aPObj-aPOmbr;
        double aLambda =  (aZLum-aPOmbr.z)  / aVect.z;
        Pt3dr aCentrLum = aPOmbr + aVect * aLambda;
//      std::cout << " === " << aNamePt << " -> " << aCentrLum << "\n";
        aPMoy = aPMoy + aCentrLum;


    }
    mXml->Export().Val().PosLum()  = aPMoy /  mXml->NameOmbre().size();
    aPMoy = aPMoy /  mXml->NameOmbre().size();
    std::cout << "For " << mName <<   " lum=> " << aPMoy << "\n";
}

void cAppli_RTI::DoPoseFromOmbre(const cDicoAppuisFlottant & aDAF,const cSetOfMesureAppuisFlottants & aMAFI)
{
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
       mVIms[aK]->DoPoseFromOmbre(aDAF,aMAFI);

/*
    for (std::list<cXml_RTI_Im>::iterator itI=mParam.RTI_Im().begin() ; itI!=mParam.RTI_Im().end() ; itI++)
    {
        const std::string & aName = itI->Name();
        cOneIm_RTI* anIm= mDicoIm[aName];
        if (anIm)
        {
            anIm->DoPoseFromOmbre(aDAF,aMAFI,*itI);
        }
    }
*/

    MakeFileXML(mParam,"RTI-PosLum.xml");
}

int RTI_PosLumFromOmbre_main(int argc,char ** argv)
{
    std::string aFullNameParam,aName3D,aName2D;
    ElInitArgMain
    (
          argc,argv,  
          LArgMain()  << EAMC(aFullNameParam,"Name of Xml Param", eSAM_IsExistFile)
                      << EAMC(aName3D,"Name of GCP File",eSAM_IsExistFile)
                      << EAMC(aName2D,"Name of Image File",eSAM_IsExistFile),
          LArgMain()  
    );

    cAppli_RTI anAppli(aFullNameParam,eRTI_PoseFromOmbre,"");

    cDicoAppuisFlottant aDAF = StdGetFromPCP(aName3D ,DicoAppuisFlottant);
    cSetOfMesureAppuisFlottants aMAFI = StdGetFromPCP(aName2D ,SetOfMesureAppuisFlottants);


    anAppli.DoPoseFromOmbre(aDAF,aMAFI);

    return EXIT_SUCCESS;

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
