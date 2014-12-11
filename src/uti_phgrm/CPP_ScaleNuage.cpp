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



#define DEF_OFSET -12349876


int ScaleNuage_main(int argc,char ** argv)
{
    std::string aNameNuage,aNameOut;

    double aSc = 1.0;
    Pt2dr  aP0(0,0);
    Pt2dr  aSz(-1,-1);
    bool   Old=false;
    bool   InDirLoc = true;


    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aNameNuage,"Input cloud name (ex: NuageImProf_LeChantier_Etape_1.xml)", eSAM_IsExistFile)
                    << EAMC(aNameOut,"Output cloud name", eSAM_IsOutputFile)
                    << EAMC(aSc,"Scaling factor", eSAM_None),
    LArgMain()  << EAM(aSz,"Sz",true)
                    << EAM(aP0,"P0",true)
                    << EAM(Old,"Old",true,"For full compatibility, def=false")
                    << EAM(InDirLoc,"InDirLoc",true,"Add input directory to output , def=true")
    );

    if(!MMVisualMode)
    {
    if (Old)
    {
        cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromFileIm(aNameNuage);
        if (aSz.x <0)
        {
            aSz = Pt2dr(aNuage->SzUnique());
        }

        cElNuage3DMaille * aRes = aNuage->ReScaleAndClip(Box2dr(aP0,aP0+aSz),aSc);
        aRes->Save(aNameOut);
    }
    else
    {
        std::string aDirBase =   InDirLoc ? DirOfFile(aNameNuage) : "";
        cXML_ParamNuage3DMaille aXML =   StdGetObjFromFile<cXML_ParamNuage3DMaille>
                                         (
                                               aNameNuage,
                                               StdGetFileXMLSpec("SuperposImage.xml"),
                                               "XML_ParamNuage3DMaille",
                                               "XML_ParamNuage3DMaille"
                                         );
         if (aSz.x < 0)
         {
              aSz = Pt2dr(aXML.NbPixel());
         }
         cXML_ParamNuage3DMaille aNewXML = CropAndSousEch(aXML,aP0,aSc,aSz);

         std::string aNameNewMasq = aDirBase + aNameOut+ "_Masq.tif";
         aNewXML.Image_Profondeur().Val().Masq() =  NameWithoutDir(aNameNewMasq);
         std::string aNameMasqueIn = DirOfFile(aNameNuage) +aXML.Image_Profondeur().Val().Masq();
         Tiff_Im aFileMasqIn(aNameMasqueIn.c_str());
         Tiff_Im aFileMasq
                 (
                     aNameNewMasq.c_str(),
                     aNewXML.NbPixel(),
                     aFileMasqIn.type_el(),
                     // GenIm::bits1_msbf,
                     Tiff_Im::No_Compr,
                     Tiff_Im::BlackIsZero

                 );
         ELISE_COPY
         (
             aFileMasq.all_pts(),
             round_ni(StdFoncChScale(aFileMasqIn.in(0),aP0,Pt2dr(aSc,aSc))),
             aFileMasq.out()
         );

         std::string aNameProfIn = DirOfFile(aNameNuage) +aXML.Image_Profondeur().Val().Image();
         Tiff_Im aFileProfIn(aNameProfIn.c_str());
         std::string aNameNewProf = aDirBase + aNameOut+ "_Prof.tif";
         aNewXML.Image_Profondeur().Val().Image() =  NameWithoutDir(aNameNewProf);
         Tiff_Im aFileProf
                 (
                     aNameNewProf.c_str(),
                     aNewXML.NbPixel(),
                     aFileProfIn.type_el(),
                     Tiff_Im::No_Compr,
                     Tiff_Im::BlackIsZero

                 );
         ELISE_COPY
         (
             aFileProf.all_pts(),
                StdFoncChScale(aFileProfIn.in(0)*aFileMasqIn.in(0),aP0,Pt2dr(aSc,aSc))
             /  Max(1e-8,StdFoncChScale(aFileMasqIn.in(0),aP0,Pt2dr(aSc,aSc))),
             aFileProf.out()
         );

         if ( aNewXML.Image_Profondeur().IsInit())
         {
           aNewXML.Image_Profondeur().Val().Correl().SetNoInit();
         }

         MakeFileXML(aNewXML,aDirBase + aNameOut+".xml");

    }
/*
    cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromFileIm(aNameNuage);
    if (aSz.x <0)
    {
        aSz = Pt2dr(aNuage->Sz());
    }

    cElNuage3DMaille * aRes = aNuage->ReScaleAndClip(Box2dr(aP0,aSz),aSc);
    aRes->Save(aNameOut);
*/

    return EXIT_SUCCESS;
    }
    else
        return EXIT_SUCCESS;
}





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
