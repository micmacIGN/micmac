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


/*


*/


int ReechInvEpip_main(int argc,char ** argv)
{
    Tiff_Im::SetDefTileFile(50000);
    std::string aDir= ELISE_Current_DIR;
    bool InParal = false;
    std::string aName1;
    std::string aName2;
    std::string anOri;
    bool CalleByP = false;
    int  aSzDecoup = 2000;
    Box2di aBoxOut;



    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aName1,"Name first image",  eSAM_IsExistFile)
                << EAMC(aName2,"Name second image", eSAM_IsExistFile)
                << EAMC(anOri,"Name second orient", eSAM_IsExistFile) ,
    LArgMain()  << EAM(aDir,"Dir",true,"directory (Def=current)")
                    << EAM(InParal,"InParal",true,"Compute in parallel (Def=true)")
                    << EAM(CalleByP,"CalleByP",true,"Internal Use", eSAM_InternalUse)
                    << EAM(aBoxOut,"BoxOut",true,"Internal Use", eSAM_InternalUse)
                    << EAM(aSzDecoup,"SzDec",true,"Sz Decoup")
    );

    std::string aNameMin = (aName1 <= aName2) ? aName1 : aName2;
    std::string aNameMax = (aName1 >  aName2) ? aName1 : aName2;

    cCpleEpip * aCplE = StdCpleEpip(aDir,anOri,aNameMin,aNameMax);

    std::string aDirIn = aDir+ aCplE->LocDirMatch(aName1);
    std::string aDirOut = aDir+  TheDIRMergeEPI()  + aName1 + "/";

    std::string aNameIn       =  aDirIn  + "NuageImProf_Chantier-Ori_Etape_Last.xml";
    std::string aNameGeomOut  =  aDirOut + "NuageImProf_LeChantier_Etape_1.xml";
    std::string aNameOut = aDirOut+"Nuage-"+aName2 +".xml";
    std::string aComBase =  MMBinFile(MM3DStr) +   " TestLib " + MakeStrFromArgcARgv(argc,argv);
    // std::string aNameOut  =  aDirOut +     "NuageImProf_LeChantier_Etape_1.xml"
    cXML_ParamNuage3DMaille aNuageXMLGeomOut = StdGetFromSI(aNameGeomOut,XML_ParamNuage3DMaille);
    Pt2di aSzOut =  aNuageXMLGeomOut.NbPixel();


    std::string aNameDistIn   = aDirIn  + "Distorsion.tif";
    std::string aNameARIn   = aDirIn  + "Score-AR.tif";

    std::string aNameDistOut  = aDirOut  + "Dist-"+aName2 +".tif";
    std::string aNameDepthOut = aDirOut  + "Depth-"+aName2 +".tif";
    std::string aNameMaskOut  = aDirOut  + "Mask-"+aName2 +".tif";
    std::string aNameAROut    = aDirOut  + "Score-AR-"+aName2 +".tif";


    bool isModified;

    Tiff_Im  aTifAROut = Tiff_Im::CreateIfNeeded(
                            isModified,
                            aNameAROut,
                            aSzOut,
                            GenIm::u_int1,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero
                      );

    Tiff_Im  aTifDistOut = Tiff_Im::CreateIfNeeded(
                            isModified,
                            aNameDistOut,
                            aSzOut,
                            GenIm::u_int1,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero
                      );
    Tiff_Im  aTifDepth = Tiff_Im::CreateIfNeeded(
                            isModified,
                            aNameDepthOut,
                            aSzOut,
                            GenIm::real4,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero
                      );
    Tiff_Im  aTifMask = Tiff_Im::CreateIfNeeded(
                            isModified,
                            aNameMaskOut,
                            aSzOut,
                            GenIm::bits1_msbf,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero
                      );


    if (CalleByP)
    {
         cXML_ParamNuage3DMaille aNuageXMLIn = StdGetFromSI(aNameIn,XML_ParamNuage3DMaille);
         cElNuage3DMaille *  aNIn = cElNuage3DMaille::FromParam(aNameIn,aNuageXMLIn,aDirIn,"",1.0,(cParamModifGeomMTDNuage*)0,true);

         cXML_ParamNuage3DMaille aNuageXMLOut =  StdGetFromSI(aNameOut,XML_ParamNuage3DMaille);
         cParamModifGeomMTDNuage aModifGOut(1.0,I2R(aBoxOut),false);
         cElNuage3DMaille *  aNOut = cElNuage3DMaille::FromParam(aNameOut,aNuageXMLOut,aDirOut,"",1.0,&aModifGOut);
         std::vector<Pt2dr> aVCont;



         Pt2di aSzOut = aBoxOut.sz();
         Box2di aBoxM1(Pt2di(0,0),aSzOut-Pt2di(1,1));
         aBoxM1.PtsDisc(aVCont,100);

         Pt2dr aRPMinIn(1e9,1e9);
         Pt2dr aRPMaxIn(-1e9,-1e9);

         for (int aKP=0 ; aKP<int(aVCont.size()) ; aKP++)
         {
             Pt3dr aP3 = aNOut->PtOfIndex(round_ni(aVCont[aKP]));
             Pt2dr aP2In = aNIn->Ter2Capteur(aP3);
             aRPMinIn.SetInf(aP2In);
             aRPMaxIn.SetSup(aP2In);
         }
         Pt2di aIPMinIn = Sup(Pt2di(0,0),round_down(aRPMinIn-Pt2dr(5,5)));
         Pt2di aIPMaxIn = Inf(aNuageXMLIn.NbPixel(),round_up(aRPMaxIn+Pt2dr(5,5)));
         Box2di aBoxIn(aIPMinIn,aIPMaxIn);
         Pt2di aSzIn = aIPMaxIn - aIPMinIn;


         if ((aSzIn.x<=0) || (aSzIn.y<=0)) return 0;

         cParamModifGeomMTDNuage aModifGIn(1.0,I2R(aBoxIn),false);
         aNIn = cElNuage3DMaille::FromParam(aNameIn,aNuageXMLIn,aDirIn,"",1.0,&aModifGIn);

         Im2D_U_INT1 aImDist_Out(aSzOut.x,aSzOut.y,255);
         Tiff_Im aTifDistIn = Tiff_Im::BasicConvStd(aNameDistIn);
         Im2D_U_INT1 aImDist_In(aSzIn.x,aSzIn.y,255);
         ELISE_COPY
         (
               aImDist_In.all_pts(),
               trans(aTifDistIn.in(),aBoxIn._p0),
               aImDist_In.out()
         );
         TIm2D<U_INT1,INT> aTDistIn(aImDist_In);
         TIm2D<U_INT1,INT> aTDistOut(aImDist_Out);

         Im2D_U_INT1 aImAR_Out(aSzOut.x,aSzOut.y,0);
         Tiff_Im aTifARIn = Tiff_Im::BasicConvStd(aNameARIn);
         Im2D_U_INT1 aImAR_In(aSzIn.x,aSzIn.y,255);
         ELISE_COPY
         (
               aImAR_In.all_pts(),
               trans(aTifARIn.in(),aBoxIn._p0),
               aImAR_In.out()
         );
         TIm2D<U_INT1,INT> aTARIn(aImAR_In);
         TIm2D<U_INT1,INT> aTAROut(aImAR_Out);


         Pt2di aPOut;
         for (aPOut.x=0 ; aPOut.x<aSzOut.x ;aPOut.x++)
         {
             for (aPOut.y=0 ; aPOut.y<aSzOut.y ;aPOut.y++)
             {
                   Pt3dr aP3 = aNOut->PtOfIndex(aPOut);
                   Pt2dr aPIn = aNIn->Terrain2Index(aP3);
                   bool Ok = aNIn->IndexHasContenuForInterpol(aPIn);
                   if (Ok)
                   {
                       aP3 = aNIn->PtOfIndexInterpol(aPIn);
                       Pt3dr aQ3 = aNOut->Euclid2ProfAndIndex(aP3);
                       aNOut->SetProfOfIndex(aPOut,aQ3.z);
                       aTDistOut.oset(aPOut,aTDistIn.getr(aPIn));
                       aTAROut.oset(aPOut,aTARIn.getr(aPIn));
                   }
                   else
                   {
                       aNOut->SetNoValue(aPOut);
                   }
             }
         }


         Im2D_Bits<1> aMaskOut = aNOut->ImMask();
         ELISE_COPY
         (
             rectangle(aBoxOut._p0,aBoxOut._p1),
             trans(aMaskOut.in(),-aBoxOut._p0),
             aTifMask.out()
         );


         Im2DGen * aIProf = aNOut->ImProf();
         ELISE_COPY
         (
             rectangle(aBoxOut._p0,aBoxOut._p1),
             trans(aIProf->in(),-aBoxOut._p0),
             aTifDepth.out()
         );

         ELISE_COPY
         (
             rectangle(aBoxOut._p0,aBoxOut._p1),
             trans(aImDist_Out.in(),-aBoxOut._p0),
             aTifDistOut.out()
         );

         ELISE_COPY
         (
             rectangle(aBoxOut._p0,aBoxOut._p1),
             trans(aImAR_Out.in(),-aBoxOut._p0),
             aTifAROut.out()
         );
    }
    else
    {
        ELISE_COPY(aTifMask.all_pts(),1,aTifMask.out());
        ELISE_COPY(aTifDepth.all_pts(),0,aTifDepth.out());

        cXML_ParamNuage3DMaille aNuageXMLOut = aNuageXMLGeomOut;

        aNuageXMLOut.Image_Profondeur().Val().Image() = NameWithoutDir(aNameDepthOut);
        aNuageXMLOut.Image_Profondeur().Val().Masq() =  NameWithoutDir(aNameMaskOut);
        MakeFileXML(aNuageXMLOut,aNameOut);

        cDecoupageInterv2D aDecoup  = cDecoupageInterv2D::SimpleDec(aSzOut,aSzDecoup,0);

        for (int aKB=0 ; aKB<aDecoup.NbInterv() ; aKB++)
        {
             std::string aCom = aComBase + " CalleByP=true "  + " BoxOut="+ToString(aDecoup.KthIntervIn(aKB));

             System(aCom);
        }
    }

    return 0;
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
