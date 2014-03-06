/*Header-MicMac-eLiSe-25/06/2007peroChImMM_main

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

void MakeMetaData_XML_GeoI(const std::string & aNameImMasq,double aResol)
{
   std::string aNameXml =  StdPrefix(aNameImMasq) + ".xml";
   if (!ELISE_fp::exist_file(aNameXml))
   {

      cFileOriMnt aFOM = StdGetObjFromFile<cFileOriMnt>
                         (
                               Basic_XML_MM_File("SampleFileOriXML.xml"),
                               StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                               "FileOriMnt",
                               "FileOriMnt"
                          );

        aFOM.NameFileMnt() = NameWithoutDir(aNameImMasq);
        aFOM.NombrePixels() = Tiff_Im(aNameImMasq.c_str()).sz();
        if (aResol>0) 
        {
           aFOM.ResolutionPlani() = Pt2dr(aResol,aResol);
        }

        MakeFileXML(aFOM,aNameXml);
   }
}
void MakeMetaData_XML_GeoI(const std::string & aNameImMasq)
{
     MakeMetaData_XML_GeoI(aNameImMasq,-1);
}



int MM2DPostSism_Main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string  aIm1,aIm2,aImMasq;
    bool Exe=true;
    double aTeta;
    int    aSzW=4;
    double aRegul=0.3;
    bool useDequant=true;
    double aIncCalc=2.0;


    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(aIm1,"Image 1")
                    << EAMC(aIm2,"Image 2"),
	LArgMain()  
                    << EAM(aImMasq,"Masq",true,"Masq of focus zone (def=none)")
                    << EAM(aTeta,"Teta",true,"Direction of seism if any (in radian)")
                    << EAM(Exe,"Exe",true,"Execute command , def=true (tuning purpose)")
                    << EAM(aSzW,"SzW",true,"Size of window (Def =4, mean 9x9)")
                    << EAM(aRegul,"Reg",true,"Regularization (Def=0.3)")
                    << EAM(useDequant,"Dequant",true,"Dequantify (Def=true)")
                    << EAM(aIncCalc,"Inc",true,"Initial uncertainty (Def=2.0")
    );
	
#if (ELISE_windows)
	replace( aIm1.begin(), aIm1.end(), '\\', '/' );
	replace( aIm2.begin(), aIm2.end(), '\\', '/' );
	replace( aImMasq.begin(), aImMasq.end(), '\\', '/' );
#endif
    std::string aDir = DirOfFile(aIm1);
    ELISE_ASSERT(aDir==DirOfFile(aIm2),"Image not on same directory !!!");


    std::string aCom =    MM3dBinFile("MICMAC")
                        + XML_MM_File("MM-PostSism.xml")
                        + " WorkDir=" + aDir
                        + " +Im1=" + aIm1
                        + " +Im2=" + aIm2
                        + " +SzW=" + ToString(aSzW)
                        + " +RegulBase=" + ToString(aRegul)
                        + " +Inc=" + ToString(aIncCalc)
                        ;


    if (EAMIsInit(&aImMasq))
    {
        ELISE_ASSERT(aDir==DirOfFile(aImMasq),"Image not on same directory !!!");
        MakeMetaData_XML_GeoI(aImMasq);
        aCom = aCom + " +UseMasq=true +Masq=" + StdPrefix(aImMasq);
    }

    if (EAMIsInit(&aTeta))
    {
        aCom = aCom + " +UseTeta=true +Teta=" + ToString(aTeta);
    }
    
    if (useDequant)
    {
        aCom = aCom + " +UseDequant=true";
    }


    if (Exe)
    {
          system_call(aCom.c_str());
    }
    else
    {
           std::cout << "COM=[" << aCom << "]\n";
    }
    return 0;
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
