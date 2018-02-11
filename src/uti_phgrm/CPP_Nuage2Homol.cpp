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

int Txt2Dat_main(int argc, char **argv){
    std::string aInTP,aOutTP;
    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aInTP,"input Tie Points (txt)")
                    << EAMC(aOutTP,"output Tie Points (dat)"),
         LArgMain()
    );
    std::cout << aInTP<<" -> "<< aOutTP<<std::endl;
    ElPackHomologue aPack = ElPackHomologue::FromFile(aInTP);
    aPack.StdPutInFile(aOutTP);
    return EXIT_SUCCESS;
}

int Nuage2Homol_main(int argc,char ** argv)
{
    std::string aNameNuage,aFullDir,aOri;
    std::string aOutHomolDirName="_nuage";//output Homol dir suffix

    bool ExpTxt=false;//Homol are in dat or txt

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aNameNuage,"Name of XML file", eSAM_IsExistFile)
                << EAMC(aFullDir,"Full Name (Dir+Pattern)", eSAM_IsPatFile)
                << EAMC(aOri,"Orientation", eSAM_IsExistDirOri),
    LArgMain()  << EAM(aOutHomolDirName, "HomolOut", true, "Output Homol directory suffix (default: _nuage)")
                << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
    );

#if (ELISE_windows)
    replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
    std::string  aDir,aPat;
    SplitDirAndFile(aDir,aPat,aFullDir);
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    // If the users enters Ori-MyOrientation/, it will be corrected into MyOrientation
    StdCorrecNameOrient(aOri,aDir);

    std::string anExt = ExpTxt ? "txt" : "dat";
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aOutHomolDirName)
            +  std::string("@")
            +  std::string(anExt);


    std::list<std::string>  L = anICNM->StdGetListOfFile(aPat);
    std::vector<ElCamera*> vCamera;
    std::vector<std::string> vName;
    for(std::list<std::string>::iterator list_iter = L.begin(); list_iter != L.end(); list_iter++)
    {
        std::cout<<(*list_iter)<<endl;

        std::string oriFileName = anICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOri+"@",(*list_iter),true);

        cResulMSO aRMso = anICNM->MakeStdOrient(oriFileName,false);
        vCamera.push_back(aRMso.Cam());
        vName.push_back(*list_iter);
    }    
    std::vector<ElPackHomologue*> vPack;
    for(size_t i=0;i<vCamera.size();++i)
    {
        for(size_t j=0;j<vCamera.size();++j)
        {
            if (i!=j)
            {
                vPack.push_back(new ElPackHomologue());
            }
        }
    }

    cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromFileIm(aNameNuage,"XML_ParamNuage3DMaille","");

    Pt2di anI;
    Pt2di aSzData = aNuage->SzData();
    for (anI.x=0 ; anI.x<aSzData.x ; anI.x++)
    {
        for (anI.y=0 ; anI.y<aSzData.y ; anI.y++)
        {
           if (aNuage->IndexHasContenu(anI))
            {
                Pt3dr aP3 = aNuage->PtOfIndex(anI);
                std::cout << aP3.x << " "<< aP3.y<<" "<<aP3.z<<std::endl;
                std::vector<Pt2dr> vPt2dr;
                for(size_t i=0;i<vCamera.size();++i)
                {
                    vPt2dr.push_back(vCamera[i]->R3toF2(aP3));
                }
                size_t index=0;
                for(size_t i=0;i<vPt2dr.size();++i)
                {
                    for(size_t j=0;j<vPt2dr.size();++j)
                    {
                        if (i != j)
                        {
                            vPack[index]->Cple_Add(ElCplePtsHomologues(vPt2dr[i],vPt2dr[j]));
                            ++ index;
                        }
                    }   
                }
            }
        }
    }

    size_t index=0;
    for(size_t i=0;i<vCamera.size();++i)
    {
        for(size_t j=0;j<vCamera.size();++j)
        {
            if (i!=j)
            {
                if (vPack[index]->size()>3)
                {
                    std::string aNameOut = aDir + anICNM->Assoc1To2(aKHOut,vName[i],vName[j],true);
                    std::cout << "export de "<<aNameOut<<std::endl;
                    vPack[index]->StdPutInFile(aNameOut);
                }
                ++index;
            }
        }
    }

    
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
