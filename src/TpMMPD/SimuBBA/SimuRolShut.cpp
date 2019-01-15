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
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "StdAfx.h"
#include "string.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"
#include "../schnaps.h"
#include "SimuBBA.h"

int SimuRolShut_main(int argc, char ** argv)
{
    std::string aPatImgs, aSH, aOri, aSHOut{"SimuRolShut"}, aDir, aImgs,aModifP,aPostfix{".thm.tif"};
    //int aSeed;
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image Pattern",eSAM_IsExistFile)
                           << EAMC(aSH, "PMul File",  eSAM_IsExistFile)
                           << EAMC(aOri, "Ori",  eSAM_IsExistDirOri)
                           << EAMC(aModifP,"File containing pose modification for each image, file size = 1 or # of images"),
                LArgMain() << EAM(aSHOut,"Out",false,"Output name of generated tie points, default=simulated")
                           << EAM(aPostfix,"Postfix",true,"Postfix of images, default=.thm.tif")
                );

    // get directory
    SplitDirAndFile(aDir,aImgs,aPatImgs);
    StdCorrecNameOrient(aOri, aDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImgs = *(aICNM->Get(aImgs));

    std::cout << aVImgs.size() << " image files.\n";

    // Generate poses corresponding to the end of exposure
    std::vector<Orientation> aVOrient;
    ReadModif(aVOrient,aModifP);

    int aSzPF = aPostfix.size();

    for(uint i=0; i<aVImgs.size();i++)
    {
        CamStenope * aCam = aICNM->StdCamStenOfNames(aVImgs[i],aOri);
        uint j = (aVOrient.size()==1)? 0 : i;
        aCam->AddToCenterOptical(aVOrient.at(j).Translation);
        aCam->MultiToRotation(aVOrient.at(j).Rotation);

        std::string aKeyOut = "NKS-Assoc-Im2Orient@-" + aOri;
        std::string aNameCamOut = aVImgs.at(i).substr(0,aVImgs.at(i).size()-aSzPF)+"_bis"+aPostfix;
        std::string aOriOut = aICNM->Assoc1To1(aKeyOut,aNameCamOut,true);
        cOrientationConique  anOC = aCam->StdExportCalibGlob();

        std::cout << "Generate " << aNameCamOut << endl;
        MakeFileXML(anOC,aOriOut);
    }


    // lecture of tie points and orientation
    cSetTiePMul * pSH = new cSetTiePMul(0);
    pSH->AddFile(aSH);
    std::map<std::string,cCelImTPM *> aVName2Im = pSH->DicoIm().mName2Im;

    // declare 2D vector for storage of cam orientation
    std::vector<std::vector<CamStenope*>> aVCam (2, std::vector<CamStenope *> (aVName2Im.size())); // real poses + simulated poses;
    for (auto &aName2Im:aVName2Im)
    {
        CamStenope * aCam0 = aICNM->StdCamStenOfNames(aName2Im.first,aOri);
        aCam0->SetNameIm(aName2Im.first);
        aVCam[0][aName2Im.second->Id()] = aCam0;

        std::string aNameCamBis = aName2Im.first.substr(0,aName2Im.first.size()-aSzPF)+"_bis"+aPostfix;
        CamStenope * aCam1 = aICNM->StdCamStenOfNames(aNameCamBis,aOri);
        aCam1->SetNameIm(aNameCamBis);
        aVCam[1][aName2Im.second->Id()] = aCam1;
    }

    // declare aVStructH to stock generated tie points
    std::vector<ElPackHomologue> aVPack (aVName2Im.size());
    std::vector<int> aVIdImS (aVName2Im.size(),-1);
    StructHomol aStructH;
    aStructH.VElPackHomol = aVPack;
    aStructH.VIdImSecond = aVIdImS;
    std::vector<StructHomol> aVStructH (aVName2Im.size(),aStructH);

    // get 2D/3D position of tie points

    // parse Configs aVCnf
    std::vector<cSetPMul1ConfigTPM *> aVCnf = pSH->VPMul();
    for(auto &aCnf:aVCnf)
    {
        std::vector<int> aVIdIm = aCnf->VIdIm();

        // parse all pts in one Config
        for(uint aKPtCnf=0; aKPtCnf<uint(aCnf->NbPts()); aKPtCnf++)
        {
            std::vector<Pt2dr> aVPtInter;
            std::vector<CamStenope*> aVCamInter; // real poses to intersect
            std::vector<CamStenope*> aVCamSimu; // simulated poses
            std::vector<int> aVIdImInter;

            // parse all imgs for one pt
            for(uint aKImCnf=0; aKImCnf<aVIdIm.size();aKImCnf++)
            {
                aVPtInter.push_back(aCnf->Pt(aKPtCnf,aKImCnf));
                aVCamInter.push_back(aVCam[0][aVIdIm[aKImCnf]]);
                aVCamSimu.push_back(aVCam[1][aVIdIm[aKImCnf]]);
                aVIdImInter.push_back(aVIdIm[aKImCnf]);
            }

            //Intersect aVPtInter:

            ELISE_ASSERT(aVPtInter.size() == aVCamInter.size(), "Size not coherent");
            ELISE_ASSERT(aVPtInter.size() > 1 && aVCamInter.size() > 1, "Nb faiseaux < 2");
            Pt3dr aPInter3D = Intersect_Simple(aVCamInter , aVPtInter);

            // reproject aPInter3D sur tout les images dans aVCamInter
            std::vector<Pt2dr> aVP2d;
            std::vector<CamStenope *> aVCamInterVu;
            std::vector<int> aVIdImInterVu;

            for(uint itVCI=0; itVCI < aVCamInter.size(); itVCI++)
            {
                CamStenope * aCam0 = aVCamInter[itVCI];
                Pt2dr aPt2d0 = aCam0->R3toF2(aPInter3D);// reprojection via real pose P0
                CamStenope * aCam1 = aVCamSimu[itVCI];
                Pt2dr aPt2d1 = aCam1->R3toF2(aPInter3D); // reprojection via simulated pose P1

                // Pl (xl,yl) = xl/X * P1 + (1-xl/X) * P0
                double aXl = aCam0->Sz().x * aPt2d0.x / (aCam0->Sz().x - aPt2d1.x + aPt2d0.x);
                double aRatio = aXl/aCam0->Sz().x;
                double aYl = aRatio * aPt2d1.y + (1-aRatio) * aPt2d0.y;
                Pt2dr aPt2d = Pt2dr (aXl, aYl);
                //std::cout << "Sz = " << aCam0->Sz() << " P0 = " << aPt2d0 << " P1 = " << aPt2d1 << " Pl = " << aPt2d << endl;

                if(aCam0->PIsVisibleInImage(aPInter3D) && aCam1->PIsVisibleInImage(aPInter3D) && IsInImage(aCam0->Sz(),aPt2d))
                {
                    aVP2d.push_back(aPt2d);
                    aVCamInterVu.push_back(aCam0);
                    aVIdImInterVu.push_back(aVIdImInter[itVCI]);
                }
            }

            // parse images to fill ElPackHomologue
            for (uint it1=0; it1 < aVCamInterVu.size(); it1++)
            {
                int aIdIm1 = aVIdImInterVu[it1];
                aVStructH[aIdIm1].IdIm=aIdIm1;

                for(uint it2=0; it2 < aVCamInterVu.size(); it2++)
                {
                    if(it1!=it2)
                    {
                        int aIdIm2 = aVIdImInterVu[it2];

                        ElCplePtsHomologues aCPH (aVP2d[it1],aVP2d[it2]);
                        aVStructH[aIdIm1].VElPackHomol[aIdIm2].Cple_Add(aCPH);
                        aVStructH[aIdIm1].VIdImSecond[aIdIm2]=aIdIm2;
                    }
                }
            }
        }
    }


    //writing of new tie points

    //key for tie points
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
            + "_"
            +  std::string(aSHOut)
            +  std::string("@")
            +  std::string("dat");


    for (uint itVSH=0; itVSH < aVStructH.size(); itVSH++)
    {
        int aIdIm1 = aVStructH.at(itVSH).IdIm;
        CamStenope * aCam1 = aVCam[0].at(aIdIm1);
        std::string aNameIm1 = aCam1->NameIm();
        if (IsInList(aVImgs,aNameIm1))
        {
            for (uint itVElPH=0; itVElPH < aVStructH.at(itVSH).VElPackHomol.size(); itVElPH++)
            {
                int aIdIm2 = aVStructH.at(itVSH).VIdImSecond.at(itVElPH);
                if (aIdIm2 == -1) continue;
                CamStenope * aCam2 = aVCam[0].at(aIdIm2);
                std::string aNameIm2 = aCam2->NameIm();
                if (IsInList(aVImgs,aNameIm2))
                {
                    std::string aHmOut= aICNM->Assoc1To2(aKHOut, aNameIm1, aNameIm2, true);
                    ElPackHomologue aPck = aVStructH.at(aIdIm1).VElPackHomol.at(aIdIm2);
                    aPck.StdPutInFile(aHmOut);
                }
            }
        }

    }

    // convert Homol folder into new format
    std::string aComConvFH = MM3dBinFile("TestLib ConvNewFH")
                           + aImgs
                           + " All SH=_"
                           + aSHOut
                           + " ExportBoth=1";
    system_call(aComConvFH.c_str());

    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
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
associés au chargement,  a  l'utilisation,  a  la modification et/ou au
développement et a  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe a
manipuler et qui le réserve donc a  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités a  charger  et  tester  l'adéquation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
a  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder a cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
