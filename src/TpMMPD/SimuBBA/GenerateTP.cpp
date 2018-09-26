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
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"
#include "../schnaps.h"

struct StructHomol {
    int IdIm;
    CamStenope * Cam;
    vector<int> VIdIm2;
    vector<ElPackHomologue> VElPackHomol;
};

int GenerateTP_main(int argc,char ** argv)
{
    string aDir,aSH,aOri,aSHOut="_Gen";

    ElInitArgMain
     (
          argc, argv,
          LArgMain() << EAMC(aDir,"Directory", eSAM_IsExistFile)
                     << EAMC(aSH, "PMul File",  eSAM_IsExistFile)
                     << EAMC(aOri, "Ori",  eSAM_IsExistDirOri),
          LArgMain() << EAM(aSHOut,"Out",false,"Output name of generated tie points, Def=Homol_Gen")
     );

    // get directory
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);


    //1. lecture of tie points and orientation

    StdCorrecNameOrient(aOri, aDir);
    const std::string  aSHInStr = aSH;
    cSetTiePMul * aSHIn = new cSetTiePMul(0);
    aSHIn->AddFile(aSHInStr);

    cout<<"Total : "<<aSHIn->DicoIm().mName2Im.size()<<" imgs"<<endl;
    std::map<std::string,cCelImTPM *> VName2Im = aSHIn->DicoIm().mName2Im;

    // load cam for all Img
    // Iterate through all elements in std::map
    std::map<std::string,cCelImTPM *>::iterator it = VName2Im.begin();
    vector<CamStenope*> aVCam (VName2Im.size());


    while(it != VName2Im.end())
    {
        //std::cout<<it->first<<" :: "<<it->second->Id()<<std::endl;
        string aNameIm = it->first;
        int aIdIm = it->second->Id();
        CamStenope * aCam = aICNM->StdCamStenOfNames(aNameIm, aOri);
        aCam->SetNameIm(aNameIm);
        aVCam[aIdIm] = aCam;
        it++;
    }

    std::cout << "Finished loading tie points! \n";
    std::cout<<"VPMul - Nb Config: "<<aSHIn->VPMul().size()<<endl;


    // declare aVCH to stock generated tie points
    vector<ElPackHomologue> aVPack (VName2Im.size());
    vector<int> aVIdIm2 (VName2Im.size(),-1);
    StructHomol aStructH;
    aStructH.VElPackHomol = aVPack;
    aStructH.VIdIm2 = aVIdIm2;
    vector<StructHomol> aVStructH (VName2Im.size(),aStructH);


    //2. get 3D position of tie points

    // parse Configs aVCnf

    std::vector<cSetPMul1ConfigTPM *> aVCnf = aSHIn->VPMul();
    for (uint aKCnf=1; aKCnf<aVCnf.size(); aKCnf++)
    {
        cSetPMul1ConfigTPM * aCnf = aVCnf[aKCnf];

        std::vector<int> aVIdIm =  aCnf->VIdIm();

        for (uint aKPtCnf=0; aKPtCnf<uint(aCnf->NbPts()); aKPtCnf++)
        {
            vector<Pt2dr> aVPtInter;
            vector<CamStenope*> aVCamInter; 
            vector<int> aVIdImInter;

            for (uint aKImCnf=0; aKImCnf<aVIdIm.size(); aKImCnf++)
            {

                aVPtInter.push_back(aCnf->Pt(aKPtCnf, aKImCnf));
                aVCamInter.push_back(aVCam[aVIdIm[aKImCnf]]);
                aVIdImInter.push_back(aVIdIm[aKImCnf]);
            }

            //Intersect aVPtInter:

            ELISE_ASSERT(aVPtInter.size() == aVCamInter.size(), "Size not coherent");
            ELISE_ASSERT(aVPtInter.size() > 1 && aVCamInter.size() > 1, "Nb faiseaux < 2");
            Pt3dr aPInter3D = Intersect_Simple(aVCamInter , aVPtInter);

            // reproject aPInter3D sur tout les images dans aVCamInter
            vector<Pt2dr> aVP2d;
            vector<CamStenope *> aVCamInterVu;
            vector<int> aVIdImInterVu;
            for (uint itVCI=0; itVCI < aVCamInter.size(); itVCI++)
            {
                CamStenope * aCam = aVCamInter[itVCI];
                Pt2dr aPt2d = aCam->R3toF2(aPInter3D);
                Pt2di aImgSz = aCam->Sz();

                //check if the point is in the camera view
                if ((aPt2d.x >=0) && (aPt2d.y >=0) && (aPt2d.x <= aImgSz.x-1) && (aPt2d.y <= aImgSz.y-1) && (aCam->Devant(aPInter3D)))
                {
                    aVP2d.push_back(aPt2d);
                    aVCamInterVu.push_back(aCam);
                    aVIdImInterVu.push_back(aVIdImInter[itVCI]);
                }

            }

            // parse images to fill ElPackHomologue
            for (uint it1=0; it1 < aVCamInterVu.size(); it1++)
            {
                int aIdIm1=aVIdImInterVu.at(it1);
                aVStructH.at(aIdIm1).IdIm=aIdIm1;
                for (uint it2=0; it2 < aVCamInterVu.size(); it2++)
                {
                    if (it1==it2) continue;

                    int aIdIm2=aVIdImInterVu.at(it2);

                    ElCplePtsHomologues aCPH (aVP2d[it1],aVP2d[it2]);
                    aVStructH.at(aIdIm1).VElPackHomol.at(aIdIm2).Cple_Add(aCPH);
                    aVStructH.at(aIdIm1).VIdIm2.at(aIdIm2)=aIdIm2;
                    //std::cout << "Add pt to IdIm1: " << aIdIm1 << "  IdIm2 : " << aIdIm2 << endl;
                }
            }


         }

    }

    std::cout << "ElPackHomologue filled !\n";

    //writing of new tie points

    //key for tie points
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aSHOut)
            +  std::string("@")
            +  std::string("dat");


    for (uint itVSH=0; itVSH < aVStructH.size(); itVSH++)
    {
        int aIdIm1 = aVStructH.at(itVSH).IdIm;
        CamStenope * aCam1 = aVCam.at(aIdIm1);
        std::string aNameIm1 = aCam1->NameIm();
        std::cout << "Master Im: " << aNameIm1 << "    IdIm1 : " << aIdIm1 << endl;
        for (uint itVElPH=0; itVElPH < aVStructH.at(itVSH).VElPackHomol.size(); itVElPH++)
        {
            int aIdIm2 = aVStructH.at(itVSH).VIdIm2.at(itVElPH);
            if (aIdIm2 == -1) continue;
            CamStenope * aCam2 = aVCam.at(aIdIm2);
            std::string aNameIm2 = aCam2->NameIm();
            //std::cout << "Second Im: " << aNameIm2  << "    IdIm2 : " << aIdIm2<< endl;
            std::string aHmOut= aICNM->Assoc1To2(aKHOut, aNameIm1, aNameIm2, true);
            ElPackHomologue aPck = aVStructH.at(aIdIm1).VElPackHomol.at(aIdIm2);
            aPck.StdPutInFile(aHmOut);
        }

    }
    std::cout << "Finished writing Homol files ! \n";


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
