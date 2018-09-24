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

int GenerateTP_main(int argc,char ** argv)
{
    string aPatImg,aDir,aImg,aSH,aOri;
    Pt2di aImgSz (5120,3840);
    ElInitArgMain
     (
          argc, argv,
          LArgMain() << EAMC(aPatImg,"Image pattern", eSAM_IsExistFile)
                     << EAMC(aSH, "PMul File",  eSAM_IsExistFile)
                     << EAMC(aOri, "Ori",  eSAM_IsExistDirOri),
          LArgMain() << EAM(aImgSz,"ImgSz",false,"Image Size, Def=[5120,3840]")
     );

    SplitDirAndFile(aDir, aImg, aPatImg);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    //1. lecture of image pattern

    std::list<std::string> aLImg = aICNM->StdGetListOfFile(aImg);
    std::cout << "Nb of Imgs: " << aLImg.size() << endl;

    //2. lecture of tie points and orientation

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
    vector<cPic*> allPics;
    while(it != VName2Im.end())
    {
        //std::cout<<it->first<<" :: "<<it->second->Id()<<std::endl;
        string aNameIm = it->first;
        int aIdIm = it->second->Id();
        CamStenope * aCam = aICNM->StdCamStenOfNames(aNameIm, aOri);
        aVCam[aIdIm] = aCam;
        allPics.push_back(new cPic(aDir,aNameIm));
        it++;
    }

    cout<<"VPMul - Nb Config: "<<aSHIn->VPMul().size()<<endl;
    std::vector<cSetPMul1ConfigTPM *> aVCnf = aSHIn->VPMul();


    //3. get 3D position of tie points
    vector<Pt3dr> aVAllPtInter;         // Coordonne 3D de tout les points dans pack

//    vector<int> aStats(aSHIn->NbIm());  // Vector contient multiplicite de pack, index d'element du vector <=> multiplicite, valeur d'element <=> nb point
//    vector<int> aStatsInRange(aSHIn->NbIm()); // Vector contient multiplicite de pack dans 1 gamme de residue defini, index d'element du vector <=> multiplicite, valeur d'element <=> nb point
//    vector<int> aStatsValid;            // Vector contient multiplicite existe de pack, valeur d'element <=> multiplicité

//    int nbPtsInRange = 0;
//    double resMax = 0.0;
//    double resMin = DBL_MAX;    

    for (uint aKCnf=1; aKCnf<aVCnf.size(); aKCnf++)
    {
        cSetPMul1ConfigTPM * aCnf = aVCnf[aKCnf];
        //cout<<"Cnf : "<<aKCnf<<" - Nb Imgs : "<<aCnf->NbIm()<<" - Nb Pts : "<<aCnf->NbPts()<<endl;
        std::vector<int> aVIdIm =  aCnf->VIdIm();

        for (uint aKPtCnf=0; aKPtCnf<uint(aCnf->NbPts()); aKPtCnf++)
        {
            vector<Pt2dr> aVPtInter;
            vector<CamStenope*> aVCamInter;


            for (uint aKImCnf=0; aKImCnf<aVIdIm.size(); aKImCnf++)
            {
                //cout<<aCnf->Pt(aKPtCnf, aKImCnf)<<" ";
                aVPtInter.push_back(aCnf->Pt(aKPtCnf, aKImCnf));
                aVCamInter.push_back(aVCam[aVIdIm[aKImCnf]]);
            }
            //cout<<endl;
            //Intersect aVPtInter:
            ELISE_ASSERT(aVPtInter.size() == aVCamInter.size(), "Size not coherent");
            ELISE_ASSERT(aVPtInter.size() > 1 && aVCamInter.size() > 1, "Nb faiseaux < 2");
            Pt3dr aPInter3D = Intersect_Simple(aVCamInter , aVPtInter);
            aVAllPtInter.push_back(aPInter3D);
         }
    }
    std::cout << "Nb of intersected pts: " << aVAllPtInter.size() << endl;
    std::cout << "aVCnf size : " << aVCnf.size() << endl;


    //4. regenerate tie points in 2D


    //read one TP in 3D
    for (uint iTP=0; iTP < aVAllPtInter.size(); iTP++)
    {
        Pt3dr aPt3d = aVAllPtInter[iTP];
        cHomol aHomol();

        //read one image
        std::map<std::string,cCelImTPM *>::iterator it = VName2Im.begin();
        //vector<CamStenope*> aVCam (VName2Im.size());
        while(it != VName2Im.end())
        {
            //std::cout<<it->first<<" :: "<<it->second->Id()<<std::endl;
            string aNameIm = it->first;
            //int aIdIm = it->second->Id();
            CamStenope * aCam = aICNM->StdCamStenOfNames(aNameIm, aOri);


            //calculate 2D position

            double aProf = aCam->ProfondeurDeChamps(aPt3d);
            Pt2dr aPt2d = aCam->NormC2M(aCam->Ter2Capteur(aPt3d));
            if (aPt2d.x >=0 && aPt2d.y >=0 && aPt2d.x <= aImgSz.x-1 && aPt2d.y <= aImgSz.y-1 && aCam->Devant(aPt3d))
            {
                std::cout << "prof: "<< aProf << "\n";
                std::cout << "R3 "<< aPt3d << " ---> F2 "<< aCam->Ter2Capteur(aPt3d);
                std::cout << " ---> M2 "<< aPt2d << "\n";

            }


            it++;
        }

    }



//    std::string aDirHomolOut = "_Gen";
//    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
//            +  std::string(aDirHomolOut)
//            +  std::string("@")
//            +  std::string("dat");
//    cout << "aKHOut" << aKHOut << endl;

//    // write tie points
//    for(ElPackHomologue::iterator iTH = aPckIn.begin(); iTH != aPckIn.end(); iTH++)
//    {
//        Pt2dr aP1 = iTH->P1();
//        Pt2dr aP2 = iTH->P2();
//        ElCplePtsHomologues aPH (aP1,aP2);
//        aPckOut.Cple_Add(aPH);
//    }

//    std::string aIm1Out = aPrefix + aIm1;
//    std::string aIm2Out = aPrefix + aIm2;
//    std::string aHmOut= aICNM->Assoc1To2(aKHOut, aIm1Out, aIm2Out, true);
//    aPckOut.StdPutInFile(aHmOut);

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
