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
#include <map>

#include "StdAfx.h"
#include "string.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"
#include "../schnaps.h"
#include "SimuBBA.h"

int SimuRolShut_main(int argc, char ** argv)
{
    std::string aPatImgs, aSH, aOri, aOriRS, aSHOut{"SimuRolShut"}, aDir, aImgs,aModifP;
    int aLine{3}, aSeed;
    std::vector<double> aNoiseGaussian(4,0.0);
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image Pattern",eSAM_IsExistFile)
                           << EAMC(aSH, "PMul File",  eSAM_IsExistFile)
                           << EAMC(aOri, "Ori",  eSAM_IsExistDirOri)
                           << EAMC(aOriRS,"Ori for modified ori files")
                           << EAMC(aModifP,"File containing pose modification for each image, file size = 1 or # of images"),
                LArgMain() << EAM(aSHOut,"Out",false,"Output name of generated tie points, default=simulated")
                           << EAM(aLine,"Line",true,"Read file containing pose modification from a certain line, def=3 (two lines for file header)")
                           << EAM(aSeed,"Seed",false,"Seed for generating gaussian noise")
                           << EAM(aNoiseGaussian,"NoiseGaussian",false,"[meanX,stdX,meanY,stdY]")
                );

    // get directory
    SplitDirAndFile(aDir,aImgs,aPatImgs);
    StdCorrecNameOrient(aOri, aDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImgs = *(aICNM->Get(aImgs));

    std::cout << aVImgs.size() << " image files.\n";

    // Generate poses corresponding to the end of exposure
    std::vector<Orientation> aVOrient;
    ReadModif(aVOrient,aModifP,aLine);

    // Copy Calib file
    cElFilename aCal = cElFilename(aDir,aICNM->StdNameCalib(aOri,aVImgs[0]));
    std::string aDirCalib, aCalib;
    SplitDirAndFile(aDirCalib,aCalib,aCal.m_basename);
    std::string aNewCalib = aDir+"Ori-"+aOriRS+"/"+aCalib;
    cElFilename aCalRS = cElFilename(aNewCalib);

    for(uint i=0; i<aVImgs.size();i++)
    {
        CamStenope * aCam = aICNM->StdCamStenOfNames(aVImgs[i],aOri);
        uint j = (aVOrient.size()==1)? 0 : i;
        aCam->AddToCenterOptical(aVOrient.at(j).Translation);
        aCam->MultiToRotation(aVOrient.at(j).Rotation);

        std::string aKeyOut = "NKS-Assoc-Im2Orient@-" + aOriRS;
        std::string aOriOut = aICNM->Assoc1To1(aKeyOut,aVImgs[i],true);
        cOrientationConique  anOC = aCam->StdExportCalibGlob();
        anOC.Interne().SetNoInit();
        anOC.FileInterne().SetVal(aNewCalib);

        std::cout << "Generate " << aOriRS << "/" << aVImgs[i] << ".xml" << endl;
        MakeFileXML(anOC,aOriOut); 
    }
    if(aCal.copy(aCalRS,true))
        std::cout << "Create Calibration file "+aNewCalib << endl;


    // gaussian noise
    if(!EAMIsInit(&aSeed))  aSeed=time(0);

    std::default_random_engine generator(aSeed);

    std::normal_distribution<double> distributionX(aNoiseGaussian[0],aNoiseGaussian[1]);
    std::normal_distribution<double> distributionY(aNoiseGaussian[2],aNoiseGaussian[3]);

    //1. lecture of tie points and orientation
    std::cout << "Loading tie points + orientation...   ";
    cSetTiePMul * pSH = new cSetTiePMul(0);
    pSH->AddFile(aSH);
    std::map<std::string,cCelImTPM *> aVName2Im = pSH->DicoIm().mName2Im;

    // load cam for all Img
    // Iterate through all elements in std::map
    vector<CamStenope*> aVCam (aVName2Im.size());//real poses
    vector<CamStenope*> aVCambis (aVName2Im.size());//generated poses
    for(auto &aName2Im:aVName2Im)
    {
        CamStenope * aCam = aICNM->StdCamStenOfNames(aName2Im.first,aOri);
        aCam->SetNameIm(aName2Im.first);
        aVCam[aName2Im.second->Id()] = aCam;

        //std::string aNamebis = aName2Im.first.substr(0,aName2Im.first.size()-aSzPF)+"_bis"+aPostfix;
        CamStenope * aCambis = aICNM->StdCamStenOfNames(aName2Im.first,aOriRS);
        aCambis->SetNameIm(aName2Im.first);
        aVCambis[aName2Im.second->Id()] = aCambis;
    }

    std::cout << "Finish loading " << pSH->VPMul().size() << " CONFIG\n";

    // declare aVStructH to stock generated tie points
    vector<ElPackHomologue> aVPack (aVName2Im.size());
    vector<int> aVIdImS (aVName2Im.size(),-1);
    StructHomol aStructH;
    aStructH.VElPackHomol = aVPack;
    aStructH.VIdImSecond = aVIdImS;
    vector<StructHomol> aVStructH (aVName2Im.size(),aStructH);

    //2. get 2D/3D position of tie points
    std::cout << "Filling ElPackHomologue...   ";

    if (EAMIsInit(&aNoiseGaussian))
    {
        std::cout << "Gaussian Noise: " << aNoiseGaussian << endl;
    }


    // parse Configs aVCnf
    std::vector<cSetPMul1ConfigTPM *> aVCnf = pSH->VPMul();
    for (auto &aCnf:aVCnf)
    {
        std::vector<int> aVIdIm =  aCnf->VIdIm();

        // Parse all pts in one Config
        for (uint aKPtCnf=0; aKPtCnf<uint(aCnf->NbPts()); aKPtCnf++)
        {
            vector<Pt2dr> aVPtInter;
            vector<CamStenope*> aVCamInter;
            vector<CamStenope*> aVCamInterbis;
            vector<int> aVIdImInter;

            // Parse all imgs for one pts
            for (uint aKImCnf=0; aKImCnf<aVIdIm.size(); aKImCnf++)
            {

                aVPtInter.push_back(aCnf->Pt(aKPtCnf, aKImCnf));
                aVCamInter.push_back(aVCam[aVIdIm[aKImCnf]]);
                aVCamInterbis.push_back(aVCambis[aVIdIm[aKImCnf]]);
                aVIdImInter.push_back(aVIdIm[aKImCnf]);
            }

            //Intersect aVPtInter:

            ELISE_ASSERT(aVPtInter.size() == aVCamInter.size(), "Size not coherent");
            ELISE_ASSERT(aVPtInter.size() > 1 && aVCamInter.size() > 1, "Nb faiseaux < 2");
            Pt3dr aPInter3D = Intersect_Simple(aVCamInter , aVPtInter);
            //std::cout << aPInter3D << endl;


            // reproject aPInter3D sur tout les images dans aVCamInter
            std::vector<Pt2dr> aVP2d;
            std::vector<CamStenope *> aVCamInterVu;
            std::vector<int> aVIdImInterVu;

            for (uint itVCI=0; itVCI < aVCamInter.size(); itVCI++)
            {
                CamStenope * aCam = aVCamInter[itVCI];
                Pt2dr aPt2d0 = aCam->R3toF2(aPInter3D);//P0

                CamStenope * aCambis = aVCamInterbis[itVCI];
                Pt2dr aPt2d1 = aCambis->R3toF2(aPInter3D);//P1

                // Pl = l*P1 + (1-l)P0
                double aY = aCam->Sz().y * aPt2d0.y /( aCam->Sz().y - aPt2d1.y + aPt2d0.y );
                double aRatio = aY / aCam->Sz().y;
                double aX = aRatio * (aPt2d1.x-aPt2d0.x) + aPt2d0.x;
                Pt2dr aPt2d = Pt2dr(aX,aY);

                if (EAMIsInit(&aNoiseGaussian))
                {
                    aPt2d.x += distributionX(generator);
                    aPt2d.y += distributionY(generator);
                }

                //std::cout << aPt2d0.x << " " << aPt2d0.y << " " << aPt2d.x << " " << aPt2d.y << " " << aPt2d1.x << " " << aPt2d1.y << " " << endl;


                //check if the point is in the camera view
                if (aCam->PIsVisibleInImage(aPInter3D) && IsInImage(aCam->Sz(),aPt2d))
                {
                    aVP2d.push_back(aPt2d);
                    //std::cout << aPt2d << endl;
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
                    aVStructH.at(aIdIm1).VIdImSecond.at(aIdIm2)=aIdIm2;
                }
            }


        }
    }

    std::cout << "ElPackHomologue filled !\n";

    //writing of new tie points
    std::cout << "Writing Homol files...   ";
    //key for tie points
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
            + "_"
            +  std::string(aSHOut)
            +  std::string("@")
            +  std::string("dat");


    for (uint itVSH=0; itVSH < aVStructH.size(); itVSH++)
    {
        int aIdIm1 = aVStructH.at(itVSH).IdIm;
        CamStenope * aCam1 = aVCam.at(aIdIm1);
        std::string aNameIm1 = aCam1->NameIm();
        if (IsInList(aVImgs,aNameIm1))
        {
            for (uint itVElPH=0; itVElPH < aVStructH.at(itVSH).VElPackHomol.size(); itVElPH++)
            {
                int aIdIm2 = aVStructH.at(itVSH).VIdImSecond.at(itVElPH);
                if (aIdIm2 == -1) continue;
                CamStenope * aCam2 = aVCam.at(aIdIm2);
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

    // write seed file
    ofstream aSeedfile;
    aSeedfile.open ("Homol_"+aSHOut+"/Seed.txt");
    std::cout << "Homol_"+aSHOut+"/Seed.txt" << endl;
    aSeedfile << aSeed << endl;
    aSeedfile.close();

    std::cout << "Finished writing Homol files ! \n";

    // convert Homol folder into new format
    std::string aComConvFH = MM3dBinFile("TestLib ConvNewFH")
                           + aImgs
                           + " All SH=_"
                           + aSHOut
                           + " ExportBoth=1";
    system_call(aComConvFH.c_str());
    std::cout << aComConvFH << endl;

    return EXIT_SUCCESS;
}

int GenerateOrient_main (int argc, char ** argv)
{
    std::string aPatImgs, aOri, aSHOut{"Modif_orient.txt"}, aDir, aImgs,aOut{"Modif_orient.txt"};
    Pt2dr aTInterv, aGauss;
    int aSeed;
    std::vector<std::string> aVTurn;
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aPatImgs,"Image Pattern, make sure images are listed in the right order",eSAM_IsExistFile)
                           << EAMC(aOri, "Ori",  eSAM_IsExistDirOri)
                           << EAMC(aTInterv, "Time Interval, interpolate to generate translation, [cadence (s), exposure time (ms)]")
                           << EAMC(aGauss,"Gaussian distribution parameters of angular velocity for rotation generation (radian/s), [mean,std]"),
                LArgMain() << EAM(aOut,"Out",true,"Output file name for genarated orientation, def=Modif_orient.txt")
                           << EAM(aSeed,"Seed",false,"Random engine, if not give, computer unix time is used.")
                           << EAM(aVTurn,"Turn",false,"List of image names representing flight turns (set the translation T(i) as T(i-1))")
                );

    // get directory
    SplitDirAndFile(aDir,aImgs,aPatImgs);
    StdCorrecNameOrient(aOri, aDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImgs = *(aICNM->Get(aImgs));


    int aNbIm = aVImgs.size();
    double aRatio = aTInterv.y/1000/aTInterv.x;

    std::vector<Pt3dr> aVTrans;
    // generation of translation, interpolation
    for(int i=0; i<aNbIm-1; i++)
    {
        CamStenope * aCam0 = aICNM->StdCamStenOfNames(aVImgs[i],aOri);
        Pt3dr aP0 = aCam0->PseudoOpticalCenter();

        CamStenope * aCam1 = aICNM->StdCamStenOfNames(aVImgs[i+1],aOri);
        Pt3dr aP1 = aCam1->PseudoOpticalCenter();

        Pt3dr aP = (aP1-aP0) * aRatio;
        //Pt3dr aP_Last = i==0 ? aP : aVTrans.back();

        if(IsInList(aVTurn,aVImgs[i]))
        {
            aP = aVTrans.back();
            std::cout << "Reset Translation for " << aVImgs[i] << endl;
        }

        aVTrans.push_back(aP);
        if(i==(aNbIm-2)) aVTrans.push_back((aP));
    }
    //std::cout << aVTrans.size() << endl;

    // generation of rotation, axis: uniform distribution, angle: gaussian distribution
    if(!EAMIsInit(&aSeed))  aSeed=time(0);
    std::default_random_engine generator(aSeed);
    std::normal_distribution<double> angle(aGauss.x,aGauss.y);
    std::uniform_real_distribution<double> axis(-1.0,1.0);

    std::vector<ElMatrix<double>> aVRot;
    for(int i=0; i<aNbIm;i++)
    {
        Pt3dr aU = Pt3dr(axis(generator),axis(generator),axis(generator)); // axis of rotation
        aU = aU / euclid(aU); // normalize

        double aTeta = angle(generator)*aTInterv.y/1000;

        Pt3dr aCol1 = Pt3dr(
                            cos(aTeta) + aU.x*aU.x*(1-cos(aTeta)),
                            aU.y*aU.x*(1-cos(aTeta)) + aU.z*sin(aTeta),
                            aU.z*aU.x*(1-cos(aTeta)) - aU.y*sin(aTeta)
                           );
        Pt3dr aCol2 = Pt3dr(
                            aU.x*aU.y*(1-cos(aTeta)) - aU.z*sin(aTeta),
                            cos(aTeta) + aU.y*aU.y*(1-cos(aTeta)),
                            aU.z*aU.y*(1-cos(aTeta)) + aU.x*sin(aTeta)
                           );
        Pt3dr aCol3 = Pt3dr(
                            aU.x*aU.z*(1-cos(aTeta)) + aU.y*sin(aTeta),
                            aU.y*aU.z*(1-cos(aTeta)) + aU.x*sin(aTeta),
                            cos(aTeta) + aU.z*aU.z*(1-cos(aTeta))
                           );

        ElMatrix<double> aRot = MatFromCol(aCol1,aCol2,aCol3);
        aVRot.push_back(aRot);
    }
    //std::cout << aVRot.size() << endl;

    ELISE_ASSERT(aVTrans.size()==aVRot.size(),"Different size for translation and rotation!");
    ofstream aFile;
    aFile.open (aOut);

    aFile << "Random engine :" << aSeed << endl;
    aFile << "#F=Tx_Ty_Tz_R00_R01_R02_R10_R11_R12_R20_R21_R22" << endl;

    for(int i=0; i<aNbIm; i++)
    {
        aFile << aVTrans.at(i).x << " "
              << aVTrans.at(i).y << " "
              << aVTrans.at(i).z << " "
              << aVRot[i](0,0) << " "
              << aVRot[i](0,1) << " "
              << aVRot[i](0,2) << " "
              << aVRot[i](1,0) << " "
              << aVRot[i](1,1) << " "
              << aVRot[i](1,2) << " "
              << aVRot[i](2,0) << " "
              << aVRot[i](2,1) << " "
              << aVRot[i](2,2) << endl;
    }

    aFile.close();

    std::cout << aOut << endl;

    return EXIT_SUCCESS;
}

int ReechRolShut_main(int argc, char ** argv)
{
    std::string aSH,aSHOut{"_Reech"},aCamVFile,aMAFIn,aMAFOut,aDir,aSHIn;
    std::vector<double> aData;
    ElInitArgMain
            (
                argc, argv,
                LArgMain() << EAMC(aCamVFile,"File containg image velocity and image depth",eSAM_IsExistFile)
                           << EAMC(aData,"[image pixel height (um), focal length (mm), rolling shutter speed (us)]"),
                LArgMain() << EAM(aSH,"SHIn",true,"Input tie point file (new format)")
                           << EAM(aSHOut,"SHOut",true,"File postfix of the output tie point file (new format), def=_Reech")
                           << EAM(aMAFIn,"MAFIn",true,"Input image measurement file")
                           << EAM(aMAFOut,"MAFOut",true,"Output image measurement file")
                );
    // get directory
    SplitDirAndFile(aDir,aSHIn,aSH);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    double h = aData.at(0)/1000/1000; // um
    double f = aData.at(1)/1000; // mm
    double T = aData.at(2)/1000000; // us

    // read file containing image velocity and depth
    std::ifstream aFile(aCamVFile.c_str());

    std::map<std::string, double> aMapReechScale;
    if(aFile)
    {
        std::cout << "Read File : " << aCamVFile << endl;
        std::string aName;
        double aDiffPos,aDiffSecond,aV,aH;

        while(aFile >> aName >> aDiffPos >> aDiffSecond >> aV >> aH)
        {
            double lambda = 1 - f*aV*T/h/aH;
            std::cout << std::setprecision(10) << aName << " " << lambda << endl;
            aMapReechScale.insert(pair<std::string, double>(aName, lambda));
        }
        aFile.close();
    }

    if(EAMIsInit(&aSH))
    {
        // load tie points
        cSetTiePMul * pSH = new cSetTiePMul(0);
        pSH->AddFile(aSH);
        //std::map<std::string,cCelImTPM *> aVName2Im = pSH->DicoIm().mName2Im;

        // modification of tie points
        std::vector<cSetPMul1ConfigTPM *> aVCnf = pSH->VPMul();
        for(auto & aCnf:aVCnf)
        {
            std::vector<int> aVIdIm =  aCnf->VIdIm();
            // Parse all pts in one Config
            for(uint aKPt=0; aKPt<uint(aCnf->NbPts());aKPt++)
            {
                // Parse all imgs for one pt
                for(uint aKIm=0;aKIm<aVIdIm.size();aKIm++)
                {
                    std::string aNameIm = pSH->NameFromId(aVIdIm.at(aKIm));
                    Pt2dr aPt = aCnf->Pt(aKPt, aKIm);
                    Pt2dr aNewPt = Pt2dr(aPt.x,aPt.y*aMapReechScale[aNameIm]);
                    //std::cout << "Name:" << aNameIm << " Pt: " << aPt << " Scale:" << aMapReechScale[aNameIm] << endl;

                    aCnf->SetPt(aKPt,aKIm,aNewPt);
                }
            }
        }


        // output modifeied tie points
        std::string aNameOut0 = cSetTiePMul::StdName(aICNM,aSHOut,"Reech",0);
        std::string aNameOut1 = cSetTiePMul::StdName(aICNM,aSHOut,"Reech",1);

        pSH->Save(aNameOut0);
        pSH->Save(aNameOut1);
    }

    if(EAMIsInit(&aMAFIn))
    {
        cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aMAFIn,SetOfMesureAppuisFlottants);
        std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();

        for(auto & aMAF : aLMAF)
        {
            std::string aNameIm = aMAF.NameIm();
            std::list<cOneMesureAF1I> & aMes = aMAF.OneMesureAF1I();
            std::cout << aNameIm << endl;

            for(auto & aOneMes : aMes)
            {
                Pt2dr aPt = aOneMes.PtIm();
                std::cout << aOneMes.NamePt() << " before:" << aOneMes.PtIm();
                Pt2dr aNewPt = Pt2dr(aPt.x,aPt.y*aMapReechScale[aNameIm]);
                aOneMes.SetPtIm(aNewPt);
                std::cout << " after:" << aOneMes.PtIm() << endl;
            }
        }
        MakeFileXML(aDico,aMAFOut);
    }

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
