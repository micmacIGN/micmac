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

#include "cNewO_BasculeTriplets.h"
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include "XML_GEN/SuperposImage.h"
#include "algo_geom/cMesh3D.h"
#include "general/ptxd.h"

cTriplet::cTriplet(cOriBascule* s1, cOriBascule* s2, cOriBascule* s3,
                   cXml_Ori3ImInit& xml, std::array<short, 3>& mask)
    : masks(mask),
      mR2on1(Xml2El(xml.Ori2On1())),
      mR3on1(Xml2El(xml.Ori3On1())) {
    mSoms[0] = s1;
    mSoms[1] = s2;
    mSoms[2] = s3;
}

cOriTriplets::cOriTriplets() :
   mCam1 (0),
   mCam2 (0)
{
}

/******************************
  Start cAppliBasculeTriplets
*******************************/
void cAppliBasculeTriplets::InitOneDir(const std::string& aPat, bool aD1) {

    cElemAppliSetFile anEASF(aPat);
    std::string aDirNKS = "";  // si NKS-Set-OfFile contient le nom absolut

    const std::vector<std::string>* aVN = anEASF.SetIm();
    std::cout << "For pattern \"" << aPat;
    std::cout << "\", found ori: \n";
    for (size_t aK = 0; aK < aVN->size(); aK++) {
        std::string aNameOri = (*aVN)[aK];
        std::cout << "  - " << aNameOri << "\n";
        CamStenope* aCS = CamOrientGenFromFile(aNameOri, anEASF.mICNM);
        SplitDirAndFile(aDirNKS, aNameOri, aNameOri);
        cOriTriplets& anOri = mOrients[mapping[aNameOri]];
        anOri.mIm = mapping[aNameOri];
        std::cout << anOri.mIm << std::endl;
        anOri.mName = aNameOri;
        anOri.mNameFull = anEASF.mDir + aDirNKS + aNameOri;
        if (aD1) {
            anOri.mCam1 = aCS;
            block1.insert(anOri.mIm);
        } else {
            anOri.mCam2 = aCS;
            block2.insert(anOri.mIm);
        }
    }
}

void cAppliBasculeTriplets::InitTriplets(bool aModeBin)
{
    cXml_TopoTriplet aXml3 =
        StdGetFromSI(mNM->NameTopoTriplet(true), Xml_TopoTriplet);

    std::vector<cXml_OneTriplet> triplets(aXml3.Triplets().begin(),
                                          aXml3.Triplets().end());
    for (auto& it3 : aXml3.Triplets()) {
        std::string aN3 = mNM->NameOriOptimTriplet(aModeBin, it3.Name1(),
                                                   it3.Name2(), it3.Name3());

        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aN3, Xml_Ori3ImInit);

        std::string names[] = {it3.Name1(), it3.Name2(), it3.Name3()};
        int in1 = 0, in2 = 0;
        std::array<short, 3> mask;
        for (int i : {0, 1 , 2}) {
            if (block1.count(names[i])) {
                in1++;
                mask[i] = 1;
            }
            if (block2.count(names[i])) {
                in2++;
                mask[i] = 2;
            }
        }
        if (!in1 || !in2) {
            continue;
        }

        std::cout << in1 << " " << in2 << std::endl;
        cOriBascule* s1 = &mAllOris[it3.Name1()];
        cOriBascule* s2 = &mAllOris[it3.Name2()];
        cOriBascule* s3 = &mAllOris[it3.Name3()];
        cTriplet b(s1, s2, s3, aXml3Ori, mask);
        ts.emplace_back(std::move(b));
        cTriplet* t = &ts.back();
        for (int i : {0, 1 , 2}) {
            mOrients[names[i]].triplets.push_back(t);
        }
    }
    std::cout << "Found " << ts.size() << " correlated triplets." << std::endl;
}

void cAppliBasculeTriplets::IsolateRotTr()
{
    std::vector<cOriTraversal> bTraversals;
    for (auto t : ts) {
        for (int i : {0, 1, 2}) {
            if (t.masks[i] == 1) {
                for (int j : {0, 1, 2}) {
                    if (t.masks[j] == 2) {
                        bTraversals.push_back({t.mSoms[i], t.mSoms[j], t.RotOfK(j)});
                    }
                }
            }
        }
    }
}


cAppliBasculeTriplets::cAppliBasculeTriplets(int argc,char ** argv) :
    mRatioOutlier(0)
{
    NRrandom3InitOfTime();

    std::string aDir;
    bool aModeBin = true;
    std::string mFullPat;

    ElInitArgMain
        (
         argc,argv,
         LArgMain()
                    << EAMC(mFullPat, "Pattern")
                    << EAMC(mOri1,"First set of image", eSAM_IsPatFile)
                    << EAMC(mOri2,"Second set of image", eSAM_IsPatFile)
                    << EAMC(mOriOut,"Orientation Dir"),
         LArgMain() << EAM(aModeBin,"Bin",true,"Binaries file, def = true",eSAM_IsBool)
                    << ArgCMA()
        );

    mEASF.Init(mFullPat);
    mNM = new cNewO_NameManager(mExtName, mPrefHom, mQuick, mEASF.mDir,
                                mNameOriCalib, "dat");
    const cInterfChantierNameManipulateur::tSet * aVIm = mEASF.SetIm();
    for (unsigned aKIm = 0; aKIm < aVIm->size(); aKIm++) {
        std::string xmlfile = "Orientation-" + (*aVIm)[aKIm] + ".xml";
        mapping[xmlfile] = (*aVIm)[aKIm];
        mAllOris[(*aVIm)[aKIm]] = cOriBascule((*aVIm)[aKIm]);
    }

    InitOneDir(mOri1, true);
    InitOneDir(mOri2, false);

    mDirOutLoc =  "Ori-" + mOriOut + "/";
    mDirOutGlob = Dir() + mDirOutLoc;

    StdCorrecNameOrient(mNameOriCalib, aDir);
    // std::cout << mNM->Dir3P() << std::endl;
    InitTriplets(aModeBin);

    IsolateRotTr();



    /*
            std::pair<ElRotation3D, ElRotation3D> aPair =
                mNM->OriRelTripletFromExisting(InOri, it3.Name1(), it3.Name2(),
                                               it3.Name3(), Ok);

            std::string aNameSauveXml = mNM->NameOriOptimTriplet(
                false, it3.Name1(), it3.Name2(), it3.Name3(), false);
            std::string aNameSauveBin = mNM->NameOriOptimTriplet(
                true, it3.Name1(), it3.Name2(), it3.Name3(), false);
            //aXml.Sigma() = {cur_sigma_t, cur_sigma_r};
            //------------
                aXml.Ori2On1() = El2Xml(
                    ElRotation3D(aPair.first.tr(), aPair.first.Mat(), true));
                aXml.Ori3On1() = El2Xml(
                    ElRotation3D(aPair.second.tr(), aPair.second.Mat(), true));
                aXml.Ori2On1() = El2Xml(RandView(
                    ElRotation3D(aPair.first.tr(), aPair.first.Mat(), true),
                    cur_sigma_t, cur_sigma_r));

            //------------
            MakeFileXML(aXml, aNameSauveXml);
            MakeFileXML(aXml, aNameSauveBin);
            */
}


////////////////////////// Mains //////////////////////////

int BasculeTriplet_main(int argc,char ** argv)
{
    cAppliBasculeTriplets aAppBascTri(argc,argv);

    return EXIT_SUCCESS;
}
