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

#include "cNewO_GenOptTriplets.h"
#include <iostream>
#include <vector>
#include "general/ptxd.h"

/******************************
  Start cAppliGenOptTriplets
*******************************/

cAppliGenOptTriplets::cAppliGenOptTriplets(int argc,char ** argv) :
    mRatioOutlier(0),
    TheRandUnif(new RandUnifQuick(100))
{
    /*std::set<double> RandSet; //:
      RandSet.insert(1.0);
      RandSet.insert(1.0);
      RandSet.insert(2.0);
      std::cout << RandSet.size() << "\n";
      getchar();
      for (int i=0; i<100000; i++)
      {
      double rndNo = TheRandUnif->Unif_0_1();
      RandSet.insert(rndNo);
      }
      std::cout << RandSet.size() << "\n";
      getchar();*/

    NRrandom3InitOfTime();

    std::string aDir;
    bool aModeBin = true;
    bool aModePerfect = true;
    double aRange = 1;

    std::vector<std::string> mSigmaTStr;
    std::vector<std::string> mSigmaRStr;
    std::vector<std::string> mRatioOutlierStr;

    ElInitArgMain
        (
         argc,argv,
         LArgMain() << EAMC(mFullPat,"Pattern of images")
                    << EAMC(InOri,"InOri that wil serve to build perfect triplets"),
         LArgMain() << EAM(mSigmaTStr,"SigmaT",true,"Sigma of the translation added noise, Def=[] no noise added")
                    << EAM(mSigmaRStr,"SigmaR",true,"Sigma of the rotation added noise, Def=[] no noise added")
                    << EAM(mRatioOutlierStr,"Ratio", true, "Good to bad triplet ratio (outliers), Def=[]")
                    << EAM(aRange,"Range",true,"Range of the noise, def = 1")
                    << EAM(aModeBin,"Bin",true,"Binaries file, def = true",eSAM_IsBool)
                    << EAM(aModePerfect,"Perfect",true,"Use perfect triplet by default, otherwise origin triplets, def = true",eSAM_IsBool)
                    << ArgCMA()
        );

    mEASF.Init(mFullPat);
    mNM = new cNewO_NameManager(mExtName, mPrefHom, mQuick, mEASF.mDir,
                                mNameOriCalib, "dat");
    // const cInterfChantierNameManipulateur::tSet * aVIm = mEASF.SetIm();

    StdCorrecNameOrient(mNameOriCalib, aDir);
    StdCorrecNameOrient(InOri, aDir);
    // std::cout << mNM->Dir3P() << std::endl;

    cXml_TopoTriplet aXml3 =
        StdGetFromSI(mNM->NameTopoTriplet(aModeBin), Xml_TopoTriplet);

    std::vector<cXml_OneTriplet> triplets(aXml3.Triplets().begin(),
                                          aXml3.Triplets().end());
    std::vector<int> ratios;
    std::vector<double> sigma_t;
    std::vector<double> sigma_r;

    ratios.push_back(0);
    if (mRatioOutlierStr.size()) {
        double ratio_sum = 0;
        size_t nsection = 0;

        ELISE_ASSERT(
            mSigmaTStr.size() == mSigmaRStr.size() &&
                mSigmaRStr.size() == mRatioOutlierStr.size(),
            "You must have a sigmaT and SigmaR for each ratio section.");
        for (auto r : mRatioOutlierStr) {
            double curR = RequireFromString<double>(r, "Section of ratio");
            ratio_sum += curR;

            ELISE_ASSERT(ratio_sum <= 1., "Sum of Ratio must be <= 1.");
            ratios.push_back(std::floor(ratio_sum * triplets.size()));
            nsection++;
        }
        for (auto r : mSigmaTStr) {
            double curST = RequireFromString<double>(r, "Value of sigmaT");
            sigma_t.push_back(curST);
        }
        for (auto r : mSigmaRStr) {
            double curSR = RequireFromString<double>(r, "Value of sigmaR");
            sigma_r.push_back(curSR);
        }
    }
    ratios.push_back(triplets.size());
    sigma_t.push_back(0);
    sigma_r.push_back(0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(triplets.begin(), triplets.end(), g);

    // Save the triplets and perturb with outliers if asked
    // For each section, apply sigmaT and SigmaT noise
    for (size_t s = 0; s < ratios.size() - 1; s++) {
        size_t start = ratios[s];
        size_t end = ratios[s + 1];
        double cur_sigma_t = sigma_t[s];
        double cur_sigma_r = sigma_r[s];
        std::cout << "---- Section [" << start << ":" << end << "[ with SigmaT: "
                  << cur_sigma_t << " SigmaR: " << cur_sigma_r << " ---- "<< std::endl;
        for (size_t j = start; j < end && end <= triplets.size(); j++) {
            auto& it3 = triplets[j];

            bool Ok;
            std::pair<ElRotation3D, ElRotation3D> aPair =
                mNM->OriRelTripletFromExisting(InOri, it3.Name1(), it3.Name2(),
                                               it3.Name3(), Ok);

            std::string aNameSauveXml = mNM->NameOriOptimTriplet(
                false, it3.Name1(), it3.Name2(), it3.Name3(), false);
            std::string aNameSauveBin = mNM->NameOriOptimTriplet(
                true, it3.Name1(), it3.Name2(), it3.Name3(), false);
            cXml_Ori3ImInit aXml;
            aXml.IsGen() = true;
            aXml.GenCat() = s;
            //aXml.Sigma() = {cur_sigma_t, cur_sigma_r};
            //------------
            if (!cur_sigma_t && !cur_sigma_r) {
                if (aModePerfect) {
                    aXml.Ori2On1() = El2Xml(ElRotation3D(
                        aPair.first.tr(), aPair.first.Mat(), true));
                    aXml.Ori3On1() = El2Xml(ElRotation3D(
                        aPair.second.tr(), aPair.second.Mat(), true));

                } else {
                    std::string aN3 = mNM->NameOriOptimTriplet(
                        aModeBin, it3.Name1(), it3.Name2(), it3.Name3());
                    cXml_Ori3ImInit aXml3Ori =
                        StdGetFromSI(aN3, Xml_Ori3ImInit);
                    aXml.Ori2On1() = aXml3Ori.Ori2On1();
                    aXml.Ori3On1() = aXml3Ori.Ori3On1();
                }

            } else {
                aXml.Ori2On1() = El2Xml(RandView(
                    ElRotation3D(aPair.first.tr(), aPair.first.Mat(), true),
                    cur_sigma_t, cur_sigma_r, aRange));

                aXml.Ori3On1() = El2Xml(RandView(
                    ElRotation3D(aPair.second.tr(), aPair.second.Mat(), true),
                    cur_sigma_t, cur_sigma_r, aRange));

                std::cout << "Perturbed S:" << s << " R=[" << it3.Name1() << ","
                          << it3.Name2() << "," << it3.Name3() << "], "
                          << aPair.first.tr() << " " << aPair.second.tr()
                          << "\n";
            }
            //------------
            MakeFileXML(aXml, aNameSauveXml);
            MakeFileXML(aXml, aNameSauveBin);
        }
    }
}

/* Kanatani : Geometric Computation for Machine Vision */
ElMatrix<double> cAppliGenOptTriplets::w2R(double w[])
{
    ElMatrix<double> aRes(3,3,0);//identity
    for (int i=0; i<3; i++)
        aRes(i,i)=1.0;

    double norm = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);

    if (norm>0)
    {
        w[0] /= norm;
        w[1] /= norm;
        w[2] /= norm;

        double s = sin(norm);
        double c = cos(norm);
        double cc = 1-c;

        aRes(0,0) = c+w[0]*w[0]*cc;  //c+n1*n1*cc
        aRes(1,0) = w[0]*w[1]*cc+w[2]*s;//n12cc+n3s
        aRes(2,0) = w[2]*w[0]*cc-w[1]*s;//n31cc-n2s

        aRes(0,1) = w[0]*w[1]*cc-w[2]*s;//n12cc-n3s;
        aRes(1,1) = c+w[1]*w[1]*cc;//c+n2*n2*cc;
        aRes(2,1) = w[1]*w[2]*cc+w[0]*s;//n23cc+n1s;

        aRes(0,2) = w[2]*w[0]*cc+w[1]*s;//n31cc+n2s;
        aRes(1,2) = w[1]*w[2]*cc+w[0]*s;//n23cc-n1s;
        aRes(2,2) = c+w[2]*w[2]*cc;//c+n3*n3*cc;

    }

    return aRes;

}

ElMatrix<double> cAppliGenOptTriplets::RandPeturbRGovindu()
{
    double aW[] = {100*PI*(TheRandUnif->Unif_0_1()-0.5)*2.0,
        100*PI*(TheRandUnif->Unif_0_1()-0.5)*2.0,
        100*PI*(TheRandUnif->Unif_0_1()-0.5)*2.0};


    return w2R(aW);
}

ElRotation3D cAppliGenOptTriplets::RandView(const ElRotation3D& view, double sigmaT,
                                            double sigmaR, double range) {
    double sigmaT_offset = sigmaT * (1.-range);
    double sigmaT_rnd = sigmaT * range;
    auto deviation = Pt3dr(
            sigmaT_offset + sigmaT_rnd * (TheRandUnif->Unif_0_1() - 0.5) * 2.,
            sigmaT_offset + sigmaT_rnd * (TheRandUnif->Unif_0_1() - 0.5) * 2.,
            sigmaT_offset + sigmaT_rnd * (TheRandUnif->Unif_0_1() - 0.5) * 2.
                           );
    std::cout << "Deviation" << deviation << std::endl;
    Pt3dr tr = view.tr() + deviation;

    double sigmaR_offset = sigmaR * (1.-range);
    double sigmaR_rnd = sigmaR * range;
    double aW[] = {
        sigmaR_offset + sigmaR_rnd * (TheRandUnif->Unif_0_1() - 0.5) * 2.0,
        sigmaR_offset + sigmaR_rnd * (TheRandUnif->Unif_0_1() - 0.5) * 2.0,
        sigmaR_offset + sigmaR_rnd * (TheRandUnif->Unif_0_1() - 0.5) * 2.0,
    };
    ElMatrix<double> WMat =
        MatFromCol(Pt3dr(     0,   aW[2], -aW[1]),
                   Pt3dr(-aW[2],       0,  aW[0]),
                   Pt3dr( aW[1],  -aW[0],      0));
    ElMatrix<double> rotation_pert = view.Mat() + view.Mat() * WMat;
    ElMatrix<double> rotation = NearestRotation(rotation_pert);

    return ElRotation3D(tr, rotation, true);
}

ElMatrix<double> cAppliGenOptTriplets::RandPeturbR()
{

    double aW[] = {NRrandom3(),NRrandom3(),NRrandom3()};

    Pt3dr aI(exp(0),exp(aW[2]),exp(-aW[1]));
    Pt3dr aJ(exp(-aW[2]),exp(0),exp(aW[0]));
    Pt3dr aK(exp(aW[1]),exp(-aW[0]),exp(0));

    ElMatrix<double> aRes = MatFromCol(aI,aJ,aK);


    return aRes;
}

/******************************
  End cAppliGenOptTriplets
*******************************/

/******************************
  Start RandUnifQuick
*******************************/

RandUnifQuick::RandUnifQuick(int Seed):
    mGen     (Seed),
    mDis01   (0.0,1.0)
{}

double RandUnifQuick::Unif_0_1()
{
    return mDis01(mGen);
}

/******************************
  end RandUnifQuick
*******************************/



////////////////////////// Mains //////////////////////////

int CPP_GenOptTriplets(int argc,char ** argv)
{
    cAppliGenOptTriplets aAppGenTri(argc,argv);

    return EXIT_SUCCESS;
}
