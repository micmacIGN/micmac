/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de
Correlation eLiSe  : ELements of an Image Software Environnement

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

#include "cNewO_SolGlobInit_RandomForest.h"
#include <math.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "general/bitm.h"
#include "general/exemple_basculement.h"
#include "general/opt_debug.h"
#include "general/photogram.h"
#include "general/ptxd.h"

using namespace SolGlobInit::RandomForest;

extern double DistBase(Pt3dr aB1, Pt3dr aB2);
extern double DistanceRot(const ElRotation3D& aR1, const ElRotation3D& aR2,
                          double aBSurH);

extern double PropPond(std::vector<Pt2df>& aV, double aProp, int* aKMed = 0);

/*static void PrintRotation(const ElMatrix<double> Mat,const std::string Msg)
{
    std::cout << " ===R" << Msg << " === \n";
    for (int l=0; l<3; l++)  {
        for (int c=0; c<3; c++) {
            std::cout << Mat(l,c) << " ";
        }
        std::cout << "\n";
    }
}*/

static cNO_CmpTriByCost TheCmp3;

static cNO_CmpTriSolByCost TheCmp3Sol;

//=====================================================================================

/******************************
  Start cNOSolIn_Triplet
*******************************/

cNOSolIn_Triplet::cNOSolIn_Triplet(RandomForest* anAppli,
                                   tSomNSI* aS1, tSomNSI* aS2, tSomNSI* aS3,
                                   const cXml_Ori3ImInit& aTrip)
    : mAppli(anAppli),
      mPdsSum(0),
      mCostPdsSum(0),
      mNb3(aTrip.NbTriplet()),
      mNumCC(IFLAG),
      mNumId(IFLAG),
      mNumTT(IFLAG),
      mR2on1(Xml2El(aTrip.Ori2On1())),
      mR3on1(Xml2El(aTrip.Ori3On1())),
      mBOnH(0.0)  // => in DostRot the translation error will not be taken into
                  // account
                  // mBOnH  (aTrip.BSurH())// => not always updated
                  // mBOnH  (0.01) //(aTrip.BSurH()) => not always updated
{
    mSoms[0] = aS1;
    mSoms[1] = aS2;
    mSoms[2] = aS3;
    residue = aTrip.ResiduTriplet();
    std::cout << residue << std::endl;
}

static double computeResiduFromPos(const cNOSolIn_Triplet* triplet) {
    double value = 0;
    std::cout << triplet->getHomolPts().size() << std::endl;
    double residues[3] = {};
    double n_res[3] = {0};
    for (uint i = 0; i < 3; i++) {
        auto r = triplet->KSom(i)->attr().CurRot().inv();
        std::cout
            << r.Mat()(0,0) << " " << r.Mat()(1,0) << " " << r.Mat()(2,0) << std::endl
            << r.Mat()(0,1) << " " << r.Mat()(1,1) << " " << r.Mat()(2,1) << std::endl
            << r.Mat()(0,2) << " " << r.Mat()(1,2) << " " << r.Mat()(2,2) << std::endl
            << r.tr() << std::endl;
        std::cout << "-----------------" << std::endl;
    }

    for (auto& pts : triplet->getHomolPts()) {
        std::vector<ElSeg3D> aVSeg;
        for (int i = 0; i < 3; i++) {
            if (!pts[i].x && !pts[i].y)
                continue;

            triplet->KSom(i)->attr().Im()->CS()->SetOrientation(triplet->KSom(i)->attr().CurRot().inv());
            aVSeg.push_back(triplet->KSom(i)->attr().Im()->CS()->Capteur2RayTer(pts[i]));
        }
        bool ISOK = false;
        Pt3dr aInt = ElSeg3D::L2InterFaisceaux(0, aVSeg, &ISOK);
        std::cout << "3D Intersection: " << aInt << std::endl;

        double residu_pts = 0;
        uint n_residu = 0;
        for (int i = 0; i < 3; i++) {
            if (!pts[i].x && !pts[i].y)
                continue;
            n_residu++;

            triplet->KSom(i)->attr().Im()->CS()->SetOrientation(triplet->KSom(i)->attr().CurRot().inv());
            Pt2dr pts_proj = triplet->KSom(i)->attr().Im()->CS()->Ter2Capteur(aInt);
            auto a = euclid(pts[i], pts_proj);
            std::cout
                << "Input point: " << pts[i] << " Output point: " << pts_proj
                << " Res: " << a
                << std::endl;
            residues[i] += a;
            n_res[i]++;
            residu_pts += a;
        }
        value += residu_pts/n_residu;
    }
    for (int i = 0; i < 3; i++) {
        std::cout << "Image: " << triplet->KSom(i)->attr().Im()->Name()
            << " Res: " << residues[i]/n_res[i] << std::endl;
    }
    value /= triplet->getHomolPts().size();
    cout.precision(12);
    std::cout << "Residu for triplet: " << value  << "  " << triplet->residue << std::endl;
    return value;
}

double cNOSolIn_Triplet::ProjTest() const {
    std::vector<ElRotation3D> aVRLoc;
    std::vector<ElRotation3D> aVRAbs;
    for (int aK = 0; aK < 3; aK++) {
        aVRLoc.push_back(RotOfK(aK));
        aVRAbs.push_back(mSoms[aK]->attr().CurRot());
    }
    double res = computeResiduFromPos(this);
    std::cout << "--- Residu: " << res << std::endl;
    return res;
}

double cNOSolIn_Triplet::CoherTest() const {
    std::vector<ElRotation3D> aVRLoc;
    std::vector<ElRotation3D> aVRAbs;
    for (int aK = 0; aK < 3; aK++) {
        aVRLoc.push_back(RotOfK(aK));
        aVRAbs.push_back(mSoms[aK]->attr().CurRot());
        // aVRAbs.push_back(mSoms[aK]->attr().TestRot());
    }

    cSolBasculeRig aSolRig = cSolBasculeRig::SolM2ToM1(aVRAbs, aVRLoc);
    double aRes = 0;

    for (int aK = 0; aK < 3; aK++) {
        const ElRotation3D& aRAbs = aVRAbs[aK];
        const ElRotation3D& aRLoc = aVRLoc[aK];
        ElRotation3D aRA2 = aSolRig.TransformOriC2M(aRLoc);

        double aD = DistanceRot(aRAbs, aRA2, mBOnH);
        aRes += aD;
        // std::cout << "RES=" << aRes  << " " << aD <<  " "<< mBOnH << "\n";
    }
    aRes = aRes / 3.0;

    return aRes;
}

void cNOSolIn_Triplet::SetArc(int aK, tArcNSI* anArc) { mArcs[aK] = anArc; }

double cNOSolIn_Triplet::CalcDistArc() {
    // Check whether sommet was orientated in this CC
    for (int aS = 0; aS < 3; aS++)
        if (this->KSom(aS)->attr().NumCC() == IFLAG) return IFLAG;

    double aDist = 0;
    // Not "solution" triplet
    if (this->NumTT() == IFLAG) {
        for (int aS = 0; aS < 3; aS++) aDist += this->KSom(aS)->attr().NumCC();
        aDist /= 3;
    } else {  // "Solution" triplet
        aDist = this->NumTT();
    }

    return aDist;
}

/******************************
  End cNOSolIn_Triplet
*******************************/

/******************************
  Start cLinkTripl
*******************************/

tSomNSI* cLinkTripl::S1() const { return m3->KSom(mK1); }
tSomNSI* cLinkTripl::S2() const { return m3->KSom(mK2); }
tSomNSI* cLinkTripl::S3() const { return m3->KSom(mK3); }

/******************************
  End cLinkTripl
*******************************/

/******************************
  Start cNOSolIn_AttrSom
*******************************/

cNOSolIn_AttrSom::cNOSolIn_AttrSom(const std::string& aName,
                                   RandomForest* anAppli)
    : mName(aName),
      mAppli(anAppli),
      mIm(new cNewO_OneIm(mAppli->NM(), mName)),
      mCurRot(ElRotation3D::Id),
      mTestRot(ElRotation3D::Id) {
}

//cNOSolIn_AttrSom::~cNOSolIn_AttrSom() { delete mIm; }

void cNOSolIn_AttrSom::AddTriplet(cNOSolIn_Triplet* aTrip, int aK1, int aK2,
                                  int aK3) {
    mLnk3.push_back(cLinkTripl(aTrip, aK1, aK2, aK3));
}

/******************************
  End cNOSolIn_AttrSom
*******************************/

/******************************
  Start cNOSolIn_AttrArc
*******************************/

cNOSolIn_AttrArc::cNOSolIn_AttrArc(cNOSolIn_AttrASym* anASym, bool OrASym)
    : mASym(anASym), mOrASym(OrASym) {}

/******************************
  End cNOSolIn_AttrArc
*******************************/

/******************************
  Start cNOSolIn_AttrASym
*******************************/

cNOSolIn_AttrASym::cNOSolIn_AttrASym() : mHeapTri(TheCmp3), mNumArc(IFLAG) {}

cLinkTripl* cNOSolIn_AttrASym::GetBestTri() {
    cLinkTripl* aLnk;
    if (mHeapTri.pop(aLnk)) return aLnk;

    return 0;
}

void cNOSolIn_AttrASym::AddTriplet(cNOSolIn_Triplet* aTrip, int aK1, int aK2,
                                   int aK3) {
    mLnk3.push_back(cLinkTripl(aTrip, aK1, aK2, aK3));
    mLnk3Ptr.push_back(new cLinkTripl(aTrip, aK1, aK2, aK3));
}

/******************************
  End cNOSolIn_AttrASym
*******************************/

/******************************
 * Start DataSet
 ******************************/

void Dataset::CreateArc(tSomNSI* aS1, tSomNSI* aS2, cNOSolIn_Triplet* aTripl,
                        int aK1, int aK2, int aK3) {
    tArcNSI* anArc = mGr.arc_s1s2(*aS1, *aS2);
    if (!anArc) {
        cNOSolIn_AttrASym* anAttrSym = new cNOSolIn_AttrASym;
        cNOSolIn_AttrArc anAttr12(anAttrSym, aS1 < aS2);
        cNOSolIn_AttrArc anAttr21(anAttrSym, aS1 > aS2);
        anArc = &(mGr.add_arc(*aS1, *aS2, anAttr12, anAttr21));
        mNbArc++;
    }
    anArc->attr().ASym()->AddTriplet(aTripl, aK1, aK2, aK3);
    aTripl->SetArc(aK3, anArc);
}
/******************************
 * end DataSet
 ******************************/

/******************************
 * Start DataTravel
 ******************************/

/*
 *   Add neighbouring/adjacent triplets of *anArc* to *mSCur3Adj*
 *
 * */
void DataTravel::AddArcOrCur(cNOSolIn_AttrASym* anArc) {
    AddArcOrCur(anArc, data.mFlagS);
}

void DataTravel::AddArcOrCur(cNOSolIn_AttrASym* anArc, int flagSommet) {
    // Adjacent triplets
    std::vector<cLinkTripl>& aLnk = anArc->Lnk3();
    for (unsigned aK = 0; aK < aLnk.size(); aK++) {
        // Test if the sommet S3 exists
        if (!aLnk.at(aK).S3()->flag_kth(flagSommet)) {
            // Add to adjacent list
            mSCur3Adj.insert(&(aLnk.at(aK)));
            continue;
        }

        // If S3 exists, try to add triplets adjacent to edges: S1-S3 and
        // S2-S3
        for (int aKArc = 0; aKArc < 3; aKArc++) {
            if (anArc == aLnk.at(aK).m3->KArc(aKArc)->attr().ASym()) {
                continue;
            }

            // Secondary triplets adjacent to S1-S3 or S2-S3
            std::vector<cLinkTripl>& aLnkSec =
                aLnk.at(aK).m3->KArc(aKArc)->attr().ASym()->Lnk3();

            for (int aT = 0; aT < int(aLnkSec.size()); aT++) {
                // Test if the "collateral" sommet S3 exists
                if (!aLnkSec.at(aT).S3()->flag_kth(flagSommet)) {
                    // Add to dynamic structure
                    mSCur3Adj.insert(&(aLnkSec.at(aT)));
                }
            }
        }
    }
}

/*
 *
 */
void DataTravel::FreeSCur3Adj(tSomNSI* aS) {
    std::vector<cLinkTripl>& aLnk = aS->attr().Lnk3();

    for (int aK = 0; aK < int(aLnk.size()); aK++) {
        std::set<cLinkTripl*>::iterator it = mSCur3Adj.find(&(aLnk[aK]));
        if (it != mSCur3Adj.end()) {
            mSCur3Adj.erase(it);
        }
    }
}

/* Randomly choose a triplet from mSCur3Adj which adj sommet has not been
 * visited */
cLinkTripl* DataTravel::GetRandTri() {
    cLinkTripl* aTri;
    if (!mSCur3Adj.size()) {
        return 0;
    }

    int aCpt = 0;
    do {
        int aRndTriIdx = NRrandom3(int(mSCur3Adj.size()) - 1);

        auto it = mSCur3Adj.begin();
        // Get the triplet
        std::advance(it, aRndTriIdx);
        aTri = *it;

        std::cout << ++aCpt << "====";
        // Remove triplet from the set => mark as explored
        mSCur3Adj.erase(it);

        // If the sommet was in the meantime added to global solution,
        // search for another one
    } while (aTri->S3()->flag_kth(data.mFlagS) && mSCur3Adj.size());

    return aTri;
}

/* Get the next Triplet from mSCur3Adj BFS style */
cLinkTripl* DataTravel::GetNextTri(int flag) {
    if (!mSCur3Adj.size()) {
        return nullptr;
    }
    cLinkTripl* aTri;
    auto it = std::min_element(mSCur3Adj.begin(), mSCur3Adj.end(),
                               [](cLinkTripl* a, cLinkTripl* b) {
                               return a->m3->NumTT() < b->m3->NumTT();
                               });
    aTri = *it;
    mSCur3Adj.erase(it);

    return aTri;
}

void DataTravel::FreeTriNumTTFlag(std::vector<cNOSolIn_Triplet*>& aV3) {
    for (auto aT : aV3) {
        aT->NumTT() = IFLAG;
    }
}

void DataTravel::FreeSomNumCCFlag(std::vector<tSomNSI*> aVS) {
    for (auto aS : aVS) {
        (*aS).attr().NumCC() = IFLAG;
    }
}

void DataTravel::FreeSomNumCCFlag() {
    for (tItSNSI anItS = data.mGr.begin(data.mSubAll); anItS.go_on(); anItS++) {
        (*anItS).attr().NumCC() = IFLAG;
    }
}

void DataTravel::resetFlags(cNO_CC_TripSom* aCC)
{
    // Free flags
    FreeAllFlag(aCC->mSoms, data.mFlagS);
    FreeAllFlag(aCC->mTri, data.mFlag3CC);

    // Free the set of current unvisted adjacent triplets
    mSCur3Adj.clear();

    // Reset the node concatenation order
    FreeSomNumCCFlag(aCC->mSoms);

    // Reset the triplet concatenation order
    FreeTriNumTTFlag(aCC->mTri);
}

/******************************
 * end DataTravel
*******************************/

/******************************
  Start cSolGlobInit_RandomForest
*******************************/

RandomForest::RandomForest(int argc, char** argv)
    : mDebug(false),
      mNbSamples(1000),
      mIterCur(0),
      mGraphCoher(true),
      mHeapTriAll(TheCmp3Sol),
      mHeapTriDyn(TheCmp3),
      mDistThresh(1e3),
      mResidu(20),
      mApplyCostPds(false),
      mAlphaProb(0.5) {

    ElInitArgMain(
        argc, argv, LArgMain() << EAMC(mFullPat, "Pattern"),
        LArgMain()
            << EAM(mDebug, "Debug", true, "Print some stuff, Def=false")
            << EAM(mNbSamples, "Nb", true, "Number of samples, Def=1000")
            << EAM(mGraphCoher, "GraphCoh", true,
                   "Graph-based incoherence, Def=true")
            /*<< EAM(mApplyCostPds, "CostPds", true,
                   "Apply Pds to cost, Def=false")*/
            << EAM(mDistThresh, "DistThresh", true,
                   "Apply distance threshold when computing incoh on samples")
            << EAM(mResidu, "Residu", true,
                   "Minimal residu considered as good. Def=20")
            << EAM(aModeBin, "Bin", true, "Binaries file, def = true",
                   eSAM_IsBool)
            /*<< EAM(mAlphaProb, "Alpha", true,
                   "Probability that a triplet at distance Dist is not an "
                   "outlier, Prob=Alpha^Dist; Def=0.5")*/
            << ArgCMA());

        mEASF.Init(mFullPat);
        // Loading of xml datafiles
        mNM = new cNewO_NameManager(mExtName, mPrefHom, mQuick, mEASF.mDir,
                mNameOriCalib, "dat");
}

static void AddVPts2Map(tMapM& aMap, ElPackHomologue& aPack, int anInd1, int anInd2) {
    for (ElPackHomologue::const_iterator itP = aPack.begin();
         itP != aPack.end(); itP++) {
        aMap.AddArc(itP->P1(), anInd1, itP->P2(), anInd2, cCMT_NoVal());
    }
}

void RandomForest::loadHomol(cNOSolIn_Triplet* aTriplet, tTriPointList& aLst) {
    //  recover tie-pts & tracks
    //
    // remembers whether inverse tie-pts exist
    std::string mHomExpIn("dat");
    std::string aKey = "NKS-Assoc-CplIm2Hom@" + mPrefHom + "@" + mHomExpIn;

    auto getKey = [=](uint8_t a, uint8_t b) {
        auto& one = aTriplet->KSom(a)->attr().Im()->Name();
        auto& two = aTriplet->KSom(b)->attr().Im()->Name();
        return mNM->ICNM()->Assoc1To2(aKey, one, two, true);
    };

    //Generate all pair id for triplet
    std::vector<std::pair<uint8_t, uint8_t>> couples;
    for (uint8_t i = 0; i < 3; i++)
        for (uint8_t j = i + 1; j < 3; j++)
            couples.push_back({i, j});

    tMapM aMap(3,false);
    //For each couple of triplet get the tie point and add it to the map
    for (auto c : couples) {
        auto key = getKey(c.first, c.second);
        auto ikey = getKey(c.second, c.first);

        ElPackHomologue aPack;
        if (ELISE_fp::exist_file(key)) {
            aPack = ElPackHomologue::FromFile(key);
            AddVPts2Map(aMap, aPack, c.first, c.second);
        } else if (ELISE_fp::exist_file(ikey)) { //Inverted
            aPack = ElPackHomologue::FromFile(ikey);
            AddVPts2Map(aMap, aPack, c.second, c.first);
        } else {
            std::cout << "NOT FOUND " << key << "\n";
        }
    }
    aMap.DoExport();

    tListM aLM = aMap.ListMerged();

    for (auto e : aLM) {
        if (e->NbSom() == 3) {
            //std::cout << "0 " << e->GetVal(0) << " 1 " << e->GetVal(1)
             //         << " 2 " << e->GetVal(2) << "\n";
            //aLst.push_back({e->GetVal(0), e->GetVal(1), e->GetVal(2)});
            aLst.push_back({e->GetVal(0), e->GetVal(1), e->GetVal(2)});
        }
    }
    ELISE_ASSERT(aLst.size() > 0, "Homol points empty for triplet");
}

void RandomForest::loadDataset(Dataset& data) {

    const cInterfChantierNameManipulateur::tSet* aVIm = mEASF.SetIm();
    //Loading of images
    for (unsigned aKIm = 0; aKIm < aVIm->size(); aKIm++) {
        const std::string& aName = (*aVIm)[aKIm];
        tSomNSI& aSom = data.mGr.new_som(cNOSolIn_AttrSom(aName, this));
        data.mMapS[aName] = &aSom;

        std::cout << data.mNbSom << "=" << aName << "\n";
        data.mNbSom++;
    }
    std::cout << "Loaded triplet" << std::endl;

    cXml_TopoTriplet aXml3 =
        StdGetFromSI(mNM->NameTopoTriplet(true), Xml_TopoTriplet);

    //Loading of triplets
    for (auto& it3 : aXml3.Triplets()) {
        ELISE_ASSERT(it3.Name1() < it3.Name2(),
                     "Incogeherence cAppli_NewSolGolInit\n");
        ELISE_ASSERT(it3.Name2() < it3.Name3(),
                     "Incogeherence cAppli_NewSolGolInit\n");

        tSomNSI* aS1 = data.mMapS[it3.Name1()];
        tSomNSI* aS2 = data.mMapS[it3.Name2()];
        tSomNSI* aS3 = data.mMapS[it3.Name3()];

        //We have the three image in our graph
        if (!aS1 || !aS2 || !aS3) continue;

        if (mDebug) {
            std::cout << "Tri=[" << it3.Name1() << "," << it3.Name2() << ","
                      << it3.Name3() << "\n";
        }
        data.mNbTrip++;

        std::string aN3 = mNM->NameOriOptimTriplet(
            aModeBin, aS1->attr().Im(), aS2->attr().Im(), aS3->attr().Im());

        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aN3, Xml_Ori3ImInit);

        cNOSolIn_Triplet* aTriplet =
            new cNOSolIn_Triplet(this, aS1, aS2, aS3, aXml3Ori);
        data.mV3.push_back(aTriplet);

        if (mDebug)
            std::cout << "% " << aN3 << " " << data.mV3.size() << " "
                      << data.mNbTrip << "\n";

        //Load tie point for the triplet :
        tTriPointList homolPts;
        loadHomol(aTriplet, homolPts);
        aTriplet->AddHomolPts(homolPts);

        ///  ADD-SOM-TRIPLET
        /*aS1->attr().AddTriplet(aTriplet, 1, 2, 0);
        aS2->attr().AddTriplet(aTriplet, 0, 2, 1);
        aS3->attr().AddTriplet(aTriplet, 0, 1, 2);*/

        aS1->attr().AddTriplet(aTriplet, 0, 1, 2);
        aS2->attr().AddTriplet(aTriplet, 1, 2, 0);
        aS3->attr().AddTriplet(aTriplet, 2, 0, 1);

        ///  ADD-EDGE-TRIPLET
        data.CreateArc(aS1, aS2, aTriplet, 0, 1, 2);
        data.CreateArc(aS2, aS3, aTriplet, 1, 2, 0);
        data.CreateArc(aS3, aS1, aTriplet, 2, 0, 1);

        // aTriplet->CheckArcsSom();
    }
    std::cout << "LOADED GRAPH " << mChrono.uval()
              << ", NbTrip=" << data.mNbTrip << "\n";

}

cLinkTripl* RandomForest::GetBestTriDyn() {
    cLinkTripl* aLnk;
    if (mHeapTriDyn.pop(aLnk)) return aLnk;

    return 0;
}

cNOSolIn_Triplet* RandomForest::GetBestTri() {
    cNOSolIn_Triplet* aLnk;
    if (mHeapTriAll.pop(aLnk)) return aLnk;

    return 0;
}

/* Create the Connected components */
void RandomForest::NumeroteCC(Dataset& data) {
    int aNumCC = 0;
    int aNumId = 0;

    DataTravel travel(data);
    for (unsigned aK3 = 0; aK3 < data.mV3.size(); aK3++) {
        cNOSolIn_Triplet* aTri0 = data.mV3[aK3];

        // If the triplet has not been marked, it's a new component
        if (aTri0->Flag().kth(data.mFlag3CC)) {
            continue;
        }

        // Create a new component
        cNO_CC_TripSom* aNewCC3S = new cNO_CC_TripSom;
        {
            aNewCC3S->mNumCC = aNumCC;  // give it a number
            data.mVCC.push_back(aNewCC3S);   // add it to the vector of component
        }
        // Quick acces to vec of tri in the CC
        std::vector<cNOSolIn_Triplet*>& aCC3 = aNewCC3S->mTri;
        // Quick access to som
        std::vector<tSomNSI*>& aCCS = aNewCC3S->mSoms;

        // Calcul des triplets
        aCC3.push_back(aTri0);  // Add triplet T0

        aTri0->Flag().set_kth_true(data.mFlag3CC);  // Mark it as explored
        aTri0->NumCC() = aNumCC;               // Put  right num to T0
        aTri0->NumId() = aNumId++;             //
        unsigned aKCur = 0;
        // Traditional loop of CC : while  no new inexplored neighboor
        while (aKCur != aCC3.size()) {
            cNOSolIn_Triplet* aTri1 = aCC3[aKCur];
            // For each edge of the current triplet
            for (int aKA = 0; aKA < 3; aKA++) {
                // Get triplet adjacent to this edge and parse them
                std::vector<cLinkTripl>& aLnk =
                    aTri1->KArc(aKA)->attr().ASym()->Lnk3();
                for (int aKL = 0; aKL < int(aLnk.size()); aKL++) {
                    // If not marked, mark it and push it in aCC3, return it was
                    // added
                    if (SetFlagAdd(aCC3, aLnk[aKL].m3, data.mFlag3CC)) {
                        aLnk[aKL].m3->NumCC() = aNumCC;
                        aLnk[aKL].m3->NumId() = aNumId;
                        aNumId++;
                    }
                }
            }
            aKCur++;
        }

        {
            // Compute the sommet of the CC, it's easy, just be careful to get
            // them only once
            int aFlagSom = data.mGr.alloc_flag_som();
            for (unsigned aKT = 0; aKT < aCC3.size(); aKT++) {
                cNOSolIn_Triplet* aTri = aCC3[aKT];
                for (int aKS = 0; aKS < 3; aKS++) {
                    // Add sommets of aTri to aNewCC3S sommets only if sommet
                    // not visited
                    SetFlagAdd(aCCS, aTri->KSom(aKS), aFlagSom);
                }
            }

            FreeAllFlag(aCCS, aFlagSom);
            data.mGr.free_flag_som(aFlagSom);
        }

        std::cout << "Nb of sommets " << aCCS.size() << " in CC " << aNumCC
                  << "\n";

        aNumCC++;
    }

    FreeAllFlag(data.mV3, data.mFlag3CC);
    std::cout << "Nb of CCs " << aNumCC << "\n";
}


static void EstimRt(cLinkTripl* aLnk) {
    // Get sommets
    tSomNSI* aS1 = aLnk->S1();
    tSomNSI* aS2 = aLnk->S2();
    tSomNSI* aS3 = aLnk->S3();

    // Get current R,t of the mother pair
    ElRotation3D aC1ToM = aS1->attr().CurRot();
    ElRotation3D aC2ToM = aS2->attr().CurRot();

    // Get rij,tij of the triplet sommets
    const ElRotation3D aC1ToL = aLnk->m3->RotOfSom(aS1);
    const ElRotation3D aC2ToL = aLnk->m3->RotOfSom(aS2);
    const ElRotation3D aC3ToL = aLnk->m3->RotOfSom(aS3);

    // Propagate R,t according to:
    // aC1ToM.tr() = T0 + aRL2M * aC1ToL.tr() * Lambda
    //
    // 1-R
    ElMatrix<double> aRL2Mprim = aC1ToM.Mat() * aC1ToL.Mat().transpose();
    ElMatrix<double> aRL2Mbis = aC2ToM.Mat() * aC2ToL.Mat().transpose();
    ElMatrix<double> aRL2M = NearestRotation((aRL2Mprim + aRL2Mbis) * 0.5);

    // 2-Lambda
    double d12L = euclid(aC2ToL.tr() - aC1ToL.tr());
    double d12M = euclid(aC2ToM.tr() - aC1ToM.tr());
    double Lambda = d12M / ElMax(d12L, 1e-20);

    // 3-t
    Pt3dr aC1ToLRot = aRL2M * aC1ToL.tr();
    Pt3dr aC2ToLRot = aRL2M * aC2ToL.tr();

    Pt3dr T0prim = aC1ToM.tr() - aC1ToLRot * Lambda;
    Pt3dr T0bis = aC2ToM.tr() - aC2ToLRot * Lambda;
    Pt3dr T0 = (T0prim + T0bis) / 2.0;

    Pt3dr aT3 = T0 + aRL2M * aC3ToL.tr() * Lambda;

    // 4- Set R3,t3
    aS3->attr().CurRot() = ElRotation3D(aT3, aRL2M * aC3ToL.Mat(), true);
    // used in coherence
    aS3->attr().TestRot() =
        ElRotation3D(aT3, aRL2M * aC3ToL.Mat(), true);

    aS3->attr().Im()->CS()->SetOrientation(aS3->attr().CurRot().inv());
}

/*
    Randomly start from triplet aSeed.
    And
**/
void RandomForest::RandomSolOneCC(Dataset& data, cNOSolIn_Triplet* aSeed, int NbSomCC) {
    int aNumCCSom = 1;
    DataTravel travel(data);

    // Mark the concatenation order of the seed triplet
    aSeed->NumTT() = aNumCCSom;

    for (int aK = 0; aK < 3; aK++) {
        // Set the current R,t of the seed
        aSeed->KSom(aK)->attr().CurRot() = aSeed->RotOfSom(aSeed->KSom(aK));

        // Mark as explored
        aSeed->KSom(aK)->flag_set_kth_true(data.mFlagS);

        // Mark the concatenation order of the node;
        // the first three nodes arise from the same triplet therefore the same
        // order
        aSeed->KSom(aK)->attr().NumCC() = aNumCCSom;
        // Id per sommet to know from which triplet it was generated => Fast
        // Tree Dist util
        aSeed->KSom(aK)->attr().NumId() = aSeed->NumId();
    }

    for (int aK = 0; aK < 3; aK++) {
        // Add the seed adjacent to the set of not visited triplets
        travel.AddArcOrCur(aSeed->KArc(aK)->attr().ASym());
    }

    int Cpt = 0;
    cLinkTripl* aTri = 0;
    while ((aTri = travel.GetRandTri()) && ((Cpt + 3) < NbSomCC)) {
        // Flag as visted
        aTri->m3->Flag().set_kth_true(data.mFlag3CC);

        // Flag triplet order
        aTri->m3->NumTT() =
            ElMax(aTri->S1()->attr().NumCC(), aTri->S2()->attr().NumCC()) + 1;

        // Flag the concatenation order of the node
        // = order of the "builder" triplet
        aTri->S3()->attr().NumCC() = aTri->m3->NumTT();

        // Id per sommet to know from which triplet it was generated => Fast
        // Tree Dist util
        aTri->S3()->attr().NumId() = aTri->m3->NumId();

        // Propagate R,t and flag sommet as visited
        EstimRt(aTri);

        // Mark sommit as vistied
        aTri->S3()->flag_set_kth_true(data.mFlagS);

        // Free mSCur3Adj from all triplets connected to S3
        travel.FreeSCur3Adj(aTri->S3());

        // Add two new edges and their respective adjacent triplets
        travel.AddArcOrCur(aTri->m3->KArc(aTri->mK1)->attr().ASym());
        travel.AddArcOrCur(aTri->m3->KArc(aTri->mK2)->attr().ASym());

        Cpt++;
    }

    std::cout << "\nIn this CC, nb of connected nodes " << Cpt + 3 << "\n";
}

/*
    On a connected component aCC :
    randomly build a tree and
*/
void RandomForest::RandomSolOneCC(Dataset& data, cNO_CC_TripSom* aCC) {
    DataTravel travel(data);
    //NRrandom3InitOfTime(); //TODO Activer pour debug
    std::cout << "DFS per CC, Nb Som " << aCC->mSoms.size() << ", nb triplets "
              << aCC->mTri.size() << "\n";

    // Select random seed triplet
    int aSeed = NRrandom3(int(aCC->mTri.size() - 1));

    cNOSolIn_Triplet* aTri0 = aCC->mTri[aSeed];
    std::cout << "Seed triplet " << aTri0->KSom(0)->attr().Im()->Name() << " "
              << aTri0->KSom(1)->attr().Im()->Name() << " "
              << aTri0->KSom(2)->attr().Im()->Name() << " "
              << aTri0->KSom(0)->attr().NumId() << " "
              << aTri0->KSom(1)->attr().NumId() << " "
              << aTri0->KSom(2)->attr().NumId() << "\n";

    // Flag as visited
    aTri0->Flag().set_kth_true(data.mFlag3CC);

    ELISE_ASSERT(aTri0 != 0, "Cannot compute seed in RandomSolOneCC");

    // Build the initial solution in this CC
    RandomSolOneCC(data, aTri0, aCC->mSoms.size());

    static int n = 0;
    GraphViz g;
    g.travelGraph(data, *aCC, aTri0);
    g.write("graph/random_tree_" + std::to_string(n++) + ".dot");

    // Calculate coherence scores within this CC
    CoherTripletsGraphBasedV2(data, aCC->mTri, aCC->mSoms.size(), aTri0->NumId());

    travel.resetFlags(aCC);
}

void RandomForest::RandomSolAllCC(Dataset& data, cNO_CC_TripSom* aCC) {
    return;
    DataTravel travel(data);
    NRrandom3InitOfTime();
    std::cout << "DFS per CC, Nb Som " << aCC->mSoms.size() << ", nb triplets "
              << aCC->mTri.size() << "\n";

    for (cNOSolIn_Triplet* aTri0 : aCC->mTri) {
        std::cout << "Seed triplet " << aTri0->KSom(0)->attr().Im()->Name()
                  << " " << aTri0->KSom(1)->attr().Im()->Name() << " "
                  << aTri0->KSom(2)->attr().Im()->Name() << " "
                  << aTri0->KSom(0)->attr().NumId() << " "
                  << aTri0->KSom(1)->attr().NumId() << " "
                  << aTri0->KSom(2)->attr().NumId() << "\n";

        // Flag as visited
        aTri0->Flag().set_kth_true(data.mFlag3CC);

        ELISE_ASSERT(aTri0 != 0, "Cannot compute seed in RandomSolOneCC");

        // Build the initial solution in this CC
        RandomSolOneCC(data, aTri0, aCC->mSoms.size());

        // Calculate coherence scores within this CC
        CoherTripletsGraphBasedV2(data, aCC->mTri, aCC->mSoms.size(), aTri0->NumId());

        travel.resetFlags(aCC);
    }
}


void RandomForest::AddTriOnHeap(Dataset& data, cLinkTripl* aLnk) {
    std::vector<cLinkTripl*> aVMaj;

    std::vector<int> aSIdx{aLnk->mK1, aLnk->mK2, aLnk->mK3};
    for (int aK = 0; aK < 3; aK++) {
        std::vector<cLinkTripl*>& aVT =
            aLnk->m3->KArc(aSIdx[aK])->attr().ASym()->Lnk3Ptr();

        for (int aT = 0; aT < int(aVT.size()); aT++) {
            // If triplet was visited continue
            if (aVT[aT]->m3->Flag().kth(data.mFlag3CC)) {
                continue;
            }

            // Add to heap
            mHeapTriDyn.push(aVT[aT]);

            // Push to aVMaj
            aVMaj.push_back(aVT[aT]);

            // Flag as marked
            aVT[aT]->m3->Flag().set_kth_true(data.mFlag3CC);

            /*std::cout << "___________ " ;
              std::cout << "added triplet Lnk "
              << aVT[aT]->S1()->attr().Im()->Name() << " "
              << aVT[aT]->S2()->attr().Im()->Name() << " "
              << aVT[aT]->S3()->attr().Im()->Name() << "\n";*/
        }
    }

    for (int aK = 0; aK < int(aVMaj.size()); aK++) {
        mHeapTriDyn.MAJ(aVMaj[aK]);
    }
}

void RandomForest::BestSolOneCC(Dataset& data, cNO_CC_TripSom* aCC) {
    int NbSomCC = int(aCC->mSoms.size());
    GraphViz g;
    DataLog<int, double, double> log;

    // Pick the  triplet
    cNOSolIn_Triplet* aTri0 = GetBestTri();//TODO tirer toujours le meme triplet
                                           //de depart pour debug
    std::cout << "Best triplet " << aTri0->KSom(0)->attr().Im()->Name() << " "
              << aTri0->KSom(1)->attr().Im()->Name() << " "
              << aTri0->KSom(2)->attr().Im()->Name() << " " << aTri0->CostArc()
              << "\n";

    log.add({aTri0->NumId(), 0., aTri0->CostArc()});

    // Flag triplet as marked
    aTri0->Flag().set_kth_true(data.mFlag3CC);

    for (int aK = 0; aK < 3; aK++) {
        // Flag sommets as explored
        aTri0->KSom(aK)->flag_set_kth_true(data.mFlagS);

        // Set the current R,t of the seed
        aTri0->KSom(aK)->attr().CurRot() = aTri0->RotOfSom(aTri0->KSom(aK));

        // PrintRotation(aTri0->KSom(aK)->attr().CurRot().Mat(),ToString(aK));
    }
    std::cout << "Final seed residue: " << computeResiduFromPos(aTri0) << std::endl;

    // initialise the starting node
    g.addTriplet(*aTri0);

    // Fill the dynamic heap with triplets connected to this triplet
    cLinkTripl* aLnk0 = new cLinkTripl(aTri0, 0, 1, 2);
    AddTriOnHeap(data, aLnk0);

    // Iterate to build the solution while updating the heap
    int Cpt = 0;
    cLinkTripl* aTriNext = 0;
    while ((aTriNext = GetBestTriDyn()) && ((Cpt + 3) < NbSomCC)) {
        // Check that the node has not been added in the meantime
        if (!aTriNext->S3()->flag_kth(data.mFlagS)) {
            std::cout << "=== Add new node " << Cpt << " "
                      << aTriNext->S1()->attr().Im()->Name() << " "
                      << aTriNext->S2()->attr().Im()->Name() << " "
                      << aTriNext->S3()->attr().Im()->Name() << " "
                      << aTriNext->m3->CostArcMed() << ", "
                      << aTriNext->m3->CostArc() << "\n";

            // Flag triplet order
            aTriNext->m3->NumTT() = ElMax(aTriNext->S1()->attr().NumCC(),
                                          aTriNext->S2()->attr().NumCC()) +
                                    1;

            // Flag the concatenation order of the node
            // = order of the "builder" triplet
            aTriNext->S3()->attr().NumCC() = aTriNext->m3->NumTT();

            // Id per sommet to know from which triplet it was generated => Fast
            // Tree Dist util
            aTriNext->S3()->attr().NumId() = aTriNext->m3->NumId();

            log.add({aTriNext->m3->NumId(), aTriNext->m3->NumTT(), aTriNext->m3->CostArc()});

            /*PrintRotation(aTriNext->S1()->attr().CurRot().Mat(),"0");
              PrintRotation(aTriNext->S2()->attr().CurRot().Mat(),"1");
              PrintRotation(aTriNext->S3()->attr().CurRot().Mat(),"2");*/

            // Propagate R,t
            EstimRt(aTriNext);

            g.addTriplet(*aTriNext->m3);

            // Mark node as vistied
            aTriNext->S3()->flag_set_kth_true(data.mFlagS);
            std::cout << "Final residue: " << computeResiduFromPos(aTriNext->m3) << std::endl;

            // Add to heap
            AddTriOnHeap(data, aTriNext);

            Cpt++;

        } else {  // however, try to adjacent triplets of that triplet

            // Add to heap
            AddTriOnHeap(data, aTriNext);
        }
    }

    g.write("graph/final.dot");
    log.write("graph/logs/final.csv");
    std::cout << "Nb final sommets=" << Cpt + 3 << ", out of " << NbSomCC
              << "\n";
}

void RandomForest::RandomSolAllCC(Dataset& data) {
    std::cout << "Nb of connected components=" << data.mVCC.size() << "\n";
    for (int aKC = 0; aKC < int(data.mVCC.size()); aKC++) {
        //RandomSolAllCC(data, data.mVCC[aKC]);
        RandomSolOneCC(data, data.mVCC[aKC]);
    }
}



//"%f,%f,%f('!')?"

void RandomForest::BestSolAllCC(Dataset& data) {
    // Add all triplets to global heap
    // (used only to get  seed)
    HeapPerSol(data);

    // Get  solution for each CC
    for (int aKC = 0; aKC < int(data.mVCC.size()); aKC++) {
        BestSolOneCC(data, data.mVCC[aKC]);

        // Save
        std::string aOutOri = "DSF_BestInit_CC" + ToString(aKC);
        Save(data, aOutOri, true);
    }

    // Free triplets
    FreeAllFlag(data.mV3, data.mFlag3CC);
}

/*
 * Incoherence score weighted by the distance in the graph
 *   Note1: a triplet may or may not have contributed to the solution
 *   Note2: a "solution" triplet contributes by adding a new sommet to the
 * solution Note3: the distance of a "solution" triplet is equivalent of the
 * order of the new sommet Note4: the distance of all other triplets is
 * equivalent of a mean distance of all three sommets
 *
 *   This function will also store the graph distances per each
 * */
// FIXME Cleanup
void RandomForest::CoherTripletsGraphBasedV2(
    Dataset& data, std::vector<cNOSolIn_Triplet*>& aV3, int NbSom,
    int TriSeedId) {
    std::cout << "Nb of sommets" << NbSom << " seed=" << TriSeedId << "\n";

    DataLog<int, double, double> log;

    // ==== Fast Tree Distance ===
    std::vector<int> aFTV1;
    std::vector<int> aFTV2;

    std::map<int, int> aCCGlob2Loc;

    NS_MMVII_FastTreeDist::cFastTreeDist aFTD(NbSom - 1);
    NS_MMVII_FastTreeDist::cAdjGraph aAdjG(NbSom - 1);

    // re-name to consecutive CC (necessary for Fast Tree Dist)
    int aItTriActive = 0;
    for (int aT = 0; aT < int(aV3.size()); aT++) {
        if ((aV3[aT]->NumTT() != IFLAG) &&
            (aV3[aT]->NumId() != IFLAG)) {
            aCCGlob2Loc[aV3[aT]->NumId()] = aItTriActive;
            aItTriActive++;
        }
    }

    cNOSolIn_Triplet* aTriSeed = aV3[TriSeedId];
    Pt3dr center;
    center = center + aTriSeed->KSom(0)->attr().CurRot().tr();
    center = center + aTriSeed->KSom(1)->attr().CurRot().tr();
    center = center + aTriSeed->KSom(2)->attr().CurRot().tr();
    center = center / 3.;

    // Build the tree for FastTree distance
    for (int aT = 0; aT < int(aV3.size()); aT++) {

        // If triplet not in solution tree avoid
        if ((aV3[aT]->NumTT() == IFLAG) ||
            (aV3[aT]->NumId() == IFLAG)) {
            continue;
        }
        // If triplet is The seed, it is the root of the tree
        if ((aV3[aT]->KSom(0)->attr().NumId() == TriSeedId) &&
            (aV3[aT]->KSom(1)->attr().NumId() == TriSeedId) &&
            (aV3[aT]->KSom(2)->attr().NumId() == TriSeedId))
            continue;

        //Get target at the first summit of the triplet
        int aS1 = aV3[aT]->KSom(0)->attr().NumId();
        if (aS1 == aV3[aT]->NumId()) aS1 = aV3[aT]->KSom(1)->attr().NumId();
        //If the first summit is the current triplet then go for the second.

        // Link aS1 and current summit in the tree
        aFTV1.push_back(aCCGlob2Loc[aS1]);
        aFTV2.push_back(aCCGlob2Loc[aV3[aT]->NumId()]);

        // std::cout << " Renamed=" << aCCGlob2Loc[aS1] << "-" <<
        // aCCGlob2Loc[aV3[aT]->NumId()] << " S1 " << aS1 << "*" <<
        // aV3[aT]->NumId() << " " << aV3[aT]->NumTT() << " ! " <<
        // aV3[aT]->KSom(0)->attr().NumCC() << "," <<
        // aV3[aT]->KSom(1)->attr().NumCC() << "," <<
        // aV3[aT]->KSom(2)->attr().NumCC() << " ========= " <<
        // aV3[aT]->KSom(0)->attr().NumId() << " " <<
        // aV3[aT]->KSom(1)->attr().NumId() << " " <<
        // aV3[aT]->KSom(2)->attr().NumId()  << "\n";
    }

    aFTD.MakeDist(aFTV1, aFTV2);
    aAdjG.InitAdj(aFTV1, aFTV2);
    aAdjG.Show();

    // int aCntFlags=0;
    for (int aT = 0; aT < int(aV3.size()); aT++) {
        auto& currentTriplet = aV3[aT];

        double aD1 = aFTD.Dist(aCCGlob2Loc[TriSeedId],
                            aCCGlob2Loc[currentTriplet->KSom(0)->attr().NumId()]);

        double aD2 = aFTD.Dist(aCCGlob2Loc[TriSeedId],
                            aCCGlob2Loc[currentTriplet->KSom(1)->attr().NumId()]);

        double aD3 = aFTD.Dist(aCCGlob2Loc[TriSeedId],
                            aCCGlob2Loc[currentTriplet->KSom(2)->attr().NumId()]);

        // std::cout << "TriNumId=" << aV3[aT]->NumId() << " | SomNumId=" <<
        // aV3[aT]->KSom(0)->attr().NumId() << " " <<
        // aV3[aT]->KSom(1)->attr().NumId() << " " <<
        // aV3[aT]->KSom(2)->attr().NumId() << "\n"; std::cout << "FTD= " << aD1
        // << " " << aD2 << " " << aD3 << "\n";

        /*std::cout << "["
          << aV3[aT]->KSom(0)->attr().NumId() << ","
          << aV3[aT]->KSom(1)->attr().NumId() << ","
          << aV3[aT]->KSom(2)->attr().NumId() << "] NumTT="
          << aV3[aT]->NumTT() << " "
          << aV3[aT]->KSom(0)->attr().Im()->Name() << " "
          << aV3[aT]->KSom(1)->attr().Im()->Name() << " "
          << aV3[aT]->KSom(2)->attr().Im()->Name() << " ";*/

        // double aDist = aV3[aT]->CalcDistArc();
        double aDist = std::ceil((aD1 + aD2 + aD3) / 3.0);
        Pt3dr center2;
        center2 = center2 + currentTriplet->KSom(0)->attr().CurRot().tr();
        center2 = center2 + currentTriplet->KSom(1)->attr().CurRot().tr();
        center2 = center2 + currentTriplet->KSom(2)->attr().CurRot().tr();
        center2 = center2 / 3.;

        double aDistAl = square_euclid(center2, center);
        std::cout << "Distance euclid: " << aDistAl << std::endl;

        if (aDist < mDistThresh) {
            double aResidue = abs(currentTriplet->ProjTest());
            //double aRot = abs(currentTriplet->CoherTest());
            // std::cout << ",Dist=" << aDist << " CohTest(à=" <<
            // aV3[aT]->CoherTest() << ",CostN=" <<  aCostCur << ",CPds=" <<
            // aCostCur/sqrt(aDist) << "\n"; std::cout <<
            // aTri->m3->Flag().set_kth_true(mFlag3CC);
            double aCostCur = 0;
            log.add({currentTriplet->NumId(), aDist, aResidue});

            // Take into account the non-visited triplets
            // TODO verifier les triplets qui sont dans la solution
            if (!ValFlag(*(aV3[aT]), data.mFlag3CC)) {
                // std::cout << "Flag0 OK " << aCntFlags++ << "\n";

                if (aDist <= 1) aDist = 1;

                double aPds = 1.0;

                if (mGraphCoher) {
                    //aPds = std::pow(0.5,aDist);
                    //aPds = std::pow(0.7,aDist);
                    //aPds = 1.0 / sqrt(aDist);
                    //aPds = 1.;
                    aPds = aDist;
                    aCostCur = aResidue;

                    /*if (aResidue > mResidu) {
                        aPds = 1./aPds;
                    }*/
                }

                // Mean
                aV3[aT]->CostPdsSum() += aCostCur;
                aV3[aT]->PdsSum() += aPds;

                // Median
                aV3[aT]->CostArcPerSample().push_back(aCostCur);
                aV3[aT]->DistArcPerSample().push_back(aPds);

                // Plot coherence vs sample vs distance
                std::cout << "==PLOT== " << aCostCur * aPds << " " << aDist
                          << " " << aPds << "\n";
            }
        }
    }
    static int n = 0;
    log.write("graph/logs/run_"+ std::to_string(n++) + ".csv");
}

/* obsolete
 * Pure incoherence/cost calculated as a function of rotational and translation
discrepancy # stores a N vector where N is the number of samples # stores only
the sum of Pds*Incoh and sum of Pds
*/
// FIXME remove if obsolete
/*
void RandomForest::CoherTriplets(std::vector<cNOSolIn_Triplet*>& aV3) {
    for (int aT = 0; aT < int(aV3.size()); aT++) {
        // Calculate the distance in the graph
        double aDist = aV3[aT]->CalcDistArc();

        // Cost
        double aCostCur = ElMin(abs(aV3[aT]->CoherTest()), 1e3);

        // Apply Pds to Cost
        // if (mApplyCostPds)
        //	aCostCur /=  sqrt(aDist);//Pds= sqrt(aDist)
        // aCostCur /= (1.0/std::pow(0.5,aDist) -1);//Pds=(1/0.5^d) -1

        // Update if distance above threshold
        if ((aDist < mDistThresh) && (aDist != IFLAG)) {
            aV3[aT]->CostPdsSum() += aCostCur * std::pow(mAlphaProb, aDist);
            aV3[aT]->PdsSum() += std::pow(mAlphaProb, aDist);

            aV3[aT]->CostArcPerSample().push_back(aCostCur);
            aV3[aT]->DistArcPerSample().push_back(aDist);

            // std::cout << "    Dist,SQRT,0.5^dist,cost=" << aDist << " " <<
            // sqrt(aDist) << " " << std::pow(mAlphaProb,aDist) << " " <<
            // aCostCur << "\n"; getchar();
        }
        // std::cout << "cost=" << aCostCur << "\n";
    }
}
*/

/* Old incoherence on all triplets in the graph */
// FIXME remove
/*
void RandomForest::CoherTriplets() {
    // std::cout << "size CostArcPerSample=" <<
    // int(mV3[0]->CostArcPerSample().size()) << "\n";

    for (int aT = 0; aT < int(mV3.size()); aT++) {
        double aCostCur = ElMin(abs(mV3[aT]->CoherTest()), 1e9);
        mV3[aT]->CostArcPerSample().push_back(aCostCur);

        // std::cout << "cost=" << aCostCur << "\n";
    }
}
*/

/* Final mean and 80% quantile incoherence computed on all triplets in the graph
 */
void RandomForest::CoherTripletsAllSamples(Dataset& data) {
    for (int aT = 0; aT < int(data.mV3.size()); aT++) {
        //if (!data.mV3[aT]->CostArcPerSample().size())
        //   continue;
        data.mV3[aT]->CostArc() = KthValProp(data.mV3[aT]->CostArcPerSample(), 0.8);
        data.mV3[aT]->CostArcMed() = MedianeSup(data.mV3[aT]->CostArcPerSample());
    }
}

/* Final incoherence computed as a weighted median
   - optionally takes into account only triplets with distance < threshold */
void RandomForest::CoherTripletsAllSamplesMesPond(Dataset& data) {
    for (int aT = 0; aT < int(data.mV3.size()); aT++) {
        /* Weighted mean */
        data.mV3[aT]->CostArc() = data.mV3[aT]->CostPdsSum() / data.mV3[aT]->PdsSum();

        std::vector<Pt2df> aVCostPds;
        for (int aS = 0; aS < int(data.mV3[aT]->CostArcPerSample().size()); aS++) {
            aVCostPds.push_back(Pt2df(data.mV3[aT]->CostArcPerSample()[aS],
                                      data.mV3[aT]->DistArcPerSample()[aS]));
            // aVCostPds.push_back (Pt2df(mV3[aT]->CostArcPerSample()[aS], 0));
            /*std::cout << std::setprecision(10);
              std::cout << " Cost|Dist|SQRTdist= " <<
              mV3[aT]->CostArcPerSample()[aS] << "|"
              << mV3[aT]->DistArcPerSample()[aS] << "|"
              << std::sqrt(mV3[aT]->DistArcPerSample()[aS]) << "\n";*/
        }

        /* Weighted median */
        if (aVCostPds.size())
            data.mV3[aT]->CostArcMed() =
                PropPond(aVCostPds, 0.1, 0);
             //MedianPond(aVCostPds,0);
    }
}

/* This heap will serve to GetBestTri when building the ultimate init solution
 */
void RandomForest::HeapPerEdge(Dataset& data) {
    // For all triplets
    for (auto aTri : data.mV3) {
        // For each edge of the current triplet
        for (int aK = 0; aK < 3; aK++) {
            std::vector<cLinkTripl*>& aLnk =
                aTri->KArc(aK)->attr().ASym()->Lnk3Ptr();

            // For all adjacent triplets to the current edge
            // Push to heap
            for (auto aTriAdj : aLnk) {
                aTri->KArc(aK)->attr().ASym()->mHeapTri.push(aTriAdj);
            }

            // Order index
            for (auto aTriAdj : aLnk) {
                aTri->KArc(aK)->attr().ASym()->mHeapTri.MAJ(aTriAdj);
            }
        }
    }
}

void RandomForest::HeapPerSol(Dataset& data) {
    // Update heap
    for (auto aTri : data.mV3) {
        mHeapTriAll.push(aTri);
    }

    // Order index
    for (auto aTri : data.mV3) {
        mHeapTriAll.MAJ(aTri);
    }
}

void RandomForest::Save(Dataset& data, std::string& OriOut, bool SaveListOfName) {
    std::list<std::string> aListOfName;

    for (tItSNSI anItS = data.mGr.begin(data.mSubAll); anItS.go_on(); anItS++) {
        if ((*anItS).flag_kth(data.mFlagS)) {
            cNewO_OneIm* anI = (*anItS).attr().Im();
            std::string aNameIm = anI->Name();
            CamStenope* aCS = anI->CS();
            ElRotation3D aROld2Cam = (*anItS).attr().CurRot().inv();

            aCS->SetOrientation(aROld2Cam);

            cOrientationConique anOC = aCS->StdExportCalibGlob();
            anOC.Interne().SetNoInit();

            std::string aFileIterne =
                mNM->ICNM()->StdNameCalib(mNameOriCalib, aNameIm);

            std::string aNameOri = mNM->ICNM()->Assoc1To1(
                "NKS-Assoc-Im2Orient@-" + OriOut, aNameIm, true);
            anOC.FileInterne().SetVal(NameWithoutDir(aFileIterne));

            // Copy interior orientation
            std::string aCom = "cp " + aFileIterne + " " + DirOfFile(aNameOri);
            System(aCom);

            aListOfName.push_back(aNameIm);

            MakeFileXML(anOC, aNameOri);
        }
    }

    if (SaveListOfName) {
        cListOfName aLOF;
        aLOF.Name() = aListOfName;
        MakeFileXML(aLOF, "ListPattern_" + OriOut + ".xml");
    }
}

void RandomForest::ShowTripletCostPerSample(Dataset& data) {
    for (auto aTri : data.mV3) {
        std::cout << "[" << aTri->KSom(0)->attr().Im()->Name() << ","
                  << aTri->KSom(1)->attr().Im()->Name() << ","
                  << aTri->KSom(2)->attr().Im()->Name() << "],\n";

        std::vector<double> aCostV = aTri->CostArcPerSample();
        std::vector<double> aDistV = aTri->DistArcPerSample();

        int aNb = int(aDistV.size());
        for (int aS = 0; aS < aNb; aS++) {
            std::cout << "[" << aCostV.at(aS) << "," << aDistV.at(aS) << "], ";
        }
        std::cout << "\n";
    }
}


void RandomForest::ShowTripletCost(Dataset& data) {
    for (auto aTri : data.mV3) {
        std::cout << "[" << aTri->KSom(0)->attr().Im()->Name() << ","
                  << aTri->KSom(1)->attr().Im()->Name() << ","
                  << aTri->KSom(2)->attr().Im()->Name() << "], "
                  << " Cost=" << aTri->CostArc()
                  << "   ,MED=" << aTri->CostArcMed() << "\n";
    }
}

// Entry point
void RandomForest::DoNRandomSol(Dataset& data) {
    // Create connected components
    NumeroteCC(data);

    // Build random inital solutions default 1000 ?
    for (int aIterCur = 0; aIterCur < mNbSamples; aIterCur++) {
        std::cout << "Iter=" << aIterCur << "\n";
        RandomSolAllCC(data);
    }

    // Calculate median/mean incoherence scores of mNbSamples
    //CoherTripletsAllSamples(data);
    CoherTripletsAllSamplesMesPond(data);

    // Print the cost for all triplets
    ShowTripletCost(data);
    if (mDebug) ShowTripletCostPerSample(data);

    // Build "most coherent" solution
    BestSolAllCC(data);

    GraphViz g;
    g.loadTotalGraph(data);
    g.write("graph/total.dot");
}

/******************************
  End cSolGlobInit_RandomForest
*******************************/

////////////////////////// Main //////////////////////////

int CPP_SolGlobInit_RandomForest_main(int argc, char** argv) {
    RandomForest aSGI(argc, argv);

    std::cout << "INIT" << std::endl;

    Dataset data;
    aSGI.loadDataset(data);

    aSGI.DoNRandomSol(data);

    return EXIT_SUCCESS;
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
