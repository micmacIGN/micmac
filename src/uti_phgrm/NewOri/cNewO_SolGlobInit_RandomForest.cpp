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
#include <fcntl.h>
#include <graphviz/types.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ios>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <ostream>
#include <ratio>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <stack>

#include "ext_stl/numeric.h"
#include "general/PlyFile.h"
#include "general/bitm.h"
#include "general/exemple_basculement.h"
#include "general/opt_debug.h"
#include "general/photogram.h"
#include "general/ptxd.h"
#include "general/util.h"

#include <time.h>

using namespace SolGlobInit::RandomForest;

extern double DistBase(Pt3dr aB1, Pt3dr aB2);
extern double DistanceRot(const ElRotation3D& aR1, const ElRotation3D& aR2,
                          double aBSurH);

extern double PropPond(std::vector<Pt2df>& aV, double aProp, int* aKMed = 0);

/* //
static void PrintRotation(const ElMatrix<double> Mat,const std::string Msg)
{
    std::cout << " ===R" << Msg << " === \n";
    for (int l=0; l<3; l++)  {
        for (int c=0; c<3; c++) {
            std::cout << Mat(l,c) << " ";
        }
        std::cout << "\n";
    }
}
// */

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
    category = aTrip.GenCat();
}
/*
// check if one point is in the image
static bool IsInImage(Pt2di aSz, Pt2dr aPt)
{
    return (aPt.x >= 0) && (aPt.x < double(aSz.x)-0.5) && (aPt.y >= 0) && (aPt.y < double(aSz.y)-0.5);
}

*/

double median(vector<double> vec)
{
        typedef vector<double>::size_type vec_sz;

        vec_sz size = vec.size();
        if (size == 0)
                throw domain_error("median of an empty vector");

        sort(vec.begin(), vec.end());

        vec_sz mid = size/2;

        return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}
static double computeResiduFromPos(const cNOSolIn_Triplet* triplet) {
    //double value = 0;
    //double number = 0;
    //double MaxDiag = 6400; //Cam light
    double MaxDiag = 2202; //1920
    {
        double x = triplet->KSom(0)->attr().Im()->CS()->Sz().x;
        double y = triplet->KSom(0)->attr().Im()->CS()->Sz().y;
        MaxDiag = sqrt(x*x + y*y);
        //std::cout << "Diag: " << MaxDiag << std::endl;
    }
    std::vector<double> res;
    //std::cout << triplet->getHomolPts().size() << std::endl;
    //double residues[3] = {};
    for (uint i = 0; i < 3; i++) {
        auto r = triplet->KSom(i)->attr().CurRot().inv();
        /*
        std::cout
            << r.Mat()(0,0) << " " << r.Mat()(1,0) << " " << r.Mat()(2,0) << std::endl
            << r.Mat()(0,1) << " " << r.Mat()(1,1) << " " << r.Mat()(2,1) << std::endl
            << r.Mat()(0,2) << " " << r.Mat()(1,2) << " " << r.Mat()(2,2) << std::endl
            << r.tr() << std::endl;
        std::cout << "-----------------" << std::endl;*/
    }

    if (!triplet->getHomolPts().size()) {
        return MaxDiag;
    }

    for (auto& pts : triplet->getHomolPts()) {
        std::vector<ElSeg3D> aVSeg;
        for (int i = 0; i < 3; i++) {
            if (!pts[i].x && !pts[i].y)
                continue;

            triplet->KSom(i)->attr().Im()->CS()->SetOrientation(triplet->KSom(i)->attr().CurRot().inv());
            //aVSeg.push_back(triplet->KSom(i)->attr().Im()->CS()->F2toRayonR3(pts[i]));
            aVSeg.push_back(triplet->KSom(i)->attr().Im()->CS()->Capteur2RayTer(pts[i]));
        }
        bool ISOK = false;
        Pt3dr aInt = ElSeg3D::L2InterFaisceaux(0, aVSeg, &ISOK);
        //std::cout << "3D Intersection: " << aInt << std::endl;
        if (!ISOK) {
            std::cout << ISOK << std::endl;
        }
        //
        double residu_pts = 0;
        uint n_residu = 0;
        for (int i = 0; i < 3; i++) {
            if (!pts[i].x && !pts[i].y) continue;

            triplet->KSom(i)->attr().Im()->CS()->SetOrientation(
                triplet->KSom(i)->attr().CurRot().inv());
            Pt2dr pts_proj =
                triplet->KSom(i)->attr().Im()->CS()->Ter2Capteur(aInt);
            auto a = euclid(pts[i], pts_proj);
            /*std::cout
              << "Input point: " << pts[i] << " Output point: " << pts_proj
              << " Res: " << a
              << std::endl;*/
            // residues[i] += a;
            a = min(a, MaxDiag);
            //if (triplet->KSom(i)->attr().Im()->CS()->PIsVisibleInImage(aInt)) {
            //} else {
            //}
            residu_pts += a;
            n_residu++;
        }

        double m = MaxDiag;
        if (n_residu) {
            m = residu_pts / n_residu;
            res.push_back(m);
        }
        if (m > MaxDiag) {
            std::cout << "!!!! Max residue " << m << "---" << std::endl;
            //m = MaxDiag;
        }

        //value += m;
        //number += 1;
    }
    /*
    for (int i = 0; i < 3; i++) {
        std::cout << "Image: " << triplet->KSom(i)->attr().Im()->Name()
            << " Res: " << residues[i]/n_res[i] << std::endl;
    }*/
    //value /= number;
    //cout.precision(12);
    //std::cout << "Residu for triplet: " << value  << "  " << triplet->residue << std::endl;
    if (res.size() > 0)
        return median(res);
    return MaxDiag;
}

double cNOSolIn_Triplet::ProjTest() const {
    std::vector<ElRotation3D> aVRLoc;
    std::vector<ElRotation3D> aVRAbs;
    for (int aK = 0; aK < 3; aK++) {
        aVRLoc.push_back(RotOfK(aK));
        aVRAbs.push_back(mSoms[aK]->attr().CurRot());
    }
    double res = computeResiduFromPos(this);
    //std::cout << "--- Residu: " << res << std::endl;
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
    //mLnk3Ptr.push_back(new cLinkTripl(aTrip, aK1, aK2, aK3));
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
 */
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
    for (auto it = mSCur3Adj.begin(); it != mSCur3Adj.end(); ) {
        if ((*it)->S3() == aS) {
            it = mSCur3Adj.erase(it);
        } else {
            it++;
        }
    }
}

/*
 * note: lambda should be a positive number
 *       min and max should be non-negative and min < max
 */
/*
static double rand_exp(double lambda, double min, double max)
{
    double u_min = exp(-min * lambda);
    double u_max = exp(-max * lambda);

    double u = u_min + (1.0 - rand() / (RAND_MAX + 1.0)) * (u_max - u_min);
    return -log(u) / lambda;
}
*/

/* Randomly choose a triplet from mSCur3Adj which adj sommet has not been
 * visited */
cLinkTripl* DataTravel::GetRandTri(bool Pond) {
    cLinkTripl* aTri;
    if (!mSCur3Adj.size()) {
        return 0;
    }
    const double a = 5;
    //const double b = 1;
    double s = mSCur3Adj.size();
    std::vector<double> i{0,
        s * 1. / 4.,
        s * 2. / 4.,
        s * 3. / 4.,
        s - 1};
    //std::vector<double> w{5, 3, 2, 1, 0};
    std::vector<double> w;
    for (auto x : i) {
        w.push_back(-a * (x - s));
        //w.push_back(-a*x*x - b*x+a*s*s+(b*s));
    }
    std::piecewise_linear_distribution<> d{i.begin(), i.end(), w.begin()};

    do {
        int aRndTriIdx;
        if (Pond) {
            if (s > 1) {
                double r = d(gen);
                //double r = rand_exp(0.001, 0, mSCur3Adj.size());

                aRndTriIdx = r;
                //std::cout << "--- " << d.max() << " " << aRndTriIdx << std::endl;
            } else {
                aRndTriIdx = 0;
            }

        } else {
            aRndTriIdx = NRrandom3(int(mSCur3Adj.size()) - 1);
        }

        auto it = mSCur3Adj.begin();
        // Get the triplet
        std::advance(it, aRndTriIdx);
        aTri = *it;

        //std::cout << ++aCpt << "====";
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
    : mOutName("DSF_BestInit_CC"),
      mDebug(false),
      mNbSamples(1000),
      mIterCur(0),
      mGraphCoher(true),
      mHeapTriAll(TheCmp3Sol),
      mHeapTriDyn(TheCmp3),
      mDistThresh(1e3),
      mResidu(20),
      mApplyCostPds(false),
      mAlphaProb(0.5),
      mR0(100) {
    ElInitArgMain(
        argc, argv, LArgMain() << EAMC(mFullPat, "Pattern"),
        LArgMain()
            << EAM(mDebug, "Debug", true, "Print some stuff, Def=false")
            << EAM(mNbSamples, "Nb", true, "Number of samples, Def=1000")
            << EAM(mOutName, "OriOut", true,
                   "Orientation output name for each block. Def=DSF_BestInit_CC")
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
            << EAM(mR0, "R0", true, "The R0 for selection, def = 100")
            << EAM(aPond, "Pond", true, "If ponderate random, def = true", eSAM_IsBool)
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

void RandomForest::loadHomol(cNOSolIn_Triplet* aTriplet, tTriPointList& aLst,
                             tTriPointList& aLstAll) {
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
            /*std::cout << "0 " << e->GetVal(0) << " 1 " << e->GetVal(1)
                      << " 2 " << e->GetVal(2) << std::endl;*/
            //aLst.push_back({e->GetVal(0), e->GetVal(1), e->GetVal(2)});
            aLst.push_back({e->GetVal(0), e->GetVal(1), e->GetVal(2)});
        }
        aLstAll.push_back({e->GetVal(0), e->GetVal(1), e->GetVal(2)});
    }
    //std::cout << aLst.size() << std::endl;
    if (aLst.size() == 0)
        std::cout << "No tie point for Triplet["
            << aTriplet->KSom(0)->attr().Im()->Name() << ","
            << aTriplet->KSom(1)->attr().Im()->Name() << ","
            << aTriplet->KSom(2)->attr().Im()->Name()
            << "]" << std::endl;
    //ELISE_ASSERT(aLst.size() > 0, "Homol points empty for triplet");
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
        tTriPointList homolAllPts;
        loadHomol(aTriplet, homolPts, homolAllPts);
        aTriplet->AddHomolPts(homolPts);
        aTriplet->AddHomolAllPts(homolAllPts);

        ///  ADD-SOM-TRIPLET
        /*aS1->attr().AddTriplet(aTriplet, 1, 2, 0);
        aS2->attr().AddTriplet(aTriplet, 0, 2, 1);
        aS3->attr().AddTriplet(aTriplet, 0, 1, 2);*/

        aS1->attr().AddTriplet(aTriplet, 1, 2, 0);
        aS2->attr().AddTriplet(aTriplet, 0, 2, 1);
        aS3->attr().AddTriplet(aTriplet, 0, 1, 2);

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

cNOSolIn_Triplet* RandomForest::GetBestTri(Dataset& data) {
    /*std::list<cNOSolIn_Triplet*> ordered(data.mV3.begin(), data.mV3.end());
    std::map<cNOSolIn_Triplet*, double> scores;
    for (auto a : ordered) {
        double score = 0;
        for (uint8_t i = 0; i < 3; i++) {
            //score += a->KSom(i)->attr().Lnk3().size();
            score += a->KArc(i)->attr().ASym()->NumArc();
        }
        scores[a] = score/3.;
    }
    ordered.sort([&scores](cNOSolIn_Triplet* a, cNOSolIn_Triplet* b){
                return scores[a] > scores[b];
                });
    return ordered.front();*/

     //Old Version
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

static std::array<ElRotation3D, 3> EstimAllRt(const cLinkTripl* aLnk) {
    // Get sommets
    const tSomNSI* aS1 = aLnk->S1();
    const tSomNSI* aS2 = aLnk->S2();
    const tSomNSI* aS3 = aLnk->S3();

    // Get current R,t of the mother pair
    const ElRotation3D aC1ToM = aS1->attr().CurRot();
    const ElRotation3D aC2ToM = aS2->attr().CurRot();

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

    Pt3dr aT1 = T0 + aRL2M * aC1ToL.tr() * Lambda;
    Pt3dr aT2 = T0 + aRL2M * aC2ToL.tr() * Lambda;
    Pt3dr aT3 = T0 + aRL2M * aC3ToL.tr() * Lambda;


    // 4- return R{1,2,3}, t{1,2,3}
    return {ElRotation3D(aT1, aRL2M * aC1ToL.Mat(), true),
            ElRotation3D(aT2, aRL2M * aC2ToL.Mat(), true),
            ElRotation3D(aT3, aRL2M * aC3ToL.Mat(), true)};
}

static void EstimRt(cLinkTripl* aLnk) {
    auto oris = EstimAllRt(aLnk);
    aLnk->S3()->attr().residue =
        ((aLnk->S1()->attr().residue + aLnk->S2()->attr().residue) / 2.) +
        aLnk->m3->Sum()[0];

    // Get sommets
    tSomNSI* aS3 = aLnk->S3();

    // 4- Set R3,t3
    aS3->attr().CurRot() = oris[2];
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
    while ((aTri = travel.GetRandTri(aPond)) && ((Cpt + 3) < NbSomCC)) {
        // Flag as visted
        aTri->m3->Flag().set_kth_true(data.mFlag3CC);

        if (!aTri->S1()->flag_kth(data.mFlagS)
            || !aTri->S2()->flag_kth(data.mFlagS)) {
            std::cout << "Error S1 or S2 not visited" << std::endl;
        }

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

                // Add two new edges and their respective adjacent triplets
        travel.AddArcOrCur(aTri->m3->KArc(aTri->mK1)->attr().ASym());
        travel.AddArcOrCur(aTri->m3->KArc(aTri->mK2)->attr().ASym());

        // Free mSCur3Adj from all triplets connected to S3
        travel.FreeSCur3Adj(aTri->S3());

        Cpt++;
    }

    std::cout << "\nIn this CC, nb of connected nodes " << Cpt + 3 << "\n";
}

struct CmpTri {
    bool operator()(cNOSolIn_Triplet* T1, cNOSolIn_Triplet* T2) const {
        //return (T1->m3->NumId()) < (T2->m3->NumId());
        //std::cout << "Pds" << T1->Pds() << std::endl;
        return (T1->Sum()[2]) < (T2->Sum()[2]);
    }
};

#include <functional>
#include <queue>
#include <vector>

/*
    On a connected component aCC :
    randomly build a tree and
*/

void RandomForest::RandomSolOneCC(Dataset& data, cNO_CC_TripSom* aCC) {
    DataTravel travel(data);
    NRrandom3InitOfTime(); //TODO Activer pour debug
    std::cout << "DFS per CC, Nb Som " << aCC->mSoms.size() << ", nb triplets "
              << aCC->mTri.size() << "\n";
    // Select random seed triplet
    size_t size = aCC->mTri.size();
    int aSeed = NRrandom3(int(size - 1));
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
    clock_t start1 = clock();
    RandomSolOneCC(data, aTri0, aCC->mSoms.size());
    clock_t end1 = clock();
    std::cout << "BuildTree " << double(end1 - start1)/CLOCKS_PER_SEC*1000 << std::endl;


    static int n = 0;
    GraphViz g;
    g.travelGraph(data, *aCC, aTri0);
    g.write(mOutName + "/graph/", "random_tree_" + std::to_string(n++) + ".dot");
    std::string aOutOri = "tempori" + std::to_string(n);
    Save(data, aOutOri, false);

    // Calculate coherence scores within this CC
    clock_t start2 = clock();
    CoherTripletsGraphBasedV2(data, aCC->mTri, aCC->mSoms.size(), aTri0->NumId());
    clock_t end2 = clock();
    std::cout << "Score " << double(end2 - start2)/CLOCKS_PER_SEC*1000 << std::endl;

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

void RandomForest::AddTriOnHeap(Dataset& data, cLinkTripl* aLnk, bool oriented) {

    std::vector<cLinkTripl*> aVMaj;
    std::vector<int> aSIdx{aLnk->mK1, aLnk->mK2, aLnk->mK3};
    for (int aK : {0, 1, 2}) {
        std::vector<cLinkTripl>& aVT =
            aLnk->m3->KArc(aSIdx[aK])->attr().ASym()->Lnk3();

        for (int aT = 0; aT < int(aVT.size()); aT++) {
            // If triplet was visited continue
            if (aVT[aT].m3->Flag().kth(data.mFlag3CC)) {
                continue;
            }

            aVT[aT].prev = aLnk;

            // Add to heap
            mHeapTriDyn.push(&(aVT[aT]));

            // Push to aVMaj
            aVMaj.push_back(&aVT[aT]);

            // Flag as marked
            aVT[aT].m3->Flag().set_kth_true(data.mFlag3CC);

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

#include <unordered_map>
template<typename T>
class DisjointSet {
    std::unordered_map<T, T> parent;
    size_t partition;

   public:
    // perform MakeSet operation
    void makeSet(std::vector<T> const& universe) {
        // create `n` disjoint sets (one for each item)
        for (auto a : universe) {
            parent[a] = a;
        }
        partition = universe.size();
    }

    // Find the root of the set in which element `k` belongs
    T Find(T k) {
        // if `k` is root
        if (parent[k] == k) {
            return k;
        }

        // recur for the parent until we find the root
        return Find(parent[k]);
    }

    // Perform Union of two subsets
    void Union(T a, T b) {
        // find the root of the sets in which elements `x` and `y` belongs
        T x = Find(a);
        T y = Find(b);

        parent[x] = y;
    }
};

void RandomForest::BestSolOneCCKurskal(Dataset& data, cNO_CC_TripSom* aCC) {
    GraphViz g;

    tHeapTriNSI mEdges(TheCmp3);
    {
        std::vector<cLinkTripl*> aVMaj;
        unsigned aKCur = 0;
        // Traditional loop of CC : while  no new inexplored neighboor
        while (aKCur != aCC->mTri.size()) {
            cNOSolIn_Triplet* aTri = aCC->mTri[aKCur];
            // For each edge of the current triplet
            for (int aKA = 0; aKA < 3; aKA++) {
                // Get triplet adjacent to this edge and parse them
                std::vector<cLinkTripl>& aLnk =
                    aTri->KArc(aKA)->attr().ASym()->Lnk3();
                for (int aKL = 0; aKL < int(aLnk.size()); aKL++) {
                    mEdges.push(&aLnk[aKL]);
                    aVMaj.push_back(&aLnk[aKL]);
                }
            }
            aKCur++;
        }

        for (int aK = 0; aK < int(aVMaj.size()); aK++) {
            mEdges.MAJ(aVMaj[aK]);
        }
    }
    DisjointSet<tSomNSI*> ds;
    ds.makeSet(aCC->mSoms);
    std::vector<cLinkTripl*> tree;

    cLinkTripl* aTriRoot = 0;
    mEdges.pop(aTriRoot);
    //ROOT
    {
        tSomNSI* pS1 = ds.Find(aTriRoot->S1());
        tSomNSI* pS2 = ds.Find(aTriRoot->S2());
        tSomNSI* pS3 = ds.Find(aTriRoot->S3());

        aTriRoot->S1()->attr().next1.push_back(aTriRoot);
        aTriRoot->S2()->attr().next2.push_back(aTriRoot);
        aTriRoot->S3()->attr().prev = aTriRoot;

        ds.Union(pS1, pS3);
        ds.Union(pS2, pS3);
    }

    cLinkTripl* aTriNext = 0;
    while (mEdges.pop(aTriNext)) {
        aTriNext->m3->KArc(0)->attr().ASym()->Lnk3();
        tSomNSI* pS1 = ds.Find(aTriNext->S1());
        tSomNSI* pS2 = ds.Find(aTriNext->S2());
        tSomNSI* pS3 = ds.Find(aTriNext->S3());

        if (pS1 != pS3 && pS2 != pS3 && pS1 == pS2) {
            tree.push_back(aTriNext);
            aTriNext->S1()->attr().next1.push_back(aTriNext);
            aTriNext->S2()->attr().next2.push_back(aTriNext);
            aTriNext->S3()->attr().prev = aTriNext;
            g.addTriplet(*aTriNext->m3);

            ds.Union(pS1, pS3);
            ds.Union(pS2, pS3);
        }
    }

    std::set<cLinkTripl*> lista, listb;
    std::set<cLinkTripl*>* pla = &lista;
    //std::set<cLinkTripl*>* plb = &listb;
    for (int aK = 0; aK < 3; aK++) {
        // Flag sommets as explored
        aTriRoot->S1()->attr().treeCost = aTriRoot->m3->Cost();
        aTriRoot->S2()->attr().treeCost = aTriRoot->m3->Cost();

        // Set the current R,t of the seed
        aTriRoot->m3->KSom(aK)->attr().CurRot() =
            aTriRoot->m3->RotOfSom(aTriRoot->m3->KSom(aK));

        for (auto a : aTriRoot->m3->KSom(aK)->attr().next1) {
            pla->insert(a);
        }
        for (auto a : aTriRoot->m3->KSom(aK)->attr().next2) {
            pla->insert(a);
        }
    }

    std::set<tSomNSI*> oriented;
    for (tSomNSI* som : aCC->mSoms) {
        if (!oriented.count(som)) {
            if (!som->attr().prev) {
                //Root node
                std::cout << "null" << std::endl;
            }
        }
    }

    g.write(mOutName + "/graph/", "final.dot");

}
class cNO_CmpSomByCost {
   public:
    bool operator()(tSomNSI* a, tSomNSI* b) {
        return a->attr().treeCost < b->attr().treeCost;
    }
};
class cNO_HeapIndSom_NSI {
   public:
    static void SetIndex(tSomNSI* aV, int i) { aV->attr().HeapIndex() = i; }
    static int Index(tSomNSI* aV) { return aV->attr().HeapIndex(); }
};

class cNO_CmpArcByCost {
   public:
    bool operator()(cNOSolIn_AttrArc* a, cNOSolIn_AttrArc* b) {
        return a->treecost < b->treecost;
    }
};
class cNO_HeapIndArc_NSI {
   public:
    static void SetIndex(cNOSolIn_AttrArc* aV, int i) { aV->HeapIndex() = i; }
    static int Index(cNOSolIn_AttrArc* aV) { return aV->HeapIndex(); }
};



struct finalTree {
    std::map<cNOSolIn_Triplet*, std::set<cNOSolIn_Triplet*, cNO_CmpTriSolByCost>> gt;

    std::map<cNOSolIn_Triplet*, cNOSolIn_Triplet*> pred;
    std::map<cNOSolIn_Triplet*, std::set<cNOSolIn_Triplet*>> next;

    //std::map<tSomNSI*, std::set<tSomNSI*>> g;
    std::map<cNOSolIn_Triplet*, std::set<cNOSolIn_Triplet*>> out;
    std::map<tSomNSI*, cNOSolIn_Triplet*> outpred;

    std::map<tSomNSI*, cNOSolIn_Triplet*> triplets;
    std::map<cNOSolIn_Triplet*, tSomNSI*> sommit;
    std::map<cNOSolIn_Triplet*, cLinkTripl*> ori;
    std::map<tSomNSI*, cLinkTripl*> orientation;

    std::map<tSomNSI*, std::set<cNOSolIn_Triplet*>> orienting;
    cNOSolIn_Triplet* root;
};


void orientFinalTree2(Dataset& data, finalTree& tree, cNOSolIn_Triplet* root, int deep = 0)
{
    if (!root)
        return;
    std::cout << "ROOT " << root->NumId()
        << " "             << root->KSom(0)->attr().Im()->Name()
        << " "             << root->KSom(1)->attr().Im()->Name()
        << " "             << root->KSom(2)->attr().Im()->Name()
        << std::endl;

    //Handle root first
    for (int aK = 0; aK < 3; aK++) {
        // Set the current R,t of the seed
        root->KSom(aK)->attr().CurRot() = root->RotOfSom(root->KSom(aK));
        root->KSom(aK)->attr().oriented = true;
        root->KSom(aK)->attr().NumCC() = deep;
        root->KSom(aK)->attr().NumId() = deep;
        root->KSom(aK)->attr().treeCost = 0;
        root->KSom(aK)->flag_set_kth_true(data.mFlagS);
    }
    std::queue<cNOSolIn_Triplet *> q;
    for (auto nt : tree.next[root]) {
        q.push(nt);
    }
    // BFS
    while (!q.empty()) {
        auto t = q.front();
        q.pop();
        ++deep;

        auto pred = tree.pred[t];
        std::cout << "Triplet " << t->NumId()
            << " "             << t->KSom(0)->attr().Im()->Name()
            << " "             << t->KSom(1)->attr().Im()->Name()
            << " "             << t->KSom(2)->attr().Im()->Name()
            << std::endl;

        int s[3] = {-1, -1, -1};
        int i = 0;
        for (int a : {0, 1, 2}) {
            bool found = false;
            for (int b : {0, 1, 2}) {
                if (t->KSom(a) == pred->KSom(b)) {
                    s[i++] = a;
                    found = true;
                    break;
                }
            }
            if (!found)
                s[2] = a;
        }

        cLinkTripl l(t, s[0], s[1], s[2]);
        cLinkTripl* link = &l;

        std::cout << link->S1()->attr().Im()->Name()
                  << " "
                  << link->S2()->attr().Im()->Name()
                  << " "
                  << link->S3()->attr().Im()->Name()
                  << std::endl;

        if (!link->S1()->flag_kth(data.mFlagS)
            && !link->S2()->flag_kth(data.mFlagS)
) {
            std::cout << "Not oriented.." << std::endl;
            q.push(t);
            continue;
        }

        //if (link->S3()->flag_kth(data.mFlagS)) {
        //    continue;
        //}

        link->S3()->attr().treeCost = link->m3->Cost();

        //Log output
        tree.outpred[link->S3()] = t;

        std::cout << link->S3()->attr().Im()->Name()  << " " << link->S3()->attr().treeCost << std::endl;
        EstimRt(link);
        link->S3()->attr().NumCC() = deep;
        link->S3()->attr().NumId() = deep;
        link->S3()->attr().oriented = true;
        link->S3()->flag_set_kth_true(data.mFlagS);

        //Triplet oriented
        t->NumTT() = deep;
        t->NumCC() = deep;

        for (auto nt : tree.next[t]) {
            q.push(nt);
        }
    }

    //set
        //aTri0->KSom(i)->flag_set_kth_true(data.mFlagS);
    //read
        //aTri->S3()->flag_kth(data.mFlagS)
}

template<typename T, typename fT>
bool is_tree(fT& tree, std::set<T*>& visited, T* root) {
    if (visited.count(root)) return false;
    visited.insert(root);

    bool r = true;
    for (auto nt : tree.next[root]) {
        r = r && is_tree(tree, visited, nt);
    }
    return r;
}
/*
void RandomForest::BestSolOneCCPrim2(Dataset& data, cNO_CC_TripSom* aCC) {
    int NbSomCC = int(aCC->mSoms.size());
    GraphViz g;
    DataLog<int, int, double, double, double, double, double, double> log(
        {"Id", "Category", "Distance", "Residue", "ResidueMedian", "Score",
         "ScoreMedian", "Accumulated"});
    finalTree tree;

    std::cout << "Generate graph gt" << std::endl;
    for (auto t : aCC->mTri) {
        for (char i : {0, 1, 2})
            for (auto& l : t->KArc(i)->attr().ASym()->Lnk3()){
                tree.gt[t].insert(l.m3);
                tree.gt[l.m3].insert(t);
            }
    }
    std::cout << "Generated graph gt" << std::endl;
    //INIT
    for (auto s : aCC->mTri) {
        s->treecost = std::numeric_limits<double>::max();
        tree.pred[s] = nullptr;
    }
    for (auto s : aCC->mSoms) {
        s->attr().treeCost = std::numeric_limits<double>::max();
    }

    cNOSolIn_Triplet* aTri0 = *std::min_element(
        aCC->mTri.begin(), aCC->mTri.end(),
        [](decltype(*begin(aCC->mTri)) a, decltype(*begin(aCC->mTri)) b) {
            size_t na = a->KArc(0)->attr().ASym()->Lnk3().size()
            + a->KArc(1)->attr().ASym()->Lnk3().size()
            + a->KArc(2)->attr().ASym()->Lnk3().size();

            size_t nb = b->KArc(0)->attr().ASym()->Lnk3().size()
            + b->KArc(1)->attr().ASym()->Lnk3().size()
            + b->KArc(2)->attr().ASym()->Lnk3().size();
            return a->Sum()[1] < b->Sum()[1] && na > nb;
        });


    std::cout << "Best triplet " << aTri0->KSom(0)->attr().Im()->Name()
              << " "             << aTri0->KSom(1)->attr().Im()->Name()
              << " "             << aTri0->KSom(2)->attr().Im()->Name()
              << " "             << aTri0->Sum()[aTri0->indexSum]
              << std::endl;
    tree.root = aTri0;
    aTri0->KArc(0)->attr();
    aTri0->treecost = 0;
    std::cout << aTri0->NumId() << std::endl;

    //End INIT
    //SEED
    //tree.pred[aTri0] = aTri0;
    std::cout << "Fin init" << std::endl;

    int Cpt = 0;
    cNO_CmpArcByCost cmp;
    ElHeap<cNOSolIn_AttrArc*, cNO_CmpArcByCost, cNO_HeapIndArc_NSI> tri(cmp);
    for (int i : {0, 1, 2}) {
        tri.push(&aTri0->KArc(i)->attr());
        tri.MAJ(&aTri0->KArc(i)->attr());
    }
    std::cout << "Added triplets" << std::endl;
    std::set<cNOSolIn_AttrArc*> visited;

    while (!tri.empty()) {
        cNOSolIn_AttrArc* t = nullptr;
        tri.pop(t);
        visited.insert(t);
        for (auto& u : t->ASym()->Lnk3()) {
            std::cout << "Check" << std::endl;
            if (u.m3 >= t->treecost + t.->Cost()) {
                u->treecost = t->treecost + t->Cost();
                std::cout << "Change cost" << std::endl;
                tree.pred[u] = t;
                std::cout << "update pred" << std::endl;

                if (!visited.count(u))
                    tri.MajOrAdd(u);
                std::cout << "Updated cost" << std::endl;
            }
        }
    }
    std::cout << "Fin Arbre" << std::endl;
    std::cout << "Invert Arbre" << std::endl;
    for (auto t : tree.pred) {
        if (t.first == t.second)
            continue;

        if (t.first && t.second)
            std::cout << t.first->NumId() << " " << t.second->NumId() << std::endl;
        if (t.second) {
            tree.next[t.second].insert(t.first);
        } else {
            std::cout << "Pas de second - racine" << std::endl;
        }
    }
    std::cout << "Fin Invert Arbre" << std::endl;

    orientFinalTree2(data, tree, aTri0);

    std::cout << "Orient Arbre" << std::endl;
    for (auto x : tree.outpred) {
        auto t = x.second;
        std::cout << t->NumId() << std::endl;
        log.add({t->NumId(), t->category,
                t->NumTT(), t->Sum()[0],
                t->Sum()[1], t->Sum()[2],
                t->Sum()[3],
                t->KSom(0)->attr().treeCost,
                });
        g.addTriplet(*t);
    }

    for (auto const& x : tree.outpred) {
        if (x.first && x.second)
            g.linkTriplets(x.second, tree.pred[x.second]);
        for (auto c : x.second) {
            g.linkTriplets(x.first, c);
        }
    }
    std::cout << "Fin log" << std::endl;

    g.write(mOutName + "/graph/", "final.dot");
    log.write(mOutName + "/graph/logs/", "final.csv");
    std::cout << "Nb final sommets=" << Cpt + 3 << ", out of " << NbSomCC
              << "\n";
}
*/

void orientFinalTree(Dataset& data, finalTree& tree, cNOSolIn_Triplet* t, int deep = 0)
{
    if (!t)
        return;

    std::cout << "Triplet " << t->NumId()
              << " "             << t->KSom(0)->attr().Im()->Name()
              << " "             << t->KSom(1)->attr().Im()->Name()
              << " "             << t->KSom(2)->attr().Im()->Name()
              << std::endl;

    if (!tree.pred.count(t)) { //Root
        for (int aK = 0; aK < 3; aK++) {
            // Set the current R,t of the seed
            t->KSom(aK)->attr().CurRot() = t->RotOfSom(t->KSom(aK));
            t->KSom(aK)->attr().oriented = true;
            t->KSom(aK)->attr().NumCC() = deep;
            t->KSom(aK)->attr().NumId() = deep;
            t->KSom(aK)->flag_set_kth_true(data.mFlagS);
        }
    } else {
        auto link = tree.ori[t];
        if (link) {
            std::cout << link->S3()->attr().Im()->Name() << std::endl;
            EstimRt(link);
            link->S3()->attr().NumCC() = deep;
            link->S3()->attr().NumId() = deep;
            link->S3()->attr().oriented = true;
            link->S3()->flag_set_kth_true(data.mFlagS);

        }
    }

    //Triplet oriented
    t->NumTT() = deep;
    t->NumCC() = deep;

    for (auto nt : tree.next[t]) {
        orientFinalTree(data, tree, nt, deep + 1);
    }
}
void RandomForest::BestSolOneCCPrim(Dataset& data, cNO_CC_TripSom* aCC) {
    int NbSomCC = int(aCC->mSoms.size());
    GraphViz g;
    DataLog<int, int, double, double, double, double, double, double> log(
        {"Id", "Category", "Distance", "Residue", "ResidueMedian", "Score",
         "ScoreMedian", "Accumulated"});

    cNOSolIn_Triplet* aTri0 = *std::min_element(
        aCC->mTri.begin(), aCC->mTri.end(),
        [](decltype(*begin(aCC->mTri)) a, decltype(*begin(aCC->mTri)) b) {
            size_t na = a->KArc(0)->attr().ASym()->Lnk3().size()
            + a->KArc(1)->attr().ASym()->Lnk3().size()
            + a->KArc(2)->attr().ASym()->Lnk3().size();

            size_t nb = b->KArc(0)->attr().ASym()->Lnk3().size()
            + b->KArc(1)->attr().ASym()->Lnk3().size()
            + b->KArc(2)->attr().ASym()->Lnk3().size();
            return a->Sum()[1] < b->Sum()[1] && na > nb;

            //return ;
        });
    finalTree tree;

    std::cout << "Best triplet " << aTri0->KSom(0)->attr().Im()->Name()
              << " "             << aTri0->KSom(1)->attr().Im()->Name()
              << " "             << aTri0->KSom(2)->attr().Im()->Name()
              << " "             << aTri0->Sum()[aTri0->indexSum]
              << std::endl;
    tree.root = aTri0;

    //INIT
    for (auto s : aCC->mSoms) {
        s->attr().treeCost = std::numeric_limits<double>::max();
        tree.triplets[s] = nullptr;
        tree.orientation[s] = nullptr;
    }

    //End INIT
    //SEED
    cLinkTripl* aLnk0 = new cLinkTripl(aTri0, 1, 2, 0);
    cLinkTripl* aLnk1 = new cLinkTripl(aTri0, 0, 2, 1);
    cLinkTripl* aLnk2 = new cLinkTripl(aTri0, 0, 1, 2);
    for (int i : {0, 1, 2}) {
        aTri0->KSom(i)->attr().treeCost = 0;
        tree.triplets[aTri0->KSom(i)] = aTri0;
        aTri0->KSom(i)->flag_set_kth_true(data.mFlagS);
    }
    tree.orientation[aTri0->KSom(0)] = aLnk0;
    tree.orientation[aTri0->KSom(1)] = aLnk1;
    tree.orientation[aTri0->KSom(2)] = aLnk2;

    //tree.pred[aTri0] = aTri0;
    std::cout << "Fin init" << std::endl;

    int Cpt = 0;
    cNO_CmpSomByCost cmp;
    //std::priority_queue<tSomNSI*, std::vector<tSomNSI*>,  cNO_CmpSomByCost> som;
    ElHeap<tSomNSI*, cNO_CmpSomByCost, cNO_HeapIndSom_NSI> som(cmp);
    som.push(aTri0->KSom(0));
    som.MAJ(aTri0->KSom(0));
    som.push(aTri0->KSom(1));
    som.MAJ(aTri0->KSom(1));
    som.push(aTri0->KSom(2));
    som.MAJ(aTri0->KSom(2));
    /*for (auto s : aCC->mSoms) {
        som.push(s);
        som.MAJ(s);
    }*/
    //for (auto s : aCC->mSoms) {  }

    std::cout << aTri0->NumId() << std::endl;
    //som.MAJ(aTri0->KSom(0));
    std::cout << "Added triplets" << std::endl;
    std::set<std::string> processed;

    while (!som.empty()) {
        tSomNSI* t = nullptr;
        som.pop(t);
        if (!t)
            continue;
        processed.insert(t->attr().Im()->Name());
        auto aT = tree.triplets[t];
        auto lT = tree.orientation[t];
        if (!lT) {
            t->attr().treeCost = std::numeric_limits<double>::max();
            std::cout << "No orientation" << std::endl;
            continue;
        }
        std::cout << t->attr().Im()->Name() << std::endl;
        std::set<cLinkTripl*, cNO_CmpTriByCost> lnks;
        for (char i : {lT->mK1, lT->mK2}) {
            for (auto& l : aT->KArc(i)->attr().ASym()->Lnk3())
                lnks.insert(&l);
        }
        for (auto& cl : lnks) {
            if (!cl->S3()->flag_kth(data.mFlagS)
              //  && cl->S3()->attr().treeCost >= cl->S1()->attr().treeCost + cl->m3->Cost()
               // && tree.g[cl->S1()].count(cl->S2())
               ) {
                auto& tS2 = tree.triplets[cl->S2()];
                bool c = tS2->KSom(0) == cl->S1() ||
                         tS2->KSom(1) == cl->S1() ||
                         tS2->KSom(2) == cl->S1();
                auto& tS1 = tree.triplets[cl->S1()];
                bool c2 = tS1->KSom(0) == cl->S2() ||
                          tS1->KSom(1) == cl->S2() ||
                          tS1->KSom(2) == cl->S2();

                if (!(c || c2)) {
                    continue;
                }
                cl->S3()->attr().treeCost = cl->S1()->attr().treeCost + cl->m3->Cost();

                tree.orienting[cl->S3()].insert(aT);
                cl->S3()->flag_set_kth_true(data.mFlagS);

                //Remove old link in tree
                tree.pred[cl->m3] = aT;
                tree.triplets[cl->S3()] = cl->m3;
                tree.orientation[cl->S3()] = cl;
                tree.ori[cl->m3] = cl;
                tree.sommit[cl->m3] = cl->S3();

                if (!processed.count(cl->S3()->attr().Im()->Name())) {
                    processed.insert(cl->S3()->attr().Im()->Name());
                    som.push(cl->S3());
                }
            }
        }
    }
    std::cout << "Fin Arbre" << std::endl;
    std::cout << processed.size() << std::endl;
    std::cout << "Fin Arbre" << std::endl;
    for (auto t : tree.pred) {
        if (t.first->NumId() == t.second->NumId())
            continue;

        std::cout << t.first->NumId() << " " << t.second->NumId() << std::endl;
        if (t.second) {
            tree.next[t.second].insert(t.first);
        } else {
            std::cout << "Pas de second" << std::endl;
        }
    }
    std::cout << "Invert Arbre" << std::endl;
    std::set<cNOSolIn_Triplet *> visited;
    std::cout << "Check tree: " << is_tree(tree, visited, aTri0) << std::endl;

    orientFinalTree(data, tree, aTri0);
    std::cout << "Orient Arbre" << std::endl;
    for (auto x : tree.next) {
        std::cout << x.first->NumId() << std::endl;
        log.add({x.first->NumId(), x.first->category,
                x.first->NumTT(), x.first->Sum()[0],
                x.first->Sum()[1], x.first->Sum()[2],
                x.first->Sum()[3],
                x.first->KSom(0)->attr().treeCost,
                });
        g.addTriplet(*x.first);
    }

    for (auto const& x : tree.next) {
        for (auto c : x.second) {
            g.linkTree(x.first, c);
        }
    }
    std::cout << "Fin log" << std::endl;

    g.write(mOutName + "/graph/", "final.dot");
    log.write(mOutName + "/graph/logs/", "final.csv");
    std::cout << "Nb final sommets=" << Cpt + 3 << ", out of " << NbSomCC
              << "\n";
}

void RandomForest::BestSolOneCC(Dataset& data, cNO_CC_TripSom* aCC, ffinalTree& tree) {
    int NbSomCC = int(aCC->mSoms.size());
    GraphViz g;
    DataLog<int, int, double, double, double, double, double, double> log(
        {"Id", "Category", "Distance", "Residue", "ResidueMedian", "Score",
         "ScoreMedian", "Accumulated"});

    // Pick the  triplet
    cNOSolIn_Triplet* aTri0 = GetBestTri(data);
    std::cout << "Best triplet " << aTri0->KSom(0)->attr().Im()->Name() << " "
              << aTri0->KSom(1)->attr().Im()->Name() << " "
              << aTri0->KSom(2)->attr().Im()->Name() << " " << aTri0->Sum()[aTri0->indexSum]
              << "\n";
    tree.root = aTri0;
    tree.pred[aTri0->KSom(1)] = aTri0->KSom(0);
    tree.pred[aTri0->KSom(2)] = aTri0->KSom(0);



    log.add({aTri0->NumId(), aTri0->category, 0., aTri0->Sum()[0],
             aTri0->Sum()[1], aTri0->Sum()[2], aTri0->Sum()[3], aTri0->Sum()[0]});

    // Flag triplet as marked
    aTri0->Flag().set_kth_true(data.mFlag3CC);
    cLinkTripl* aLnk0 = new cLinkTripl(aTri0, 1, 2, 0);

    aTri0->KSom(0)->attr().prev = aLnk0;

    for (int aK = 0; aK < 3; aK++) {
        // Flag sommets as explored
        aTri0->KSom(aK)->flag_set_kth_true(data.mFlagS);
        aTri0->KSom(aK)->attr().treeCost = aTri0->Cost();
        tree.ori[aTri0->KSom(aK)] = aLnk0;

        aTri0->KSom(aK)->attr().residue = aTri0->Sum()[0];
        aTri0->KSom(aK)->attr().NumCC() = 0;
        aTri0->KSom(aK)->attr().NumId() = aTri0->NumId();

 //       tree.ori[aTri0->KSom(aK)] = aTri0;

        // Set the current R,t of the seed
        aTri0->KSom(aK)->attr().CurRot() = aTri0->RotOfSom(aTri0->KSom(aK));

        // PrintRotation(aTri0->KSom(aK)->attr().CurRot().Mat(),ToString(aK));
    }
    std::cout << "Final seed residue: " << computeResiduFromPos(aTri0) << std::endl;

    // initialise the starting node
    g.addTriplet(*aTri0);
    aTri0->KSom(0)->attr().treeCost = 0;

    // Fill the dynamic heap with triplets connected to this triplet
    AddTriOnHeap(data, aLnk0);

    // Iterate to build the solution while updating the heap
    int Cpt = 0;
    cLinkTripl* aTriNext = 0;
    //TODO remove cpt+3
    while ((aTriNext = GetBestTriDyn())) {
        // Check that the node has not been added in the meantime
        if (!aTriNext->S3()->flag_kth(data.mFlagS))
        {  // however, try to adjacent triplets of that triplet
            std::cout << "=== Add new node " << Cpt << " "
                      << aTriNext->S1()->attr().Im()->Name() << " "
                      << aTriNext->S2()->attr().Im()->Name() << " "
                      << aTriNext->S3()->attr().Im()->Name() << " "
                      << aTriNext->m3->Sum()[1] << ", "
                      << aTriNext->m3->Sum()[2] << "\n";

            // Flag triplet order
            aTriNext->m3->NumTT() = ElMax(aTriNext->S1()->attr().NumCC(),
                                          aTriNext->S2()->attr().NumCC()) +
                                    1;
            //Choose tree
            tSomNSI* prev = aTriNext->S1();
            if (aTriNext->S2()->attr().treeCost < aTriNext->S1()->attr().treeCost)
                prev = aTriNext->S2();
            aTriNext->S3()->attr().treeCost = prev->attr().treeCost + aTriNext->m3->Cost();
            tree.pred[aTriNext->S3()] = prev;
            tree.ori[aTriNext->S3()] = aTriNext;

            // Flag the concatenation order of the node
            // = order of the "builder" triplet
            aTriNext->S3()->attr().NumCC() = aTriNext->m3->NumTT();

            // Id per sommet to know from which triplet it was generated => Fast
            // Tree Dist util
            aTriNext->S3()->attr().NumId() = aTriNext->m3->NumId();
            aTriNext->S3()->attr().treeCost = aTriNext->m3->Cost();

            //Accumulate residue from each branch
//            aTriNext->m3->Sum()[4] = aTriNext->m3->Sum()[0] + aTriNext->prev->m3->Sum()[4];
            /*PrintRotation(aTriNext->S1()->attr().CurRot().Mat(),"0");
              PrintRotation(aTriNext->S2()->attr().CurRot().Mat(),"1");
              PrintRotation(aTriNext->S3()->attr().CurRot().Mat(),"2");*/

            // Propagate R,t
            EstimRt(aTriNext);


            log.add({aTriNext->m3->NumId(), aTriNext->m3->category,
                     aTriNext->m3->NumTT(), aTriNext->m3->Sum()[0],
                     aTriNext->m3->Sum()[1], aTriNext->m3->Sum()[2],
                     aTriNext->m3->Sum()[3],
                     aTriNext->S3()->attr().residue,
                     });

            tree.g[aTriNext->S3()].insert(aTriNext->S1());
            tree.g[aTriNext->S1()].insert(aTriNext->S3());

            tree.g[aTriNext->S3()].insert(aTriNext->S2());
            tree.g[aTriNext->S2()].insert(aTriNext->S3());

            g.addTriplet(*aTriNext->m3);
            //g.linkTriplets(common, aTriNext->m3);

            tree.triplets[aTriNext->m3->KSom(0)].insert(aTriNext->m3);
            tree.triplets[aTriNext->m3->KSom(1)].insert(aTriNext->m3);
            tree.triplets[aTriNext->m3->KSom(2)].insert(aTriNext->m3);

            // Mark node as vistied
            aTriNext->S3()->flag_set_kth_true(data.mFlagS);
            std::cout << "Final residue: " << computeResiduFromPos(aTriNext->m3) << std::endl;

            // Add to heap
            AddTriOnHeap(data, aTriNext);

            Cpt++;

        } else {
            aTriNext->viewed = true;

            // Add to heap
            AddTriOnHeap(data, aTriNext, false);
            /*if (aTriNext->m3->Cost() < aTriNext->S3()->attr().treeCost) {
                std::cout << "Better triplet for point: " << std::endl
                          << aTriNext->S3()->attr().treeCost
                          << " -> "
                          << aTriNext->m3->Cost()
                          << std::endl
                          << aTriNext->S1()->attr().Im()->Name() << " "
                          << aTriNext->S2()->attr().Im()->Name() << " "
                          << aTriNext->S3()->attr().Im()->Name() << " "
                          << std::endl;
                EstimRt(aTriNext);
                aTriNext->S3()->attr().treeCost = aTriNext->m3->Cost();

                log.add({aTriNext->m3->NumId(), aTriNext->m3->category,
                     aTriNext->m3->NumTT(), aTriNext->m3->Sum()[0],
                     aTriNext->m3->Sum()[1], aTriNext->m3->Sum()[2],
                     aTriNext->m3->Sum()[3],
                     aTriNext->S3()->attr().residue,
                     });

                 g.addTriplet(*aTriNext->m3);
            }*/
        }
    }
    std::cout << "Finish tree " << std::endl;
    for (auto t : tree.pred) {
        if (t.first == t.second)
            continue;

        std::cout << t.first->attr().Im()->Name() << " " << t.second->attr().Im()->Name() << std::endl;
        if (t.second) {
            tree.next[t.second].insert(t.first);
        } else {
            std::cout << "Pas de second" << std::endl;
        }
    }
    std::set<tSomNSI *> visited;
    std::cout << "Check tree: " << is_tree(tree, visited, aTri0->KSom(0)) << std::endl;


    for (auto const& x : tree.next) {
        for (auto c : x.second) {
            g.linkTree(x.first, c);
        }
    }

    g.linkTree(aTri0->KSom(0), aTri0->KSom(1));
    g.linkTree(aTri0->KSom(0), aTri0->KSom(2));

    g.write(mOutName + "/graph/", "final.dot");
    log.write(mOutName + "/graph/logs/", "final.csv");
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

/*
 *
 *
cSetOfMesureAppuisFlottants aSOMAF;
        for
                (
                 std::list<cSaisiePointeIm>::const_iterator itSP = mSOSPI.SaisiePointeIm().begin();
                 itSP != mSOSPI.SaisiePointeIm().end();
                 itSP++
                 )
        {
            cMesureAppuiFlottant1Im aMAF;
            aMAF.NameIm() = itSP->NameIm();

            for
                    (
                     std::list<cOneSaisie>::const_iterator itS=itSP->OneSaisie().begin();
                     itS!=itSP->OneSaisie().end();
                     itS++
                     )
            {
                if (itS->Etat()==eEPI_Valide)
                {
                    cOneMesureAF1I aM;
                    aM.NamePt() = itS->NamePt();
                    aM.PtIm() = itS->PtIm();
                    aMAF.OneMesureAF1I().push_back(aM);
                }
            }

            aSOMAF.MesureAppuiFlottant1Im().push_back(aMAF);
        }

        MakeFileXML(aSOMAF, mSauv2D);


*/

//TODO
// Read View parameters
// Execute Campari
// Export 2D point from image
// Projection from 2D to 3D points
//

void create3Dpts(ffinalTree& tree, tSomNSI* oa, tSomNSI* ob, std::string mSauv2D,
                 std::string mSauv3D) {
    size_t indexPts = 0;

    auto t = tree.ori[ob]->m3;

    cSetOfMesureAppuisFlottants aSOMAF;
    std::map<std::string, cMesureAppuiFlottant1Im> imgs;
    cDicoAppuisFlottant aDico;
        for (auto pts : t->getHomolAllPts())
        {
            size_t curPts = indexPts++;
            std::string ptsName = std::to_string(curPts);

            std::vector<ElSeg3D> aVSeg;
            for (int i = 0; i < 3; i++) {
                if (!pts[i].x && !pts[i].y)
                    continue;

                auto name = t->KSom(i)->attr().Im()->Name();
                imgs[name].NameIm() = name;
                cOneMesureAF1I aM;
                aM.NamePt() = ptsName;
                aM.PtIm() = pts[i];
                imgs[name].OneMesureAF1I().push_back(aM);

                t->KSom(i)->attr().Im()->CS()->SetOrientation(t->KSom(i)->attr().CurRot().inv());
                //aVSeg.push_back(triplet->KSom(i)->attr().Im()->CS()->F2toRayonR3(pts[i]));
                aVSeg.push_back(t->KSom(i)->attr().Im()->CS()->Capteur2RayTer(pts[i]));
            }

            bool ISOK = false;
            Pt3dr aInt = ElSeg3D::L2InterFaisceaux(0, aVSeg, &ISOK);

            cOneAppuisDAF anAP;
            anAP.Pt() = aInt;
            anAP.NamePt() = ptsName;
            anAP.Incertitude() = Pt3dr(1,1,1);

            aDico.OneAppuisDAF().push_back(anAP);
        }

    for (auto e : imgs) {
        aSOMAF.MesureAppuiFlottant1Im().push_back(e.second);
    }

    MakeFileXML(aSOMAF, mSauv2D);
    MakeFileXML(aDico, mSauv3D);
}

std::string exec(const std::string& cmd) {
    std::array<char, 1024> buffer;
    std::cout << cmd << std::endl;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

finalScene RandomForest::dfs(Dataset& data, ffinalTree& tree, tSomNSI* node) {
    if (!node) //Nothing get out
        return {};

    std::string curNodeName = node->attr().Im()->Name();
    auto node_ori = tree.ori[node];
    if (!tree.next.count(node)) { // Leaf nothing todo
        return {{node_ori->m3}, {node}};
    }

    std::cout << curNodeName << std::endl;
    std::cout << node_ori->S1()->attr().Im()->Name() << "/ "
              << node_ori->S2()->attr().Im()->Name() << "/ "
              << node_ori->S3()->attr().Im()->Name() << "/ "
              << std::endl;

    finalScene merging;
    merging.ts.insert(node_ori->m3);
    merging.ss.insert(node);

    std::string m2d = curNodeName + "_2d.xml";
    std::string m3d = curNodeName + "_3d.xml";
    size_t n = 0;
    for (auto& e : tree.next[node]) {
        auto r = dfs(data, tree, e);

        auto i = curNodeName + std::to_string(n++);
        for (auto e : r.ts) merging.ts.insert(e);
        for (auto e : r.ss) merging.ss.insert(e);

        //SAVE orientation
        for (auto e : r.ss) {
            e->flag_set_kth_true(data.mFlagS);
        }
        std::string pattern = "(" + node->attr().Im()->Name() + "|" +
                              e->attr().Im()->Name() + ")";

        create3Dpts(tree, node, e, m2d, m3d);
        std::cout << pattern << std::endl;
        Save(data, i, false);
        //TODO BAR -> robuste bascule for adding new triplet
        //std::cout <<
        //    exec("mm3d BAR \"" + pattern + "\" Ori-" + i + " " + m3d + " " + m2d + " Out=" + i + "-Bar");

        for (auto e : merging.ss) {
            e->flag_set_kth_false(data.mFlagS);
        }

        //std::string inOri = "" + i + "-Bar";
        /*std::cout << "Reading output ori from: " << inOri << std::endl;
        for (auto e : merging.ss) {
            std::string imgName = e->attr().Im()->Name();
            auto aCam = mNM->ICNM()->StdCamStenOfNames(imgName, inOri);
            e->attr().CurRot() = aCam->Orient();
        }
        */
    }


        //TODO Campari -> bundle
    // TODO Propagate parameters
   // std::cout <<
   // exec("mm3d Campari \"" + pattern + "\" Ori-" + curNodeName + "-Bar " + curNodeName + "-Camp SH=5Pts");
    //TODO READ new orientation

    //Read new orientation from files
    return merging;
}

void RandomForest::hierarchique(Dataset& data, ffinalTree& tree) {
    auto all = dfs(data, tree, tree.root->KSom(0));
    //TODO global campari on all data
}

void RandomForest::BestSolAllCC(Dataset& data) {
    // Add all triplets to global heap
    // (used only to get  seed)
    HeapPerSol(data);

    // Get  solution for each CC
    for (int aKC = 0; aKC < int(data.mVCC.size()); aKC++) {
        //BestSolOneCCKurskal(data, data.mVCC[aKC]);
        ffinalTree tree;
        BestSolOneCC(data, data.mVCC[aKC], tree);
        //BestSolOneCCPrim(data, data.mVCC[aKC]);
        std::cout << "Finish MSP" << std::endl;
        // Save
        std::string aOutOri = mOutName + ToString(aKC);
        //std::string aOutOri = "DSF_BestInit_CC" + ToString(aKC);
        Save(data, aOutOri, true);

        FreeAllFlag(data.mVCC[aKC]->mSoms, data.mFlagS);

        hierarchique(data, tree);
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

    DataLog<int, double, double> log({"Id", "Distance", "Residue"});

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
//    aAdjG.Show();

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

        //double aDistAl = square_euclid(center2, center);
        //std::cout << "Distance euclid: " << aDistAl << std::endl;

        if (aDist < 1) continue;
        if (aDist < mDistThresh) {
            //double aRot = abs(currentTriplet->CoherTest());
            // std::cout << ",Dist=" << aDist << " CohTest(à=" <<
            // aV3[aT]->CoherTest() << ",CostN=" <<  aCostCur << ",CPds=" <<
            // aCostCur/sqrt(aDist) << "\n"; std::cout <<
            // aTri->m3->Flag().set_kth_true(mFlag3CC);


            // Take into account the non-visited triplets
            // TODO verifier les triplets qui sont dans la solution
            if (!ValFlag(*(aV3[aT]), data.mFlag3CC)) {
                // std::cout << "Flag0 OK " << aCntFlags++ << "\n";
                //clock_t start1 = clock();
                double aResidue = currentTriplet->ProjTest();
                //clock_t end1 = clock();
                //std::cout << "ProjTest " << double(end1 - start1)/CLOCKS_PER_SEC << std::endl;
                log.add({currentTriplet->NumId(), aDist, aResidue});

                double score = aResidue / (aResidue + mR0);

                // Median
                aV3[aT]->Data()[0].push_back(aResidue);
                aV3[aT]->Data()[1].push_back(aDist);
                aV3[aT]->Data()[2].push_back(score);

                // Plot coherence vs sample vs distance
                //std::cout << "==PLOT== " << aCostCur * aPds << " " << aDist
                //         << " " << aPds << "\n";
            }
        }
    }
    static int n = 0;
    log.write(mOutName + "/graph/logs/runs/", "run_"+ std::to_string(n++) + ".csv");
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

static double computeAsymResidue(const tTriPointList& pts,
        const cNOSolIn_Triplet* tA,
        const std::array<ElRotation3D, 3>& oriA,
        const cNOSolIn_Triplet* tB,
        const std::vector<int>& index,
        const std::array<ElRotation3D, 3>& oriB) {
    const double MaxDiag = 6400;
    double value = 0;
    double number = 0;
    for (auto p : pts) {
        //Generate 3D point from first triplet A
        std::vector<ElSeg3D> aVSeg;
        for (int i = 0; i < 3; i++) {
            auto* v = tA->KSom(i);
            if (!p[i].x && !p[i].y) {
                continue;
            }
            v->attr().Im()->CS()->SetOrientation(oriA[i].inv());
            aVSeg.push_back(v->attr().Im()->CS()->Capteur2RayTer(p[i]));
        }

        bool ISOK = false;
        Pt3dr aPts = ElSeg3D::L2InterFaisceaux(0, aVSeg, &ISOK);

        //Compute reprojection error on triplet B
        double residu_pts = 0;
        uint n_residu = 0;
        for (int i = 0; i < 3; i++) {
            if (!p[i].x && !p[i].y) continue;

            auto* v = tB->KSom(index[i]);
            auto ori = oriB[index[i]];
            v->attr().Im()->CS()->SetOrientation(ori.inv());

            Pt2dr pts_proj = v->attr().Im()->CS()->Ter2Capteur(aPts);
            auto a = euclid(p[i], pts_proj);

            if (!v->attr().Im()->CS()->PIsVisibleInImage(aPts)) {
                a = min(a, MaxDiag);
            }
            residu_pts += a;
            n_residu++;
        }

        double m = MaxDiag;
        if (n_residu) {
            m = residu_pts / n_residu;
        }
        value += m;
        number += 1;
    }
    if (number)
        value /= number;

    return value;
}

static double computeDoubleResidue(const cNOSolIn_Triplet* tA,
                                   const std::array<ElRotation3D, 3>& oriA,
                                   const cNOSolIn_Triplet* tB,
                                   const std::array<ElRotation3D, 3>& oriB) {
    std::vector<tSomNSI*> views;
    std::vector<int> indexA;
    std::vector<int> indexB;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (tA->KSom(i) == tB->KSom(j)) {
                views.push_back(tA->KSom(i));
                indexA.push_back(i);
                indexB.push_back(j);
                break;
            }
        }
    }

    if (views.size() != 2) {
        std::cout << "Error number of common view." << std::endl;
    }

    double residuA =
        computeAsymResidue(tA->getHomolPts(), tA, oriA, tB, indexB, oriB);
    double residuB =
        computeAsymResidue(tB->getHomolPts(), tB, oriB, tA, indexA, oriA);

    return (residuA + residuB)/2.;
    //return residuB;
}


//Complexity N * edge
void RandomForest::PreComputeTriplets(Dataset& data) {

    for (int aKC = 0; aKC < int(data.mVCC.size()); aKC++) {
        for (auto t : data.mVCC[aKC]->mTri) {
            for (int aK = 0; aK < 3; aK++) {
                // Set the current R,t of the origin
                t->KSom(aK)->attr().CurRot() = t->RotOfSom(t->KSom(aK));
            }
            //auto aResidue = t->ProjTest();

            for (uint8_t i = 0; i < 3; i++) {
                auto* a = t->KArc(i)->attr().ASym();
                for (auto& t1 : a->Lnk3()) {
                    if (t == t1.m3) {
                        t1.Pds() = 0;
                        continue;
                    }
                    auto oriB = EstimAllRt(&t1);

                    std::array<ElRotation3D, 3> oriA {
                        t->KSom(0)->attr().CurRot(),
                        t->KSom(1)->attr().CurRot(),
                        t->KSom(2)->attr().CurRot()
                    };

                    double aResidue = computeDoubleResidue(t, oriA, t1.m3, oriB);
                    // double aResidueold = t1.m3->ProjTest();
                    // double aResidueold = 0;
                    /*std::cout << aResidue << " - " << aResidueold << std::endl; std::cout << "["
                                            << t->KSom(0)->attr().Im()->Name() << ","
                                            << t->KSom(1)->attr().Im()->Name() << ","
                                            << t->KSom(2)->attr().Im()->Name() << "]"
                                            << std::endl;
                      std::cout << "["
                                << t1.m3->KSom(0)->attr().Im()->Name() << ","
                                << t1.m3->KSom(1)->attr().Im()->Name() << ","
                                << t1.m3->KSom(2)->attr().Im()->Name() << "]"
                                << std::endl;*/
                    // t1.Pds() = aResidue;
                    t1.Pds() = aResidue;
                    t1.m3->pondSum += aResidue;
                    t1.m3->pondN += 1;
                }
            }
            //t->confiance = aResidue;
        }
    }
}

/* Final mean and 80% quantile incoherence computed on all triplets in the graph
 */
double median(std::vector<double> &v)
{
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}


void RandomForest::CoherTripletsAllSamples(Dataset& data) {
    for (int aT = 0; aT < int(data.mV3.size()); aT++) {
        if (data.mV3[aT]->Data()[0].size() == 0
            || data.mV3[aT]->Data()[2].size() == 0) {
            for (uint8_t i = 0; i<4; i++)
                data.mV3[aT]->Sum()[i] = 1000000;
            continue;
        }
        data.mV3[aT]->Sum()[0] = mean(data.mV3[aT]->Data()[0]);
        data.mV3[aT]->Sum()[1] = MedianeSup(data.mV3[aT]->Data()[0]);
        data.mV3[aT]->Sum()[2] = mean(data.mV3[aT]->Data()[2]);
        auto a = data.mV3[aT]->Data()[2];
        std::sort(a.begin(), a.end());
        data.mV3[aT]->Sum()[3] = a.back();
    }
}

/* Final incoherence computed as a weighted median
   - optionally takes into account only triplets with distance < threshold */
/*
void RandomForest::CoherTripletsAllSamplesMesPond(Dataset& data) {
    for (int aT = 0; aT < int(data.mV3.size()); aT++) {
        / * Weighted mean */
/*
        //data.mV3[aT]->CostArc() =
        //    data.mV3[aT]->CostPdsSum() / data.mV3[aT]->PdsSum();

        std::vector<Pt2df> aVCostPds;
        for (int aS = 0; aS < int(data.mV3[aT]->CostArcPerSample().size());
             aS++) {
            aVCostPds.push_back(Pt2df(data.mV3[aT]->CostArcPerSample()[aS],
                                      data.mV3[aT]->DistArcPerSample()[aS]));
            // aVCostPds.push_back (Pt2df(mV3[aT]->CostArcPerSample()[aS], 0));
            // *std::cout << std::setprecision(10);
              std::cout << " Cost|Dist|SQRTdist= " <<
              mV3[aT]->CostArcPerSample()[aS] << "|"
              << mV3[aT]->DistArcPerSample()[aS] << "|"
              << std::sqrt(mV3[aT]->DistArcPerSample()[aS]) << "\n";*/
/*        }
/
        // Weighted median */
 /*       if (aVCostPds.size())
            data.mV3[aT]->CostArcMed() = ;
        // MedianPond(aVCostPds, 0);
    }
}*/

/* This heap will serve to GetBestTri when building the ultimate init solution
 */
void RandomForest::HeapPerEdge(Dataset& data) {
    // For all triplets
    for (auto aTri : data.mV3) {
        // For each edge of the current triplet
        for (int aK = 0; aK < 3; aK++) {
            std::vector<cLinkTripl>& aLnk =
                aTri->KArc(aK)->attr().ASym()->Lnk3();

            // For all adjacent triplets to the current edge
            // Push to heap
            for (auto aTriAdj : aLnk) {
                aTri->KArc(aK)->attr().ASym()->mHeapTri.push(&aTriAdj);
            }

            // Order index
            for (auto aTriAdj : aLnk) {
                aTri->KArc(aK)->attr().ASym()->mHeapTri.MAJ(&aTriAdj);
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

void RandomForest::Save(Dataset& data, const std::string& OriOut, bool SaveListOfName) {
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

        std::vector<double> aResidueV = aTri->Data()[0];
        std::vector<double> aDistV = aTri->Data()[1];

        int aNb = int(aDistV.size());
        for (int aS = 0; aS < aNb; aS++) {
            std::cout << "[" << aResidueV.at(aS) << "," << aDistV.at(aS) << "], ";
        }
        std::cout << "\n";
    }
}


void RandomForest::ShowTripletCost(Dataset& data) {
    for (auto aTri : data.mV3) {
        std::cout << "[" << aTri->KSom(0)->attr().Im()->Name() << ","
                  << aTri->KSom(1)->attr().Im()->Name() << ","
                  << aTri->KSom(2)->attr().Im()->Name() << "], "
                  << " Cost=" << aTri->Sum()[2]
                  << "   ,MED=" << aTri->Sum()[3] << "\n";
    }
}

// Entry point
void RandomForest::DoNRandomSol(Dataset& data) {
    // Create connected components
    clock_t start1 = clock();
    NumeroteCC(data);
    clock_t end1 = clock();
    std::cout << "NumeroteCC " << double(end1 - start1)/CLOCKS_PER_SEC << std::endl;
    {
        DataLog<std::string, std::string, std::string, std::string> log(
            {"Id", "Image1", "Image2", "Image3"});
        for (auto& trip : data.mV3) {
            log.add({std::to_string(trip->NumId()),
                     trip->KSom(0)->attr().Im()->Name(),
                     trip->KSom(1)->attr().Im()->Name(),
                     trip->KSom(2)->attr().Im()->Name()});
        }
        log.write(mOutName + "/graph/logs/", "all.csv");
    }

    std::cout << "CREATE CC" << std::endl;

    if (aPond) {
        clock_t start2 = clock();
        PreComputeTriplets(data);
        clock_t end2 = clock();
        std::cout << "Precompute " << double(end2 - start2)/CLOCKS_PER_SEC << std::endl;
    }

    // Build random inital solutions default 1000 ?
    for (int aIterCur = 0; aIterCur < mNbSamples; aIterCur++) {
        std::cout << "Iter=" << aIterCur << "\n";
        clock_t start3 = clock();
        RandomSolAllCC(data);
        clock_t end3 = clock();
        std::cout << "Iter " << double(end3 - start3)/CLOCKS_PER_SEC*1000 << std::endl;
    }

    std::cout << "GENERATED Trees" << std::endl;

    for (auto& t : data.mV3) {
        DataLog<size_t, double, double, double> log({"Id", "Distance", "Residue", "Score"});
        for (size_t i = 0; i < t->Data()[0].size(); i++) {
            log.add({i, t->Data()[1][i], t->Data()[0][i], t->Data()[2][i]});
        }
        log.write(mOutName + "/graph/logs/triplets/", std::to_string(t->NumId()) + ".csv");
    }

    // Calculate median/mean incoherence scores of mNbSamples
    CoherTripletsAllSamples(data);
    //CoherTripletsAllSamplesMesPond(data);
    //std::cout << "Ponderate Trees" << std::endl;

    // Print the cost for all triplets
    ShowTripletCost(data);
    if (mDebug) ShowTripletCostPerSample(data);

    // Build "most coherent" solution
    BestSolAllCC(data);

    GraphViz g;
    g.loadTotalGraph(data);
    g.write(mOutName + "/graph/", "total.dot");
    logTotalGraph(data, mOutName + "/graph/logs/", "total.csv");
}
void RandomForest::logTotalGraph(Dataset& data, std::string dir, std::string filename) {
    DataTravel travel(data);
    DataLog<int, int, double, double, double, double, double> log(
        {"Id", "Category", "Distance", "Residue", "ResidueMedian", "Score",
         "ScoreMedian"});
    // Clear the sommets
    travel.mVS.clear();
    auto flag = data.mAllocFlag3.flag_alloc();

    cNOSolIn_Triplet* aTri0 = data.mV3[0];

    // initialise the starting node
    double distance = mean(aTri0->Data()[1]);

    log.add({aTri0->NumId(), aTri0->category, distance, aTri0->Sum()[0],
             aTri0->Sum()[1], aTri0->Sum()[2], aTri0->Sum()[3]});

    std::vector<cNOSolIn_Triplet*> aCC3;
    // Add first triplet
    aCC3.push_back(aTri0);
    aTri0->Flag().set_kth_true(flag);  // explored

    unsigned aKCur = 0;
    // Visit all sommets (not all triplets)
    while (aKCur != aCC3.size()) {
        cNOSolIn_Triplet* aTri1 = aCC3[aKCur];
        // For each edge of the current triplet
        for (int aKA = 0; aKA < 3; aKA++) {
            // Get triplet adjacent to this edge and parse them
            std::vector<cLinkTripl>& aLnk =
                aTri1->KArc(aKA)->attr().ASym()->Lnk3();

            for (unsigned aKL = 0; aKL < aLnk.size(); aKL++) {
                // If not marked, mark it and push it in aCC3, return it was
                // added
                if (SetFlagAdd(aCC3, aLnk[aKL].m3, flag)) {
                    // travel.mVS[aLnk[aKL].S3()->attr().Im()->Name()] =
                    //    aLnk[aKL].S3();

                    std::cout << aCC3.size() << "=["
                              << aLnk[aKL].S1()->attr().Im()->Name();
                    std::cout << "," << aLnk[aKL].S2()->attr().Im()->Name();
                    std::cout << "," << aLnk[aKL].S3()->attr().Im()->Name()
                              << "=====\n";
                    double distance = mean(aLnk[aKL].m3->Data()[1]);
                    log.add({aLnk[aKL].m3->NumId(), aLnk[aKL].m3->category,
                             distance, aLnk[aKL].m3->Sum()[0],
                             aLnk[aKL].m3->Sum()[1], aLnk[aKL].m3->Sum()[2],
                             aLnk[aKL].m3->Sum()[3]});
                }
            }
        }
        aKCur++;
    }

    // Unflag all triplets do pursue with DFS
    for (unsigned aK3 = 0; aK3 < data.mV3.size(); aK3++) {
        data.mV3[aK3]->Flag().set_kth_false(flag);
    }

    data.mAllocFlag3.flag_free(flag);
    log.write(dir, filename);
}

/******************************
  End cSolGlobInit_RandomForest
*******************************/

////////////////////////// Main //////////////////////////

int CPP_SolGlobInit_RandomForest_main(int argc, char** argv) {
    RandomForest aSGI(argc, argv);
    srand(time(NULL));

    std::cout << "INIT" << std::endl;

    Dataset data;
    std::cout << "INIT Dataset" << std::endl;
    aSGI.loadDataset(data);
    std::cout << "LOADED Dataset" << std::endl;

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
