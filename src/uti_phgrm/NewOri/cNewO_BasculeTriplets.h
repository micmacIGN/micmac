#pragma once
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


#include "NewOri.h"
//#include "general/CMake_defines.h"
#include "XML_GEN/SuperposImage.h"
#include "XML_GEN/all.h"
#include "general/bitm.h"
#include "graphes/cNewO_BuildOptions.h"
#include <array>
#include <random>
#include <vector>
#define TREEDIST_WITH_MMVII false
#include  "../../../MMVII/include/TreeDist.h"

struct cTriplet;
struct cOriBascule;



struct cOriBascule {
    cOriBascule(){}
    cOriBascule(std::string name)
        :  mName(name),
           mXmlFile("Orientation-" + name + ".xml")
    {}

    std::string mName;
    std::string mXmlFile;
    std::vector<cTriplet*> ts;

};

struct cOriTraversal {
    cOriBascule* b1;
    cOriBascule* b2;
    ElRotation3D rot;
};

struct cTriplet {
    cTriplet(cOriBascule* s1, cOriBascule* s2, cOriBascule* s3,
             cXml_Ori3ImInit& xml, std::array<short, 3>& mask);

    cXml_Ori3ImInit t;
    std::array<short, 3> masks; // true = first block

    const ElRotation3D& RotOfSom(const cOriBascule* aS) const {
        if (aS == mSoms[0]) return ElRotation3D::Id;
        if (aS == mSoms[1]) return mR2on1;
        if (aS == mSoms[2]) return mR3on1;
        ELISE_ASSERT(false, " RotOfSom");
        return ElRotation3D::Id;
    }

    const ElRotation3D& RotOfK(int aK) const {
        switch (aK) {
            case 0:
                return ElRotation3D::Id;
            case 1:
                return mR2on1;
            case 2:
                return mR3on1;
        }
        ELISE_ASSERT(false, " RotOfSom");
        return ElRotation3D::Id;
    }

    const ElRotation3D RotOfJToK(int aJ, int aK) const {
        switch (aK) {
            case 0:
                switch (aJ) {
                    case 0:
                        return ElRotation3D::Id;
                    case 1:
                        return mR2on1;
                    case 2:
                        return mR3on1;
                }
                break;
            case 1:
                switch (aJ) {
                    case 0:
                        return mR2on1.inv();
                    case 1:
                        return ElRotation3D::Id;
                    case 2:
                        return mR2on1.inv() * mR3on1;
                }
                break;
            case 2:
                switch (aJ) {
                    case 0:
                        return mR3on1.inv();
                    case 1:
                        return mR3on1.inv() * mR2on1;
                    case 2:
                        return ElRotation3D::Id;
                }
                break;
        }

        ELISE_ASSERT(false, " RotOfSom");
        return ElRotation3D::Id;
    }

    ElRotation3D mR2on1;
    ElRotation3D mR3on1;
    float mBOnH;

    cOriBascule* mSoms[3];
};

class cOriTriplets
{
    public :
        cOriTriplets();
        CamStenope * mCam1;
        CamStenope * mCam2;
        std::string mNameFull;
        std::string mName;
        std::string mIm;
        std::vector<cTriplet*> triplets;
};


class cAppliBasculeTriplets : public cCommonMartiniAppli
{
	public:
		cAppliBasculeTriplets(int argc,char ** argv);

        void InitOneDir(const std::string & aPat, bool aD1);

        void InitTriplets(bool aModeBin);

        void IsolateRotTr();

        const std::string & Dir() {return  mDir ;}

	private:
        std::string       mOri1;
        std::string       mOri2;
        std::string       mOriOut;

        std::string       mDir;
        std::string       mDirOutLoc;
        std::string       mDirOutGlob;

        std::set<std::string> block1;
        std::set<std::string> block2;
        std::map<std::string, std::string> mapping;

        std::map<std::string, cOriBascule> mAllOris;
        std::map<std::string, cOriTriplets> mOrients;
        std::vector<cTriplet> ts;

		ElMatrix<double> RandPeturbR();
        ElMatrix<double> RandPeturbRGovindu();
        ElRotation3D RandView(const ElRotation3D &view, double sigmaT, double sigmaR);
        ElMatrix<double> w2R(double[]);

		int    mNbTri;
        //outlier ratio, if 0.3 => 30% of outliers will be present among the triplets
		double mRatioOutlier;
        cElemAppliSetFile    mEASF;
        cNewO_NameManager  * mNM;

		//RandUnifQuick * TheRandUnif;
};

/*
class RandUnifQuick
{
    public:
        RandUnifQuick(int Seed);
        double Unif_0_1();
        ~RandUnifQuick() {}

    private:
        std::mt19937                     mGen;
        std::uniform_real_distribution<> mDis01;

};
*/

/*
double RandUnif_C()
{
   return (RandUnif_0_1()-0.5) * 2.0;
}
*/

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
