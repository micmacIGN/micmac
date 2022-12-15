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




#include "cNewO_ExportTriplets.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

/******************************
  Start cAppliGenOptTriplets
*******************************/
class cNO_Triplet {
   public:
    cNO_Triplet(cNewO_NameManager& aNM, std::string& image1,
                std::string& image2, std::string& image3,
                const cXml_Ori3ImInit& aTrip)
        : names{image1, image2, image3},
        mIm{  new cNewO_OneIm(aNM, image1),
                new cNewO_OneIm(aNM, image2),
                new cNewO_OneIm(aNM, image3)
          },
          mNb3(aTrip.NbTriplet()),
          mR2on1(Xml2El(aTrip.Ori2On1())),
          mR3on1(Xml2El(aTrip.Ori3On1()))
     {}

    int Nb3() const { return mNb3; }
    ElTabFlag& Flag() { return mTabFlag; }
    std::string names[3];
    cNewO_OneIm* mIm[3];

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

   private:
    int mNb3;
    ElTabFlag mTabFlag;
    ElRotation3D mR2on1;
    ElRotation3D mR3on1;
};



cAppliExportTriplets::cAppliExportTriplets(int argc, char** argv) {
    std::string aDir;
    std::string prefix = "Test";
    bool aModeBin = true;

        ElInitArgMain
        (
            argc, argv,
            LArgMain() << EAMC(mFullPat, "Pattern"),
            LArgMain() << EAM(aModeBin,"Bin",true,"Binaries file, def = true",eSAM_IsBool)
                    << EAM(prefix, "OriPrefOut", true,
                            "Orientation prefix output name for each triplet. Def=Test")
                    << ArgCMA()
        );
    mEASF.Init(mFullPat);
    mNM = new cNewO_NameManager(mExtName, mPrefHom, mQuick, mEASF.mDir,
                                mNameOriCalib, "dat");
    //const cInterfChantierNameManipulateur::tSet * aVIm = mEASF.SetIm();

    //StdCorrecNameOrient(mNameOriCalib, aDir);
    //StdCorrecNameOrient(InOri, aDir);
    std::cout << mNM->Dir3P() << std::endl;

    cXml_TopoTriplet aXml3 =
        StdGetFromSI(mNM->NameTopoTriplet(true), Xml_TopoTriplet);

    std::vector<cXml_OneTriplet> triplets(aXml3.Triplets().begin(),
                                          aXml3.Triplets().end());
    size_t index = 0;
    for (auto t : triplets) {
        std::string aN3 =
            mNM->NameOriOptimTriplet(aModeBin, t.Name1(), t.Name2(), t.Name3());

        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aN3, Xml_Ori3ImInit);
        cNO_Triplet triplet(*mNM, t.Name1(), t.Name2(), t.Name3(), aXml3Ori);
        std::cout << index << std::endl;
        Save(index++, &triplet, prefix);
    }
}

void cAppliExportTriplets::Save(size_t index, cNO_Triplet* triplet, std::string prefixOriOut) {
    std::list<std::string> aListOfName;

    for (size_t i = 0; i < 3; i++) {
        std::string aNameIm = triplet->names[i];

        CamStenope* aCS = triplet->mIm[i]->CS();
        ElRotation3D aROld2Cam = triplet->RotOfK(i).inv();

        aCS->SetOrientation(aROld2Cam);

        cOrientationConique anOC = aCS->StdExportCalibGlob();
        anOC.Interne().SetNoInit();

        std::string aFileIterne =
            mNM->ICNM()->StdNameCalib(mNameOriCalib, aNameIm);

        std::string aNameOri = mNM->ICNM()->Assoc1To1(
            "NKS-Assoc-Im2Orient@-" + prefixOriOut + "-" + std::to_string(index), aNameIm, true);
        anOC.FileInterne().SetVal(NameWithoutDir(aFileIterne));

        // Copy interior orientation
        std::string aCom = "cp " + aFileIterne + " " + DirOfFile(aNameOri);
        System(aCom);

        aListOfName.push_back(aNameIm);

        MakeFileXML(anOC, aNameOri);
    }
}


/******************************
  End cAppliGenOptTriplets
*******************************/

////////////////////////// Mains //////////////////////////

int CPP_ExportTriplets(int argc,char ** argv)
{
    cAppliExportTriplets aAppExpTri(argc,argv);

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
