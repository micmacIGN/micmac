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

#ifndef _NEW_PERF_TRI_H
#define _NEW_PERF_TRI_H

#include "NewOri.h"
#include <random>

class RandUnifQuick;

class cAppliGenOptTriplets : public cCommonMartiniAppli
{
	public:
		cAppliGenOptTriplets(int argc,char ** argv);
	    ~cAppliGenOptTriplets() {  };

        static ElMatrix<double> w2R(double[]);
	private:
        
		ElMatrix<double> RandPeturbR();
        ElMatrix<double> RandPeturbRGovindu();


        ElRotation3D OriRelPairFromExisting(std::string& InOri,const std::string& N1,const std::string& N2);

		std::string mFullPat; 
		std::string InOri;


		int    mNbTri;
		double mSigma;//bruit
		double mRatioOutlier;//outlier ratio, if 0.3 => 30% of outliers will be present among the triplets

		cElemAppliSetFile    mEASF;
        cNewO_NameManager  * mNM;

		RandUnifQuick * TheRandUnif;
};

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

#endif 
