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
#include "TpPPMD.h"

inline double PdsV(double aDxy)
{
	return (aDxy==0) ? 2 : 1;
}

int  TD_Match1_main(int argc,char ** argv)
{
	std::string aName ="/home/prof/Bureau/CalibCoul/_DSC0323_Bayer_8Bits.tif";
    cTD_Im  anIm = cTD_Im::FromString(aName);
    
    Pt2di aSz = anIm.Sz();
    cTD_Im  anImOut(aSz.x,aSz.y);
    for (int anX=1 ; anX<aSz.x -1; anX++)
    {
		for (int anY=1 ; anY<aSz.y -1; anY++)
		{
		    double aVal = 0;
		    for (int aDx=-1; aDx <= 1 ; aDx++)
		    {
			   for (int aDy=-1; aDy <= 1 ; aDy++)
		       {
				   double aP = PdsV(aDx) * PdsV(aDy);
				   aVal += aP * anIm.GetVal(anX+aDx,anY+aDy);
			   }
			}
		    anImOut.SetVal(anX,anY,aVal/16.0);
		}
	}
    
  
    std::string NameOut = "/home/prof/Bureau/CalibCoul/DeBayer.tif";
    
    anImOut.Save(NameOut);
    
    return EXIT_SUCCESS;
}


int  TD_Match2_main(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}
/************************************************/

/************************************************/

class cCelMatch3
{
        public :


        ///================================================
        ///===== PRE-REQUIREMENT FOR 2D PROG DYN , transfer
        ///===== from 3D volume to buffer of computation 
        ///================================================
        void InitTmp(const cTplCelNapPrgDyn<cCelMatch3> & aCel)
        {
            *this = aCel.ArgAux();
        }
        private :
};


/* Class that will be used to instantiate cProg2DOptimiser
*/

class cTD_Match3_main
{
     public :
        int ToI(const double & aFl) {return round_ni(1000 *aFl);}

        ///===== PRE-REQUIREMENT FOR 2D PROG DYN , transfer

            typedef  cCelMatch3 tArgCelTmp; // Type temporary when parsing a line
            typedef  cCelMatch3 tArgNappe;  // Type where is stored permently  the 3D value 

      ///-------- Not a PRE-REQUIREMENT but Helpull

            typedef  cTplCelNapPrgDyn<tArgNappe>    tCelNap;
            typedef  cTplCelOptProgDyn<tArgCelTmp>  tCelOpt;

      ///-------- REQUIREMENT : kernell of computation
           void DoConnexion
           (
                  const Pt2di & aPIn, const Pt2di & aPOut,
                  ePrgSens aSens,int aRab,int aMul,
                  tCelOpt*Input,int aInZMin,int aInZMax,
                  tCelOpt*Ouput,int aOutZMin,int aOutZMax
           );

      /// Required : but do generally nothing

          void GlobInitDir(cProg2DOptimiser<cTD_Match3_main> &) 
          {
              std::cout << "One New Dir\n";
          }





       //====================================
        cTD_Match3_main(int argc,char ** argv);

        Output  GrayW() {return (mW ? mW->ogray() : Output::onul(1));}

        void TestOnePx(int aPx);

        Im2D_U_INT1 mIm1;
        Im2D_U_INT1 mIm2;
        Im2D_REAL4  mImCor;
        Im2D_INT2   mPxOpt;
        Im2D_REAL4  mCorOpt;



        Pt2di       mSz1;
        Pt2di       mSz2;
        Pt2di       mPxInterv;
        Video_Win*  mW;
        int         mSzW;
        double      mNbPix;
        int         mNbJump;
        double      mRegulCost;
};

void cTD_Match3_main::DoConnexion
     (
         const Pt2di & aPIn,   // Point of input column
         const Pt2di & aPOut,  // Point of output column
         ePrgSens aSens,       // Is it forward or backwrad,  need to indicate it in udate
         int ,
         int ,
         tCelOpt*Input,int aInZMin,int aInZMax,
         tCelOpt*Ouput,int aOutZMin,int aOutZMax
     )
{

// =====================
    for (int aZOut=aOutZMin ; aZOut<aOutZMax ; aZOut++)  // Parse all the input Z
    {
           int aDZMin,aDZMax;
           // Compute the inteval of deta-z that are inside the "jump" and inside Z-In
           ComputeIntervaleDelta
            (
                aDZMin,aDZMax,aZOut,mNbJump,
                aOutZMin,aOutZMax,
                aInZMin,aInZMax
            );
            for (int aDZ = aDZMin; aDZ<= aDZMax ; aDZ++)
            {
                 int aZIn = aZOut + aDZ;
                 int  aICost = ToI(mRegulCost * ElAbs(aDZ) );  // Compute the regularization cost
                 Ouput[aZOut].UpdateCostOneArc(Input[aZIn],aSens,aICost);  // Update the out cost
            }
    }
}

void cTD_Match3_main::TestOnePx(int aPx)
{

     Fonc_Num aFTransIm2 = trans(mIm2.in(0),Pt2di(aPx,0));

   // aFTransIm2 = trans(mIm1.in(0),Pt2di(aPx,0));

     Fonc_Num  aF1 = rect_som(mIm1.in(0),mSzW) / mNbPix;
     Fonc_Num  aF2 = rect_som(aFTransIm2,mSzW) / mNbPix;
     Fonc_Num  aF11 = rect_som(Square(mIm1.in(0)),mSzW) / mNbPix;
     Fonc_Num  aF22 = rect_som(Square(aFTransIm2),mSzW) / mNbPix;
     Fonc_Num  aF12 = rect_som(mIm1.in(0)*aFTransIm2,mSzW) / mNbPix;

     aF11 = aF11 - Square(aF1);
     aF22 = aF22 - Square(aF2);
     aF12 = aF12 - aF1 * aF2;

     Fonc_Num  aFCor = aF12 / sqrt(Max(1e-5,aF11*aF22));
     
     ELISE_COPY
     (
         mIm1.all_pts(),
         aFCor,
         mImCor.out() | ( GrayW() << (Max(0 ,Min(255,(1+mImCor.in()) * 100))))
     );

     ELISE_COPY
     (
         select(mIm1.all_pts(), mImCor.in() > mCorOpt.in()),
         Virgule(mImCor.in(),aPx),
         Virgule(mCorOpt.out(),mPxOpt.out())
     );
}


cTD_Match3_main::cTD_Match3_main(int argc,char ** argv) :
    mIm1      (1,1),
    mIm2      (1,1),
    mImCor    (1,1),
    mPxOpt    (1,1),
    mCorOpt   (1,1),
    mPxInterv (-50,50),
    mW        (0),
    mSzW      (3),
    mNbJump   (1),
    mRegulCost (0.1)
{
    std::string mNameIm1,mNameIm2;
    bool Visu=true;

    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(mNameIm1,"Name Im1")
                      << EAMC(mNameIm2,"Name Im2"),
           LArgMain() << EAM(mPxInterv,"PxI",true, "Paralax intervall")
                      << EAM(Visu,"Visu",true,"Interactive visualisation")
                      << EAM(mSzW,"SzW",true,"Size of correlation window")
                      << EAM(mRegulCost,"Regul",true,"Regularisation factor")
                      << EAM(mNbJump,"NbJ",true,"Nb Jump")
    );

    mNbPix = ElSquare(1+2*mSzW);
    Tiff_Im aTif1(mNameIm1.c_str());
    Tiff_Im aTif2(mNameIm2.c_str());
    mSz1 = aTif1.sz();
    mSz2 = aTif2.sz();

    if (Visu)
       mW = Video_Win::PtrWStd(mSz1);

    mIm1 = Im2D_U_INT1(mSz1.x,mSz1.y);
    ELISE_COPY
    (
        mIm1.all_pts(),
        aTif1.in(),
        mIm1.out() | GrayW()
    );
    mPxOpt.Resize(mSz1);
    mImCor.Resize(mSz1);
    mCorOpt =  Im2D_REAL4(mSz1.x,mSz1.y,-1e10);
   
    mIm2 =  Im2D_U_INT1::FromFileStd(mNameIm2);
    // if (mW) mW->clik_in();

   Im2D_INT2 aNapInf(mSz1.x,mSz1.y, mPxInterv.x);   // Z Min has constant valure 
   Im2D_INT2 aNapSup(mSz1.x,mSz1.y,1+ mPxInterv.y);  // Z Max has "opposite" constant value

   //  Create the strtucture
   cProg2DOptimiser<cTD_Match3_main> aPrgD
                                     (
                                         *this,
                                         aNapInf,  //  ZMIn 
                                         aNapSup,  // ZMax
                                         0,1       // Allow to create more value
                                     );

   cDynTplNappe3D<tCelNap> &  aSparseVol = aPrgD.Nappe();
   tCelNap ***  aSparsPtr = aSparseVol.Data() ;

    // Fill the data term with correlation
    for (int aPx= mPxInterv.x ; aPx <= mPxInterv.y ; aPx++)
    {
        TestOnePx(aPx); // Compute correlation for 1 paralax
        float **  aDataCor = mImCor.data();
        for (int anX=0 ; anX<mSz1.x ; anX++)
        {
            for (int anY=0 ; anY<mSz1.y ; anY++)
            {
                int aICost =  ToI (1-aDataCor[anY][anX]);
                aSparsPtr[anY][anX][aPx].SetOwnCost(aICost);   // Fill the cube 
            }
        }
    }

    aPrgD.DoOptim(5);  // Main call to optimize, parameter = Number of direction

    aPrgD.TranfereSol(mPxOpt.data()); // Copy the value of solution
    
    Tiff_Im::Create8BFromFonc("PxPrgD.tif",mSz1,mPxOpt.in()+128);
}



int  TD_Match3_main(int argc,char ** argv)
{
    std::cout << "TD_Match3_main \n";
    cTD_Match3_main  aM3(argc,argv);
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
