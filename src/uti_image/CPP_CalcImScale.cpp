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
#include "im_tpl/algo_filter_exp.h"


#define DEF_OFSET -12349876


class cCalcImScale
{
    public :
        cCalcImScale(int argc,char** argv);
    private :

       std::string NameIMax(const std::string & aNature) {return  "IndMax-" + aNature + mNameIm + ".tif";}

       double  SigmaOfK(int aK) const;
       // Compute grad then filter
       void  OneBoxScaleGradScale(const Box2di & aBoxIn,const Box2di & aBoxOut,int aNum);


       void  OneBoxScaleOfGrad(const Box2di & aBoxIn,const Box2di & aBoxOut);
       void  OneBoxLapl(const Box2di & aBoxIn,const Box2di & aBoxOut);
       std::string mNameIm;
       int         mNumF;
       int         mNbByOct;
       int         mNbPow;
       Pt2di       mSzGlob;
       int         mSzMax;
       int         mSzBrd;
       Im2D_REAL4  mIm0;
};

double  cCalcImScale::SigmaOfK(int aK) const
{
    return pow(2.0,aK/double(mNbByOct));
}

void  cCalcImScale::OneBoxScaleGradScale(const Box2di & aBoxIn,const Box2di & aBoxOut,int aNum)
{

   bool IsModified;
   std::string aNameIMax =  NameIMax("SGradS");

    Tiff_Im aTifInd= Tiff_Im::CreateIfNeeded(IsModified,aNameIMax,mSzGlob,GenIm::u_int1,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);

    Pt2di aSz = aBoxIn.sz();
    mIm0.Resize(aBoxIn.sz());

    Im2D_REAL4   aImValMax(aSz.x,aSz.y,-1.0);
    Im2D_U_INT1  aImIndMax(aSz.x,aSz.y,255);


    Tiff_Im aTF = Tiff_Im::StdConvGen(mNameIm,1,true);

    Fonc_Num aF = trans(aTF.in_proj(),aBoxIn.P0());
    // aF = deriche(aF,0.5,10);
    // aF = Polar_Def_Opun::polar(aF,0.0).v0();

    ELISE_COPY(mIm0.all_pts(),aF,mIm0.out());

    for (int aK=0 ; aK< mNbPow ; aK++)
    {
        double aPow  =  SigmaOfK(aK)* 1.1;
        Pt2di aSzRed =  round_ni(Pt2dr(aSz)/aPow);

        Im2D_REAL4  aImRed(aSzRed.x,aSzRed.y);
        Im2D_REAL4  aImRed2(aSzRed.x,aSzRed.y);
        Fonc_Num aFIn = StdFoncChScale(mIm0.in_proj(), Pt2dr(0,0), Pt2dr(aPow,aPow));

        ELISE_COPY ( aImRed.all_pts(), aFIn,aImRed.out() );


        Im2D_REAL4  aImFilter(aSzRed.x,aSzRed.y);

        // int aSzF = 1;
        // double aNbW = ElSquare(1+2*aSzF);

        ELISE_COPY(aImRed.all_pts(),aFIn,aImRed.out());
        ELISE_COPY(aImRed.all_pts(),Square(aImRed.in()),aImRed2.out());

        double aSzF = 1.5;
        int    aNbIter = 2;

        FilterGauss(aImRed,aSzF,aNbIter);
        FilterGauss(aImRed2,aSzF,aNbIter);

        // Fonc_Num  aFFilter = rect_som(Square(aImRed.in_proj()),aSzF) / aNbW  - Square(rect_som(aImRed.in_proj(),aSzF) / aNbW);
        ELISE_COPY
        (
             aImRed.all_pts(),
             aImRed2.in() - Square(aImRed.in()),
             aImFilter.out()
        );

        // Tiff_Im::Create8BFromFonc("FIl_" + ToString(aK) + ".tif",aSzRed,aImFilter.in());

        Fonc_Num aFFilterReech = StdFoncChScale(aImFilter.in_proj(), Pt2dr(0,0), Pt2dr(1/aPow,1/aPow));


        Im2D_REAL4  aImR1(aSz.x,aSz.y);
        ELISE_COPY(aImR1.all_pts(),aFFilterReech,aImR1.out());
        ELISE_COPY
        (
             select(mIm0.all_pts(),aImR1.in()>aImValMax.in()),
             aImR1.in(),
             aImValMax.out()  |  (aImIndMax.out() << aK)
        );
        std::cout << "SGS ******   Remain " << mNbPow-aK << " in current box \n";

        Tiff_Im::CreateFromIm(aImR1,"FIl_" + ToString(aK) + ".tif");
    }

    ELISE_COPY
    (
         aBoxOut.Flux(),
         trans(aImIndMax.in(),-aBoxIn.P0()),
         // trans(mIm0.in(),-aBoxIn.P0()),
         aTifInd.out()
    );
}


void  cCalcImScale::OneBoxScaleOfGrad(const Box2di & aBoxIn,const Box2di & aBoxOut)
{

   bool IsModified;
   std::string aNameIMax =  NameIMax("ScOfGrad");

    Tiff_Im aTifInd= Tiff_Im::CreateIfNeeded(IsModified,aNameIMax,mSzGlob,GenIm::u_int1,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);

    Pt2di aSz = aBoxIn.sz();
    mIm0.Resize(aBoxIn.sz());

    Im2D_REAL4   aImValMax(aSz.x,aSz.y,-1.0);
    Im2D_U_INT1  aImIndMax(aSz.x,aSz.y,255);


    Tiff_Im aTF = Tiff_Im::StdConvGen(mNameIm,1,true);

    Fonc_Num aF = trans(aTF.in_proj(),aBoxIn.P0());
    aF = deriche(aF,0.5,10);
    aF = Polar_Def_Opun::polar(aF,0.0).v0();

    ELISE_COPY(mIm0.all_pts(),aF,mIm0.out());

    for (int aK=0 ; aK< mNbPow ; aK++)
    {
        int aNbIter = ElMax(2,4-aK);
        double aP0 =  SigmaOfK(aK-1);
        double aP1 =  SigmaOfK(aK);
        double aSz = sqrt(ElSquare(aP1) - ElSquare(aP0));
        FilterGauss(mIm0,aSz,aNbIter);
        // FilterGauss(I
        ELISE_COPY
        (
             select(mIm0.all_pts(),mIm0.in()>aImValMax.in()),
             mIm0.in(),
             aImValMax.out()  |  (aImIndMax.out() << aK)
        );
        std::cout << "Remain " << mNbPow-aK << " in current box \n";
    }

    ELISE_COPY
    (
         aBoxOut.Flux(),
         trans(aImIndMax.in(),-aBoxOut.P0()),
         aTifInd.out()
    );
}



void  cCalcImScale::OneBoxLapl(const Box2di & aBoxIn,const Box2di & aBoxOut)
{

   bool IsModified;
   std::string aNameIMax =  NameIMax("Lapl");

    Tiff_Im aTifInd= Tiff_Im::CreateIfNeeded(IsModified,aNameIMax,mSzGlob,GenIm::u_int1,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);

    Pt2di aSz = aBoxIn.sz();
    mIm0.Resize(aBoxIn.sz());

    Im2D_REAL4   aImValMax(aSz.x,aSz.y,-1.0);
    Im2D_U_INT1  aImIndMax(aSz.x,aSz.y,255);


    Tiff_Im aTF = Tiff_Im::StdConvGen(mNameIm,1,true);

    Fonc_Num aF = trans(aTF.in_proj(),aBoxIn.P0());
    ELISE_COPY(mIm0.all_pts(),aF,mIm0.out());
    FilterGauss(mIm0,1.0,3);

    for (int aK=0 ; aK< mNbPow ; aK++)
    {
        Im2D_REAL4  mImPrec = mIm0.dup();
        int aNbIter = ElMax(2,4-aK);
        double aP0 =  SigmaOfK(aK-1);
        double aP1 =  SigmaOfK(aK);
        // double aSz = aP1 -aP0;
        // double aSz = sqrt(ElSquare(aP1) - ElSquare(aP0));
        double aSz = sqrt(ElSquare(aP1) - ElSquare(aP0));
        FilterGauss(mIm0,aSz,aNbIter);
        // FilterGauss(I

        ELISE_COPY(mImPrec.all_pts(),Abs(mImPrec.in()-mIm0.in()),mImPrec.out());
        ELISE_COPY
        (
             select(mIm0.all_pts(),mImPrec.in()>aImValMax.in()),
             mImPrec.in(),
             aImValMax.out()  |  (aImIndMax.out() << aK)
        );
        std::cout << "Remain " << mNbPow-aK << " in current box \n";
        
    }

    ELISE_COPY
    (
         aImIndMax.all_pts(),
         rect_min(rect_max(rect_min(aImIndMax.in_proj(),1),2),1),
         aImIndMax.out()
    );

    ELISE_COPY
    (
         aBoxOut.Flux(),
         trans(aImIndMax.in(),-aBoxIn.P0()),
         aTifInd.out()
    );
}





cCalcImScale::cCalcImScale(int argc,char** argv) :
     mNbByOct  (5),
     mNbPow    (20),
     mSzMax    (3000),
     mSzBrd    (100),
     mIm0      (1,1)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mNameIm,"Image name", eSAM_IsExistFile)
                    << EAMC(mNumF,"Num Filtr 0 SGS ,  1  Grad S, 2 Lapl"),
        LArgMain()  << EAM(mNbByOct,"NbByO",true,"Number by octave, def = 5")
                    << EAM(mSzMax,"SzMax",true,"Sz for tiling")
   );

   Tiff_Im aTF = Tiff_Im::StdConvGen(mNameIm,1,true);
   mSzGlob = aTF.sz();
   cDecoupageInterv2D aDec =   cDecoupageInterv2D::SimpleDec(mSzGlob,mSzMax,mSzBrd);

   for (int aKInterv = 0 ; aKInterv < aDec.NbInterv() ; aKInterv++)
   {
       if (mNumF==1)
           OneBoxScaleOfGrad(aDec.KthIntervIn(aKInterv),aDec.KthIntervOut(aKInterv));
       else if (mNumF==2)
           OneBoxLapl(aDec.KthIntervIn(aKInterv),aDec.KthIntervOut(aKInterv));
       else if ((mNumF>=3) && (mNumF<10))
            OneBoxScaleGradScale(aDec.KthIntervIn(aKInterv),aDec.KthIntervOut(aKInterv),mNumF);
       else 
           ELISE_ASSERT(false,"Bad Num filter");
   }
}

int CPP_CalcImScale(int argc,char** argv)
{
    cCalcImScale anAppli(argc,argv);

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
