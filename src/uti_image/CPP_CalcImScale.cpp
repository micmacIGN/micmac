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

       std::string NameIMax(const std::string & aNature) 
       {
            return  "IndMax-" + std::string(mMaxGlob?"Glob-":"Loc-" ) + aNature + mNameIm + ".tif";
       }

       double  SigmaOfK(int aK) const;
       void    AddFoncInd(Fonc_Num aF,int aK);
       // Compute grad then filter
       std::string  OneBoxScaleOfGrad();

       std::string  OneBoxScaleGradScale();
       std::string  OneBoxLapl();

       std::string mNameIm;
       int         mNumF;
       int         mNbByOct;
       int         mNbPow;
       Pt2di       mSzGlob;
       int         mSzMax;
       int         mSzBrd;
       Im2D_REAL4  mIm0;
       Im2D_U_INT1 mIndMax;
       Im2D_REAL4  mImVMax;

       Im2D_REAL4  mImNM2;
       Im2D_REAL4  mImNM1;
       Im2D_REAL4  mImNM0;
       Im2D_U_INT1 mMaxLocPrec;

       Box2di      mBoxIn;
       Box2di      mBoxOut;
       Pt2di       mSzIn;
       bool        mMaxGlob;
};

double  cCalcImScale::SigmaOfK(int aK) const
{
    return pow(2.0,aK/double(mNbByOct));
}

void  cCalcImScale::AddFoncInd(Fonc_Num aF,int aK)
{
   //Pt2di aPBug(626,596);
   if (mMaxGlob)
   {
       ELISE_COPY
       (
          select(mIm0.all_pts(),aF>mImVMax.in()),
          aF,
          mImVMax.out()  |  (mIndMax.out() << aK)
       );
       return;
   }

   ELISE_COPY(mImNM0.all_pts(),aF,mImNM0.out());

   //std::cout << "LocVBUG= " << mImVMax.GetR(aPBug) << "\n";

   if (aK>=2)
   {
      // K=0 pas de max loc
      // K=1 max loc , calculable a K=2, et utilisable a partir de K=2 aussi
      // puisque la max lox "barre" la route aux suivants
      ELISE_COPY
      (
         select
         (
             mIm0.all_pts(),
                (mImNM2.in()<mImNM1.in()) 
             && (mImNM0.in() <= mImNM1.in())
         ),
         1,
         mMaxLocPrec.out()
      );
   }
   ELISE_COPY
   (
       select(mIm0.all_pts(),mImNM0.in()>mImVMax.in() && (!mMaxLocPrec.in())),
       aF,
       mImVMax.out()  |  (mIndMax.out() << aK)
   );


   Im2D_REAL4 aLastINM2 = mImNM2;
   mImNM2               = mImNM1;
   mImNM1               = mImNM0;
   mImNM0               = aLastINM2;
 
}

std::string   cCalcImScale::OneBoxScaleGradScale()
{
    for (int aK=0 ; aK< mNbPow ; aK++)
    {
        double aPow  =  SigmaOfK(aK)* 1.1;
        Pt2di aSzRed =  round_ni(Pt2dr(mSzGlob)/aPow);
        double aSzF = 1.5;
        int    aNbIter = 2;

        Im2D_REAL4  aImRed(aSzRed.x,aSzRed.y);
        Fonc_Num aFIn = StdFoncChScale(mIm0.in_proj(), Pt2dr(0,0), Pt2dr(aPow,aPow));
        ELISE_COPY ( aImRed.all_pts(), aFIn,aImRed.out() );

        Im2D_REAL4  aImRed2(aSzRed.x,aSzRed.y);
        ELISE_COPY(aImRed.all_pts(),Square(aImRed.in()),aImRed2.out());

        FilterGauss(aImRed,aSzF,aNbIter);
        FilterGauss(aImRed2,aSzF,aNbIter);
        Im2D_REAL4  aImFilter(aSzRed.x,aSzRed.y);
        ELISE_COPY
        (
             aImRed.all_pts(),
             aImRed2.in() - Square(aImRed.in()),
             aImFilter.out()
        );


        Fonc_Num aFFilterReech = StdFoncChScale(aImFilter.in_proj(), Pt2dr(0,0), Pt2dr(1/aPow,1/aPow));
        Im2D_REAL4  aImR1(mSzGlob.x,mSzGlob.y);
        ELISE_COPY(aImR1.all_pts(),aFFilterReech,aImR1.out());
        // Tiff_Im::CreateFromIm(aImR1,"FIl_" + ToString(aK) + ".tif");

        AddFoncInd(aImR1.in(),aK);
        std::cout << "SGS ******   Remain " << mNbPow-aK << " in current box \n";

    }


    return std::string("SGradS");
}


std::string  cCalcImScale::OneBoxScaleOfGrad()
{

    Fonc_Num aF = deriche(mIm0.in_proj(),0.5,10);
    aF = Polar_Def_Opun::polar(aF,0.0).v0();
    ELISE_COPY(mIm0.all_pts(),aF,mIm0.out());

    for (int aK=0 ; aK< mNbPow ; aK++)
    {
        int aNbIter = ElMax(2,4-aK);
        double aP0 =  SigmaOfK(aK-1);
        double aP1 =  SigmaOfK(aK);
        double aSz = sqrt(ElSquare(aP1) - ElSquare(aP0));
        FilterGauss(mIm0,aSz,aNbIter);
        AddFoncInd(mIm0.in(),aK);
        std::cout << "Remain " << mNbPow-aK << " in current box \n";
    }

    return std::string("ScOfGrad");
}



std::string  cCalcImScale::OneBoxLapl()
{
    FilterGauss(mIm0,1.0,3);

    for (int aK=0 ; aK< mNbPow ; aK++)
    {
        Im2D_REAL4  mImPrec = mIm0.dup();

        int aNbIter = ElMax(2,4-aK);
        double aP0 =  SigmaOfK(aK-1);
        double aP1 =  SigmaOfK(aK);
        double aSz = sqrt(ElSquare(aP1) - ElSquare(aP0));

        FilterGauss(mIm0,aSz,aNbIter);

        ELISE_COPY(mImPrec.all_pts(),Abs(mImPrec.in()-mIm0.in()),mImPrec.out());
        AddFoncInd(mImPrec.in(),aK);
        std::cout << "Remain " << mNbPow-aK << " in current box \n";
        
    }

    // Filtrage de Ind   ouverture puis fermeture
    ELISE_COPY
    (
         mIndMax.all_pts(),
         rect_min(rect_max(rect_min(mIndMax.in_proj(),1),2),1),
         mIndMax.out()
    );


    return std::string("Lapl");
}





cCalcImScale::cCalcImScale(int argc,char** argv) :
     mNbByOct  (5),
     mNbPow    (20),
     mSzMax    (3000),
     mSzBrd    (100),
     mIm0      (1,1),
     mIndMax   (1,1),
     mImVMax   (1,1),
     mImNM2    (1,1),
     mImNM1    (1,1),
     mImNM0  (1,1),
     mMaxLocPrec  (1,1),
     mMaxGlob  (false)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(mNameIm,"Image name", eSAM_IsExistFile)
                    << EAMC(mNumF,"Num Filtr 0 SGS ,  1  Grad S, 2 Lapl"),
        LArgMain()  << EAM(mNbByOct,"NbByO",true,"Number by octave, def = 5")
                    << EAM(mSzMax,"SzMax",true,"Sz for tiling")
                    << EAM(mNbPow,"NbP",true,"Number of power")
                    << EAM(mMaxGlob,"Glob",true,"Max Glob/Firs max loc")
   );

   Tiff_Im aTF = Tiff_Im::StdConvGen(mNameIm,1,true);
   mSzGlob = aTF.sz();
   cDecoupageInterv2D aDec =   cDecoupageInterv2D::SimpleDec(mSzGlob,mSzMax,mSzBrd);


   for (int aKInterv = 0 ; aKInterv < aDec.NbInterv() ; aKInterv++)
   {
       mBoxIn = aDec.KthIntervIn(aKInterv);
       mBoxOut = aDec.KthIntervOut(aKInterv);
       mSzIn   = mBoxIn.sz();

       mIm0.Resize(mSzIn);
       mIndMax.Resize(mSzIn);
       mImVMax.Resize(mSzIn);

       ELISE_COPY
       (
            mIm0.all_pts(),
            Virgule(trans(aTF.in(),mBoxIn.P0()),255,-1),
            Virgule(mIm0.out(),mIndMax.out(),mImVMax.out())
       );
       if (! mMaxGlob)
       {
          mImNM2.Resize(mSzIn);
          mImNM1.Resize(mSzIn);
          mImNM0.Resize(mSzIn);
          mMaxLocPrec.Resize(mSzIn);
          mMaxLocPrec.raz();
       }
     

       std::string aPost;
       if (mNumF==1)
           aPost = OneBoxScaleOfGrad();
       else if (mNumF==2)
           aPost = OneBoxLapl();
       else if ((mNumF>=3) && (mNumF<10))
           aPost = OneBoxScaleGradScale();
       else 
           ELISE_ASSERT(false,"Bad Num filter");
       std::string aNameIMax =  NameIMax(aPost);

       bool IsModified;
       Tiff_Im aTifInd= Tiff_Im::CreateIfNeeded(IsModified,aNameIMax,mSzGlob,GenIm::u_int1,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
       ELISE_COPY
       (
           mBoxOut.Flux(),
           trans(mIndMax.in(),-mBoxIn.P0()),
           aTifInd.out()
       );

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
