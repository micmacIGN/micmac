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
#include <algorithm>

/*
Parametre de Tapas :
  
   - calibration In : en base de donnees ou deja existantes.


*/

// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876

#define  NbModele 10

class cAppli_HomCorOri
{
    public :
         cAppli_HomCorOri(int argc,char ** argv);
         void DoMatch();
         void Load();
         void ComputeOrtho();
    private :

         void LoadImage(const std::string & aName,Im2D_REAL4,bool Resize);


         Pt2dr Im1ToIm2(const Pt2di aP);
         Pt2dr RRIm1ToIm2(const Pt2di aP);
         

         std::string mDir;
         cInterfChantierNameManipulateur * mICNM;
         std::string mNameIm1;
         std::string mNameIm2;
         std::string mOri;
         std::string mDirMatch;

         int mNumFinal;
         int mZoomFinal;
         int mNumCorrel;
         double mDownSize;
         double mDynVisu;

         cElNuage3DMaille * mNuage;
         Pt2dr              mVPxT;;
         CamStenope       * mCS2;
         Im2D_REAL4        mImCorrel;
         Im2D_REAL4        mImProf;
         Im2D_REAL4        mImPxT;
         TIm2D<float,double> mTPx;
         Video_Win *       mW;

         Im2D_REAL4        mIm1;
         Im2D_REAL4        mIm2Ori;
         Im2D_REAL4        mIm2;
         Im2D_Bits<1>      mMasq;
         bool              mMatch;
};

Fonc_Num Correl(Fonc_Num aMasq,Fonc_Num aF1,Fonc_Num aF2,int aSzW)
{
     Symb_FNum aSFM(aMasq);
     Symb_FNum aSF1(aF1);
     Symb_FNum aSF2(aF2);

     Symb_FNum SAllF(Virgule(aSFM,aSF1*aSFM,aSF2*aSFM,Square(aSF1)*aSFM,Square(aSF2)*aSFM,aSF1*aSF2*aSFM));
     Symb_FNum SMoy = rect_som(SAllF,aSzW);


     Symb_FNum  aSBrut =  SMoy.kth_proj(0);
     Symb_FNum  aS =  Max(aSBrut,1e-5);
     Symb_FNum  aS1 = SMoy.kth_proj(1) / aS;
     Symb_FNum  aS2 = SMoy.kth_proj(2) / aS;
     Symb_FNum  aS11 = SMoy.kth_proj(3) / aS - ElSquare(aS1);
     Symb_FNum  aS22 = SMoy.kth_proj(4) / aS - ElSquare(aS2);
     Symb_FNum  aS12 = SMoy.kth_proj(5) / aS - aS1 * aS2;

     Symb_FNum  aCor =  (aS12 / sqrt(Max(1e-5,aS11*aS22))) * (aSBrut>0);

     return Max(0,Min(255,128*(1+aCor)));

}


Pt2dr cAppli_HomCorOri::Im1ToIm2(const Pt2di aP)
{
    Pt3dr aP3 = mNuage->PtOfIndex(aP);
    Pt2dr aRes =  mCS2->R3toF2(aP3);

    return aRes +  mVPxT * mTPx.get(aP);
}

Pt2dr cAppli_HomCorOri::RRIm1ToIm2(const Pt2di aP)
{
    return Im1ToIm2(aP) / (mZoomFinal * mDownSize);
}

void cAppli_HomCorOri::LoadImage(const std::string & aName,Im2D_REAL4 anIm,bool Resize)
{
    std::string aNameFile = mDir+StdNameImDeZoom(aName,mZoomFinal);
    Tiff_Im aTif = Tiff_Im::StdConv(aNameFile);

    if (Resize)
    {
        Pt2di aSz = round_ni(Pt2dr(aTif.sz()) /mDownSize);
        anIm.Resize(aSz);
    }

    ELISE_COPY
    (
         anIm.all_pts(),
         StdFoncChScale(aTif.in(0), Pt2dr(0,0), Pt2dr(mDownSize,mDownSize)),
         anIm.out()
    );
   //  ELISE_COPY (anIm.all_pts(),Min(255,anIm.in()/mDynVisu),mW->ogray());
}


void cAppli_HomCorOri::ComputeOrtho() 
{
    ELISE_COPY(mIm2.all_pts(),-1,mIm2.out());

    Pt2di aSz = mIm2.sz();

    // TIm2D<float,double> aTIm1(mIm1);
    TIm2D<float,double> aTIm2(mIm2);
    TIm2D<float,double> aTIm2Ori(mIm2Ori);
    TIm2D<float,double> aTImC(mImCorrel);
    TIm2DBits<1>        aTMasq(mMasq);

    


    Pt2di aP;
    for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
    {
       for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
       {
            if ((aTImC.get(aP) > 196) && mNuage->IndexHasContenu(aP))
            {
                aTMasq.oset(aP,1);
                Pt2dr aPIm2 = RRIm1ToIm2(aP);

                aTIm2.oset(aP,aTIm2Ori.getr(aPIm2,-10.0));
            }
            else
            {
                aTMasq.oset(aP,0);
            }
       }
    }

    Im2D_REAL4 aGX1(aSz.x,aSz.y);
    Im2D_REAL4 aGY1(aSz.x,aSz.y);
    Im2D_REAL4 aGX2(aSz.x,aSz.y);
    Im2D_REAL4 aGY2(aSz.x,aSz.y);
    ELISE_COPY(aGX1.all_pts(),deriche(mIm1.in_proj(),1.0),Virgule(aGX1.out(),aGY1.out()));
    ELISE_COPY(aGX2.all_pts(),deriche(mIm2.in_proj(),1.0),Virgule(aGX2.out(),aGY2.out()));


    ELISE_COPY 
    (
         mIm2.all_pts(),
         Min(255,Virgule(mIm2.in(),mIm1.in(),mIm1.in())/mDynVisu),
         mW->orgb()
    );
    std::cout << "LLLLL \n";
    getchar();

    ELISE_COPY 
    (
         mIm2.all_pts(),
         Min(255,(mIm2.in()/mDynVisu)),
         mW->ogray()
    );


    

    for (int aSzW=1 ; aSzW<10 ; aSzW ++)
    {
         ELISE_COPY
         (
              mIm2.all_pts(),
              Min(Correl(mMasq.in(0),mIm1.in(0),mIm2.in(0),aSzW),mImCorrel.in()),
              mW->ogray() | mImCorrel.out()
         );

         ELISE_COPY
         (
              mIm2.all_pts(),
              Min(Correl(mMasq.in(0),aGX1.in(0),aGX2.in(0),aSzW),mImCorrel.in()),
              mW->ogray() | mImCorrel.out()
         );

         ELISE_COPY
         (
              mIm2.all_pts(),
              Min(Correl(mMasq.in(0),aGY1.in(0),aGY2.in(0),aSzW),mImCorrel.in()),
              mW->ogray() | mImCorrel.out()
         );

         ELISE_COPY ( mImCorrel.all_pts(), mImCorrel.in() > 196, mMasq.out());
         getchar();
    }


}



void cAppli_HomCorOri::Load()
{
    std::string aName = mDir+mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+mOri,mNameIm2,true);
    mCS2 = CamOrientGenFromFile(aName,mICNM);

    mNuage = cElNuage3DMaille::FromFileIm(mDirMatch+"Scaled.xml");
    mVPxT = mNuage->Params().PM3D_ParamSpecifs().DirTrans().Val();

    std::cout << "PXXXXtt " << mVPxT << "\n";
    mImCorrel.Resize(mNuage->SzUnique());
    mImProf.Resize(mNuage->SzUnique());
    mImPxT.Resize(mNuage->SzUnique());
    mTPx = TIm2D<float,double>(mImPxT);
    mIm1.Resize(mNuage->SzUnique());
    mIm2.Resize(mNuage->SzUnique());
    mMasq = Im2D_Bits<1>(mNuage->SzUnique().x,mNuage->SzUnique().y);


    mW = Video_Win::PtrWStd(mNuage->SzUnique());

    ELISE_COPY
    (
         mImCorrel.all_pts(),
         StdFoncChScale
         (
             Tiff_Im::StdConv(mDirMatch+ "Correl_Geom-Im_Num_"+ToString(mNumCorrel)+ ".tif").in(0),
             Pt2dr(0,0),
             Pt2dr(mDownSize,mDownSize)
         ),
         mImCorrel.out()
    );
    ELISE_COPY
    (
         mImPxT.all_pts(),
         StdFoncChScale
         (
             Tiff_Im::StdConv(mDirMatch+ "Px2_Num"+ ToString(mNumFinal) + "_DeZoom"+ ToString(mZoomFinal)+ "_Geom-Im.tif").in(0),
             Pt2dr(0,0),
             Pt2dr(mDownSize,mDownSize)
         ),
         mImPxT.out()
    );

    ELISE_COPY(mImProf.all_pts(),mNuage->ImProf()->in(),mImProf.out());

    // ELISE_COPY ( mImProf.all_pts(), mImProf.in() * 20, mW->ocirc());
    // ELISE_COPY ( mImCorrel.all_pts(), mImCorrel.in() > 196, mW->odisc());

    LoadImage(mNameIm1,mIm1,false);
    LoadImage(mNameIm2,mIm2Ori,true);

}

void cAppli_HomCorOri::DoMatch()
{
    if (! mMatch) return;

    std::string aCom =     MMBinFile(MM3DStr)
                         + std::string(" MICMAC ")
                         + XML_MM_File("MM-CalibEpip.xml ")
                         + " +Im1=" + mNameIm1
                         + " +Im2=" + mNameIm2
                         + " +AeroIn=-" + mOri
                         + " WorkDir=" + mDir
                       ;

   System(aCom);

   aCom =   MMBinFile(MM3DStr)
          + std::string(" ScaleNuage ")
          + mDirMatch + "NuageImProf_Geom-Im_Etape_"+ ToString(mNumFinal) + ".xml " 
          +  " Scaled "
          + ToString(mDownSize);

   System(aCom);
}


cAppli_HomCorOri::cAppli_HomCorOri (int argc,char ** argv) :
    mNumFinal  (11),
    mZoomFinal (2),
    mNumCorrel (mNumFinal-1),
    mDownSize  (3.0),
    mDynVisu   (255),
    mNuage     (0),
    mImCorrel  (1,1),
    mImProf    (1,1),
    mImPxT     (1,1),
    mTPx       (mImPxT),
    mIm1       (1,1),
    mIm2Ori    (1,1),
    mIm2       (1,1),
    mMasq      (1,1),
    mMatch     (false)
{
    MMD_InitArgcArgv(argc,argv);
    

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(mNameIm1,"First Image")
                    << EAMC(mNameIm2,"Second Images")
                    << EAMC(mOri,"Orientation"),
	LArgMain()  << EAM(mMatch,"Match",true,"Do matching, def = true")	
    );

    mDir = DirOfFile(mNameIm1);
    mNameIm1 = NameWithoutDir(mNameIm1);
    mNameIm2 = NameWithoutDir(mNameIm2);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mDirMatch = mDir + "CalibEPi" + mNameIm1 + "-" + mNameIm2 + "/";

}


int MMHomCorOri_main(int argc,char ** argv)
{
   cAppli_HomCorOri anAppli (argc,argv);

   anAppli.DoMatch();
   anAppli.Load();
   anAppli.ComputeOrtho();


   BanniereMM3D();

   return 1;
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
