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

static const int TheSeuilC = 196;

class cAppli_HomCorOri
{
    public :
         cAppli_HomCorOri(int argc,char ** argv);
         void DoMatch();
         void Load();
         void ComputeOrtho();
         void ComputePts();
         void SauvPack();
         bool AllDone() const;
    private :
         Pt3dr ComputeOneBox(const Box2di &);
         double ScorePts(const Pt2di &);

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
         Pt2dr              mVPxT;
         CamStenope       * mCS2;
         Im2D_REAL4        mImCorrel;
         Im2D_REAL4        mImProf;
         TIm2D<float,double> mTProf;
         Im2D_REAL4        mImPxT;
         TIm2D<float,double> mTPx;
         Video_Win *       mW;

         Im2D_REAL4        mIm1;
         Im2D_REAL4        mIm2Ori;
         Im2D_REAL4        mIm2;
         Im2D_U_INT1       mMasq;
         TIm2D<U_INT1,INT> mTMasq;
         bool              mMatch;
         std::string       mNameHom12;
         std::string       mNameHom21;
         ElPackHomologue   mPack;
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

    return aRes +  mVPxT * (mTPx.get(aP) * mDownSize);
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

inline double PdsErr(double anEr) {return 1/(1+ElSquare(anEr));}

double cAppli_HomCorOri::ScorePts(const Pt2di & aP)
{
    int aLarge = 10;
    double aRes = 0;
    Pt2di aSz = mTPx.sz();
    Pt2di aPLarg(aLarge,aLarge);

    Pt2di aP0 = Sup(Pt2di(0,0),aP-aPLarg);
    Pt2di aP1 = Inf(aSz,aP+aPLarg);

    double aProf = mTProf.get(aP);
    double aPax =  mTPx.get(aP);
    Pt2di aQ;
    for (aQ.x=aP0.x ; aQ.x<aP1.x; aQ.x++)
    {
        for (aQ.y=aP0.y ; aQ.y<aP1.y; aQ.y++)
        {
             double aEPr = ElAbs(aProf-mTProf.get(aQ));
             double aEPax = ElAbs(aPax-mTPx.get(aQ)) * 10;

             aRes += PdsErr(aEPr) * PdsErr(aEPax);
        }
    }


    return aRes / ( 2*aLarge);
}

Pt3dr cAppli_HomCorOri::ComputeOneBox(const Box2di & aBox)
{
    // mW->draw_rect(I2R(aBox),mW->pdisc()(P8COL::green));
    Pt2di aP;
    int aNbTirage = 10;

    int aNbOk=0;
    int aNb=0;
    for (aP.x=aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
    {
        for (aP.y=aBox._p0.y ; aP.y<aBox._p1.y ; aP.y++)
        {
            aNb++;
            if (mTMasq.get(aP))
            {
                aNbOk++;
            }
        }
    }

    if (aNbOk <(aNb/3))
    {
         return Pt3dr(0,0,-1);
    }

    double aProba = double(aNbTirage) / aNbOk;

    double aScMax = -1;
    Pt2di aPMax(0,0);
    int aNbT=0;

    for (aP.x=aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
    {
        for (aP.y=aBox._p0.y ; aP.y<aBox._p1.y ; aP.y++)
        {
            if (mTMasq.get(aP)  && (NRrandom3()<aProba))
            {
                double aSc = ScorePts(aP);
                aNbT++;
                if (aSc > aScMax)
                {
                    aScMax = aSc;
                    aPMax = aP;
                }
            }
        }
    }
    return Pt3dr(aPMax.x,aPMax.y,aScMax);
}

class cCmpPt3drOnZ
{
    public :
      bool operator () (const Pt3dr & aP1,const Pt3dr & aP2) {return aP1.z>aP2.z;}
};

void cAppli_HomCorOri::ComputePts()
{
    Pt2di aSz = mIm2.sz();
    int aNbTarget =10000;
    double aLarg = sqrt((aSz.x*aSz.y) /double(aNbTarget));
    int aNbX = round_up(aSz.x/aLarg);
    int aNbY = round_up(aSz.y/aLarg);

    double aSom=0;

    std::vector<Pt3dr> aVP;

    for (int aKx=0 ; aKx<aNbX; aKx++)
    {
        for (int aKy=0 ; aKy<aNbY; aKy++)
        {
             int aX0 = (aKx * aSz.x) /aNbX;
             int aX1 = ((aKx+1) * aSz.x) /aNbX;
             int aY0 = (aKy * aSz.y) /aNbY;
             int aY1 = ((aKy+1) * aSz.y) /aNbY;
             Pt3dr aRes  = ComputeOneBox(Box2di(Pt2di(aX0,aY0),Pt2di(aX1,aY1)));
             if (aRes.z >0)
             {
                aSom += aRes.z;
                aVP.push_back(aRes);
             }
        }
    }
    cCmpPt3drOnZ aCmp;
    std::sort(aVP.begin(),aVP.end(),aCmp);

    for (int aK=0 ; aK<int(aVP.size() * 0.9) ; aK++)
    {
         // std::cout << " " << aVP[aK] << "\n";
         Pt2di aP = round_ni(Pt2dr(aVP[aK].x,aVP[aK].y));
         // mW->draw_circle_abs(Pt2dr(aP),2.0,mW->pdisc()(P8COL::red));

         mPack.Cple_Add(ElCplePtsHomologues(Pt2dr(aP)*(mZoomFinal*mDownSize),Im1ToIm2(aP)));
    }
}

bool cAppli_HomCorOri::AllDone() const
{
// std::cout << mNameHom12 << " " << mNameHom21 << "\n";
     return ELISE_fp::exist_file(mNameHom12) && ELISE_fp::exist_file(mNameHom21);

}

void cAppli_HomCorOri::SauvPack()
{
   mPack.StdPutInFile(mNameHom12);
   mPack.SelfSwap();
   mPack.StdPutInFile(mNameHom21);
   mPack.SelfSwap();
}


void cAppli_HomCorOri::ComputeOrtho()
{
    ELISE_COPY(mIm2.all_pts(),-1,mIm2.out());

    Pt2di aSz = mIm2.sz();

    // TIm2D<float,double> aTIm1(mIm1);
    TIm2D<float,double> aTIm2(mIm2);
    TIm2D<float,double> aTIm2Ori(mIm2Ori);
    TIm2D<float,double> aTImC(mImCorrel);




    Pt2di aP;
    for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
    {
       for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
       {
            if ((aTImC.get(aP) > TheSeuilC) && mNuage->IndexHasContenu(aP))
            {
                mTMasq.oset(aP,1);
                Pt2dr aPIm2 = RRIm1ToIm2(aP);

                aTIm2.oset(aP,aTIm2Ori.getr(aPIm2,-10.0));
            }
            else
            {
                mTMasq.oset(aP,0);
            }
       }
    }

    Im2D_REAL4 aGX1(aSz.x,aSz.y);
    Im2D_REAL4 aGY1(aSz.x,aSz.y);
    Im2D_REAL4 aGX2(aSz.x,aSz.y);
    Im2D_REAL4 aGY2(aSz.x,aSz.y);
    ELISE_COPY(aGX1.all_pts(),deriche(mIm1.in_proj(),1.0),Virgule(aGX1.out(),aGY1.out()));
    ELISE_COPY(aGX2.all_pts(),deriche(mIm2.in_proj(),1.0),Virgule(aGX2.out(),aGY2.out()));


/*
    ELISE_COPY
    (
         mIm2.all_pts(),
         Min(255,Virgule(mIm2.in(),mIm1.in(),mIm1.in())/mDynVisu),
         mW->orgb()
    );
*/

    static const int aNbW = 5;
    int aTabSzW[aNbW] = {1,2,4,6,9};

    for (int aKW=0 ; aKW<aNbW ; aKW ++)
    {
         int aSzW = aTabSzW[aKW];
         ELISE_COPY
         (
              mIm2.all_pts(),
              Min(Correl(mMasq.in(0),mIm1.in(0),mIm2.in(0),aSzW),mImCorrel.in()),
              mImCorrel.out()
         );

         ELISE_COPY
         (
              mIm2.all_pts(),
              Min(Correl(mMasq.in(0),aGX1.in(0),aGX2.in(0),aSzW),mImCorrel.in()),
               mImCorrel.out()
         );

         ELISE_COPY
         (
              mIm2.all_pts(),
              Min(Correl(mMasq.in(0),aGY1.in(0),aGY2.in(0),aSzW),mImCorrel.in()),
              mImCorrel.out()
         );

         ELISE_COPY ( mImCorrel.all_pts(), mImCorrel.in() > TheSeuilC, mMasq.out());
    }

    ELISE_COPY(mMasq.border(1),0,mMasq.out());

    FiltrageCardCC(true,mTMasq,1,2,1000);
    ELISE_COPY(mMasq.all_pts(),mMasq.in()==1,mMasq.out());
    // ELISE_COPY(mMasq.all_pts(),mMasq.in(),mW->odisc());
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
    mTProf = TIm2D<float,double>(mImProf);
    mImPxT.Resize(mNuage->SzUnique());
    mTPx = TIm2D<float,double>(mImPxT);
    mIm1.Resize(mNuage->SzUnique());
    mIm2.Resize(mNuage->SzUnique());
    mMasq.Resize(mNuage->SzUnique());
    mTMasq = TIm2D<U_INT1,INT>(mMasq);


    // mW = Video_Win::PtrWStd(mNuage->SzUnique());

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
    ELISE_COPY(mImCorrel.all_pts(), mImCorrel.in()<TheSeuilC,mMasq.out());

    ELISE_COPY
    (
         mImPxT.all_pts(),
         StdFoncChScale
         (
             //Tiff_Im::StdConv(mDirMatch+ "Px2_Num"+ ToString(mNumFinal) + "_DeZoom"+ ToString(mZoomFinal)+ "_Geom-Im.tif").in(0),
             Tiff_Im::StdConv(LocPx2FileMatch(mDirMatch, mNumFinal, mZoomFinal)).in(0),
             Pt2dr(0,0),
             Pt2dr(mDownSize,mDownSize)
         ),
         mImPxT.out()
    );

    ELISE_COPY(mImProf.all_pts(),mNuage->ImProf()->in(),mImProf.out());

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
    mDownSize  (3.0),
    mDynVisu   (255),
    mNuage     (0),
    mImCorrel  (1,1),
    mImProf    (1,1),
    mTProf     (mImProf),
    mImPxT     (1,1),
    mTPx       (mImPxT),
    mIm1       (1,1),
    mIm2Ori    (1,1),
    mIm2       (1,1),
    mMasq      (1,1),
    mTMasq     (mMasq),
    mMatch     (true)
{
    MMD_InitArgcArgv(argc,argv);


    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(mNameIm1,"First Image", eSAM_IsExistFile)
                    << EAMC(mNameIm2,"Second Image", eSAM_IsExistFile)
                    << EAMC(mOri,"Orientation", eSAM_IsExistFile),
                LArgMain()  << EAM(mMatch,"Match",true,"Do matching (Def = true)", eSAM_IsBool)
                    << EAM(mZoomFinal,"ZoomF", true, "Zoom Final",eSAM_IsPowerOf2)
    );

    if (mZoomFinal==2)
    {
    }
    else if (mZoomFinal==4)
    {
        mNumFinal = 10;
    }
    else
    {
    }
    mNumCorrel = mNumFinal-1;

    mDir = DirOfFile(mNameIm1);
    mNameIm1 = NameWithoutDir(mNameIm1);
    mNameIm2 = NameWithoutDir(mNameIm2);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mDirMatch = mDir + "CalibEPi" + mNameIm1 + "-" + mNameIm2 + "/";

    mNameHom12 =
             mICNM->Dir()
           + mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@-DenseM@dat",mNameIm1,mNameIm2,true);

    mNameHom21 =
             mICNM->Dir()
           + mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@-DenseM@dat",mNameIm2,mNameIm1,true);
}


int MMHomCorOri_main(int argc,char ** argv)
{
   cAppli_HomCorOri anAppli (argc,argv);

   if (! anAppli.AllDone() && !MMVisualMode)
   {
       anAppli.DoMatch();
       anAppli.Load();
       anAppli.ComputeOrtho();
       anAppli.ComputePts();
       anAppli.SauvPack();

       BanniereMM3D();
   }

   return 0;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
