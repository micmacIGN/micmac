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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>


#define BORDER 400
#define SZMARK 30

class cSimulImage
{
    public :
       cSimulImage (int argc,char ** argv);
       void NoOp() {}
    private  :

        Pt2dr Freem2Fid(const Pt2dr & aFreem,bool WithM)
        {
            Pt2dr aAmpl = mMil ;
            if (WithM) aAmpl = aAmpl + Pt2dr(mPSzM);
            return  Pt2dr(mPBord) + mMil + aAmpl.mcbyc(aFreem);
        }
        bool              mSymX;
        bool              mSymY;
        std::string       mFullName;
        std::string       mDir;
        std::string       mName;
        std::string       mNameOut;
        Pt2di             mSz;
        Pt2dr             mMil;
        int               mBorder;
        Pt2di             mPBord;
        int               mSzMark;
        Pt2di             mPSzM;
        Pt2di             mSzWBord;
        std::vector<Im2D_U_INT1>       mIm;



        double            mScaleOri;

        double            mScale;
        double            mTeta;
        double            mResolMicr;
        bool              mFidCentered;
        Pt2dr             mTr;

        ElSimilitude      mSimOld2New;
        ElSimilitude      mSimNew2Old;

        void CalcSim();
};

void cSimulImage::CalcSim()
{
    mSimOld2New = ElSimilitude::SimOfCentre(mMil+Pt2dr(mPBord),Pt2dr::FromPolar(mScale,mTeta));
    mSimOld2New = ElSimilitude(mTr,Pt2dr(1,0)) * mSimOld2New;

    mSimNew2Old = mSimOld2New.inv();
}

cSimulImage::cSimulImage(int argc,char ** argv) :
    mBorder  (BORDER),
    mPBord   (mBorder,mBorder),
    mSzMark  (SZMARK),
    mPSzM    (mSzMark,mSzMark),
    mScaleOri (2.0),
    //  mResolMicr  (4.8),
    mResolMicr  (6.410256),
    mFidCentered (false)
{
    NRrandom3InitOfTime();
    std::cout << "TEST RANDOM " << NRrandom3() << "\n";
    MMD_InitArgcArgv(argc,argv);

    mSymX = NRrandom3()<0.5;
    mSymY = NRrandom3()<0.5;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(mFullName,"Name of Image"),
	LArgMain()  << EAM(mNameOut,"Out",true)
	            << EAM(mSymX,"SymX",true)
	            << EAM(mSymY,"SymY",true)
    );	

    SplitDirAndFile(mDir,mName,mFullName);

    if (!EAMIsInit(&mNameOut))
    {
        mNameOut = StdPrefix(mName) + "_Out.tif";
    }

    //   ====== CHARGEMENT DE L'IMAGE =================


    Tiff_Im aTifIn = Tiff_Im::StdConvGen(mFullName.c_str(),3,false);

    mSz  =  round_ni(Pt2dr(aTifIn.sz()) / mScaleOri);



    mSzWBord = mSz +  mPBord*2;

    for (int aK=0 ; aK<3 ; aK++)
    {
       mIm.push_back(Im2D_U_INT1(mSzWBord.x,mSzWBord.y,0));
    }

    Fonc_Num aFFile = StdFoncChScale(aTifIn.in(0),Pt2dr(0,0),Pt2dr(mScaleOri,mScaleOri),Pt2dr(1,1));
 
    ELISE_COPY
    (
         rectangle(mPBord,mPBord+mSz),
         trans(aFFile,-mPBord),
	 Virgule(mIm[0].out(),mIm[1].out(),mIm[2].out())
    );

    mMil = Pt2dr(mSz)/2.0;


    //   ====== CREATION DES MARQUES =================

    {
       cElBitmFont & aFt = cElBitmFont::BasicFont_10x8();
       for (int aK=0 ; aK < 8 ; aK++)
       {
            // Pt2di aPFReem = TAB_8_NEIGH[aK];
            // Pt2dr aPFid = Pt2dr(mPBord) + mMil + mMil.mcbyc(Pt2dr(TAB_8_NEIGH[aK]));
            Pt2dr aPFid = Freem2Fid(Pt2dr(TAB_8_NEIGH[aK]),false) ;
            std::cout << "PF = " << aPFid << "\n";

            Fonc_Num aFRay = sqrt(Square(FX-aPFid.x)+Square(FY-aPFid.y));
            ELISE_COPY
            (
               rectangle(Pt2di(aPFid)-mPSzM,Pt2di(aPFid)+mPSzM),
               Max(0,Min(255,128.0+ 64*sin(aFRay/2) + 64*(frandr()-0.5))),
               mIm[0].out()|mIm[1].out()|mIm[2].out()
            );
            Im2D_Bits<1> aImC = aFt.ImChar('0'+aK);

            int aZoom=2;
            Pt2di aSzF = aImC.sz();
            Pt2di aP0 = Pt2di(aPFid)-mPSzM+Pt2di(2,2);
            ELISE_COPY
            (
               rectangle(aP0,aP0+aSzF*aZoom),
               255 * aImC.in(0)[Virgule((FX-aP0.x)/aZoom,(FY-aP0.y)/aZoom)],
               mIm[0].out()|mIm[1].out()|mIm[2].out()
            );
       }
    }

    //   ====== INVERSION OU SYMETRIE =================

    {
        std::vector<Im2D_U_INT1> aDup;
        for (int aK=0 ; aK<3 ; aK++)
        {
           aDup.push_back(Im2D_U_INT1(mSzWBord.x,mSzWBord.y,0));
        }
        Fonc_Num  aChX = FX;
        Fonc_Num  aChY = FY;

        if (mSymX)
        {
              aChX  = mSzWBord.x-1-FX;
        }
        if (mSymY)
        {
              aChY  = mSzWBord.y-1-FY;
        }

        ELISE_COPY
        (
             mIm[0].all_pts(),
             Virgule(mIm[0].in(),mIm[1].in(),mIm[2].in())[Virgule(aChX,aChY)],
             Virgule(aDup[0].out(),aDup[1].out(),aDup[2].out())
        );

        mIm = aDup;
    }

    //   ====== Changement de geometrie "simulation" d'imprecision du scan =================

    mTeta = 0.15 * (NRrandom3()-0.5);
    mScale = 1.0 + 0.15 * (NRrandom3()-0.5);
    mTr    = Pt2dr(0,0);
    CalcSim();
    
    Pt2dr aP0(1e5,1e5);
    Pt2dr aP1(-1e5,-1e5);
    for (int aK=0 ; aK < 8 ; aK++)
    {
         Pt2dr aPFid = Freem2Fid(Pt2dr(TAB_8_NEIGH[aK]),true) ;
         aPFid = mSimOld2New(aPFid);
         aP0.SetInf(aPFid);
         aP1.SetSup(aPFid);
    }

    mTr  = - aP0;
    CalcSim();



    for (int aK=0 ; aK<3 ; aK++)
    {
       cCubicInterpKernel  aKer(-0.5);
       Pt2di aSzOut = round_ni(aP1-aP0);
       Im2D_U_INT1 aImOut(aSzOut.x,aSzOut.y);
       TIm2D<U_INT1,INT> aTOut(aImOut);
       TIm2D<U_INT1,INT> aTIn(mIm[aK]);

       Pt2di aP;
       for (aP.x=0 ; aP.x<aSzOut.x; aP.x++)
       {
           for (aP.y=0 ; aP.y<aSzOut.y; aP.y++)
           {
               // aTOut.oset(aP,aTIn.getprojR(mSimNew2Old(Pt2dr(aP))));
               aTOut.oset(aP,ElMax(0,ElMin(255,round_ni(aTIn.getr(aKer,mSimNew2Old(Pt2dr(aP)),0)))));
           }
       }
       mIm[aK] = aImOut;
    }

                 //  OK l'image est prete 

    std::string aFullNameOut = mDir+mNameOut;
    Tiff_Im aTRes
            (
                 aFullNameOut.c_str(),
                 mIm[0].sz(),
                 GenIm::u_int1,
                 Tiff_Im::No_Compr,
                 Tiff_Im::RGB
            );
    ELISE_COPY
    (
        aTRes.all_pts(),
         Virgule(mIm[0].in(),mIm[1].in(),mIm[2].in()),
        aTRes.out()
    );
    // Tiff_Im::CreateFromIm(mIm,mDir+mNameOut);

    std::string aNameJpg = StdPrefix(mNameOut) + ".jpg";

    std::string aCom = "convert " + mDir+mNameOut + " -quality 95 " + mDir+aNameJpg;
    VoidSystem(aCom.c_str());

    //   ====== INVERSION OU SYMETRIE =================

    cMesureAppuiFlottant1Im aNewPointe;
    aNewPointe.NameIm() = aNameJpg;

    cMesureAppuiFlottant1Im aPointeMM;
    aPointeMM.NameIm() = "Glob";

    for (int aK=0 ; aK < 8 ; aK++)
    {
         Pt2dr aPFreemOri = Pt2dr(TAB_8_NEIGH[aK]);
         Pt2dr aPFreem = aPFreemOri;
         if (mSymX )  aPFreem.x *=-1;
         if (mSymY )  aPFreem.y *=-1;

         Pt2dr aPFidOld = Freem2Fid(aPFreem,false) ;
         Pt2dr aPFidNew = mSimOld2New(aPFidOld);


         aPFidOld = aPFidOld - Pt2dr(mPBord);
         if (mFidCentered)
            aPFidOld = aPFidOld -mMil;
         aPFidOld = Pt2dr(round_ni(aPFidOld*mScaleOri*(mResolMicr/1000.0)));
         

         std::string aNamePt = "P"+ToString(aK);

         cOneMesureAF1I aMesNew;
         aMesNew.PtIm() = aPFidNew;
         aMesNew.NamePt() = aNamePt;
         aNewPointe.OneMesureAF1I().push_back(aMesNew);

         cOneMesureAF1I aMesMM;
         aMesMM.PtIm() = aPFidOld;
         aMesMM.NamePt() = aNamePt;
         aPointeMM.OneMesureAF1I().push_back(aMesMM);

         std::cout << "K=" << aK << " " << aPFidNew << " " <<  aPFidOld << "\n";
    }

    std::string aDirOI = "Ori-InterneScan/";
    ELISE_fp::MkDirSvp(mDir+aDirOI);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    std::pair<std::string,std::string> aPair = aICNM->Assoc2To1("Key-Assoc-STD-Orientation-Interne",aNameJpg,true);


    MakeFileXML(aNewPointe,aPair.second);
    MakeFileXML(aPointeMM,aPair.first);
    
    
}


int main(int argc,char ** argv)
{
    cSimulImage aSim(argc,argv);
    aSim.NoOp();
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
