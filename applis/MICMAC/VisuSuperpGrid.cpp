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
#include "MICMAC.h"
#include "im_tpl/image.h"

using namespace NS_ParamMICMAC;




class cAppliVisu
{
    public :
        cAppliVisu
        (
             cInterfModuleImageLoader  & aIMIL1,
             cInterfModuleImageLoader  & aIMIL2,
             const std::string  &        aNameDir,
             const std::string  &        aNameXml,
             double                      aZContr
        );

        void Load(Pt2di aP0Im1);

        void AnswerClik();
        void ShowProfil(Pt2dr aP0);
        void ShowGraph(const std::vector<double> &,int);

        void InitImContr(double aDyn);

    private :
        
        cInterfModuleImageLoader & mIMIL1;
        Pt2di         mSzFile1;
        cInterfModuleImageLoader & mIMIL2;
        cDbleGrid     mDbleGr;
        double        mZoom;
        Pt2di         mSzWIm;
        Video_Win     mW1;
        Video_Win     mW2;
        Video_Win     mW12;

        double        mZContr;
        Pt2di         mSzContr;
        Video_Win     mWContr;
        Im2D_REAL4    mImC;

        double        mZoomGr;
        int           mSzGr;
        Video_Win     mWGraph;

        Im2D_REAL4          mIm1;
        TIm2D<REAL4,REAL8>  mTIm1;
        Im2D_REAL4          mIm2;
        TIm2D<REAL4,REAL8>  mTIm2;
        Im2D_REAL4          mIm2G1;
        TIm2D<REAL4,REAL8>  mTIm2G1;
        Pt2di         mP0Im1;
        Pt2di         mP0Im2;
        Box2dr        mBoxIm1;
        Box2dr        mBoxIm2;

        Pt2dr  G1toG2(const Pt2dr & aP2)  const
        {
             return mDbleGr.Direct(aP2+mP0Im1)-mP0Im2;
        }
};

static cDbleGrid::cXMLMode aXMM;
cAppliVisu::cAppliVisu
(
             cInterfModuleImageLoader  & aIMIL1,
             cInterfModuleImageLoader  & aIMIL2,
             const std::string  & aNameDir,
             const std::string  & aNameXml,
             double               aZContr
) :
  mIMIL1    (aIMIL1),
  mSzFile1  (Std2Elise(mIMIL1.Sz(1))),
  mIMIL2    (aIMIL2),
  mDbleGr   (aXMM,aNameDir,aNameXml),
  mZoom     (1.0),
  mSzWIm    (300,300),
  mW1       (Video_Win::WStd(mSzWIm,mZoom)),
  mW2       (mW1,Video_Win::eDroiteH,mSzWIm),
  mW12      (mW2,Video_Win::eDroiteH,mSzWIm),
  mZContr   (aZContr),
  mSzContr  (mSzFile1/mZContr),
  mWContr   (mW1,Video_Win::eBasG,mSzContr),
  mImC      (mSzContr.x,mSzContr.y),
  mZoomGr   (10.0),
  mSzGr     (600),
  mWGraph   (mWContr,Video_Win::eDroiteH,Pt2di(mSzGr,300)),
  mIm1      (mSzWIm.x,mSzWIm.y),
  mTIm1     (mIm1),
  mIm2      (2,2),
  mTIm2     (mIm2),
  mIm2G1    (mSzWIm.x,mSzWIm.y),
  mTIm2G1   (mIm2G1)
{
    ELISE_COPY(mW1.all_pts(),P8COL::red,mW1.odisc());
    ELISE_COPY(mW2.all_pts(),P8COL::green,mW2.odisc());
    ELISE_COPY(mW12.all_pts(),P8COL::yellow,mW12.odisc());
   

   
    ELISE_COPY(mWContr.all_pts(),P8COL::magenta,mWContr.odisc());

    ELISE_COPY(mWGraph.all_pts(),(1+sin(FX/30.0))*128,mWGraph.ogray());
}

Fonc_Num AUC(Fonc_Num aF) 
{
   return Max(0,Min(255,aF));
}

double TheGamma=2.0;
Fonc_Num Gamma(Fonc_Num aF,double aGamma)
{
    return 255* pow(aF/255.0,1/aGamma);
}


void cAppliVisu::InitImContr(double aDyn)
{
  LoadAllImCorrel(mImC,&mIMIL1,round_ni(mZContr),Pt2di(0,0));
  ELISE_COPY
  (
      mImC.all_pts(),
      Gamma(AUC(mImC.in()/aDyn),TheGamma),
      mWContr.ogray() | mImC.oclip()
  );
}



void LoadOneIm
     (
        Im2D_REAL4 anIm,
        Video_Win aW,
        cInterfModuleImageLoader & aIMIL,
        Pt2di aP0
     )
{
  LoadAllImCorrel(anIm,&aIMIL,1,aP0);
   double aMinI1,aMaxI1;

   ELISE_COPY
   (
       anIm.all_pts(),
       anIm.in(),
          VMin(aMinI1)
       |  VMax(aMaxI1)
       |  (aW.odisc() << P8COL::cyan)
   );
   ELISE_COPY
   (
       anIm.all_pts(),
       Gamma(10.0 + (anIm.in()-aMinI1) * (234.0 /Max(1,(aMaxI1- aMinI1))),TheGamma),
       anIm.out() | aW.ogray()
   );
}


void CorrecFromInertie
     (
        const RMat_Inertie & aMat,
        Im2D_REAL4 anIm,
        Video_Win aW
     )
{
   ELISE_COPY
   (
      anIm.all_pts(),
      aMat.s1() +(anIm.in()-aMat.s2())*sqrt(aMat.s11()/ElMax(1.0,aMat.s22())),
         anIm.out()
      | (aW.ogray()<<Max(0,Min(255,anIm.in())))
   );
}
void cAppliVisu::Load(Pt2di aP0Im1)
{
   ELISE_COPY(mImC.all_pts(),mImC.in(),mWContr.ogray());
   // Chargement de I1 
   mP0Im1 = aP0Im1;
   LoadOneIm(mIm1,mW1,mIMIL1,aP0Im1);
   mBoxIm1 = Box2dr (mP0Im1,mP0Im1+mIm1.sz());

   mWContr.draw_rect
   (
        mBoxIm1._p0/mZContr,
        mBoxIm1._p1/mZContr,
        mWContr.pdisc()(P8COL::yellow)
   );

   // Chargement de Im2
   mBoxIm2 = mDbleGr.ImageOfBox(mBoxIm1);
   mP0Im2 = round_down(mBoxIm2._p0);

   mIm2.Resize(round_up(mBoxIm2._p1-mP0Im2));
   mTIm2 = mIm2;
   LoadOneIm(mIm2,mW2,mIMIL2,mP0Im2);
  

   // Chargement de I2 en Geom1
   RMat_Inertie aMat;
   Pt2di aP1;
   for (aP1.y=0 ; aP1.y<mSzWIm.y; aP1.y++)
   {
       for (aP1.x=0 ; aP1.x<mSzWIm.x; aP1.x++)
       {
           double aV2 = mTIm2.getprojR(G1toG2(aP1));
           mTIm2G1.oset(aP1,aV2);
           aMat.add_pt_en_place(mTIm1.get(aP1),aV2);
       }
       ELISE_COPY
       (
          rectangle(Pt2di(0,aP1.y),Pt2di(mSzWIm.x,aP1.y+1)),
          mIm2G1.in(),
          mW2.ogray()
       );
  }
   aMat=aMat.normalize();
   CorrecFromInertie(aMat,mIm2,mW2);
   CorrecFromInertie(aMat,mIm2G1,mW2);

   // Visu  en multi canal

   ELISE_COPY
   (
       mIm1.all_pts(),
       AUC(Virgule(mIm1.in(),mIm2G1.in(),mIm2G1.in())),
       mW12.orgb()
   );
}

void cAppliVisu::ShowProfil(Pt2dr aP0)
{
   Pt2dr aP1 =  mW1.disp().clik()._pt;
   mW1.draw_seg(aP0,aP1,mW1.pdisc()(P8COL::red));
   double aD= euclid(aP0,aP1);
   int aNbPts = round_ni(aD*mZContr);

   std::vector<double> mV1;
   std::vector<double> mV2;
   cCubicInterpKernel aKer(-0.5);
   for (int aK=0 ; aK<=aNbPts ; aK++)
   {
       Pt2dr aQ1 = barry(1-aK/double(aNbPts),aP0,aP1);
       Pt2dr aQ2 = G1toG2(aQ1);
       mV1.push_back(mTIm1.getr(aKer,aQ1,0.0));
       mV2.push_back(mTIm2.getr(aKer,aQ2,0.0));
   }
   mWGraph.clear();
   for (int aK=0 ; aK<=60 ; aK++)
   {
      mWGraph.draw_seg
      (
         Pt2dr(aK*mZContr,0),
         Pt2dr(aK*mZContr,255),
         mWGraph.pdisc()(P8COL::yellow)
      );
   }
   ShowGraph(mV1,P8COL::red);
   ShowGraph(mV2,P8COL::green);
}

void cAppliVisu::ShowGraph
     (
        const std::vector<double> & aV,
        int aCoul
     )
{
    for (int aK=1; aK<int(aV.size()) ; aK++)
    {
       mWGraph.draw_seg
       (
           Pt2dr(aK-1,280-aV[aK-1]),
           Pt2dr(aK,280-aV[aK]),
           mWGraph.pdisc()(aCoul)
       );
    }
}

void cAppliVisu::AnswerClik()
{
    Clik aCl = mW1.disp().clik();

    if (aCl._w==mWContr)
    {
         Load(aCl._pt*mZContr-mIm1.sz()/2);
    }

    if ((aCl._w==mW1) ||(aCl._w==mW2)||(aCl._w==mW12))
    {
         ShowProfil(aCl._pt);
    }
}

int main(int argc,char ** argv)
{
   cAppliMICMAC & aAPM = *(cAppliMICMAC::Alloc(argc,argv,eAllocAM_VisuSup));

   std::string aNameDir = aAPM.WorkDir();
   int aDZ = aAPM.VSG_DeZoomContr().Val();

   cout << aAPM.LastMAnExp().NameXMLFinal() << "\n";
   std::cout << aAPM.WorkDir() << "-"
             << aAPM.PDV1()->Name() << "\n";

    cAppliVisu anAppli
    (
         *(aAPM.PDV1()->IMIL()),
         *(aAPM.PDV2()->IMIL()),
         aAPM.FullDirGeom(),
         aAPM.LastMAnExp().NameXMLFinal(),
         aDZ
    );

    double aDyn = aAPM.VSG_DynImRed().Val();
    anAppli.InitImContr(aDyn);
    while (1)
    {
       anAppli.AnswerClik();
    }
/*
    while (1)
    {
       int x,y;
       cin >> x >> y;
       anAppli.Load(Pt2di(x,y));
    }
*/
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
