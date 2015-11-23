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


/*  Pre-requis
      bool  aParam.GZC_V4()  const;
      bool  aParam.GZC_PtIn(const Pt2di &) const;
      void  aParam.GZC_OnNewPt(const Pt2di &);
      bool  aParam.GZC_StopOnNewStep() const;
      bool  aParam.GZC_ArcIn(const Pt2di &,const int &) const;

*/
template <class tParam> void     GenZonec
                                 (
                                      const Pt2di & aPGerm,   
                                      tParam & aParam
                                 )
{
   Pt2di * aTabV = aParam.GZC_V4() ? TAB_4_NEIGH : TAB_8_NEIGH ;
   int aNbV = aParam.GZC_V4() ? 4 : 8;

   std::vector<Pt2di>  aVec1;
   std::vector<Pt2di>  aVec2;

   std::vector<Pt2di> * aVCur = &aVec1;
   std::vector<Pt2di> * aVNext = &aVec2;

   if (aParam.GZC_PtIn(aPGerm))
   {
      aParam.GZC_OnNewPt(aPGerm);
      aVCur->push_back(aPGerm);
   }
   int aNbStep = 1;

   while (! aVCur->empty())
   {
       if (aParam.GZC_StopOnNewStep())
          return;

       int aNbCur = (int)aVCur->size();
       for (int aKp=0 ; aKp<aNbCur ; aKp++)
       {
           Pt2di aP = (*aVCur)[aKp];
           for (int aKv=0; aKv<aNbV ; aKv++)
           {
                 Pt2di aPV = aP+aTabV[aKv];
                 if (aParam.GZC_PtIn(aPV) && aParam.GZC_ArcIn(aP,aKv))
                 {
                    aParam.GZC_OnNewPt(aPV);
                    aVNext->push_back(aPV);
                 }
           }
       }

       ElSwap(aVNext,aVCur);
       aVNext->clear();
       aNbStep++;
   }
}



class cAnalyPbTopo
{
    public :

          static const int LabInit = 255;
          static const int LabPoub = 254;
          cAnalyPbTopo
          (
             cAnaTopoXmlBascule & Res,
             cElNuage3DMaille & aNuageInit,
             cElNuage3DMaille &aNuageTarget,
             const std::string & aNameOut,
             bool                aShow,
             bool                aDebug
          );
       // Utilisation pour composante connexe
         bool  GZC_V4()  const {return false;}
         bool  GZC_PtIn(const Pt2di & aP) const
         {
            return mTImLab.get(aP)==LabInit;
         }
         void  GZC_OnNewPt(const Pt2di & aP)
         {
               mTImLab.oset(aP,mNumZoneC);
               mCurATB.BoxGlob()._p0.SetInf(aP);
               mCurATB.BoxGlob()._p1.SetSup(aP);
               mCurATB.NbGlob() ++;
               if (mTImDef0.get(aP))
               {
                   mGotMasq = true;
                   mCurATB.BoxMasq()._p0.SetInf(aP);
                   mCurATB.BoxMasq()._p1.SetSup(aP);
                   mCurATB.GermMasq() = aP;
                   mCurATB.NbMasq() ++;
               }
         }
         bool  GZC_StopOnNewStep() const {return false;}
         bool  GZC_ArcIn(const Pt2di & aP,const int & aFlag) const
         {
               return (mDataF[aP.y][aP.x] &  (1<<aFlag)) != 0;
         }
    private  :
         void MakeOneLine(std::vector<Pt2dr> & aV,int aLine);
         void TestFlag(int aK);
         int  NextNumZoneC();




         cElNuage3DMaille  & mN0;
         Pt2di               mSz0;
         cElNuage3DMaille  & mNTarget;
         Im2D_Bits<1>        mImDef0;
         TIm2DBits<1>        mTImDef0;
         Im2D_U_INT1         mImFlag;
         Im2D_U_INT1         mImLab;
         TIm2D<U_INT1,INT>   mTImLab;
         U_INT1 **           mDataF;
         std::vector<std::vector<Pt2dr>  > mVPts;
         Pt2di               mPCur;
         double              mDistSeuil;
         int                 mNumZoneC;
         int                 mNbMaxZone; // Pour l'insant fixe en dur a 2
         cOneZonzATB         mCurATB;
         cAnaTopoBascule     mATB;
         bool                mGotMasq;
         bool                mShow;
         bool                mDebug;

};

void cAnalyPbTopo::MakeOneLine(std::vector<Pt2dr> & aVec,int anY)
{
    aVec.clear();
    Pt2di aP(0,anY);
    for ( ; aP.x<mSz0.x ; aP.x++)
    {
         Pt3dr aPTer = mN0.PtOfIndex(aP);
         aVec.push_back(mNTarget.Terrain2Index(aPTer));
    }
}


void cAnalyPbTopo::TestFlag(int aK)
{
    /*  Supprime les flag des arcs correspondant a des distances trop grandes */

    Pt2di aV = TAB_8_NEIGH[aK];
    if (aV.y> 0) return;
    if ((aV.y==0) &&  (aV.x>0)) return;

    Pt2di aPPrec = mPCur + aV;
    if (aPPrec.x >= mSz0.x) return;

    Pt2dr aProjCur = mVPts[1][mPCur.x];
    Pt2dr aProjNext = mVPts[1+aV.y][mPCur.x+aV.x];

    double aD = dist8(aProjCur-aProjNext);
    if (aD>mDistSeuil)
    {
         // Suppression des flags
         mDataF[mPCur.y][mPCur.x] &= ~(1<<aK);
         mDataF[aPPrec.y][aPPrec.x] &= ~(1<<((aK+4)%8));
    }
}

bool operator == (const cOneZonzATB & aZ1, const cOneZonzATB & aZ2)
{
   return aZ1.Num() == aZ2.Num();
}

int  cAnalyPbTopo::NextNumZoneC()
{
  if (int(mATB.OneZonzATB().size()) <= mNbMaxZone) return (int)(mATB.OneZonzATB().size() + 1);

   
   cOneZonzATB aMinZ =  *(mATB.OneZonzATB().begin());
   for 
   (
             std::list<cOneZonzATB>::const_iterator itZ=mATB.OneZonzATB().begin();
             itZ !=mATB.OneZonzATB().end();
             itZ++
   )
   {
        if (aMinZ.NbMasq() > itZ->NbMasq()) 
           aMinZ = *itZ;
   }

   mATB.OneZonzATB().remove(aMinZ);

   //  std::cout << "ZZZZZZzz  " <<  mATB.OneZonzATB().size() << "\n";

   cCC_NoActionOnNewPt aCCNo;
   OneZC
   (
       aMinZ.GermGlob(),false,
       mTImLab,aMinZ.Num(),LabPoub,
       mTImLab,aMinZ.Num(),
       aCCNo
   );
   


   return aMinZ.Num();
}


cAnalyPbTopo::cAnalyPbTopo
(
     cAnaTopoXmlBascule & aRes,
     cElNuage3DMaille & aNuageInit,
     cElNuage3DMaille &aNuageTarget,
     const std::string & aNameOut,
     bool                aShow,
     bool                aDebug

) :
    mN0       (aNuageInit),
    mSz0      (mN0.SzUnique()),
    mNTarget  (aNuageTarget),
    mImDef0   (mSz0.x,mSz0.y),
    mTImDef0  (mImDef0),
    mImFlag   (mSz0.x,mSz0.y,255),
    mImLab    (mSz0.x,mSz0.y,LabPoub),
    mTImLab   (mImLab),
    mDataF    (mImFlag.data()),
    mVPts     (2),
    // mDistSeuil (5),
    // mDistSeuil (sqrt(euclid(mN0.SzGeom()))),
    // mDistSeuil ( ElMin(mNTarget.SzGeom().x,mNTarget.SzGeom().y) / 2),
    mDistSeuil (mNTarget.SeuilDistPbTopo()),
    mNumZoneC (1),
    mNbMaxZone (2),
    mShow     (aShow),
    mDebug    (aDebug)
{
    if (mDebug) 
    {
       std::cout << "SEUIL " << mDistSeuil << "\n";
    }
    ELISE_COPY(mImDef0.all_pts(),mN0.ImDef().in(),mImDef0.out());

    mN0.ProfBouchePPV();

    MakeOneLine(mVPts[0],0);

    for (mPCur.y = 1; mPCur.y<mSz0.y  ; mPCur.y++)
    {
        MakeOneLine(mVPts[1],mPCur.y);

        for ( mPCur.x=1 ; mPCur.x<mSz0.x ; mPCur.x++)
        {
            for (int aK=0 ; aK<8 ; aK++)
            {
                TestFlag(aK);
            }
            // TestFlag(mPCur,
        }

        mVPts[0] = mVPts[1];
    }

    if (mDebug)
    {
      Tiff_Im::CreateFromIm(mImFlag,"TopoFlag.tif");
    }

    ELISE_COPY(mImFlag.all_pts(),nflag_open_sym(mImFlag.in(0)),mImFlag.out());
    ELISE_COPY(mImLab.interior(1),LabInit,mImLab.out());


    // Im2D_Bits<1>

    
   
    // for (mPCur.y = 0; mPCur.y<mSz0.y  ; mPCur.y++)
    for (mPCur.y = mSz0.y-1; mPCur.y>=0  ; mPCur.y--)
    {
        for ( mPCur.x=0 ; mPCur.x<mSz0.x ; mPCur.x++)
        {
            if (GZC_PtIn(mPCur))
            {
               mGotMasq = false;
               mCurATB.BoxGlob()._p0 = mPCur;
               mCurATB.BoxGlob()._p1 = mPCur;
               mCurATB.GermGlob()    = mPCur;

               mCurATB.BoxMasq()._p0 =  Pt2di( 1000000, 1000000);
               mCurATB.BoxMasq()._p1 = -Pt2di( 1000000, 1000000);
               mCurATB.NbGlob() = 0;
               mCurATB.NbMasq() = 0;

               mCurATB.Num()     = mNumZoneC;
               GenZonec(mPCur,*this);
               mCurATB.BoxGlob()._p1 = mCurATB.BoxGlob()._p1 + Pt2di(1,1);
               mCurATB.BoxMasq()._p1 = mCurATB.BoxMasq()._p1 + Pt2di(1,1);

               if (mDebug)
               {
                   std::cout << mCurATB.BoxGlob() << " " << mCurATB.BoxMasq() << "\n";
               }
               mATB.OneZonzATB().push_back(mCurATB);
               mNumZoneC = NextNumZoneC();
            }
        }
    }
    ELISE_COPY
    (
         select(mImLab.all_pts(), (mImLab.in()==LabPoub) || (mImLab.in()==LabInit)|| (!mImDef0.in())),
         0,
         mImLab.out()
    );

    if (mDebug)
    {
        std::cout << "NB ZONE " << mATB.OneZonzATB().size() << "\n";
        for 
        (
            std::list<cOneZonzATB>::const_iterator itZ=mATB.OneZonzATB().begin();
            itZ!=mATB.OneZonzATB().end();
            itZ++
        )
        {
            std::cout << " Zone " << itZ->Num() 
                      << " NBG " <<itZ->NbGlob() 
                      << " NBM " << itZ->NbMasq() 
                      << "\n";
        }
    }

    if (mATB.OneZonzATB().size() > 0)
    {
        aRes.ResFromAnaTopo() = true;
        aRes.OneZonXmlAMTB().clear();
        int aCpt=1;
        for 
        (
             std::list<cOneZonzATB>::const_iterator itZ=mATB.OneZonzATB().begin();
             itZ !=mATB.OneZonzATB().end();
             itZ++
        )
        {
            std::string aPost = "-Zone"+ ToString(aCpt);
            aCpt++;
            std::string aNameMasq =  aNameOut+ aPost + "-Masq.tif";
            Tiff_Im::CreateFromFonc(aNameMasq,mImLab.sz(),mImLab.in()==itZ->Num(),GenIm::bits1_msbf);
            cXML_ParamNuage3DMaille aNewXML = mN0.Params();
            aNewXML.Image_Profondeur().Val().Masq() = NameWithoutDir(aNameMasq);
            std::string aNameXml =  aNameOut+ aPost + ".xml";
            MakeFileXML(aNewXML,aNameXml);

            cOneZonXmlAMTB aOZ;
            aOZ.NameXml() = NameWithoutDir(aNameXml);
            aRes.OneZonXmlAMTB().push_back(aOZ);
        }
    }
    else
    {
    }

    if (mShow)
    {
        Video_Win aW = Video_Win::WStd(mSz0,0.3);
        ELISE_COPY
        (
            aW.all_pts(),
            its_to_rgb
            (
               Virgule
               (
                    (mImDef0.in() + 0.5) *128 ,
                    47 + mImLab.in()*95,
                    100
               )
            ),
            aW.orgb()
        );
        for 
        (
             std::list<cOneZonzATB>::const_iterator itZ=mATB.OneZonzATB().begin();
             itZ !=mATB.OneZonzATB().end();
             itZ++
        )
        {
             std::cout << "NbPts, masq " << itZ->NbMasq() << " glob " << itZ->NbGlob() << "\n";
             aW.draw_rect
             (
                 Pt2dr(itZ->BoxGlob()._p0),
                 Pt2dr(itZ->BoxGlob()._p1),
                 aW.pdisc()(itZ->Num())
             );
             aW.draw_rect
             (
                 Pt2dr(itZ->BoxMasq()._p0),
                 Pt2dr(itZ->BoxMasq()._p1),
                 Line_St(aW.pdisc()(itZ->Num()),3)
             );
             aW.draw_circle_loc
             (
                 Pt2dr(itZ->GermGlob()),
                 10.0,
                 aW.pdisc()(itZ->Num())
             );
             aW.draw_circle_loc
             (
                 Pt2dr(itZ->GermMasq()),
                 5.0,
                 aW.pdisc()(itZ->Num())
             );
        }
        Tiff_Im::CreateFromIm(mImFlag,aNameOut+"-Flag.tif");
        Tiff_Im::CreateFromIm(mImLab,aNameOut+"-Label.tif");
        MakeFileXML(mATB,aNameOut+"-DetailledMTD.xml");
        std::cout << "NumZonec " <<  mNumZoneC << " Seuil " <<  mDistSeuil << "\n";
        aW.clik_in();
    }

}


int TopoSurf_main(int argc,char ** argv)
{
    // std::string aName0 = "/media/data2/Cylindres/Colonne-Ramasseum-2011/Fusion-QuickMac/Nuage-Depth-IMG_0115.CR2.xml";
    // std::string aNameProj = "/media/data2/Cylindres/Colonne-Ramasseum-2011/MEC-Malt/NuageImProf_STD-MALT_Etape_8.xml";

    std::string aName0 ;
    std::string aNameProj ;
    std::string aNameOut ;
    bool        Show=false;
    bool        Debug=false;
    bool        PbTopo;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aName0,"Name Nuage In",eSAM_IsPatFile)
                    << EAMC(aNameProj,"Name Targeted Proj",eSAM_IsPatFile),
        LArgMain()  << EAM(aNameOut,"Out",true,"Name Result")
                    << EAM(Show,"Show",true,"Visualize result in Window (Def = false)")
                    << EAM(PbTopo,"PbTopo",true,"Analye Pb Topo , Def depends from Target topology")
                    << EAM(Debug,"Debug",true,"For debug ( ;-)")

    );


    cElNuage3DMaille *  aNProj =NuageWithoutData(aNameProj);

    if (!EAMIsInit(&PbTopo))
    {
       PbTopo = aNProj->SeuilDistPbTopo() > 0;
    }


    if (! EAMIsInit(&aNameOut))
       aNameOut = DirOfFile(aName0) + "TopoBasc-" + StdPrefix(NameWithoutDir(aName0)) ;

    cAnaTopoXmlBascule aRes;
    aRes.ResFromAnaTopo() = false;
    cOneZonXmlAMTB aOZ;
    aOZ.NameXml() = aName0; 
    aRes.OneZonXmlAMTB().push_back(aOZ);

    if (PbTopo)
    {
       cElNuage3DMaille *  aN0 = cElNuage3DMaille::FromFileIm(aName0);
       cAnalyPbTopo anAPT(aRes,*aN0,*aNProj,aNameOut,Show,Debug);
    }

     MakeFileXML(aRes,aNameOut+"-MTD.xml");


     return EXIT_SUCCESS;
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
