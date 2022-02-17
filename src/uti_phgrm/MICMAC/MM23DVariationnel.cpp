/*Header-MicMac-eLiSe-25/06/2007peroChImMM_main

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


#if (0)
std::string StdNameMDTPCOfFile(const std::string & aName,bool XML);
cMetaDataPartiesCachees * StdGetMDTPCOfFile(const std::string & aName,bool SVP=true);
Pt2di DecalageFromPC(const std::string & aN1,const std::string & aN2);
void MakeMetaData_XML_GeoI(const std::string & aNameImMasq,double aResol);
void MakeMetaData_XML_GeoI(const std::string & aNameImMasq);


/***********************************************************/
/*                                                         */
/*          FUSION DEPLACEMENTS                            */
/*                                                         */
/***********************************************************/

class cAppliFusionDepl;
class cOneImageAFD;

class cCpleImAFD
{
    public :
         cCpleImAFD(cOneImageAFD*,cOneImageAFD*,const std::string & aDir);
         void Reload();
         bool  Loaded() const {return mLoaded;}
         Pt3dr GetDepl(Pt2di aP0,Pt2di aP1);
    private :
         cAppliFusionDepl &        mAppli;
         std::string               mDir;
         std::string               mNameX;
         Tiff_Im                   mTifX;
         std::string               mNameY;
         Tiff_Im                   mTifY;
         cOneImageAFD *            mI1;
         cOneImageAFD *            mI2;
         Box2di                    mLocBox;
         Im2D<REAL4,REAL8>         mDepX;
         TIm2D<REAL4,REAL8>        mTDepX;
         Im2D<REAL4,REAL8>         mDepY;
         TIm2D<REAL4,REAL8>        mTDepY;
         bool                      mLoaded;
         Box2di                    mCurBox;
         Pt2di                     mDecGl;
};

class cOneImageAFD
{
     public :
          cOneImageAFD(const std::string & anIm,bool Avant,cAppliFusionDepl &);
          friend class cAppliFusionDepl;
          cAppliFusionDepl & Appli();
          // const Pt2di & P0() {return mP0;}
          // const Pt2di & P1() {return mP1;}
          bool  Loaded() const {return mLoaded;}
          void Reload(const Box2di &);
          const Box2di & CurBox() const {return mCurBox;}
          const Box2di & LocBox() const {return mLocBox;}
          bool  Avant() const {return mAvant;}
          INT  PC(const Pt2di &);
     private :
          cAppliFusionDepl &        mAppli;
          string                    mNameIm;
          string                    mNamePC;
          bool                      mAvant;
          cMetaDataPartiesCachees * mMDP;
          Box2di                    mLocBox;
          bool                      mLoaded;
          Box2di                    mCurBox;
          Im2D_U_INT1               mImPC;
          TIm2D<U_INT1,INT>         mTImPC;
          Pt2di                     mDecGl;
};



class cAppliFusionDepl
{
     public :
          cAppliFusionDepl(int argc,char ** argv);
        
          std::string  NamePx(int aNumPx);
     private :
          void  DoOneBox(const Box2di &);

          void AddIm(const std::string &,bool Avant);
          void AddCple(cOneImageAFD * , cOneImageAFD *);

          cInterfChantierNameManipulateur * mICNM;
          std::vector<cOneImageAFD *> mImAvant;
          std::vector<cOneImageAFD *> mImApres;
          std::vector<cOneImageAFD *> mAllIm;

          std::vector<cCpleImAFD *>   mCples;

          std::string                 mPatAvant;
          std::string                 mPatApres;
          Box2di                      mGlobBox;
          // Pt2di                       mSz;
          std::string                 mKeyDirDepl;
          int                         mNumEt;
          string                      mDirFusion;
 
          Tiff_Im *  FileResult(const std::string & aName);

          Tiff_Im *                 mFilePx1;
          Tiff_Im *                 mFilePx2;
          Im2D<REAL4,REAL8>         mDep1;
          TIm2D<REAL4,REAL8>        mTDep1;
          Im2D<REAL4,REAL8>         mDep2;
          TIm2D<REAL4,REAL8>        mTDep2;
};

       // =============  cCpleImAFD ==================

cCpleImAFD::cCpleImAFD(cOneImageAFD* anI1,cOneImageAFD* anI2,const std::string & aDir) :
   mAppli   (anI1->Appli()),
   mDir     (aDir),
   mNameX   (mDir+ "/" + mAppli.NamePx(1)),
   mTifX    (mNameX.c_str()),
   mNameY   (mDir+ "/" + mAppli.NamePx(2)),
   mTifY    (mNameY.c_str()),
   mI1      (anI1),
   mI2      (anI2),
   mLocBox  (Inf(mI1->LocBox(),mI2->LocBox())),
   mDepX    (1,1),
   mTDepX   (mDepX),
   mDepY    (1,1),
   mTDepY   (mDepY)
{
}

void cCpleImAFD::Reload()
{
    mLoaded = mI1->Loaded() && mI2->Loaded();
    if (!mLoaded) return;

    mLoaded =   (! InterVide(mI1->CurBox(),mI2->CurBox()));
    if (!mLoaded) return;

    mCurBox = Inf(mI1->CurBox(),mI2->CurBox());

    mDecGl = mCurBox._p0 - mLocBox._p0;

    mDepX.Resize(mCurBox.sz());
    mTDepX =  TIm2D<REAL4,REAL8>(mDepX);
    mDepY.Resize(mCurBox.sz());
    mTDepY =  TIm2D<REAL4,REAL8>(mDepY);
    
    ELISE_COPY(mDepX.all_pts(),trans(mTifX.in(),mDecGl),mDepX.out());
    ELISE_COPY(mDepY.all_pts(),trans(mTifY.in(),mDecGl),mDepY.out());

    std::cout << "ReloadCple " <<  mDir << "\n";
}

Pt3dr cCpleImAFD::GetDepl(Pt2di aP0,Pt2di aP1)
{
    aP0 = Sup(aP0,mCurBox._p0);
    aP1 = Inf(aP1,mCurBox._p1-Pt2di(1,1));
    if ((aP0.x>=aP1.x) ||  (aP0.y>=aP1.y))
       return Pt3dr(0,0,0);

    std::vector<double> aVx;
    std::vector<double> aVy;
  
    Pt2di aP;
    for (aP.x =aP0.x ; aP.x <= aP1.x ; aP.x++)
    {
        for (aP.y =aP0.y ; aP.y <= aP1.y ; aP.y++)
        {
            if (mI1->PC(aP) < 5)
            {
               aVx.push_back(mTDepX.get(aP-mCurBox._p0));
               aVy.push_back(mTDepY.get(aP-mCurBox._p0));
            }
        }
    }
// std::cout << aP0 << aP1 << aVx.size() << "\n";
    if (aVx.size() == 0)
      return Pt3dr(0,0,0);

    Pt3dr aRes = Pt3dr(MedianeSup(aVx),MedianeSup(aVy),aVx.size());

    if (mI1->Avant())
       aRes = Pt3dr(-aRes.x,-aRes.y,aRes.z);

    return aRes;
}

       // =============  cOneImageAFD ==================

cOneImageAFD::cOneImageAFD(const std::string & anIm,bool Avant,cAppliFusionDepl & anAppli) :
    mAppli      (anAppli),
    mNameIm     (anIm),
    // mNamePC     (DirOfFile(mNameIm) + "PC_"+NameWithoutDir(mNameIm)),
    mNamePC     (StdNameMDTPCOfFile(mNameIm,false)),
    mAvant      (Avant),
    mMDP        (StdGetMDTPCOfFile(anIm,false)),
    mLocBox    (mMDP->Offset(),mMDP->Offset()+mMDP->Sz()),
    mImPC       (1,1),
    mTImPC      (mImPC)
{
}

cAppliFusionDepl & cOneImageAFD::Appli()
{
    return mAppli;
}


void cOneImageAFD::Reload(const Box2di & aBoxGlob)
{
   mLoaded = (! InterVide(aBoxGlob,mLocBox));
   if (! mLoaded) return;
   mCurBox  = Inf(mLocBox,aBoxGlob);
   mImPC.Resize(mCurBox.sz());
   mTImPC =  TIm2D<U_INT1,INT>(mImPC);
   Tiff_Im aTifPC(mNamePC.c_str());

   // std::cout << mP0 << mP1 << " SS " << mP1-mP0 << "\n";
   // std::cout << mCurBox._p0 << " " << mTImPC.sz() << " " << aTifPC.sz() << " \n";

   mDecGl = mCurBox._p0 - mLocBox._p0;

   ELISE_COPY
   (
       mImPC.all_pts(),
       trans(aTifPC.in(),mDecGl),
       mImPC.out()
   );
   std::cout << "Reload " << mNameIm << mCurBox << "\n";
}

INT  cOneImageAFD::PC(const Pt2di & aPGlob)
{
   return mTImPC.get(aPGlob-mCurBox._p0,500);
}


       // =============  cAppliFusionDepl ==================


void cAppliFusionDepl::AddCple(cOneImageAFD * anI1, cOneImageAFD * anI2)
{
     std::string aDir = mICNM->Assoc1To2(mKeyDirDepl,anI1->mNameIm,anI2->mNameIm,true);
     std::string aNameFile = aDir + "/"+NamePx(1);
     if (ELISE_fp::exist_file(aNameFile))
     {
        mCples.push_back(new cCpleImAFD(anI1,anI2,aDir));
     }

}

void cAppliFusionDepl::AddIm(const std::string & aPat,bool Avant)
{
     cElemAppliSetFile anEASF(aPat);
     if (mICNM==0)
         mICNM = anEASF.mICNM;

     const std::vector<std::string> * aSetIm = anEASF.SetIm();

     for (int aKIm=0 ; aKIm<int(aSetIm->size()) ; aKIm++)
     {
         cOneImageAFD * anIm = new cOneImageAFD((*aSetIm)[aKIm],Avant,*this);
         mAllIm.push_back(anIm);
         if (Avant)
            mImAvant.push_back(anIm);
         else
            mImApres.push_back(anIm);
         if (aKIm==0)
            mGlobBox = anIm->LocBox();
         else             
            mGlobBox = Sup(mGlobBox,anIm->LocBox());
     }

}

Tiff_Im *  cAppliFusionDepl::FileResult(const std::string & aName)
{
   bool IsNew;
   return new Tiff_Im
              ( 
                       Tiff_Im::CreateIfNeeded
                       (
                           IsNew,
                           mDirFusion+aName,
                           mGlobBox.sz(),
                           GenIm::real4,
                           Tiff_Im::No_Compr,
                           Tiff_Im::BlackIsZero
                       )
              );
}

std::string cAppliFusionDepl::NamePx(int aNumPx)
{
   return "Px"+ ToString(aNumPx)  +"_Num"+ ToString(mNumEt) + "_DeZoom1_LeChantier.tif";
}


double ScoreDepl(const std::vector<Pt3dr> & aVP,const Pt3dr & aP)
{
    double aRes=0;

    for (int aKp=0 ; aKp<int(aVP.size()) ; aKp++)
    {
        const Pt3dr & aQ = aVP[aKp];
        aRes += aQ.z * (ElAbs(aQ.x-aP.x) + ElAbs(aQ.y-aP.y));
    }
    return aRes;
}

void  cAppliFusionDepl::DoOneBox(const Box2di & aBoxGlob)
{
    mDep1.Resize(aBoxGlob.sz());
    mTDep1 = TIm2D<REAL4,REAL8>(mDep1);
    mDep2.Resize(aBoxGlob.sz());
    mTDep2 = TIm2D<REAL4,REAL8>(mDep2);

    for (int aKIm=0 ; aKIm<int(mAllIm.size()) ; aKIm++)
    {
       mAllIm[aKIm]->Reload(aBoxGlob);
    }

    for (int aKCpl=0 ; aKCpl<int(mCples.size()) ; aKCpl++)
    {
       mCples[aKCpl]->Reload();
    }

    int aStepCalc = 1;
    int aStepMed  = 4;
    Pt2di aPStepMed(aStepMed,aStepMed);
    Pt2di aPStepCalc(aStepCalc,aStepCalc);
    Pt2di aP;
    for (aP.x= aBoxGlob._p0.x +aStepCalc; aP.x <aBoxGlob._p1.x ; aP.x+= 2*aStepCalc+1)
    {
        for (aP.y= aBoxGlob._p0.y +aStepCalc; aP.y <aBoxGlob._p1.y ; aP.y+= 2*aStepCalc+1)
        {
            std::vector<Pt3dr> aVDepl;
            for (int aKCpl=0 ; aKCpl<int(mCples.size()) ; aKCpl++)
            {
                Pt3dr aDepl = mCples[aKCpl]->GetDepl(aP-aPStepMed,aP+aPStepMed);
                if (aDepl.z >0)
                {
                   aVDepl.push_back(aDepl);
                }
            }

            if (aVDepl.size() > 0)
            {
                double aScMin = 1e30;
                int aKmin = -1;
                for (int aKp=0 ; aKp<int(aVDepl.size()) ; aKp++)
                {
                    double aSc = ScoreDepl(aVDepl,aVDepl[aKp]);
                    if (aSc<aScMin)
                    {
                       aScMin = aSc;
                       aKmin = aKp;
                    }
                }
                if (aKmin>=0)
                {
                    Pt2di aPC1 = aP- aPStepCalc; 
                    Pt2di aPC2 = aP+ aPStepCalc; 
                    Pt2di aQ;
                    for (aQ.x=aPC1.x ; aQ.x<=aPC2.x ; aQ.x++)
                    {
                       for (aQ.y=aPC1.y ; aQ.y<=aPC2.y ; aQ.y++)
                       {
                          if (mTDep1.inside(aQ))
                          {
                              mTDep1.oset(aQ,aVDepl[aKmin].x);
                              mTDep2.oset(aQ,aVDepl[aKmin].y);
                          }
                       }
                    }
                }
                    // std::cout << "Depl " << aVDepl[aKmin] << "\n";
            }
        }
        std::cout << "FusionDepl Reste " << aBoxGlob._p1.x - aP.x << " X=" << aP.x  << "\n";
    }

    ELISE_COPY
    (
        rectangle(aBoxGlob._p0,aBoxGlob._p1),
        trans(Virgule(mDep1.in(),mDep2.in()),-aBoxGlob._p0),
        Virgule(mFilePx1->out(),mFilePx2->out())
    );
}



cAppliFusionDepl::cAppliFusionDepl(int argc,char ** argv) :
    mICNM       (0),
    mKeyDirDepl ("Loc-Assoc-Dir-Depl"),
    mNumEt      (8),
    mDep1       (1,1),
    mTDep1      (mDep1),
    mDep2       (1,1),
    mTDep2      (mDep2)
{
    std::string aImMasq;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mPatAvant,"Images before")
                    << EAMC(mPatApres,"Image after"),
        LArgMain()  << EAM(aImMasq,"YYYYY",true,"XXXXXXX")
     );

     AddIm(mPatAvant,true);
     AddIm(mPatApres,false);

     for (int aKAv=0 ; aKAv<int(mImAvant.size()) ; aKAv++)
     {
         cOneImageAFD * anImAv = mImAvant[aKAv];
         for (int aKAp=0 ; aKAp<int(mImApres.size()) ; aKAp++)
         {
             cOneImageAFD * anImAp = mImApres[aKAp];
             AddCple(anImAv,anImAp);
             AddCple(anImAp,anImAv);
         }
     }

     mDirFusion = mICNM->Dir() + "Fusion-Depl/";
     ELISE_fp::MkDir(mDirFusion);


     mFilePx1 = FileResult("Px1.tif");
     mFilePx2 = FileResult("Px2.tif");
     std::cout << "AppliP0P1 " << mGlobBox << "\n";


     // DoOneBox(Box2di(Pt2di(500,500),Pt2di(1500,2500)));
     DoOneBox(Box2di(Pt2di(0,0),mFilePx1->sz()));
}

int FusionDepl_Main(int argc,char ** argv)
{
    cAppliFusionDepl anAppli(argc,argv);
    return EXIT_SUCCESS;
}
#endif

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
