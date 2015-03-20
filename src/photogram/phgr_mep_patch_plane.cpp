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

/*
    Temps pour 1000 SVD :       0.040899
    Temps pour 1000 Solve(8)  : 0.00645113
    Temps pour 1000000  AddEq  : 0.113502

Le parametre de taille n'a quasiment pas d'influence dans OneIterNbSomGlob 
    OneIterNbSomGlob(aHom,20,P8COL::green,Show);

Par contre, si on supprime TestEvalHomographie, le temps es / par 60.

Conclusion, pour optimiser les chances il faut tenter bcp de triangle, en faisant varier
les tailles d'initialisation. Pour chaque taille on prend un approch progressive, mais 
on ne fait qu'un seul test de TestEvalHomographie,





  A faire rajouter une observation.
  Mesure les temps de calcul des différentes briques :

     * calcul de SVD . Si calcul de SVD rapide, dq on fait une estim hom, on tente sa chance sur residu 3D:q
     * inversion
     * ajout une obs


  Algo :

    - a partir de germe (4 point)
    - file d'attente (heap) sur le critere de l'ecart / homogr courante

*/

#include "StdAfx.h"

class cAttrSomOPP
{
    public :

       cAttrSomOPP() {}

       cAttrSomOPP(const Pt2dr & aP1,const Pt2dr & aP2,double aPds) :
          mP1 (aP1),
          mP2 (aP2),
          mPds (aPds)
       {
       }

       ElCplePtsHomologues Cple() {return ElCplePtsHomologues(mP1,mP2,1.0);}

       Pt2dr mP1;
       Pt2dr mP2;
       double mPds;
       double mDist;

};

class cAttrArcSomOPP
{
     public :
     private :
};



typedef  ElSom<cAttrSomOPP ,cAttrArcSomOPP>   tSomOPP;
typedef  ElArc<cAttrSomOPP,cAttrArcSomOPP>    tArcOPP;
typedef  ElGraphe<cAttrSomOPP,cAttrArcSomOPP> tGrOPP;


Pt2dr POfSomPtr(const tSomOPP * aSom) {return aSom->attr().mP1;}

class cOriPlanePatch
{
    public :
        cOriPlanePatch(   double aFoc,
                          const ElPackHomologue & aPack,
                          Video_Win * aW,
                          Pt2dr       aP0W,
                          double      aScaleW
                      );
         void operator()(tSomOPP*,tSomOPP*,bool);  // Delaunay call back


         void TestPt();
         void TestHomogr();
         void TestOneGermGlob(const std::vector<tSomOPP*>  & aVSom,bool Show);
         void    ResetHom();
         cElHomographie SolveHom();
         void OneIterNbSomGlob(cElHomographie & aHom,int aNbGlob,int aNbCoul,bool Show);
    private  :
         void  TestEvalHomographie(const cElHomographie &,bool Show);
         void AddHom(tSomOPP*);
         void ShowPoint(const Pt2dr &,double aRay,int coul) const;
         void ShowPoint(const tSomOPP &,double aRay,int coul) const;
         void ShowSeg(const Pt2dr & aP1,const Pt2dr& aP2,int aCoul) const;
         Pt2dr ToW(const Pt2dr & aP) const;
         Pt2dr FromW(const Pt2dr & aP) const;
         tSomOPP * GetSom(int aCoul);



         double                 mFoc;
         ElPackHomologue        mPack;
         Video_Win *            mW;
         Pt2dr                  mP0W;
         double                 mScaleW;
         std::vector<tSomOPP *> mVSom;
         int                    mNbSom;
         tGrOPP                 mGrOPP;
         int                    mFlagVisitH;
         L2SysSurResol          mSysHom;
         bool                   mModeAff;
         cInterfBundle2Image *  mIBI_Lin;
         
};

    //  ==============  Graphisme   ================

Pt2dr cOriPlanePatch::ToW(const Pt2dr & aP) const { return (aP-mP0W) *mScaleW; }
Pt2dr cOriPlanePatch::FromW(const Pt2dr & aP) const { return aP/mScaleW + mP0W; }
void cOriPlanePatch::ShowPoint(const Pt2dr & aP,double aRay,int aCoul) const
{
   if (mW && (aCoul>=0)) 
      mW->draw_circle_abs(ToW(aP),aRay,mW->pdisc()(aCoul));
}
void  cOriPlanePatch::ShowPoint(const tSomOPP & aS,double aRay,int aCoul) const
{
    ShowPoint(aS.attr().mP1,aRay,aCoul);
}

void cOriPlanePatch::ShowSeg(const Pt2dr & aP1,const Pt2dr& aP2,int aCoul) const
{
    if (mW) mW->draw_seg(ToW(aP1),ToW(aP2),mW->pdisc()(aCoul) );
}
   // ==========================================================


void cOriPlanePatch::operator()(tSomOPP* aS1,tSomOPP* aS2,bool)
{
    ShowSeg(aS1->attr().mP1,aS2->attr().mP1,P8COL::red);

    if (! mGrOPP.arc_s1s2(*aS1,*aS2))
    {
        cAttrArcSomOPP anAttr;
        mGrOPP.add_arc(*aS1,*aS2,anAttr,anAttr);
    }
}



tSomOPP * cOriPlanePatch::GetSom(int aCoul)
{
   Clik aCl = mW->clik_in();
   Pt2dr aP = FromW(aCl._pt);
   double aDistMin = 1e20;
   tSomOPP * aRes = 0;
   
   for (int aK=0 ; aK<int(mVSom.size()) ; aK++)
   {
       double aD = euclid(aP,mVSom[aK]->attr().mP1);
       if (aD<aDistMin)
       {
           aDistMin = aD;
           aRes = mVSom[aK];
       }
   }

   ShowPoint(*aRes,3.0,P8COL::white);
   return aRes;
}


// (a + b x1 + c y1) = x2 (1+g x1 + h y1)
// (d + e x1 + f y1) = y2 (1+g x1 + h y1)

void cOriPlanePatch::AddHom(tSomOPP* aSom)
{
     static double aCoeff[8];

     aSom->flag_set_kth_true(mFlagVisitH);
     const cAttrSomOPP & anAttr = aSom->attr();
     double aX1 =  anAttr.mP1.x;
     double aY1 =  anAttr.mP1.y;
     double aX2 =  anAttr.mP2.x;
     double aY2 =  anAttr.mP2.y;
    
     aCoeff[0] = 1;
     aCoeff[1] = aX1;
     aCoeff[2] = aY1;
     aCoeff[3] = aCoeff[4] = aCoeff[5] = 0;
     if ( mModeAff)
     {
         aCoeff[6] =  aCoeff[7] = 0.0;
     }
     else
     {
        aCoeff[6]  = aX2 * aX1;
        aCoeff[7] =  aX2 * aY1;
     }
     mSysHom.GSSR_AddNewEquation(1.0,aCoeff,aX2,0);


     aCoeff[0] = aCoeff[1] = aCoeff[2] = 0;
     aCoeff[3] = 1;
     aCoeff[4] = aX1;
     aCoeff[5] = aY1;
     if (mModeAff)
     {
         aCoeff[6] =  aCoeff[7] = 0.0;
     }
     else
     {
        aCoeff[6]  = aY2 * aX1;
        aCoeff[7]  = aY2 * aY1;
     }
     mSysHom.GSSR_AddNewEquation(1.0,aCoeff,aY2,0);
}


void cOriPlanePatch::ResetHom()
{
    mSysHom.GSSR_Reset(true);
    mSysHom.SetPhaseEquation(0);
    for (int aK=0 ; aK< mNbSom ; aK++)
    {
        mVSom[aK]->flag_set_kth_false(mFlagVisitH);
    }
}


void  cOriPlanePatch::TestEvalHomographie(const cElHomographie & aHom,bool Show)
{
     cResMepRelCoplan aRMC =  ElPackHomologue::MepRelCoplan(1.0,aHom,tPairPt(Pt2dr(0,0),Pt2dr(0,0)));
     const std::list<cElemMepRelCoplan>  & aLSolPl = aRMC.LElem();

     ElRotation3D aBestR = ElRotation3D::Id;
     double aBestScore = 1e20;
     for (std::list<cElemMepRelCoplan>::const_iterator itS = aLSolPl.begin() ; itS != aLSolPl.end() ; itS++)
     {
        if ( itS->PhysOk())
        {
             ElRotation3D aR = itS->Rot();
             aR = aR.inv();
             double aScore = LinearCostMEP(mPack,aR,0.1);
             if (aScore<aBestScore)
             {
                 aBestScore = aScore;
                 aBestR = aR;
             }
        }
     }
     double anEr = mIBI_Lin->ErrInitRobuste(aBestR);
     anEr =  mIBI_Lin->ResiduEq(aBestR,anEr);
     double anEr0 = anEr;
     for (int aK=0 ; aK<5 ; aK++)
     {
           aBestR = mIBI_Lin->OneIterEq(aBestR,anEr);
     }
     if (Show)
        std::cout << "COST Hom " << anEr0*mFoc << " => " << anEr*mFoc << "\n";
}

cElHomographie cOriPlanePatch::SolveHom()
{ 
    if (mModeAff)
    {
        static double aCoeff[8];
        for (int aCstr=6 ; aCstr<8 ; aCstr++)
        {
            for (int aK=0 ; aK< 8 ; aK++)
                 aCoeff[aK] = (aK==aCstr);
            mSysHom.GSSR_AddNewEquation(1.0,aCoeff,0,0);
        }
    }
    Im1D_REAL8 aSol =      mSysHom.GSSR_Solve (0);
    double * aDS = aSol.data();

    cElComposHomographie aHX(aDS[1],aDS[2],aDS[0]);
    cElComposHomographie aHY(aDS[4],aDS[5],aDS[3]);
    cElComposHomographie aHZ(aDS[6],aDS[7],     1);

    return cElHomographie(aHX,aHY,aHZ);
}

void cOriPlanePatch::OneIterNbSomGlob(cElHomographie & aHom,int aNbGlob,int aCoul,bool Show)
{
    int aNbIn = 0;
    std::vector<double> aVDist;
    for (int aK=0 ; aK< mNbSom ; aK++)
    {
       tSomOPP * aS = mVSom[aK];
       if (! aS->flag_kth(mFlagVisitH))
       {
           cAttrSomOPP & anAttr = aS->attr();
           anAttr.mDist = square_euclid(anAttr.mP2-aHom.Direct(anAttr.mP1));
           aVDist.push_back(anAttr.mDist);
       }
       else
          aNbIn ++;
    }
    double aVSeuil = KthVal(aVDist,aNbGlob-aNbIn);
    
    for (int aK=0 ; aK< mNbSom ; aK++)
    {
        tSomOPP * aS = mVSom[aK];
        if ((!aS->flag_kth(mFlagVisitH)) &&  (aS->attr().mDist<=aVSeuil))
        {
             AddHom(aS);
             if (Show) ShowPoint(*aS,3.0,aCoul);
        }
    }
    aHom = SolveHom();
    TestEvalHomographie(aHom,Show);
}

void cOriPlanePatch::TestOneGermGlob(const std::vector<tSomOPP*>  & aVSom,bool Show)
{
    mModeAff = false;
    ResetHom();
    for (int aK=0 ; aK<int (aVSom.size()) ; aK++)
    {
        AddHom(aVSom[aK]);
    }
    cElHomographie aHom = SolveHom();

    OneIterNbSomGlob(aHom,8,P8COL::red,Show);
/*
    OneIterNbSomGlob(aHom,10,P8COL::red,Show);
    OneIterNbSomGlob(aHom,12,P8COL::red,Show);
    OneIterNbSomGlob(aHom,14,P8COL::red,Show);
    OneIterNbSomGlob(aHom,16,P8COL::red,Show);
*/
    OneIterNbSomGlob(aHom,20,P8COL::green,Show);
    OneIterNbSomGlob(aHom,50,P8COL::blue,Show);
    OneIterNbSomGlob(aHom,100,P8COL::cyan,Show);
    OneIterNbSomGlob(aHom,200,P8COL::black,Show);
}




void cOriPlanePatch::TestHomogr()
{
    mW->clik_in();
    for (int aK=0 ; aK<int(mVSom.size()) ; aK++)
    {
        ShowPoint(*(mVSom[aK]),3.0,P8COL::green);
    }
    mModeAff = true;
    int aNbPts = mModeAff ? 3 : 4;
    Pt2dr aC (0,0);
    std::vector<tSomOPP*> aVTest;
    for (int aK=0 ; aK< aNbPts ; aK++)
    {
        tSomOPP * aSom =  GetSom(P8COL::white);
        aVTest.push_back(aSom);
        aC  = aC +  aSom->attr().mP1;
    }

    TestOneGermGlob(aVTest,true);

    ElTimer aChrono;
    for (int aK=0 ; aK<100 ; aK++)
        TestOneGermGlob(aVTest,false);
    std::cout  << "TIME GLOB " << aChrono.uval() << "\n";
}

/*
    ResetHom();
    TestOneGerm(aVTest);
    cElHomographie aHom = SolveHom();

    for (int aK=0 ; aK<int(mVSom.size()) ; aK++)
    {
        Pt2dr aP1 = mVSom[aK]->attr().mP1;
        Pt2dr aP2 = mVSom[aK]->attr().mP2;
        Pt2dr aQ2 = aHom.Direct(aP1);

        double aResidu = euclid(aQ2,aP2) * mFoc;
        int aCoul = P8COL::red;

        if (aResidu  < 10)
        {
            aCoul =P8COL::green;
        }
        ShowPoint(aP1,3.0,aCoul);
    }
}
*/

#if (0)

    aC = aC /aNbPts ;

    double aDMax = 6;
    double ErrProp = 1e-2;


    ElTimer aChrono;
    cElHomographie aHom  = cElHomographie::Id();
    

    for (int aNbIt = 0 ; aNbIt<4 ; aNbIt ++)
    {
        aHom = cElHomographie(aPack,true);
        ElPackHomologue  aNewPack;

        std::vector<tSomOPP*> aVTest;
        for (int aK=0 ; aK<int(mVSom.size()) ; aK++)
        {
             Pt2dr aP1 = mVSom[aK]->attr().mP1;
             Pt2dr aP2 = mVSom[aK]->attr().mP2;
             Pt2dr aQ2 = aHom.Direct(aP1);

             double aResidu = euclid(aQ2,aP2) * mFoc;
             double aDC = euclid(aP1,aC) * mFoc;

             int aCoul = P8COL::red;

             if (aResidu  < (aDMax + aDC * ErrProp))
             {
                // double aPds = 1/(1+ElSquare(aD/aDPond));
                double aPds = 1.0;
                ElCplePtsHomologues aCple(aP1,aP2,aPds);
                aVTest.push_back(mVSom[aK]);
                aNewPack.Cple_Add(aCple);
                aCoul = P8COL::yellow;
             }
             // if (aResidu < 3) aCoul = P8COL::white;

             ShowPoint(aP1,3.0,aCoul);
        }
ResetHom();
TestOneGerm(aVTest);
TestOneGerm(aVTest);
SolveHom();

         
         getchar();
         // std::cout << "END IT " << aNbIt << "\n";
         aPack = aNewPack;
    }
    
    std::cout << "TTT " << aChrono.uval() << "\n";

    ElTimer aChronoSVD;
    for (int aK=0 ; aK<1000  ; aK++) 
    {
        cResMepRelCoplan aRMC =  ElPackHomologue::MepRelCoplan(1.0,aHom,tPairPt(Pt2dr(0,0),Pt2dr(0,0)));
    }
    std::cout << "tSVD  " << aChronoSVD.uval() << "\n";
    
    ElTimer aChronoAddE;
    L2SysSurResol aSys(8);
    aSys.SetPhaseEquation(0);
    for (int aNb=0 ; aNb<1000 ; aNb++)
    {
        double aCoeff[8];
        for (int aX=0 ; aX<8 ; aX++)
        {
            aCoeff[aX] = NRrandC();
        }
        for (int anE=0 ; anE<1000 ; anE++)
            aSys.GSSR_AddNewEquation(1.0,aCoeff,5,0);
    }
    std::cout << "ADDe  " << aChronoAddE.uval() << "\n";
    
    ElTimer aChronoSolve;
    for (int aK=0 ; aK<1000  ; aK++) 
    {
         aSys.GSSR_Solve (0);

    }
    std::cout << "tSVD  " << aChronoSolve.uval() << "\n";
}
    
#endif





void  cOriPlanePatch::TestPt()
{
    GetSom(P8COL::yellow);
}



cOriPlanePatch::cOriPlanePatch
( 
      double                  aFoc,
      const ElPackHomologue & aPack,
      Video_Win * aW,
      Pt2dr       aP0W,
      double      aScaleW
)  :
   mFoc         (aFoc),
   mPack        (aPack),
   mW           (aW),
   mP0W         (aP0W),
   mScaleW      (aScaleW),
   mFlagVisitH  (mGrOPP.alloc_flag_som()),
   mSysHom      (8),
   mModeAff     (false),
   mIBI_Lin     (cInterfBundle2Image::LinearDet(mPack,mFoc))
{
    // if (mW) mW->clear();
    for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    {
         tSomOPP & aSom = mGrOPP.new_som(cAttrSomOPP(itP->P1(),itP->P2(),itP->Pds()));
         mVSom.push_back(&aSom);
         if (mW) ShowPoint(aSom,3.0,P8COL::green);
    }
    mNbSom = mVSom.size();

    std::cout << "ENETR DELAU , Nb " << aPack.size() << " \n";

    ElTimer aChrono;
    Delaunay_Mediatrice
    (
         &(mVSom[0]),
         &(mVSom[0])+mVSom.size(),
          POfSomPtr,
       *this,
       1e10,
       (tSomOPP **) 0
    );



    if (aW)
    {
        while (1)
        {
              TestHomogr();
        }
    }
    std::cout << "Time Delaunay " << aChrono.uval() << "\n";
    getchar();
}




void TestOriPlanePatch
     (
         double                  aFoc,
         const ElPackHomologue & aPack,
         Video_Win * aW,
         Pt2dr       aP0W,
         double      aScaleW
            
     )
{
    cOriPlanePatch anOPP(aFoc,aPack,aW,aP0W,aScaleW);

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
