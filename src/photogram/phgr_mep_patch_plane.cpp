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
         void TestOneHomogr(std::vector<tSomOPP*>  & aVSom);

    private  :
         void ShowPoint(const Pt2dr &,double aRay,int coul) const;
         void ShowPoint(const tSomOPP &,double aRay,int coul) const;
         void ShowSeg(const Pt2dr & aP1,const Pt2dr& aP2,int aCoul) const;
         Pt2dr ToW(const Pt2dr & aP) const;
         Pt2dr FromW(const Pt2dr & aP) const;
         tSomOPP * GetPt(int aCoul);



         double                 mFoc;
         Video_Win *            mW;
         Pt2dr                  mP0W;
         double                 mScaleW;
         std::vector<tSomOPP *> mVSom;
         tGrOPP                 mGrOPP;
         int                    mFlagVisitH;
         L2SysSurResol          mSysHom;
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



tSomOPP * cOriPlanePatch::GetPt(int aCoul)
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



void cOriPlanePatch::TestOneHomogr(std::vector<tSomOPP*>  & aVSom)
{
    mSysHom.Reset();
    
}








void cOriPlanePatch::TestHomogr()
{
    Pt2dr aC (0,0);
    ElPackHomologue aPack;
    for (int aK=0 ; aK<4 ; aK++)
    {
        ElCplePtsHomologues   aCple = GetPt(P8COL::white)->attr().Cple();
        aPack.Cple_Add(aCple);
        aC  = aC + aCple.P1();
    }
    aC = aC /4;

    double aDMax = 6;
    double ErrProp = 1e-2;


    ElTimer aChrono;

    for (int aNbIt = 0 ; aNbIt<4 ; aNbIt ++)
    {
        cElHomographie aHom(aPack,true);
        ElPackHomologue  aNewPack;

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
                aNewPack.Cple_Add(aCple);
                aCoul = P8COL::yellow;
             }
             // if (aResidu < 3) aCoul = P8COL::white;

             ShowPoint(aP1,3.0,aCoul);
        }

        // std::cout << "END IT " << aNbIt << "\n";
        aNewPack = aPack;
    }
    std::cout << "TTT " << aChrono.uval() << "\n";
}




void  cOriPlanePatch::TestPt()
{
    GetPt(P8COL::yellow);
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
   mW           (aW),
   mP0W         (aP0W),
   mScaleW      (aScaleW),
   mFlagVisitH  (mGrOPP.alloc_flag_som()),
   mSysHom      (8)
{
    // if (mW) mW->clear();
    for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    {
         tSomOPP & aSom = mGrOPP.new_som(cAttrSomOPP(itP->P1(),itP->P2(),itP->Pds()));
         mVSom.push_back(&aSom);
         if (mW) ShowPoint(aSom,3.0,P8COL::green);
    }

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
