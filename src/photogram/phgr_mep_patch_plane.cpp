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
        cOriPlanePatch( const ElPackHomologue & aPack,
                          Video_Win * aW,
                          Pt2dr       aP0W,
                          double      aScaleW
                      );
         void operator()(tSomOPP*,tSomOPP*,bool);  // Delaunay call back

    private  :
         void ShowPoint(const Pt2dr &,double aRay,int coul) const;
         void ShowPoint(const tSomOPP &,double aRay,int coul) const;
         void ShowSeg(const Pt2dr & aP1,const Pt2dr& aP2,int aCoul) const;
         Pt2dr ToW(const Pt2dr & aP) const;



         Video_Win *            mW;
         Pt2dr                  mP0W;
         double                 mScaleW;
         std::vector<tSomOPP *> mVSom;
         tGrOPP                 mGrOPP;
};

    //  ==============  Graphisme   ================

Pt2dr cOriPlanePatch::ToW(const Pt2dr & aP) const { return (aP-mP0W) *mScaleW; }
void cOriPlanePatch::ShowPoint(const Pt2dr & aP,double aRay,int aCoul) const
{
   if (mW) 
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
}

cOriPlanePatch::cOriPlanePatch
( 
      const ElPackHomologue & aPack,
      Video_Win * aW,
      Pt2dr       aP0W,
      double      aScaleW
)  :
   mW       (aW),
   mP0W     (aP0W),
   mScaleW  (aScaleW)
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



    std::cout << "Time Delaunay " << aChrono.uval() << "\n";
    getchar();
}




void TestOriPlanePatch
     (
         const ElPackHomologue & aPack,
         Video_Win * aW,
         Pt2dr       aP0W,
         double      aScaleW
            
     )
{
    cOriPlanePatch anOPP(aPack,aW,aP0W,aScaleW);

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
