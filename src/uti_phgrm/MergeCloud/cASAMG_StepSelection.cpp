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


// Recouvrt :
// Soldat-Temple-Hue =>  7.21886
// 12Tets  =>     25.9573
// Lafleur =>  12.8814
// Nubienne => 11.7624

#include "MergeCloud.h"


void cASAMG::InitGlobHisto()
{
    ELISE_COPY
    (
           mImQuality.all_pts().chc(mImQuality.in()),
           1,
           mHisto.histo()
    );

    mNbTot=0;
    ELISE_COPY(mHisto.all_pts(),mHisto.in(),sigma(mNbTot));
}

void cASAMG::InitNewStep(int aCurNiv)
{
   mQualOfNiv = 0;
   mNbOfNiv = 0;
   for (int aNiv= aCurNiv  ; aNiv<= mMaxNivH ; aNiv++)
   {
       mQualOfNiv += mDH[aNiv] * mAppli->GainQual(aNiv);
       mNbOfNiv   += mDH[aNiv];
   }
}

void cASAMG::SuppressPix(const Pt2di &aP,const int & aMarq)
{
    mTLabFin.oset(aP,aMarq);
    int aNiv = mTQual.get(aP);
    mDH[aNiv]--;
}

bool cASAMG::IsCurSelectable() const
{
   return true;
}

void cASAMG::FinishNewStep(int aNiv)
{
    
}


int CptGlob=0;
int CptDet=0;


void cASAMG::SetSelected(int aNivSel,int aNivElim,tMCSom * aSom)
{

    // 1 - On cree le masq de bits des pixel a effacer
    Im2D_Bits<1> aMasq2Sel(mSz.x,mSz.y,0);
    TIm2DBits<1> aTM2Sel(aMasq2Sel);
    
    mNivSelected = aNivSel;
    Pt2di aP;
    for (aP.x=0 ; aP.x < mSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
        {
            if ((mTQual.get(aP) >= aNivElim) &&  (mTLabFin.get(aP)==eLFNoAff))
            {
                 aTM2Sel.oset(aP,1);
                 SuppressPix(aP,eLFMaster);
                 for (int aK=0 ; aK<4 ; aK++)
                 {
                     Pt2di aV = aP + TAB_4_NEIGH[aK];
                     int aLab = mTLabFin.get(aV);
                     if ( (aLab==eLFMasked) || (aLab==eLFMaster))
                     {
                         aTM2Sel.oset(aV,1);
                     }
                 }
            }
        }
    }
    ELISE_COPY(aMasq2Sel.border(1),0,aMasq2Sel.out());

    // 2 - On cree un nuage raster des points a selectionner pour ne faire qu'une fois le passage au 3D
    cRawNuage aNuagMaster(mSz);

    for (aP.x=0 ; aP.x < mSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
        {
            if (aTM2Sel.get(aP))
            {
                aNuagMaster.SetPt(aP,mStdN->PtOfIndex(aP));
            }
        }
    }
    Video_Win * aW = mAppli->Param().VisuSelect().Val() ?  TheWinIm() : 0;
    if (aW)
    {
        InspectQual(false);
        ELISE_COPY
        (
             aMasq2Sel.all_pts(),
             nflag_close_sym(flag_front8(aMasq2Sel.in(0))),
             aW->out_graph(Line_St(aW->pdisc()(P8COL::red)))
        );
        aW->clik_in();
    }
    // Im2D_Bits<1> aMasqInter(mSz.x,mSz.y,0);
    // ELISE_COPY(aMasqInter.all_pts(),rect_som(aMasq2Sel.in(0),1)==9,aMasqInter.out());

    // ElTimer aChrono;
    for (tArcIter itA = aSom->begin(mAppli->SubGrAll()) ; itA.go_on() ; itA++)
    {
          ProjectElim((*itA).s2().attr(),aMasq2Sel,aNuagMaster);
          tMCSom & aS2 = (*itA).s2();
          double aQIn = aS2.attr()->mQualOfNiv;
          double aNbIn = aS2.attr()->mNbOfNiv;
          aS2.attr()->InitNewStep(aNivElim);
          double aQOut = aS2.attr()->mQualOfNiv;
          double aNbOut = aS2.attr()->mNbOfNiv;
          if (mAppli->Param().VisuSelect().Val())
          {
              std::cout << aS2.attr()->mIma->mNameIm << " IN " << aQIn << " "<< aNbIn << " Out " << aQOut << " " << aNbOut << "\n";
          }
    }
    // std::cout << "TIME-Proj=" << aChrono.uval() << "\n";
    // std::cout << "STAT prop: " << double(CptDet)/ CptGlob << " For tot " << CptGlob << "\n";



    InitNewStep(aNivElim);
}

static double Epsilon = 1e-5;

void cASAMG::DoOneTri(const Pt2di & aP0,const Pt2di & aP1,const Pt2di & aP2,const cRawNuage &aNuage,const cRawNuage & aMasterNuage)
{

   CptGlob++;
   Pt3dr aQ0 = aNuage.GetPt(aP0);
     
   double aDP00;
   double anInt00 = InterioriteEnvlop(Pt2di(round_ni(aQ0.x),round_ni(aQ0.y)),aQ0.z,aDP00);

   if (anInt00<  -mAppli->Param().ElimDirectInterior().Val())
       return;

   Pt3dr aQ1 = aNuage.GetPt(aP1);
   Pt3dr aQ2 = aNuage.GetPt(aP2);

   Pt2dr aA0(aQ0.x,aQ0.y);
   Pt2dr aA1(aQ1.x,aQ1.y);
   Pt2dr aA2(aQ2.x,aQ2.y);


   Pt2dr  aV01 =  aA1-aA0;
   Pt2dr  aV02 =  aA2-aA0;

   double aDet = aV01 ^ aV02;

   if (aDet<=0) return;


   Pt2di aPInf = round_down(Inf(aA0,Inf(aA1,aA2)));
   aPInf = Sup(aPInf,Pt2di(0,0));
   Pt2di aPSup = round_up(Sup(aA0,Sup(aA1,aA2)));
   aPSup = Inf(aPSup,mSz-Pt2di(1,1));


   Pt2di anA;
   for (anA.x=aPInf.x ; anA.x<=aPSup.x ; anA.x++)
   {
       for (anA.y=aPInf.y ; anA.y<=aPSup.y ; anA.y++)
       {
           if (mTMasqN.get(anA))
           {
               Pt2dr aV = Pt2dr(anA) - aA0;
               REAL  aPds1 =  (aV ^ aV02) / aDet;
               if (aPds1> -Epsilon)
               {
                     REAL  aPds2 = (aV01 ^ aV) / aDet;
                     if (aPds2 > -Epsilon)
                     {
                        REAL aPds0 = (1-aPds1-aPds2);
                        if ((aPds0 > -Epsilon) && (mTLabFin.get(anA)==eLFNoAff))
                        {
                           double aZ = aPds0 * aQ0.z + aPds1 * aQ1.z + aPds2 * aQ2.z;
                           if (InterioriteEnvlop(anA,aZ,aDP00)>0)
                           {
                              SuppressPix(anA,eLFMasked);
                           }
                        }
                     }
               }
           }
       }
   }

   CptDet++;
}


void cASAMG::ProjectElim(cASAMG * aN2,Im2D_Bits<1> aMasq2Sel,const cRawNuage & aMasterNuage)
{
    // std::cout <<  "ProjectElim " << aN2->mIma->mNameIm << "\n";
    TIm2DBits<1> aTM2Sel(aMasq2Sel);
    cRawNuage aNuagSec(mSz);
   

    // On pre calcule les projections

    {
        Pt2di aP;
        for (aP.x=0 ; aP.x < mSz.x ; aP.x++)
        {
            for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
            {
                if (aTM2Sel.get(aP))
                {
                   Pt3dr aPE = aMasterNuage.GetPt(aP);
                   // Pt3dr aPProj = aN2->mStdN->Euclid2ProfAndIndex(aPE);
                   Pt3dr aPProj = aN2->mStdN->Euclid2ProfPixelAndIndex(aPE);
                   aNuagSec.SetPt(aP,aPProj);
                }
            }
        }
    }
    // On map les triangles 
    Pt2di aP00;
    for (aP00.x=0 ; aP00.x < mSz.x ; aP00.x++)
    {
        for (aP00.y=0 ; aP00.y<mSz.y ; aP00.y++)
        {
            if (aTM2Sel.get(aP00))
            {
                Pt2di aP11 = aP00 + Pt2di(1,1);
                if (aTM2Sel.get(aP11))
                {
                    Pt2di aP10 = aP00 + Pt2di(1,0);
                    if (aTM2Sel.get(aP10))
                        aN2->DoOneTri(aP00,aP10,aP11,aNuagSec,aMasterNuage);
                    Pt2di aP01 = aP00 + Pt2di(0,1);
                    if (aTM2Sel.get(aP01))
                        aN2->DoOneTri(aP00,aP11,aP01,aNuagSec,aMasterNuage);
                }
            }
        }
    }

    Video_Win * aW = mAppli->Param().VisuElim().Val() ?  TheWinIm() : 0;
    if (aW)
    {
        aN2->InspectQual(false);
        ELISE_COPY
        (
             select(aN2->mImLabFin.all_pts(),aN2->mImLabFin.in()!=eLFNoAff),
             1+aN2->mImLabFin.in(),
             aW->odisc()
        );
        //Tiff_Im::Create8BFromFonc("TestProj.tif",mSz,aN2->mImLabFin.in()+aN2->mMasqN.in());
        aW->clik_in();
        //getchar();
    }

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
