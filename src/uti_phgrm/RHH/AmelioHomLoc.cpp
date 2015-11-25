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
#include "ReducHom.h"

NS_RHH_BEGIN


/*************************************************/
/*                                               */
/*                  cImagH                       */
/*                                               */
/*************************************************/

double  cAppliReduc::ErrorSolLoc()
{
     double aSomEr = 0.0;
     double aSomP  = 0.0;
     for (int aKIm1=0 ; aKIm1<int(mIms.size()) ; aKIm1++)
     {
          cImagH * anI1 = mIms[aKIm1];
          cElHomographie  aCurH1 = anI1->HF()->HomCur();
                
          if (anI1->C2CI())
          {
             double aPdsE = 1 /  anI1->PdsEchant();
             const std::vector<cLink2Img*> & aVL = anI1->VLink();
             for (int aKL=0 ; aKL<int(aVL.size()) ; aKL++)
             {
                 cLink2Img * aLnk = aVL[aKL];
                 cImagH* anI2 = aLnk->Dest();
                 cElHomographie  aCurH2 = anI2->HF()->HomCur();
                 cElHomographie  aCurH2Inv = aCurH2.Inverse();
                 if (anI2->C2CI())
                 {
                    cElHomographie aH12 = aLnk->Hom12();
                    const std::vector<Pt3dr> & anEch = aLnk->EchantP1();

                    for (int aKEch = 0 ; aKEch< int(anEch.size()) ; aKEch++)
                    {
                        Pt3dr  aP3d =   anEch[aKEch];
                        Pt2dr aP1(aP3d.x,aP3d.y);
                        Pt2dr aP2 = aH12.Direct(aP1);
                        double aPds = aP3d.z * aPdsE;

                        Pt2dr aRes = aCurH2Inv.Direct(aCurH1.Direct(aP1)) - aP2;

                        double anEr = square_euclid(aRes);

                        aSomEr+= anEr * aPds;
                        aSomP+= aPds;
                    }
                 }
             }
          }
     }
     return sqrt(aSomEr/aSomP);
}

class cCmpPtrImOnGain
{
    public :
       bool operator () (cImagH * aI1,cImagH * aI2) const
       {
         return aI1->GainLoc() > aI2->GainLoc();
       }
};

 // cEqHomogFormelle   cHomogFormelle

void cAppliReduc::AmelioHomLocal(cImagH & anIm)
{
    double aPdsLVMStd = 0.1;
    double aPdsFreezC = 100;
    double aPdsEvol = 10;;

    int aNbIterSupl =  3;
    int aMaxIterProgr = 6;

    const std::vector<cLink2Img*> & aVLImC = mImCAmel->VLink();

    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        mIms[aKIm]->HF()->ReinitHom(cElHomographie::Id());
        mIms[aKIm]->GainLoc() = 0;
        mIms[aKIm]->InitLoc() = false;
    }

    std::vector<cImagH *> aVIms ;

    for (int aKL=0 ; aKL<int(aVLImC.size()) ; aKL++)
    {
         cLink2Img * aLnK = aVLImC[aKL];
         cImagH * anIm = aLnK->Dest();
         anIm->GainLoc() = aLnK->PdsEchant() + 1e-7;
         anIm->C2CI() = true;
         aVIms.push_back(anIm);
         anIm->HF()->ReinitHom(aLnK->Hom12().Inverse());
    }
    mImCAmel->GainLoc() = 1e10;
    mImCAmel->InitLoc() = true;
    mImCAmel->C2CI() = true;
    int aNbIm2Init = (int)aVIms.size();


   cCmpPtrImOnGain aCmpPtrIm;
   std::sort(aVIms.begin(),aVIms.end(),aCmpPtrIm);

   int aNbIterProgr = ElMin(aMaxIterProgr,round_up(aVIms.size()/3.0));
   int aNbIterTot = aNbIterProgr + aNbIterSupl;

   double aErrorIn = ErrorSolLoc();
   if (Show(eShowGlob))
       std::cout << "ERROR IN " <<  aErrorIn << "\n";
   for (int aNbIter =0 ; aNbIter < aNbIterTot ; aNbIter ++)
   {
        if (aNbIter < aNbIterProgr)
        {
            int aK0 = (aNbIter *aNbIm2Init) / aNbIterProgr;
            int aK1 = ((aNbIter+1) *aNbIm2Init) / aNbIterProgr;
            for (int aKIm=aK0; aKIm<aK1 ; aKIm++)
            {
                ElPackHomologue aPack;
                cImagH * anIm =  aVIms[aKIm];
                const std::vector<cLink2Img*> & aVL = anIm->VLink();
                int aNbInit=0;
                for (int aKL=0 ; aKL<int(aVL.size()) ; aKL++)
                {
                     cLink2Img * aLnK = aVL[aKL];
                     cImagH * anI2 = aLnK->Dest();
                     if (anI2->InitLoc())
                     {
                        const std::vector<Pt3dr> &  anEch = aLnK->EchantP1();
                        cElHomographie aH = anI2->HF()->HomCur() * aLnK->Hom12();

                        for (int aKP=0 ; aKP<int(anEch.size()) ; aKP++)
                        {
                            const Pt3dr & aP3d = anEch[aKP];
                            Pt2dr aP1 (aP3d.x,aP3d.y);
                            double aPds = aP3d.z;
                            Pt2dr aP2 = aH.Direct(aP1);
                            aPack.Cple_Add(ElCplePtsHomologues(aP1,aP2,aPds));
                        }
                        aNbInit++;
                     }
                }
                cElHomographie aNewH(aPack,true);
                anIm->HF()->ReinitHom(aNewH);
                if (Show(eShowDetail)) std::cout << anIm->Name() << " : " << aNbInit << "\n";
            }

            for (int aKIm=aK0; aKIm<aK1 ; aKIm++)
            {
                aVIms[aKIm]->InitLoc() = true;
            }
            if (Show(eShowDetail)) std::cout << "==============================\n";
        }


        if (mDoCompensLoc)
        {
            mSetEq.SetPhaseEquation();

            double aSomEr=0;
            double aSomP=0;


             for (int aKIm1=0 ; aKIm1<int(mIms.size()) ; aKIm1++)
             {
                  cImagH * anI1 = mIms[aKIm1];
                  anI1->AddViscositty((anI1== mImCAmel) ? aPdsFreezC : aPdsLVMStd);
                  cElHomographie  aCurH1 = anI1->HF()->HomCur();
                
                  if (anI1->InitLoc())
                  {
                       double aPdsE = aPdsEvol /  anI1->PdsEchant();
                       const std::vector<cLink2Img*> & aVL = anI1->VLink();
                       for (int aKL=0 ; aKL<int(aVL.size()) ; aKL++)
                       {
                            cLink2Img * aLnk = aVL[aKL];
                            cImagH* anI2 = aLnk->Dest();
                            cElHomographie  aCurH2 = anI2->HF()->HomCur();
                            cElHomographie  aCurH2Inv = aCurH2.Inverse();
                            if (anI2->InitLoc())
                            {
                                double aSomRes = 0;
                                double aSomCtrl = 0;
                                cElHomographie aH12 = aLnk->Hom12();
                                const std::vector<Pt3dr> & anEch = aLnk->EchantP1();
                                cEqHomogFormelle * anEq = aLnk->EqHF();
                                int aNbPts = (int)anEch.size();

                                for (int aKEch = 0 ; aKEch<int(aNbPts) ; aKEch++)
                                {
                                     Pt3dr  aP3d =  anEch[aKEch];
                                     Pt2dr aP1(aP3d.x,aP3d.y);
                                     Pt2dr aP2 = aH12.Direct(aP1);
                                     double aPds = aP3d.z * aPdsE;

                                     Pt2dr aRes = anEq->StdAddLiaisonP1P2(aP1,aP2,aPds,false);
                                     Pt2dr aCtrl = aCurH2Inv.Direct(aCurH1.Direct(aP1)) - aP2;
                                     aSomRes += euclid(aRes);
                                     aSomCtrl += euclid(aCtrl);


                                     double anEr = square_euclid(aRes);

                                     aSomEr+= anEr * aPds;
                                     aSomP+= aPds;
                                }

/*
                            std::cout  << anEq
                                       << " N12=" << anI1->Name() << " " << anI2->Name() 
                                       << " ; RES = " << aSomRes/aNbPts << " Ctrl=" << aSomCtrl/aNbPts << "\n";
*/
                            }
                       }
                  }
              // getchar();

              // anI->HF()->SetModeCtrl(cNameSpaceEqF::eHomFigee);
             }
             if (Show(eShowDetail)) std::cout << "ERR = " << sqrt(aSomEr/aSomP) << "\n";

             mSetEq.SolveResetUpdate();
         }
   }
   for (int aKIm1=0 ; aKIm1<int(mIms.size()) ; aKIm1++)
   {
        cImagH * anI1 = mIms[aKIm1];
        anI1->H2ImC() = anI1->HF()->HomCur();
   }
   double aErrorOut = ErrorSolLoc();
   if (Show(eShowGlob))
      std::cout << "ERROR OUT " <<  aErrorOut << " DELTA=" << aErrorOut - aErrorIn << "\n";
    
    
}





NS_RHH_END


/*
*/
/*
*/




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
