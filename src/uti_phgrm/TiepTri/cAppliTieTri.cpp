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


#include "TiepTri.h"

class cCmpPt2diOnEuclid
{
   public : 
       bool operator () (const Pt2di & aP1, const Pt2di & aP2)
       {
                   return euclid(aP1) < euclid(aP2) ;
       }
};

std::vector<Pt2di> VoisinDisk(double aDistMin,double aDistMax)
{
   std::vector<Pt2di> aResult;
   int aDE = round_up(aDistMax);
   Pt2di aP;
   for (aP.x=-aDE ; aP.x <= aDE ; aP.x++)
   {
       for (aP.y=-aDE ; aP.y <= aDE ; aP.y++)
       {
            double aD = euclid(aP);
            if ((aD <= aDistMax) && (aD>aDistMin))
               aResult.push_back(aP);
       }
   }
   return aResult;
}


cAppliTieTri::cAppliTieTri
(
              cInterfChantierNameManipulateur * anICNM,
              const std::string & aDir,
              const std::string & anOri,
              const cXml_TriAngulationImMaster & aTriang
)  :
     mICNM          (anICNM),
     mDir           (aDir),
     mOri           (anOri),
     mWithW         (false),
     mDisExtrema    (TT_DIST_EXTREMA),
     mDistRechHom   (TT_DIST_RECH_HOM),
     mNivInterac    (0),
     mCurPlan       (Pt3dr(0,0,0),Pt3dr(1,0,0),Pt3dr(0,1,0)),
     mSeuilDensite  (TT_DefSeuilDensiteResul),
     mDefStepDense  (TT_DefStepDense),
     mNbTri         (0),
     mNbPts         (0),
     mTimeCorInit   (0.0),
     mTimeCorDense  (0.0)
{
   mMasIm = new cImMasterTieTri(*this,aTriang.NameMaster());

   for (int aK=0 ; aK<int(aTriang.NameSec().size()) ; aK++)
   {
      mImSec.push_back(new cImSecTieTri(*this,aTriang.NameSec()[aK],aK));
   }

   mVoisExtr = VoisinDisk(0.5,mDisExtrema);
   cCmpPt2diOnEuclid aCmp;
   std::sort(mVoisExtr.begin(),mVoisExtr.end(),aCmp);

   mVoisHom = VoisinDisk(-1,mDistRechHom);

   cSinCardApodInterpol1D * aSinC = new cSinCardApodInterpol1D(cSinCardApodInterpol1D::eTukeyApod,5.0,5.0,1e-4,false);
   mInterpol = new cTabIM2D_FromIm2D<tElTiepTri>(aSinC,1000,false);

}


void cAppliTieTri::DoAllTri(const cXml_TriAngulationImMaster & aTriang)
{
    int aNbTri = aTriang.Tri().size();

    for (int aK=0 ; aK<int(aTriang.Tri().size()) ; aK++)
    {
        DoOneTri(aTriang.Tri()[aK],aK);
        if ( (aK%20)==0)
        {
            std::cout << "Av = "  << (aNbTri-aK) * (100.0/aNbTri) << "% "
                      << " NbP/Tri " << double(mNbPts) / mNbTri
                      << "\n";
        }
    }

    for (int aKT= 0; aKT< int(mVGlobMIRMC.size()) ; aKT++)
    {
        cOneTriMultiImRechCorrel & aTMIRC = mVGlobMIRMC[aKT];
        const std::vector<cResulMultiImRechCorrel<double>*>& aVMC = aTMIRC.VMultiC() ;
        for (int aKP=0 ; aKP<int(aVMC.size()) ; aKP++)
        {
             cResulMultiImRechCorrel<double> & aRMIRC =  *(aVMC[aKP]);
             Pt2dr aPMaster (aRMIRC.PMaster().mPt);
             const std::vector<int> &   aVInd = aRMIRC.VIndex();
             int aNbIm = aVInd.size();
             ELISE_ASSERT(aNbIm==int(aRMIRC.VRRC().size()),"Incoh size in cAppliTieTri::DoAllTri");
             for (int aKI=0 ; aKI<aNbIm ; aKI++)
             {
                 const cResulRechCorrel<double> & aRRC = aRMIRC.VRRC()[aKI];

                 // std::cout << "Corr " << aRRC.mCorrel << " " << aRMIRC.IsInit() << " KT=" << aTMIRC.KT() << "\n";
                 if (aRRC.IsInit())
                 {
                    cImSecTieTri * anIm = mImSec[aVInd[aKI]];
                    
                    anIm->PackH().Cple_Add(ElCplePtsHomologues(aPMaster,aRRC.mPt)) ;
                 }
                 else
                 {
                      ELISE_ASSERT(false,"Incoh init in cAppliTieTri::DoAllTri");
                      // getchar();
                 }
             }
        }
    }

    for (int aKIm=0 ; aKIm<int(mImSec.size()) ; aKIm++)
    {
          
    }
}

void cAppliTieTri::RechHomPtsDense(cResulMultiImRechCorrel<double> & aRMIRC)
{
     std::vector<cResulRechCorrel<double> > & aVRRC = aRMIRC.VRRC();


     for (int aKNumIm = 0 ; aKNumIm <int(aVRRC.size())  ; aKNumIm++)
     {
         cResulRechCorrel<double> & aRRC = aVRRC[aKNumIm];
         int aKIm = aRMIRC.VIndex()[aKNumIm];
         aRRC = mLoadedImSec[aKIm]->RechHomPtsDense(aRMIRC.PMaster().mPt,aRRC);
     }
}

void cAppliTieTri::PutInGlobCoord(cResulMultiImRechCorrel<double> & aRMIRC)
{
     aRMIRC.PMaster().mPt = aRMIRC.PMaster().mPt + mMasIm->Decal();
     std::vector<cResulRechCorrel<double> > & aVRRC = aRMIRC.VRRC();
     for (int aKIm=0 ; aKIm<int(mLoadedImSec.size()) ; aKIm++)
     {
         cResulRechCorrel<double> & aRRC = aVRRC[aKIm];
         aRRC.mPt = aRRC.mPt + Pt2dr(mLoadedImSec[aKIm]->Decal());
     }
}

void cAppliTieTri::DoOneTri(const cXml_Triangle3DForTieP & aTri,int aKT )
{

 // if (505!=aKT) return;
    
    // Verification du triangle  

     // std::cout << "TRI " << aTri.P1() << aTri.P2() << aTri.P3() << "\n";

    // 
    if (!  mMasIm->LoadTri(aTri)) return;

    mNbTri++;

    mCurPlan = cElPlan3D(aTri.P1(),aTri.P2(),aTri.P3());
    mLoadedImSec.clear();
    for (int aKNumIm=0 ; aKNumIm<int(aTri.NumImSec().size()) ; aKNumIm++)
    {
        int aKIm = aTri.NumImSec()[aKNumIm];
        if ( mImSec[aKIm]->LoadTri(aTri))
        {
            mLoadedImSec.push_back(mImSec[aKIm]);
        }
    }

    if (mLoadedImSec.size() == 0)
       return;

    if (0 && (mNivInterac==2))  // Version interactive
    {
         while (mWithW) 
         {
              cIntTieTriInterest aPI= mMasIm->GetPtsInteret();
              for (int aKIm=0 ; aKIm<int(mLoadedImSec.size()) ; aKIm++)
              {
                  mLoadedImSec[aKIm]->RechHomPtsInteretBilin(aPI,mNivInterac);
              }
         }
    }
    else
    {
         const std::list<cIntTieTriInterest> & aLIP =  mMasIm->LIP();
         ElTimer aChrono;
         for (std::list<cIntTieTriInterest>::const_iterator itI=aLIP.begin(); itI!=aLIP.end() ; itI++)
         {
              cResulMultiImRechCorrel<double> * aRMIRC = new cResulMultiImRechCorrel<double>(*itI);
              for (int aKIm=0 ; (aKIm<int(mLoadedImSec.size()))  ; aKIm++)
              {
                  cResulRechCorrel<double> aRes = mLoadedImSec[aKIm]->RechHomPtsInteretBilin(*itI,mNivInterac);
                  if (aRes.IsInit())
                  {
                     aRMIRC->AddResul(aRes,mLoadedImSec[aKIm]->Num());
                  }
              }
              if (aRMIRC->IsInit())
              {
                  mVCurMIRMC.push_back(aRMIRC);
              }
              else
              {
                  delete aRMIRC;
              }
         }
         mTimeCorInit += aChrono.uval();
    }

    FiltrageSpatialRMIRC(mSeuilDensite);

    {
       ElTimer aChrono;
       for (int aKR = 0 ; aKR<int(mVCurMIRMC.size()) ; aKR++)
       {
            RechHomPtsDense(*mVCurMIRMC[aKR]);
       }
       mTimeCorDense += aChrono.uval();
    }


    if (mMasIm->W())
    {
        mMasIm->W()->disp().clik();
    }

    for (int aKp=0 ; aKp<int(mVCurMIRMC.size()) ; aKp++)
    {
        PutInGlobCoord(*mVCurMIRMC[aKp]);
        // mVGlobMIRMC.push_back(mVCurMIRMC[aKp]); 
    }

//   std::cout << "NBPPSS " << mVCurMIRMC.size() << "\n";
    mNbPts += mVCurMIRMC.size();
    mVGlobMIRMC.push_back(cOneTriMultiImRechCorrel(aKT,mVCurMIRMC));
    mVCurMIRMC.clear();
}

class cCmpPtrRMIRC
{
    public :
          bool operator() (cResulMultiImRechCorrel<double> * aRMIRC1, cResulMultiImRechCorrel<double> * aRMIRC2)
          {
               return aRMIRC1->Score() > aRMIRC2->Score();
          }
};

void   cAppliTieTri::FiltrageSpatialRMIRC(const double & aDist)
{
     double aSqDist = aDist *aDist;
     cCmpPtrRMIRC  aCmp;
     std::sort(mVCurMIRMC.begin(),mVCurMIRMC.end(),aCmp);
     std::vector<cResulMultiImRechCorrel<double>*> aNewV;

     for (int aKR1=0 ;aKR1<int(mVCurMIRMC.size()) ; aKR1++)
     {
         cResulMultiImRechCorrel<double> * aR1 = mVCurMIRMC[aKR1];

         if (aR1)
         {
            aNewV.push_back(aR1);
            for (int aKR2=aKR1+1 ; aKR2<int(mVCurMIRMC.size()) ; aKR2++)
            {
                cResulMultiImRechCorrel<double> * aR2 = mVCurMIRMC[aKR2];
                if (aR2 && (aR1->square_dist(*aR2) < aSqDist))
                {
                    mVCurMIRMC[aKR2] = 0;
                    delete aR2;
                }
            }

            if (mMasIm->W())
            {
               Video_Win * aW = mMasIm->W();
               aW->draw_circle_loc(Pt2dr(aR1->PMaster().mPt),aDist,aW->pdisc()(P8COL::yellow));
            }
         }
     }
     mVCurMIRMC = aNewV;
}

void  cAppliTieTri::SetSzW(Pt2di aSzW, int aZoom)
{
    mSzW = aSzW;
    mZoomW = aZoom;
    mWithW = true;
}




cInterfChantierNameManipulateur * cAppliTieTri::ICNM()      {return mICNM;}
const std::string &               cAppliTieTri::Ori() const {return mOri;}
const std::string &               cAppliTieTri::Dir() const {return mDir;}

Pt2di cAppliTieTri::SzW() const {return mSzW;}
int   cAppliTieTri::ZoomW() const {return mZoomW;}
bool  cAppliTieTri::WithW() const {return mWithW;}


cImMasterTieTri * cAppliTieTri::Master() {return mMasIm;}

const std::vector<Pt2di> &   cAppliTieTri::VoisExtr() const { return mVoisExtr; }
const std::vector<Pt2di> &   cAppliTieTri::VoisHom() const { return mVoisHom; }


bool &   cAppliTieTri::Debug() {return mDebug;}
const double &   cAppliTieTri::DistRechHom() const {return mDistRechHom;}

int  &   cAppliTieTri::NivInterac() {return mNivInterac;}
const cElPlan3D & cAppliTieTri::CurPlan() const {return mCurPlan;}
tInterpolTiepTri * cAppliTieTri::Interpol() {return mInterpol;}

double &   cAppliTieTri::SeuilDensite() {return mSeuilDensite;}
int    &   cAppliTieTri::DefStepDense() {return mDefStepDense;}



/***************************************************************************/

cIntTieTriInterest::cIntTieTriInterest(const Pt2di & aP,eTypeTieTri aType) :
   mPt   (aP),
   mType (aType)
{
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
aooter-MicMac-eLiSe-25/06/2007*/
