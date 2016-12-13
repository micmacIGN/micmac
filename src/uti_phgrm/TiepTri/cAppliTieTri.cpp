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

/*
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
*/


cAppliTieTri::cAppliTieTri
(
   const cParamAppliTieTri & aParam,
   cInterfChantierNameManipulateur * anICNM,
   const std::string & aDir,
   const std::string & anOri,
   const cXml_TriAngulationImMaster & aTriang
)  :
     cParamAppliTieTri (aParam),
     mICNM          (anICNM),
     mDir           (aDir),
     mOri           (anOri),
     mWithW         (false),
     mDisExtrema    (TT_DIST_EXTREMA),
     mDistRechHom   (TT_DIST_RECH_HOM),
     mNivInterac    (0),
     mCurPlan       (Pt3dr(0,0,0),Pt3dr(1,0,0),Pt3dr(0,1,0)),
     mNbTriLoaded   (0),
     mNbPts         (0),
     mTimeCorInit   (0.0),
     mTimeCorDense  (0.0),
     mHasPtSelecTri (false),
     mHasNumSelectImage (false),
     mKeyMasqIm         ("NKS-Assoc-STD-Masq")
{
   mMasIm = new cImMasterTieTri(*this,aTriang.NameMaster());

   for (int aK=0 ; aK<int(aTriang.NameSec().size()) ; aK++)
   {
      mImSec.push_back(new cImSecTieTri(*this,aTriang.NameSec()[aK],aK));
   }

   mVoisExtr = SortedVoisinDisk(0.5,mDisExtrema,true);
   mVoisHom = SortedVoisinDisk(-1,mDistRechHom,false);

   cSinCardApodInterpol1D * aSinC = new cSinCardApodInterpol1D(cSinCardApodInterpol1D::eTukeyApod,5.0,5.0,1e-4,false);
   mInterpolSinC = new cTabIM2D_FromIm2D<tElTiepTri>(aSinC,1000,false);


   mInterpolBilin = new cInterpolBilineaire<tElTiepTri>;
   mInterpolBicub = new cTplCIKTabul<tElTiepTri,tElTiepTri>(10,8,-0.5);


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
                      << " NbP/Tri " << double(mNbPts) / mNbTriLoaded
                      << "\n";
        }
    }
    std::cout << "NB TRI LOADED = " << mNbTriLoaded << "\n";

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

                 //std::cout << "Corr " << aRRC.mCorrel << " " << aRMIRC.IsInit() << " KT=" << aTMIRC.KT() << "\n";
                 if (aRRC.IsInit())
                 {
                    cImSecTieTri * anIm = mImSec[aVInd[aKI]];
                    anIm->PackH().Cple_Add(ElCplePtsHomologues(aPMaster,aRRC.mPt)) ;
                 }
                 else
                 {
                      // Ce cas peut sans doute se produire legitimment si la valeur entiere etait trop proche
                      // de la frontiere et que toute les  tentative etaient en dehors
                      //ELISE_ASSERT(false,"Incoh init in cAppliTieTri::DoAllTri");
                      // getchar();
                 }
             }
        }
    }

    cout<<"Write pts homo to disk:..."<<endl;
    for (int aKIm=0 ; aKIm<int(mImSec.size()) ; aKIm++)
    {
        cImSecTieTri* aImSec = mImSec[aKIm];
        cout<<"  ++ Im2nd : "<<aImSec->Num();
        cout<<" - Nb Pts= "<<aImSec->PackH().size()<<endl;
        std::string pic1 = Master()->NameIm();
        std::string pic2 = aImSec->NameIm();
        cHomolPackTiepTri aPack(pic1, pic2, aKIm, mICNM);
        aPack.Pack() = aImSec->PackH();
        std::string aHomolOut = "_TiepTri";
        aPack.writeToDisk(aHomolOut);
    }
}

void cAppliTieTri::RechHomPtsDense(cResulMultiImRechCorrel<double> & aRMIRC)
{
     std::vector<cResulRechCorrel<double> > & aVRRC = aRMIRC.VRRC();


     for (int aKNumIm = 0 ; aKNumIm <int(aVRRC.size())  ; aKNumIm++)
     {
         cResulRechCorrel<double> & aRRC = aVRRC[aKNumIm];
         int aKIm = aRMIRC.VIndex()[aKNumIm];

         // aRRC = mImSecLoaded[aKIm]->RechHomPtsDense(aRMIRC.PMaster().mPt,aRRC);
         aRRC = mImSec[aKIm]->RechHomPtsDense(aRMIRC.PMaster().mPt,aRRC);
     }
}

void cAppliTieTri::PutInGlobCoord(cResulMultiImRechCorrel<double> & aRMIRC)
{
     aRMIRC.PMaster().mPt = aRMIRC.PMaster().mPt + mMasIm->Decal();
     std::vector<cResulRechCorrel<double> > & aVRRC = aRMIRC.VRRC();
     for (int aKNumIm=0 ; aKNumIm<int(aVRRC.size()) ; aKNumIm++)
     {
         cResulRechCorrel<double> & aRRC = aVRRC[aKNumIm];
         int aKIm = aRMIRC.VIndex()[aKNumIm];
         aRRC.mPt = aRRC.mPt + Pt2dr(mImSec[aKIm]->Decal());
     }
}

void cAppliTieTri::DoOneTri(const cXml_Triangle3DForTieP & aTri,int aKT )
{

 // if (505!=aKT) return;

    // Verification du triangle

     // std::cout << "TRI " << aTri.P1() << aTri.P2() << aTri.P3() << "\n";

    if (!  mMasIm->LoadTri(aTri)) return;

    mNbTriLoaded++;

    mCurPlan = cElPlan3D(aTri.P1(),aTri.P2(),aTri.P3());
    mImSecLoaded.clear();
    for (int aKNumIm=0 ; aKNumIm<int(aTri.NumImSec().size()) ; aKNumIm++)
    {
        int aKIm = aTri.NumImSec()[aKNumIm];
        if ( mImSec[aKIm]->LoadTri(aTri))
        {
            mImSecLoaded.push_back(mImSec[aKIm]);
        }
    }

    if (mImSecLoaded.size() == 0)
       return;

    if (mNivInterac==2)  // Version interactive
    {
         while (mWithW)
         {
              cIntTieTriInterest aPI= mMasIm->GetPtsInteret();
              for (int aKIm=0 ; aKIm<int(mImSecLoaded.size()) ; aKIm++)
              {
                  mImSecLoaded[aKIm]->RechHomPtsInteretBilin(aPI,mNivInterac);  //1pxl/2 -> pxl entier-> sub pxl
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
              for (int aKIm=0 ; (aKIm<int(mImSecLoaded.size()))  ; aKIm++)
              {
                  cResulRechCorrel<double> aRes = mImSecLoaded[aKIm]->RechHomPtsInteretBilin(*itI,mNivInterac);
                  if (aRes.IsInit())
                  {
                     aRMIRC->AddResul(aRes,mImSecLoaded[aKIm]->Num());
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

    FiltrageSpatialRMIRC(mDistFiltr);

    {
       ElTimer aChrono;
       for (int aKR = 0 ; aKR<int(mVCurMIRMC.size()) ; aKR++)
       {
            RechHomPtsDense(*(mVCurMIRMC[aKR]));    //recherche dense with Interpolation sin, 0.125->1/32
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


void cAppliTieTri::SetPtsSelect(const Pt2dr & aP)
{
    mHasPtSelecTri = true;
    mPtsSelectTri = aP;
}

bool cAppliTieTri::HasPtSelecTri() const {return mHasPtSelecTri;}
const Pt2dr & cAppliTieTri::PtsSelectTri() const {return mPtsSelectTri;}

bool cAppliTieTri::NumImageIsSelect(const int aNum) const
{
   if (!mHasNumSelectImage) return true;
   return BoolFind(mNumSelectImage,aNum);
}


void cAppliTieTri::SetNumSelectImage(const std::vector<int> & aVNum)
{
     mHasNumSelectImage = true;
     mNumSelectImage = aVNum;
}


const std::string &  cAppliTieTri::KeyMasqIm() const
{
    return mKeyMasqIm;
}
void cAppliTieTri::SetMasqIm(const  std::string  & aKeyMasqIm)
{
    mKeyMasqIm = aKeyMasqIm;
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

tInterpolTiepTri * cAppliTieTri::Interpol() 
{
   if (mNumInterpolDense==0) return mInterpolBilin;
   if (mNumInterpolDense==1) return mInterpolBicub;
   if (mNumInterpolDense==2) return mInterpolSinC;

   ELISE_ASSERT(false,"AppliTieTri::Interp");
   return 0;
}




/***************************************************************************/

cIntTieTriInterest::cIntTieTriInterest(const Pt2di & aP,eTypeTieTri aType,const double & aQualFast) :
   mPt       (aP),
   mType     (aType),
   mFastQual (aQualFast),
   mSelected (true)
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
