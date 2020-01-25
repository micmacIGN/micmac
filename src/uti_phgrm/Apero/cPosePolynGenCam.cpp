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

#include "Apero.h"


/***********************************************************************/
/*                                                                     */
/*                      cGenPoseCam                                    */
/*                                                                     */
/***********************************************************************/


cGenPoseCam::cGenPoseCam(cAppliApero & anAppli,const std::string & aName) :
    mAppli (anAppli),
    mName  (aName),
    mNumTmp(-12345678),
    mCdtImSec (0),
    mRotIsInit (false),
    mPreInit   (false),
    mOrIntM2C     (ElAffin2D::Id()),
    mOrIntC2M     (ElAffin2D::Id()),
    mMasqH       (mAppli.MasqHom(aName)),
    mTMasqH      (mMasqH ? new  TIm2DBits<1>(*mMasqH) : 0),
    mSomPM       (0),
    mLastEstimProfIsInit (false),
    mLasEstimtProf       (-1),
    mNbPtsMulNN   (-1),
    mNbPLiaisCur  (0),
    mCurLayer      (0),
    mSom           (0)
{
   tGrApero::TSom & aSom = mAppli.Gr().new_som(this);
   SetSom(aSom);
}

void cGenPoseCam::SetSom(tGrApero::TSom & aSom)
{
   mSom = & aSom;
}

tGrApero::TSom * cGenPoseCam::Som()
{
   return mSom;
}

void cGenPoseCam::UseRappelOnPose() const 
{
}



bool cGenPoseCam::PreInit() const { return mPreInit; }
bool cGenPoseCam::RotIsInit()  const { return mRotIsInit; }

cAnalyseZoneLiaison  &  cGenPoseCam::AZL() {return mAZL;}
double               &  cGenPoseCam::QualAZL() {return mQualAZL;}



int     &  cGenPoseCam::NbPLiaisCur() {return mNbPLiaisCur;}


bool cGenPoseCam::HasMasqHom() const { return mMasqH !=0; }

void    cGenPoseCam::VirtualInitAvantCompens()
{
}

void    cGenPoseCam::InitAvantCompens()
{
    VirtualInitAvantCompens();
    if (PMoyIsInit())
    {
       mLasEstimtProf =   ProfMoyHarmonik();
       mLastEstimProfIsInit = true;
    }
    mPMoy = Pt3dr(0,0,0);
    mMoyInvProf =0;
    mSomPM = 0;
}

double  cGenPoseCam::SomPM() const {return mSomPM;}

bool     cGenPoseCam::PMoyIsInit() const
{
   return mSomPM != 0;
}

Pt3dr   cGenPoseCam::GetPMoy() const
{
   ELISE_ASSERT(PMoyIsInit(),"cPoseCam::GetPMoy");
   return mPMoy / mSomPM;
}


double   cGenPoseCam::ProfMoyHarmonik() const
{
   ELISE_ASSERT(PMoyIsInit(),"cPoseCam::ProfMoyHarmonik");
   return 1.0 / (mMoyInvProf/mSomPM);
}


void  cGenPoseCam::SetNbPtsMulNN(int aNbNN)
{
  mNbPtsMulNN = aNbNN;
}

int   cGenPoseCam::NbPtsMulNN() const
{
   return mNbPtsMulNN;
}


void cGenPoseCam::VirtualAddPMoy
     (
           const Pt2dr & aPIm,
           const Pt3dr & aP,
           int aKPoseThis,
           const std::vector<double> * aVPds,
           const std::vector<cGenPoseCam*>*
     )
{
}



void    cGenPoseCam::AddPMoy
        (
             const Pt2dr & aPIm,
             const Pt3dr & aP,
             double aBSurH,
             int aKPoseThis,
             const std::vector<double> * aVPds,
             const std::vector<cGenPoseCam*> *aVPose
        )
{

   double aPds = (aBSurH-mAppli.Param().LimInfBSurHPMoy().Val());
   aPds /= (mAppli.Param().LimSupBSurHPMoy().Val() - mAppli.Param().LimInfBSurHPMoy().Val());
   if (aPds<0) return;

    const cBasicGeomCap3D * aCS = GenCurCam() ;
    if (mTMasqH)
    {
      Pt2di aPIm =  round_ni(aCS->Ter2Capteur(aP));

      if (! mTMasqH->get(aPIm,0))
         return;
    }
    double aProf = aCS->ProfondeurDeChamps(aP);


    mPMoy = mPMoy + aP * aPds;
    mMoyInvProf  += (1/aProf) * aPds;
    mSomPM  += aPds ;

    VirtualAddPMoy(aPIm,aP,aKPoseThis,aVPds,aVPose);
}


int & cGenPoseCam::NumTmp()
{
   return mNumTmp;
}

cPoseCdtImSec *  & cGenPoseCam::CdtImSec() {return mCdtImSec;}



const  std::string & cGenPoseCam::Name() const {return mName;}

cPoseCam * cGenPoseCam::DownCastPoseCamNN()
{
    cPoseCam * aRes =  DownCastPoseCamSVP();
    ELISE_ASSERT(aRes!=0,"cGenPoseCam::DownCastPoseCamNN");
    return aRes;
}

const cPoseCam * cGenPoseCam::DownCastPoseCamNN() const
{
    const cPoseCam * aRes =  DownCastPoseCamSVP();
    ELISE_ASSERT(aRes!=0,"cGenPoseCam::DownCastPoseCamNN");
    return aRes;
}

cPoseCam * cGenPoseCam::DownCastPoseCamSVP()
{
   return 0;
}
const cPoseCam * cGenPoseCam::DownCastPoseCamSVP() const
{
   return 0;
}


cCalibCam *  cGenPoseCam::CalibCam() const {return 0;}

cCalibCam *  cGenPoseCam::CalibCamNN()  const
{
    cCalibCam * aRes = CalibCam() ;
    ELISE_ASSERT(aRes!=0,"cGenPoseCam::CalibCamNN");
    return aRes;
}

const cBasicGeomCap3D * cGenPoseCam::GenCurCam () const { return PDVF()->GPF_CurBGCap3D(); }
cBasicGeomCap3D * cGenPoseCam::GenCurCam ()  {return  PDVF()->GPF_NC_CurBGCap3D();}



bool  cGenPoseCam::IsInZoneU(const Pt2dr & aP) const
{
    return GenCurCam()->CaptHasData(aP);
}


void cGenPoseCam::ResetStatR()
{
  mStatRSomP =0;
  mStatRSomPR =0;
  mStatRSom1 =0;
}

void cGenPoseCam::AddStatR(double aPds,double aRes)
{
  mStatRSomP += aPds;
  mStatRSomPR += aPds * aRes;
  mStatRSom1 += 1;
}

void cGenPoseCam::GetStatR(double & aSomP,double & aSomPR,double & aSom1) const
{
   aSomP = mStatRSomP;
   aSomPR = mStatRSomPR;
   aSom1  = mStatRSom1;
}


Pt3dr cGenPoseCam::CurCentreOfPt(const Pt2dr & aPt) const
{
    return GenCurCam()->OpticalCenterOfPixel (aPt);
}

void cGenPoseCam::Trace() const
{
}

const ElAffin2D &  cGenPoseCam::OrIntM2C() const
{
   return mOrIntM2C;
}
const ElAffin2D &  cGenPoseCam::OrIntC2M() const
{
   return mOrIntC2M;
}


void cGenPoseCam::ResetPtsVu()
{
   mPtsVu.clear();
}

void cGenPoseCam::AddPtsVu(const Pt3dr & aP)
{
   mPtsVu.push_back(aP);
}


const std::vector<Pt3dr> &  cGenPoseCam::PtsVu() const
{
    return mPtsVu;
}

bool cGenPoseCam::CanBeUSedForInit(bool OnInit) const
{
   return OnInit ? RotIsInit() : PreInit() ;
}


cOneImageOfLayer * cGenPoseCam::GetCurLayer()
{
   if (mCurLayer==0)
   {
      std::cout << "FOR NAME POSE " << mName << "\n";
      ELISE_ASSERT(false,"Cannot get layer");
   }
   return mCurLayer;
}

void cGenPoseCam::SetCurLayer(cLayerImage * aLI)
{
    mCurLayer = aLI->NamePose2Layer(mName);
}


void cGenPoseCam::C2MCompenseMesureOrInt(Pt2dr & aPC)
{
   aPC = mOrIntC2M(aPC);
}

bool cGenPoseCam::AcceptPoint(const Pt2dr & aP) const
{
   return true;
}





/***********************************************************************/
/*                                                                     */
/*                      cPosePolynGenCam                               */
/*                                                                     */
/***********************************************************************/

cPosePolynGenCam::cPosePolynGenCam(cAppliApero & anAppli,const std::string & aNameIma,const std::string & aDirOri) :
    cGenPoseCam (anAppli,aNameIma),
    mNameOri    (mAppli.ICNM()->Assoc1To1( "NKS-Assoc-Im2GBOrient@"+aDirOri,mName,true)),
    mCam        (cPolynomial_BGC3M2D::NewFromFile(mNameOri)),
    mCamF       (mAppli.SetEq(),*mCam,false,false,false)
{
    
    mRotIsInit = true;
    mPreInit = true;
}

cGenPDVFormelle *  cPosePolynGenCam::PDVF() { return & mCamF; }
const cGenPDVFormelle *  cPosePolynGenCam::PDVF() const { return & mCamF; }


cPolynBGC3M2D_Formelle *   cPosePolynGenCam::PolyF() 
{
   return &mCamF;
}

Pt2di cPosePolynGenCam::SzCalib() const 
{
   return mCam->SzBasicCapt3D();
}


/***********************************************************************/
/*                                                                     */
/*                      cAppliApero                                    */
/*                                                                     */
/***********************************************************************/

void cAppliApero::AddObservationsContrCamGenInc
     (
           const std::list<cContrCamGenInc> & aLC,
           bool IsLastIter,
           cStatObs & aSO
     )
{
     for (std::list<cContrCamGenInc>::const_iterator itC=aLC.begin() ; itC!=aLC.end() ; itC++)
     {
            AddObservationsContrCamGenInc(*itC,IsLastIter,aSO);
     }
}

void cAppliApero::AddObservationsContrCamGenInc
     (
         const cContrCamGenInc & aCCIG,
         bool IsLastIter,
         cStatObs & aSO
      )
{
    cSetName *  aSelector = mICNM->KeyOrPatSelector(aCCIG.PatternApply());

    for (int aK=0 ; aK<int(mVecPolynPose.size()) ; aK++)
    {
        cPosePolynGenCam * aGPC = mVecPolynPose[aK];
        if (aSelector->IsSetIn(aGPC->Name()))
        {
             cPolynBGC3M2D_Formelle * aPF = aGPC->PolyF();

             if (aCCIG.PdsAttachToId().IsInit())
             {
                 aPF->AddEqAttachGlob
                 (
                     aCCIG.PdsAttachToId().Val()*aGPC->SomPM(),
                     false,
                     20,
                     (CamStenope *) 0
                 );
             }

             if (aCCIG.PdsAttachToLast().IsInit())
             {
                 aPF->AddEqAttachGlob
                 (
                     aCCIG.PdsAttachToLast().Val()*aGPC->SomPM(),
                     true,
                     20,
                     (CamStenope *) 0
                 );
             }

             if (aCCIG.PdsAttachRGLob().IsInit())
             {
                 aPF->AddEqRotGlob(aCCIG.PdsAttachRGLob().Val()*aGPC->SomPM());
             }

        }
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
