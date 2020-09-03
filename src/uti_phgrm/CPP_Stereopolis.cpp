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
#include "BlockCam.h"

static const Pt3di  PColRed(255,0,0);
static const Pt3di  PColGreen(0,255,0);
static const Pt3di  PColBlue(0,0,255);
static const Pt3di  PColBlack(0,0,0);
static const Pt3di  PColOrange(255,128,0);
static const Pt3di  PColCyan(0,0,255);
static const Pt3di  PColGrayMed(128,128,12);

typedef std::map<std::string,std::pair<Pt3dr,Pt3dr>> tMapInc;

// std::map<std::string,std::pair<Pt3dr,Pt3dr>>  
tMapInc StdGetSCenterOPK(const std::string &  aDir);

    // StdGetSCenterOPK("Ori-4MesInc/");

/*****************************************/
/*                                       */
/*                ::                     */
/*                                       */
/*****************************************/


void CreatPlyOnePt(const std::string& aName, const Pt3dr & aPt,const Pt3di aCoul,double aRay,int aNbByRay)
{
    cPlyCloud        aPlyFile;
    aPlyFile.AddSphere(aCoul,aPt,aRay,aNbByRay);
    aPlyFile.PutFile(aName);
}

class cBlockATS
{
    public :
        cBlockATS(const cBlockATS *) = delete;
        cBlockATS(cGS_1BlC *);
        const ElRotation3D & OriC2S() const {return mOriC2S;}
        const Pt3dr & LevA() const {return mLevA;}
        const Pt3dr & Pos() const {return mOriC2W.tr();}
        const ElMatrix<double> &  BoreS() const {return mBoreS;}
        void  CalculNext(const cBlockATS & aNext);

        // Calcul le boresight relatif
        void  SetEstimBoreS(const  ElMatrix<double> & aMat);
        void  SetInc(const tMapInc &);

        const Pt3dr & OPK_V() const {return mOPK;}   // Vehicule
              Pt3dr   OPK_W() const {return   mOriC2W.Mat() * mOPK;}   // Word
        const cGS_1BlC & StdBl() const {return *mStdBl;}
        const std::string & NameIm() const {return mStdBl->mCamSec->mName;}
        bool   WithInc () const {return mWithInc;}
        Pt3dr  IncXYZ () const {return mIncXYZ;}
        Pt3dr  IncOPK () const {return mIncOPK;}

        double PdsPos() const  {return mPdsPos;}
        void  SetPdsPos(double aPds)  {mPdsPos=aPds;}
        double PdsOri() const  {return mPdsOri;}
        void  SetPdsOri(double aPds)  {mPdsOri=aPds;}
        Pt3di StdCoul() {return (mPdsPos >0) ? PColGreen : PColRed;}
    private :
        cGS_1BlC *     mStdBl;
        ElRotation3D   mOriC2W;  // Orientation Centrale 3 World
        ElRotation3D   mOriS2W;  // Orientation Secondary = cam
        ElRotation3D   mOriC2S;  // Orientation Centrale relatively to Cam 

        // ElRotation3D   mOriCNext2C;  // Orientation Centrale 3 World
        // ElRotation3D   mOriSNext2S;  // Orientation Centrale 3 World

       // Relative position of C(=central) in S (=second=camera) repair
        Pt3dr             mLevA;  // alwo known as lever arm
        ElMatrix<double>  mBoreS;  // alwo known as boresight

       // Handles Boresigth As  
        Pt3dr             mOPK;  // Omega Phi Kapa as delta between Boresight and average Boresight

        //  Inc on orients
        bool   mWithInc;
        Pt3dr  mIncXYZ;
        Pt3dr  mIncOPK;
        double mPdsPos;
        double mPdsOri;
};

void  cBlockATS::CalculNext(const cBlockATS & aNext)
{
/*
    mOriCNext2C  = mOriC2W.inv() * aNext.mOriC2W;
    mOriSNext2S  = mOriS2W.inv() * aNext.mOriS2W;
    std::cout <<  "TTRRRRRR "<< mOriCNext2C.tr() << mOriSNext2S.tr() << "\n";
*/
}
  
void  cBlockATS::SetInc(const tMapInc & aMap)
{
    tMapInc::const_iterator anIt = aMap.find(NameIm());

    if (anIt==aMap.end())
    {
       return;
    }
    mWithInc = true;
    mIncXYZ = anIt->second.first;
    mIncOPK = anIt->second.second;

//  std::cout << "JJJJ " << (anIt==aMap.end()) << " <- " << NameIm() << mIncXYZ << mIncOPK << "\n";
}

cBlockATS::cBlockATS(cGS_1BlC * aBl) :
    mStdBl       (aBl),
    mOriC2W      (mStdBl->mCamC->mCamS->Orient().inv()),
    mOriS2W      (mStdBl->mCamSec->mCamS->Orient().inv()),
    mOriC2S      (mOriS2W.inv() * mOriC2W),
    // mOriCNext2C  (ElRotation3D::Id),
    // mOriSNext2S  (ElRotation3D::Id),
    mLevA        (mOriC2S.tr()),
    mBoreS       (mOriC2S.Mat()),
    mWithInc     (false),
    mPdsPos      (1.0),
    mPdsOri      (1.0)
{
}

void  cBlockATS::SetEstimBoreS(const  ElMatrix<double> & aMat)
{
     if (true)
     {
     // BOres  = RCam-1 * RIns
     //   RCam  * BOres  = RIns
     //   RCam * BOres  = RIns * OPK
     //   OPK = RIns -1 *  RCam * BOres 
    
        ElMatrix<double> aRes = mOriC2W.inv().Mat() * mOriS2W.Mat()   * aMat ;
        ElRotation3D aR(Pt3dr(0,0,0),aRes,true);
        mOPK = Pt3dr(aR.teta12(),aR.teta02(),aR.teta01());

        // mOPK =  mOriC2W.Mat() * mOPK;
     }
     else
     {
     }
     
}


class cAnalyseTrajStereopolis : public cGS_Appli
{
    public :
        cAnalyseTrajStereopolis(int argc,char ** argv);
    private :
        template <class Type>  void Stat(const Type & Fctr,const std::string &Name);
        template <class Type,class TypeSel>  void StatWithSel(const Type & Fctr,const TypeSel & aSel,const std::string &Name);
        
        double ScoreOneLA(const Pt3dr &) const;  // Score 4 One LevArm, using sum euclid dist
        double ScoreOneBoreS(const ElMatrix<double> &) const;
        std::vector<cBlockATS*> mVBlATS;
        cPlyCloud              mPly_LACloud;
        cPlyCloud              mPly_TrajCOpt;
        cPlyCloud              mPly_TrajSeg;
        cPlyCloud              mPly_TrajLevArm;
        cPlyCloud              mPly_TrajBoreS_V;
        cPlyCloud              mPly_TrajBoreS_W;
        cPlyCloud              mPly_TrajNameCam;

        cPlyCloud              mPly_TrajIncPos;
        cPlyCloud              mPly_TrajIncOPK;
        Pt3dr                  mLAEstim;
        ElMatrix<double>       mBoresEstim;
        std::map<std::string,std::pair<Pt3dr,Pt3dr>>  mMapInc;
        bool                                          mWithInc;
        cMasqBin3D *                                  mMasq3D;
};

template <class Type,class TypeSel>  
    void cAnalyseTrajStereopolis::StatWithSel
         (
             const Type & aFctr,
             const TypeSel & aSel,
             const std::string &Name
         )
{
    std::vector<Pt3dr> aVP;
    Pt3dr aSomP(0,0,0);
    int aNbSel = 0;
    for (const auto & aBl : mVBlATS)
    {
        if (aSel(aBl))
        {
            Pt3dr aP = aFctr(*aBl);
            aP = Pt3dr(ElAbs(aP.x),ElAbs(aP.y),ElAbs(aP.z));
            aVP.push_back(aP);
            aSomP  =  aSomP + aP;
            aNbSel ++;
        }
    }

    ELISE_ASSERT(aNbSel!=0,"No Block in StatWithSel");
    aSomP = aSomP / aNbSel;
    std::cout << "========== STAT FOR " << Name << " =================\n";
    std::cout <<  "   MoyAbs= " << euclid(aSomP) << " " <<  aSomP << "\n";
    Pt3dr aPMed = P3DMed(aVP,[](const Pt3dr & aP) {return aP;});
    std::cout <<  "   MedAbs= " << euclid(aPMed) << " " <<  aPMed << "\n";
}

bool SelPds(cBlockATS * aBl) {return aBl->PdsPos()>0;}
bool SelInc(cBlockATS * aBl) {return aBl->WithInc() && (aBl->PdsPos()>0);}
template <class Type>  void cAnalyseTrajStereopolis::Stat(const Type & Fctr,const std::string &Name)
{
    StatWithSel(Fctr,[](cBlockATS *) {return true;},Name);
}
        // template <class Type,class Sel>  void StatWithSel(const Type & Fctr,const Type & aSel,const std::string &Name);

double cAnalyseTrajStereopolis::ScoreOneLA(const Pt3dr & aLA) const
{
   double aSomEcP = 0;
   double aSomP = 0;
   for (const auto & aBl : mVBlATS)
   {
       double aPds = aBl->PdsPos();
       aSomP += aPds;
       aSomEcP += euclid(aLA-aBl->LevA()) * aPds;
   }

   return aSomEcP / aSomP;
}

double cAnalyseTrajStereopolis::ScoreOneBoreS(const  ElMatrix<double> & aBoreS) const
{
   double aRes = 0;
   for (const auto & aBl : mVBlATS)
       aRes += sqrt(aBoreS.L2(aBl->BoreS()));

   return aRes / mVBlATS.size();
}
/*
*/




cAnalyseTrajStereopolis::cAnalyseTrajStereopolis(int argc,char ** argv) :
    cGS_Appli    (argc,argv,cGS_Appli::eComputeBlini),
    mBoresEstim  (3,3),
    mWithInc     (false),
    mMasq3D      (nullptr)
{
     if (EAMIsInit(&mNameMasq3D))
     {
         mMasq3D =  cMasqBin3D::FromSaisieMasq3d(mNameMasq3D);
     }
     if (EAMIsInit(&mDirInc))
     {
         mWithInc = true;
         mMapInc =  StdGetSCenterOPK("Ori-"+mDirInc + "/");
     }
     int    aNbByRayPly = 4;
     double aRayPly = 0.01;

     ELISE_ASSERT(mSetBloc.size()==2,"cAnalyseTrajStereopolis require excactly 2 cam in bloc");

     for (const auto & aPtrBl : mVBlocs)
     {
          cBlockATS * aPtrBlA = new cBlockATS(aPtrBl);
          mVBlATS.push_back(aPtrBlA);
         
          if (mWithInc)
             aPtrBlA->SetInc(mMapInc);
          // std::cout << "TR = " <<  mVBlATS.back().OriC2S().tr() << "\n";
          Pt3dr aLA =  aPtrBlA->LevA();

          if (mMasq3D)
          {
              double aPds = mMasq3D->IsInMasq(aPtrBlA->Pos());
              aPtrBlA->SetPdsOri(aPds);
              aPtrBlA->SetPdsPos(aPds);
          }
          mPly_LACloud.AddPt(aPtrBlA->StdCoul(),aLA);
     }
     mPly_LACloud.PutFile("AnTS_LevArmCloud.ply");
     for (int aK=1 ; aK<int(mVBlATS.size()) ; aK++)
     {
        mVBlATS[aK-1]->CalculNext(*mVBlATS[aK]);
     }
    

     // Calcul d'un point moyen par min de som(abs(dist)) 
     {
         double aScCMin = 1e20;
         Pt3dr aLAMinDEucl;
         for (const auto & aBl : mVBlATS)
         {
              double aScC =  ScoreOneLA(aBl->LevA());
              if (aScC<aScCMin)
              {
                   aScCMin = aScC;
                   aLAMinDEucl = aBl->LevA();
              }
         }
         CreatPlyOnePt("AnTS_LevArmDAbs.ply",aLAMinDEucl,PColBlue,aRayPly,aNbByRayPly);
         mLAEstim = aLAMinDEucl;
     }
     if (0)  // Pb Resolu, a priori inutile
     {
        Pt3dr aMedLA = P3DMed(mVBlATS,[](const cBlockATS * aBl) {return aBl->LevA() ;});
        CreatPlyOnePt("AnTS_LevArmMed.ply",aMedLA,PColOrange,aRayPly,aNbByRayPly);
     }

     // Calcul d'un borseight moyen par min de som(abs(dist)) 
     {
         double aScCMin = 1e20;
         ElMatrix<double> aBoreSMinDist(3,3);
         for (const auto & aBl : mVBlATS)
         {
              double aScC =  ScoreOneBoreS(aBl->BoreS());
              if (aScC<aScCMin)
              {
                   aScCMin = aScC;
                   aBoreSMinDist = aBl->BoreS();
              }
         }
         mBoresEstim = aBoreSMinDist;

        cPlyCloud       aPly_BoreSCloud;
        for (auto & aBl : mVBlATS)
        {
            aBl->SetEstimBoreS(mBoresEstim);
            aPly_BoreSCloud.AddPt(aBl->StdCoul(),aBl->OPK_V());
        }
        aPly_BoreSCloud.PutFile("AnTS_OPKCloud.ply");

        std::cout << "DISP LA " << ScoreOneLA(mLAEstim) << "\n";
        std::cout << "DISP BORESIGHT " << ScoreOneBoreS(mBoresEstim) << "\n";
     }


     {
         for (int aKp=0 ; aKp<int(mVBlATS.size()) ; aKp++)
         {
// std::cout << "WWWWWWww " << mVBlATS.at(aKp)->OPK();
         }

         double aMulPos = 100;
         double aMulOPK = 200;

         Pt3di aCoulTraj = PColOrange;
         for (int aKp=0 ; aKp<int(mVBlATS.size()) ; aKp++)
         {
             // const cBlockATS & aBl = mVBlocs.at(aKp);
             const cBlockATS & aBl = * mVBlATS.at(aKp);
             Pt3di aColPos = (aBl.PdsPos() > 0) ? PColGreen : PColRed ;
             mPly_TrajCOpt.AddSphere(aColPos,aBl.Pos(),0.10,4);
             Pt3dr anEcart = aBl.LevA() - mLAEstim;
             mPly_TrajLevArm.AddSeg(PColRed,aBl.Pos(),aBl.Pos() + anEcart*aMulPos,100);

             mPly_TrajBoreS_V.AddSeg(PColBlue  ,aBl.Pos(),aBl.Pos() + aBl.OPK_V() * aMulOPK  ,100);
             mPly_TrajBoreS_W.AddSeg(PColCyan,aBl.Pos(),aBl.Pos() + aBl.OPK_W() * aMulOPK  ,100);
             if (aKp>=1)
             {
                const cBlockATS & aBlPrec = *mVBlATS.at(aKp-1);
                mPly_TrajSeg.AddSeg(aCoulTraj,aBl.Pos(),aBlPrec.Pos(),100);
             }
             mPly_TrajNameCam.PutString
             (
               aBl.StdBl().mTimeId,
               aBl.Pos() + Pt3dr(0,0,2),
               Pt3dr(1,0,0),
               Pt3dr(0,1,0),
               PColGrayMed,
               0.3,
               0.05,
               1 // Number of pix ply for 1 pix digit
             );
             if (aBl.WithInc())
             {
               mPly_TrajIncPos.AddSeg(Pt3di(255,0,128),aBl.Pos(),aBl.Pos() + aBl.IncXYZ()*aMulPos,100);
               mPly_TrajIncOPK.AddSeg(Pt3di(0,128,255),aBl.Pos(),aBl.Pos() + aBl.IncOPK()*aMulOPK,100);
             }
         }
         mPly_TrajCOpt.PutFile("AnTS_TrajCenter.ply");
         mPly_TrajSeg.PutFile("AnTS_TrajLine.ply");
         mPly_TrajLevArm.PutFile("AnTS_TrajLevArm.ply");
         mPly_TrajBoreS_V.PutFile("AnTS_TrajBoreS_Vehicule.ply");
         mPly_TrajBoreS_W.PutFile("AnTS_TrajBoreS_Word.ply");
         mPly_TrajNameCam.PutFile("AnTS_TrajNameCam.ply");
         if (mWithInc)
         {
              mPly_TrajIncPos.PutFile("AnTS_TrajIncXYZ.ply");
              mPly_TrajIncOPK.PutFile("AnTS_TrajIncOPK.ply");
         }
     }
     StatWithSel ( [this](const cBlockATS & aBl){return aBl.LevA() - mLAEstim;},SelPds, "Lever-Arm");
     StatWithSel ( [](const cBlockATS & aBl){return aBl.OPK_V() ;},SelPds, "Bore-Sight");

     StatWithSel ( [](const cBlockATS & aBl){return aBl.IncXYZ() ;},SelInc, "XYZInc");
     StatWithSel ( [](const cBlockATS & aBl){return aBl.IncOPK() ;},SelInc, "WKPInc");
}

int AnalyseTrajStereopolis_main(int argc,char ** argv)
{
    // GetSCenterOPK("Ori-4MesInc/Sensib-ConvName.txt","Ori-4MesInc/Sensib-Data.dmp");
    cAnalyseTrajStereopolis anAppli(argc,argv);

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
