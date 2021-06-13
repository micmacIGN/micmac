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

class  cAppliFilterGeo3;

class cMoyGeo3
{
   public :
       cMoyGeo3() :
          mPds (0)
       {
       }
       void SetPt(const Pt3dr & aPt,double aPds)
       {
            mPt = aPt;
            mPds = aPds;
       }
       double Pds() const {return mPds;}
       const Pt3dr & Pt()  const
       { 
            ELISE_ASSERT(mPds!=0,"cMoyGeo3::Pt");
            return mPt;
       }
   private :
        Pt3dr mPt;
        double mPds;
};

class cOneGeo3
{
    public :
        friend class cAppliFilterGeo3;
        cOneGeo3(const std::string& aName);

        void Compile(double aT0,double aT1,double mDT);

        // Calcul le cost et memorise le KMin / CostMin
        double CostK(int aK,const Pt3dr & aPond);
        void ShowDiff(int aKGlob);
    private :
        std::string      mName;
        cDicoGpsFlottant mDicoOri;
        std::vector<cOneGpsDGF> &  mVObs;
        std::vector<cMoyGeo3>      mVMoy;
// Box de temps et d'espace
        Pt3dr            mP0;
        Pt3dr            mP1;
        double           mT0;
        double           mT1;
        int              mKMin;
        double           mCostKMin;
        Pt3dr            mPMed;
}; 


bool CmpcOneGeo3OnTime(const cOneGpsDGF & aG1,const cOneGpsDGF & aG2)
{
   return aG1.TimePt() < aG2.TimePt();
}

void cOneGeo3::Compile(double aT0,double aT1,double aDT)
{
    int aK0 =0;
    int aK1 =0;
    int aNb = mVObs.size();

    std::vector<double> aMedX,aMedY,aMedZ;

    for (double aT= aT0 ; aT<aT1 ; aT+= aDT)
    {

         while((aK1<aNb) && (mVObs.at(aK1).TimePt() <(aT+aDT)))
             aK1++;

         std::vector<double> aVX,aVY,aVZ;
         for (int aK=aK0 ; aK<aK1 ; aK++)
         {
             const cOneGpsDGF &  anObs = mVObs.at(aK);
             if (anObs.TagPt()==1)
             {
                aVX.push_back(anObs.Pt().x);
                aVY.push_back(anObs.Pt().y);
                aVZ.push_back(anObs.Pt().z);
             }
         }
         mVMoy.push_back(cMoyGeo3());
         if (aVX.size() != 0)
         {
             double x = MedianeSup(aVX);
             double y = MedianeSup(aVY);
             double z = MedianeSup(aVZ);
             mVMoy.back().SetPt(Pt3dr(x,y,z),aVX.size());

             aMedX.push_back(x);
             aMedY.push_back(y);
             aMedZ.push_back(z);
         }

         aK0 = aK1;
    }
    mPMed = Pt3dr(MedianeSup(aMedX),MedianeSup(aMedY),MedianeSup(aMedZ));
}


void cOneGeo3::ShowDiff(int aKGlob)
{

   std::cout << std::setprecision(2) << " " 
             << mName << " : " 
             << " " << 1e3*  euclid(mVMoy.at(aKGlob).Pt() -  mVMoy.at(mKMin).Pt() )
             << " " << 1e3*  euclid(mVMoy.at(aKGlob).Pt() -  mPMed)
             << " " << 1e3*  euclid(mVMoy.at(mKMin).Pt() -  mPMed)
             << "\n";
}

double cOneGeo3::CostK(int aK,const Pt3dr & aPond)
{
   Pt3dr aP0 =  mVMoy.at(aK).Pt();
   double aSomDif=0;
   double aSomPds=0;
   for (const auto & anObs : mVMoy)
   {
      double aPds = anObs.Pds();
      if (aPds)
      {
         const Pt3dr & aP = anObs.Pt();
         double aDif =  ElAbs(aP0.x-aP.x) * aPond.x
                      + ElAbs(aP0.y-aP.y) * aPond.y
                      + ElAbs(aP0.z-aP.z) * aPond.z;
 
         aSomDif += aPds * aDif;
         aSomPds += aPds;
      }
   }

   double aCost =  aSomDif/aSomPds;
   if (aCost<mCostKMin)
   {
      mCostKMin = aCost;
      mKMin = aK;
   }

   return aCost;
}


cOneGeo3::cOneGeo3(const std::string& aName)  :
       mName     (aName),
       mDicoOri  (StdGetFromPCP(aName,DicoGpsFlottant)),
       mVObs     (mDicoOri.OneGpsDGF()),
       mKMin     (-1),
       mCostKMin (1e30)
{
    std::sort(mVObs.begin(),mVObs.end(),CmpcOneGeo3OnTime);
    // Calcul des intervalle de temps et d'espace
    mP0 = mP1 = mDicoOri.OneGpsDGF().at(0).Pt();
    mT0 = mT1 = mDicoOri.OneGpsDGF().at(0).TimePt();
    for (const auto & anObs :  mDicoOri.OneGpsDGF())
    {
        mP0 = Inf(mP0,anObs.Pt());
        mP1 = Sup(mP1,anObs.Pt());
        mT0 = ElMin(mT0,anObs.TimePt());
        mT1 = ElMax(mT1,anObs.TimePt());
    }

    std::cout << "Geo3 : " << mName << " NbObs " << mDicoOri.OneGpsDGF().size() << " DT=" << mT1-mT0 << " DP=" << mP1-mP0<< "\n";

}

class cAppliFilterGeo3
{
   public :
        cAppliFilterGeo3(int argc,char ** argv);
   private :
       std::string              mPat;
       std::string              mDir;
       cElemAppliSetFile        mEASF;
       std::string              mShow;
       std::vector<cOneGeo3 *>  mV3;
       double                   mT0;
       double                   mT1;
       double                   mDT;
       double                   mPdsZ;
       int                      mNbM;
       std::vector<double>      mVPMin;
};


cAppliFilterGeo3::cAppliFilterGeo3(int argc,char ** argv) :
    mDT   (10.0),
    mPdsZ (2.0)
{
	
    ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mPat, "Dir+Pat"),
          LArgMain() << EAM(mDT,"DT",true,"Time intervall for merging, in sec, def=?")
                     << EAM(mShow,"Show",true,"X or Y or Z")
                     << EAM(mPdsZ,"PdsZ",true,"Weight of alti/plani, def=2")
    );

    mDT = mDT / (24 * 3600); // Conversion en jours

    mEASF.Init(mPat);
    mDir = mEASF.mDir;
    //  const cInterfChantierNameManipulateur::tSet *  aSetFile = SetIm();

    mT0 = 1e30;
    mT1 = -1e30;

    for (const auto & aName : *(mEASF.SetIm()))
    {
        mV3.push_back(new cOneGeo3(mDir+aName));
        mT0 = ElMin(mT0,mV3.back()->mT0);
        mT1 = ElMax(mT1,mV3.back()->mT1);
    }
    std::cout << "Delta-TIME " << mT1-mT0 << "\n";

    for (const auto & aPG : mV3)
    {
       aPG->Compile(mT0,mT1,mDT);
       ELISE_ASSERT(aPG->mVMoy.size()==mV3.at(0)->mVMoy.size(),"Incohen in moy in cAppliFilterGeo3");
    }

    mNbM = mV3.at(0)->mVMoy.size();

    int aNbNN=0;
    for (int aK=0 ; aK< mNbM ; aK++)
    {
        double aPMin = 1e30;
        for (const auto & aPG : mV3)
        {
            aPMin  = ElMin(aPMin,aPG->mVMoy.at(aK).Pds());
        }
        mVPMin.push_back(aPMin);
        if (aPMin!=0)
           aNbNN++;
        // std::cout << "PM = " << aPMin << "\n";
    }
    std::cout << " Prop not null " << aNbNN / double(mNbM) <<  " NbNN=" << aNbNN << "\n";

    double aMinGlob = 1e30;
    int aKGlob = -1;
    for (int aK=0 ; aK< mNbM ; aK++)
    {
       if (mVPMin[aK] !=0)
       {
           double aCostK= 0;
           for (const auto & aPG : mV3)
           {
               aCostK += aPG->CostK(aK,Pt3dr(1,1,mPdsZ));
           }
           if (aCostK < aMinGlob)
           {
              aMinGlob = aCostK;
              aKGlob = aK;
           }
       }
    }
    
    std::cout << "================ DIF  G/L  G/M  L/M ============= \n";
    for (const auto & aPG : mV3)
    {
        aPG->ShowDiff(aKGlob) ;
    }
    std::cout << "================ GLOBAL ============= \n";
    for (const auto & aPG : mV3)
    {
        std::cout << std::setprecision(12) << aPG->mName << " : " << aPG->mVMoy.at(aKGlob).Pt() << "\n";
    }
    std::cout << "================ Local (ByGeo3) ============= \n";
    for (const auto & aPG : mV3)
    {
        std::cout << std::setprecision(12) << aPG->mName << " : " << aPG->mVMoy.at(aPG->mKMin).Pt() << "\n";
    }
    std::cout << "================ Median  ============= \n";
    for (const auto & aPG : mV3)
    {
        std::cout << std::setprecision(12) << aPG->mName << " : " << aPG->mPMed << "\n";
    }



        // std::cout << std::setprecision(12) << mName << " : " << mVMoy.at(aKGlob).Pt() << " Loc: " <<  mVMoy.at(mKMin).Pt() << "\n";
    //     aPG->ShowRes(aKGlob);
}




int CPP_FilterGeo3(int argc,char ** argv)
{
   cAppliFilterGeo3 anAppli(argc,argv);

   return EXIT_SUCCESS;
}
    

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
