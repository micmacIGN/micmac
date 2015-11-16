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

#include "general/all.h"
#include "private/all.h"
#include "XML_GEN/all.h"


/*
 
     Principe general :

      1 - on calcule l'offset de temps en regardant la diff la
      + frequente


      2- On calcule ensuite un graphe, entr les diff images
      des diff cannaux, ou deux images sont voisines si elles
      ont le bon offset a un seuil pres


      3- 
*/

using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;

namespace NS_Resynchr
{

class cAppliResynchIm;
class cOneCh;
class cOneIm;


class cOneIm
{
    public :
       cOneIm(const std::string&,const cOneCh&,const cAppliResynchIm &);
       const cMetaDataPhoto & MDP() {return mMDP;}
       const std::string & Name() {return  mName;}
       std::string FullName() const;
       const cOneCh&  Ch() const {return mCh;}


       double DifInDate(const cOneIm &) const;
       double EcartSync(const cOneIm &) const;
       void MakeVois(cOneIm * anIm,double aSeuil);

       std::vector<cOneIm *> CC();
       void ReName(int aNumC);

    private :

       std::string    mName;
       const cOneCh&  mCh;
       cMetaDataPhoto mMDP;
       bool           mMarqued;
       std::vector<cOneIm *> mVois;
};

class cOneCh
{
    public :
       cOneCh
       (
           const cAppliResynchIm & anArg,
	   const cOneResync  &,
	   int   aNumCh
       );
       const std::string & Dir() const {return mDir;}

       double  GetDifTime(const cOneCh & aCh2);  // T1 - T2
       double & TMoy() {return mTMoy;}
       const double & TMoy() const {return mTMoy;}
       void MakeVois(cOneCh & anIm,double aSeuil);

       void MakeCC(std::vector<std::vector<cOneIm *> > &);

       int NumCh() const {return mNumCh;}

       void ReName(const std::string & ,int aNumC) const;

    private :

       void GetInterv
            (
                int & aKMin, int & aKMax,
		const std::vector<double> & aVD ,int aK,double aSeuil
            );

       double  GainIndex(const std::vector<double> & aVD ,int aK);
       double  GainCpleIndex(const std::vector<double> & aVD ,int aK1,int aK2);


       const cAppliResynchIm &  mAppli;
       const cReSynchronImage & mRSI;
       const cOneResync       & mOR;
       std::string              mDir;
       int                      mNumCh;
       std::vector<cOneIm *>    mVIm;
       int                      mNbIm;
       cElRegex *               mAutoRename;

       double                   mEcMaxRA;
       double                   mSigmaRA;
       double                   mEcartMoyRA;

       double                   mTMoy;
};

class cAppliResynchIm
{
    public :
       cAppliResynchIm(int argc,char ** argv);
       const cReSynchronImage & RSI() const {return mArg;}
       int NbCh() const                     {return mNbCh;}
       int Exe() const                      {return mExe;}
    private :
       bool CCIsOk(const std::vector<cOneIm *> & aCC);


       std::string mNameParam;
       int  mExe;
       cReSynchronImage mArg;
       int              mNbCh;
       std::vector<cOneCh *> mVCH;
       double  mEcMin;
       double  mEcMax;
};

/********************************************************/
/*                                                      */
/*              cOneIm                                  */
/*                                                      */
/********************************************************/

cOneIm::cOneIm
(
    const std::string& aName,
    const cOneCh& aCh,
    const cAppliResynchIm &
)  :
   mName    (aName),
   mCh      (aCh),
   mMDP     (cMetaDataPhoto::CreateExiv2(aCh.Dir()+aName)),
   mMarqued (false)
{
   // std::cout << ToString(mMDP.Date()) << "\n";
}

double cOneIm::DifInDate(const cOneIm & anIm) const
{
  return mMDP.Date().DifInSec(anIm.mMDP.Date());
}

double cOneIm::EcartSync(const cOneIm & anIm) const
{
   return ElAbs(mCh.TMoy()-anIm.mCh.TMoy() - DifInDate(anIm));
}

void cOneIm::MakeVois(cOneIm * anIm,double aSeuil)
{
    if (EcartSync(*anIm)<aSeuil)
    {
        // std::cout << mName << " / " << anIm->mName << "\n";
        mVois.push_back(anIm);
	anIm->mVois.push_back(this);
    }
}

std::vector<cOneIm *> cOneIm::CC()
{
   std::vector<cOneIm *> aRes;
   int aK0=0;

   if (! mMarqued) 
   {
      mMarqued=true;
      aRes.push_back(this);
   }
   while (aK0!=int(aRes.size()))
   {
       for (int aKV=0 ; aKV<int(aRes[aK0]->mVois.size()) ; aKV++)
       {
           cOneIm *aV = aRes[aK0]->mVois[aKV];
           if (! aV->mMarqued)
	   {
	       aV->mMarqued = true,
	       aRes.push_back(aV);
	   }
       }
       aK0++;
   }

   return aRes;
}

void cOneIm::ReName(int aNumC)
{
   mCh.ReName(mName,aNumC);
}


std::string cOneIm::FullName()const  
{
  return  mCh.Dir()+ mName;
}
/********************************************************/
/*                                                      */
/*              cOneCh                                  */
/*                                                      */
/********************************************************/

cOneCh::cOneCh
(
    const cAppliResynchIm & anAppli,
    const cOneResync  &     anOR,
    int                     aNumCh
) :
  mAppli      (anAppli),
  mRSI        (mAppli.RSI()),
  mOR         (anOR),
  mDir        (mOR.Dir()),
  mNumCh      (aNumCh),
  mAutoRename (new cElRegex(mOR.PatRename(),10)),
  mEcMaxRA    (mRSI.EcartRechAuto().Val()),
  mSigmaRA    (mRSI.SigmaRechAuto().Val()),
  mEcartMoyRA (mRSI.EcartCalcMoyRechAuto().Val()),
  mTMoy       (0)
{
   std::list<std::string> aLN = RegexListFileMatch(mDir,mOR.PatSel(),1,false);

   mNbIm=0;
   for (std::list<std::string>::iterator itS=aLN.begin();itS!=aLN.end();itS++)
   {
      mVIm.push_back(new cOneIm(*itS,*this,anAppli));
      mNbIm++;
   }
   if (mNbIm==0)
   {
       std::cout << "Dir=" << mDir << "\n";
       std::cout << "Pat=" << mOR.PatSel() << "\n";
       ELISE_ASSERT(false,"Aucune  image dans cOneCh::cOneCh");
   }
}

void cOneCh::GetInterv
     (
         int & aKMin, int & aKMax,
        const std::vector<double> & aVD ,int aK,double aSeuil
     )
{
    aKMin = aK;
    while ((aKMin>=0) && (ElAbs(aVD[aKMin]-aVD[aK])<aSeuil))
        aKMin--;

    int aSzV = aVD.size();
    aKMax = aK;
    while ((aKMax<aSzV) && (ElAbs(aVD[aKMax]-aVD[aK])<aSeuil))
         aKMax++;
}

double  cOneCh::GainCpleIndex(const std::vector<double> & aVD ,int aK1,int aK2)
{
   return exp(-ElSquare(aVD[aK1]-aVD[aK2])/ (2*ElSquare(mSigmaRA)));
}

double  cOneCh::GainIndex(const std::vector<double> & aVD ,int aK)
{
    int aKMin,aKMax;
    GetInterv(aKMin,aKMax,aVD,aK,mEcMaxRA);

    double aGain=0.0;
    for (int aK2=aKMin+1 ; aK2<aKMax ; aK2++)
        aGain += GainCpleIndex(aVD,aK,aK2);

    return aGain;
}

void cOneCh::MakeVois(cOneCh & aCh2,double aSeuil)
{
    for (int aK1=0 ; aK1<mNbIm ; aK1++)
    {
       for (int aK2=0 ; aK2<aCh2.mNbIm ; aK2++)
       {
	  mVIm[aK1]->MakeVois(aCh2.mVIm[aK2],aSeuil);
       }
    }
}


double  cOneCh::GetDifTime(const cOneCh & aCh2)
{
    std::vector<double> aVD; 

    for (int aK1=0 ; aK1<mNbIm ; aK1++)
    {
       for (int aK2=0 ; aK2<aCh2.mNbIm ; aK2++)
       {
	  aVD.push_back(mVIm[aK1]->DifInDate(*(aCh2.mVIm[aK2])));
       }
    }

    std::sort(aVD.begin(),aVD.end());

    int aSzV = aVD.size();

    int aKGainMax =-1;
    double aGainMax = 0;
    for (int aK=0 ; aK<aSzV ; aK++)
    {
       double aGain = GainIndex(aVD,aK);
       if (aGain > aGainMax)
       {
           aGainMax = aGain;
	   aKGainMax  = aK;
       }
    }


    int aK1,aK2;
    GetInterv(aK1,aK2,aVD,aKGainMax,mEcartMoyRA);
    double aS0=0;
    double aSV=0;

    for (int aK = aK1+1 ; aK<aK2 ; aK++)
    {
        aS0 ++;
	aSV += aVD[aK];
    }

    double aRes = aSV /aS0;
    // std::cout << aK1 << " " << aK2 << "\n";
    // std::cout << aKGainMax  << " " << aVD[aKGainMax ] << " " << aRes << "\n";

    return aRes;
}



void cOneCh::MakeCC(std::vector<std::vector<cOneIm *> >  & aRes )
{
    for (int aK1=0 ; aK1<mNbIm ; aK1++)
    {
        std::vector<cOneIm *> aCC = mVIm[aK1]->CC();
	if (aCC.size() !=0)
	   aRes.push_back(aCC);
    }
    return;
}

void cOneCh::ReName(const std::string & aName,int aNumC) const
{
   std::string aNew =  MatchAndReplace
                       (
		           *mAutoRename,
			   aName+"@"+ToString(aNumC),
			   mOR.Rename()
                       );
   std::string aCom= std::string("mv ")
                     + ToStrBlkCorr(mOR.Dir()+aName) + " "
		     + ToStrBlkCorr(mOR.Dir()+aNew);
   if (mAppli.Exe())
   {
      std::cout <<  aCom << "\n";
      int aRes = system(aCom.c_str());
      ELISE_ASSERT(aRes==0,"Cannot exec move");
   }
   else
   {
      std::cout << "  " << aName  << " -> " << aNew << ";" << mOR.Dir() << "\n";
   }
}
/********************************************************/
/*                                                      */
/*             cAppliResynchIm                          */
/*                                                      */
/********************************************************/

bool cAppliResynchIm::CCIsOk(const std::vector<cOneIm *> & aCC)
{
   if (int(aCC.size()) != mNbCh)
      return false;
   for (int aK1=0 ; aK1<int(aCC.size()) ; aK1++)
   {
      if (aCC[aK1]->Ch().NumCh() != aK1)
         return false;

      for (int aK2=aK1+1 ; aK2<int(aCC.size()) ; aK2++)
      {
          double aEc = aCC[aK1]->EcartSync(*aCC[aK2]);
	  if (aEc>mEcMin)
	  {
	     return false;
          }
      }
   }

   return true;
}

cAppliResynchIm::cAppliResynchIm
(
   int argc,
   char ** argv
)  :
   mExe (0)
{
   ElInitArgMain
   (
       argc,argv,
       LArgMain() << EAM(mNameParam),
       LArgMain() << EAM(mExe,"Exe",true)
  );
  std::cout << "NP : " << mNameParam << "\n";

  mArg = StdGetObjFromFile<cReSynchronImage>
         (
	     mNameParam,
	     "include/XML_GEN/SuperposImage.xml",
	     "ReSynchronImage",
	     "ReSynchronImage"
	 );

  mEcMin = mArg.EcartMin();
  mEcMax = mArg.EcartMax();

  mNbCh = mArg.OneResync().size();
  ELISE_ASSERT(mNbCh>=2,"Moins de 2 Cannaux !!\n");

  {
    int aK=0;
    for
    (
      std::list<cOneResync>::const_iterator itR=mArg.OneResync().begin();
      itR!=mArg.OneResync().end();
      itR++
    )
    {
       mVCH.push_back(new cOneCh(*this,*itR,aK));
       aK++;
    }
  }

  for (int aK1=0 ; aK1<mNbCh; aK1++)
  {
     for (int aK2=aK1+1 ; aK2<mNbCh; aK2++)
     {
        double aDif = mVCH[aK1]->GetDifTime(*mVCH[aK2]);
	mVCH[aK1]->TMoy() += aDif/(mNbCh);
	mVCH[aK2]->TMoy() -= aDif/(mNbCh);
     }
  }

  //for (int aK1=0 ; aK1<mNbCh; aK1++)
 //     std::cout << aK1 << " TM= " <<  mVCH[aK1]->TMoy() << "\n";

  if (0) // Verification
  {
     for (int aK1=0 ; aK1<mNbCh; aK1++)
     {
        for (int aK2=aK1+1 ; aK2<mNbCh; aK2++)
        {
           double aDif = mVCH[aK1]->GetDifTime(*mVCH[aK2]);
	    std::cout << aK1 << " " 
	              << aK2 << " "
		      << aDif << " "
		      << (aDif-( mVCH[aK1]->TMoy()-mVCH[aK2]->TMoy())) << "\n";
        }
     }
  }


  // Creation du graphe de voisinage
  for (int aK1=0 ; aK1<mNbCh; aK1++)
  {
      for (int aK2=aK1+1 ; aK2<mNbCh; aK2++)
      {
          mVCH[aK1]->MakeVois(*mVCH[aK2],mEcMax);
      }
  }


  std::vector<std::vector<cOneIm *> > aVCC; 
  for (int aK1=0 ; aK1<mNbCh; aK1++)
      mVCH[aK1]->MakeCC(aVCC);

  bool GotPb=false;
  for (int aKC=0 ; aKC< int(aVCC.size()) ; aKC++)
  {
     std::vector<cOneIm *> aCC = aVCC[aKC];
     if (! CCIsOk(aCC))
     {
        std::cout << "==========================\n";
        GotPb = true;
        for (int aKI=0 ; aKI<int(aCC.size()) ; aKI++)
        {
            std::cout << aCC[aKI]->FullName() << "\n";
        }
     }
  }
  if (GotPb)
  {
      std::cout << "==========================\n";
      std::cout << "Resynchronisation ambigues/impossible pour les groupes ci dessus\n";
      getchar();
  }

  for (int aKC=0 ; aKC< int(aVCC.size()) ; aKC++)
  {
     std::vector<cOneIm *> aCC = aVCC[aKC];
     if ( CCIsOk(aCC))
     {
        for (int aKI=0 ; aKI<int(aCC.size()) ; aKI++)
        {
	     aCC[aKI]->ReName(aKC);
        }
        std::cout << "==========================\n";
     }
  }
}


};


using namespace NS_Resynchr;

int main(int argc,char ** argv)
{
   cAppliResynchIm aARI(argc,argv);
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
