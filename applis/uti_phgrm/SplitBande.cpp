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
#include <algorithm>

using namespace NS_ParamChantierPhotogram;

class cSom;
class cSplitBande;

class cSom
{
     public :
       cSom(const cSplitBande & aSB,const std::string & aName,const cMetaDataPhoto & aMDP) :
          mSB   (&aSB),
          mName (aName),
          mMDP  (aMDP)
       {
       }

       const cSplitBande * mSB;
       std::string         mName;
       cMetaDataPhoto      mMDP;
       int                 mNumBande;
       bool                mOkBnd;

       double DS(const cSom & aS2)
       {
           return mMDP.Date().DifInSec(aS2.mMDP.Date());
       }
};

bool operator < (const cSom & aS1,const cSom & aS2)
{
   return aS1.mMDP.Date() < aS2.mMDP.Date();
}


class cSplitBande
{
    public :
        friend class cSom;
        cSplitBande(int argc,char ** argv);
        void DoAll();
        double  TMoy();
        double  TCut();

    private :
        std::string mDir;
        std::string mPat;
  
        double mDelta;
        cInterfChantierNameManipulateur * mICNM;
        std::list<std::string>  mLFile;
        std::vector<cSom >    mVC;
        int                    mNbSom;
        double                 mPropCutEx;
        double                 mAddMoy;
        double                 mMulMoy;
        bool                   mExe;
        std::string            mKCH;
        std::string            mSep;
        int                    mNum0;
        int                    mNbDig;

};

double  cSplitBande::TMoy()
{
    // Calcul du temps moyen

    std::vector<double> aVDif;
    for (int aK=1 ; aK<mNbSom ; aK++)
    {
        aVDif.push_back(mVC[aK].DS(mVC[aK-1]));
        std::cout << aVDif.back() << "\n";;
    }
    std::sort(aVDif.begin(),aVDif.end());
    double aNbD=0;
    double aSomD=0;
    for (int aK= round_ni(mPropCutEx * mNbSom) ; aK< round_ni((1-mPropCutEx) * (mNbSom-1)) ; aK++)
    {
         aNbD++;
         aSomD += aVDif[aK];
    }
    return aSomD / aNbD;
}

double  cSplitBande::TCut()
{
   return mAddMoy + TMoy() * mMulMoy;
}


cSplitBande::cSplitBande(int argc,char ** argv) :
     mNbSom (-1),
     mPropCutEx   (0.1),
     mAddMoy      (3.0),
     mMulMoy      (1.2),
     mExe         (false),
     mKCH         ("NKS-Assoc-AddPost"),
     mSep         ("_"),
     mNum0        (0),
     mNbDig       (3)
{
    int mVitAff = 10;
      
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(mDir)
                    << EAM(mPat),
        LArgMain()  << EAM(mDelta,"Delta",true)
                    << EAM(mSep,"Sep",true)
                    << EAM(mExe,"Exe",true)
                    << EAM(mNum0,"Num0",true)
                    << EAM(mNbDig,"NbDig",true)
        
    );

    cTplValGesInit<std::string>  aTplFCND;
    mICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,mDir,aTplFCND);

    mLFile =  mICNM->StdGetListOfFile(mPat);
    mNbSom =  mLFile.size();

    std::cout << "Nb Images = " <<  mNbSom << "\n";

    int aCpt = 0;
    for 
    (
         std::list<std::string>::const_iterator itS=mLFile.begin();
         itS!=mLFile.end();
         itS++
    )
    {
         const cMetaDataPhoto &   aMDP = cMetaDataPhoto::CreateExiv2(mDir+*itS);
         mVC.push_back(cSom(*this,*itS,aMDP));
         if ((aCpt %mVitAff) == (mVitAff-1)) 
         {
            std::cout << "Load  : remain " << (mNbSom-aCpt) << " to do\n";
         }
         aCpt++;
    }
    std::sort(mVC.begin(),mVC.end());

    double aTCut = TCut();

    int aNumB = mNum0;
    int aNbInB = 1;
    mVC[0].mNumBande = aNumB;
    for (int aK=1 ; aK<mNbSom ; aK++)
    {
        double aDif = mVC[aK].DS(mVC[aK-1]);
        if (aDif > aTCut )
        {
             std::cout << " NAME " << mVC[aK].mName 
                       << " Dif "  << aDif  
                       << " Nb " << aNbInB << "\n";
             aNbInB = 1;
             aNumB ++ ;
        }
        else
        {
             aNbInB++;
        }
        mVC[aK].mNumBande = aNumB;
    }
    if (aNbInB != 1)
    {
       std::cout << "Last Bande " << aNbInB << "\n";
    }

    for (int aK=0 ; aK<mNbSom ; aK++)
    {
        std::string aKey = mKCH + "@" + mSep + ToStringNBD(mVC[aK].mNumBande,mNbDig);
        std::string aNewN =  mICNM->Assoc1To1(aKey,mVC[aK].mName,true);

        std::string aCom = "mv " + mDir+mVC[aK].mName + " " + mDir+aNewN ;
        std::cout << aCom << "\n";
        if (mExe)
        {
           VoidSystem(aCom.c_str());
        }
    }
}

void cSplitBande::DoAll()
{
}




int main(int argc,char ** argv)
{
    cSplitBande aGr(argc,argv);
    aGr.DoAll();
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
