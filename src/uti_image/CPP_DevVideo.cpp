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
#include <algorithm>

class cOneImageVideo;
class cAppliDevideo;

// ffmpeg -i MVI_0001.MOV  -ss 30 -t 20 Im%5d_Ok.png

// Im*_Ok => OK 
// Im*_Nl => Image Nulle (eliminee)


#define DEF_OFSET -12349876

const std::string TmpVid() {return "Tmp-Vid";}

int DevOneImPtsCarVideo_main(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}


class cOneImageVideo
{
    public :
        cOneImageVideo(const std::string & aNameIm,cAppliDevideo *,int aK);
        const std::string & NameOk()    const {return mNameOk;}
        void LoadPts();
        void  UpDateMaxLoc(cOneImageVideo & anOIV);
        void  Show();
        bool  Ok() const;
        void  SetNotOk() ;
        bool  IsMaxLoc() const;
        int   SzSift() const;
        int   TimeNum() const;
    private :

        cAppliDevideo *  mAppli;
        std::string      mNameInit;
        std::string      mNameOk;
        std::string      mNameNl;
        std::string      mNamePtsSift;
        int              mSzSift;
        bool             mIsMaxLoc;
        bool             mOk;
        int              mTimeNum;
};

class cCmpOIV_SzSift
{
    public :
       bool operator() (cOneImageVideo * anOIV1 ,cOneImageVideo * anOIV2)
       {
           return anOIV1->SzSift() < anOIV2->SzSift();
       }
 
};


class cAppliDevideo
{
    public :
        cAppliDevideo(int argc,char ** argv);
        const std::string & Dir() {return mEASF.mDir;}
        std::string CalcName(const std::string & aName,const std::string aNew);
    private :
        std::string NamePat(const std::string & aPref) const;
        std::string mFullName;
        cElemAppliSetFile mEASF;
        std::string mPrefix;
        std::string mPostfix;
        std::string mMMPatImDev;
        std::string mMMPatImOk;
        cElRegex *  mAutoMM;
        const std::vector<std::string> * mVName;
        std::vector<cOneImageVideo*>      mVIms;
        std::vector<cOneImageVideo*>      mVImsOk;
        double      mRate;
        double      mRateVideoInit;
        double      mStdJump;
};


// =============  cOneImageVideo ===================================

cOneImageVideo::cOneImageVideo(const std::string & aNameIm,cAppliDevideo * anAppli,int anTimeNum) :
   mAppli       (anAppli),
   mNameInit    (aNameIm),
   mNameOk      (mAppli->CalcName(aNameIm,"Ok")),
   mNameNl      (mAppli->CalcName(aNameIm,"Nl")),
   mNamePtsSift (mAppli->Dir()+  "Pastis/LBPp"+ mNameOk  + ".dat"),
   mIsMaxLoc    (true),
   mOk          (true),
   mTimeNum       (anTimeNum)
{
    // std::cout << mNameInit   << "  " << mNameIm << "\n";
    if (mNameInit!= mNameOk)
       ELISE_fp::MvFile(anAppli->Dir()+mNameInit,anAppli->Dir()+mNameOk);
}

void cOneImageVideo::LoadPts()
{
    mSzSift = sizeofile(mNamePtsSift.c_str());
}

void  cOneImageVideo::Show()
{
    std::cout << (mIsMaxLoc? "###" : (mOk ? "ooo" : "---")) << mNameOk << " " << mSzSift << "\n";
}

void  cOneImageVideo::UpDateMaxLoc(cOneImageVideo & anOIV)
{
    if (mSzSift < anOIV.mSzSift)  mIsMaxLoc = false;
    if (mSzSift > anOIV.mSzSift)  anOIV.mIsMaxLoc = false;
}

bool cOneImageVideo::Ok() const {return mOk;}
bool cOneImageVideo::IsMaxLoc() const {return mIsMaxLoc;}
int cOneImageVideo::SzSift() const {return mSzSift;}
int cOneImageVideo::TimeNum() const {return mTimeNum;}
void cOneImageVideo::SetNotOk() 
{
   ELISE_fp::MvFile(mAppli->Dir()+mNameOk,mAppli->Dir()+mNameNl);
   mOk=false;
}
// =============  cAppliDevideo ===================================

cAppliDevideo::cAppliDevideo(int argc,char ** argv) :
     mPrefix ("Im"),
     mPostfix ("png"),
     mRate           (4.0),
     mRateVideoInit  (24.0)
{
    bool DoSIFT = true;
    ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(mFullName,"Full name (Dir+Pat)", eSAM_IsPatFile) ,
           LArgMain() << EAM(mRate,"Rate",true,"Rate final Def=4")
    );
    mStdJump = mRateVideoInit/mRate;

    mEASF.Init(mFullName);
    
    mMMPatImDev = NamePat("Ok|Nl");
    mMMPatImOk = NamePat("Ok");

    mAutoMM =  new cElRegex(mMMPatImDev,10);
    mVName = mEASF.mICNM->Get(mMMPatImDev);
    ELISE_fp::MkDir(Dir()+TmpVid() +"/");

    // bool MisOneSift = false;
    for (int aK=0 ; aK<int(mVName->size()) ; aK++)
    {
        mVIms.push_back(new cOneImageVideo((*mVName)[aK],this,aK));
        // MisOneSift = MisOneSift || mVIms.back()->MisSift();
        // std::string aNamePts = mVIms.back()->NameDigeo();
    }

    std::string aComTapioca = MM3dBinFile("Tapioca") + " Line " + QUOTE(mMMPatImOk) +  " -1 1";
    if (DoSIFT)
       System(aComTapioca);


    // Calcule sz sift et init max loc
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
        mVIms[aK]->LoadPts();
    }
    for (int aK=1 ; aK<int(mVIms.size()) ; aK++)
    {
        mVIms[aK-1]->UpDateMaxLoc(*(mVIms[aK]));
         
    }

    //  Supression des pts en commencant par le  - de pts sift

    std::vector<cOneImageVideo*>  aVSort =  mVIms;
    cCmpOIV_SzSift aCmp;
    std::sort(aVSort.begin(),aVSort.end(),aCmp);

    for (int aKS=0 ; aKS<int(aVSort.size()) ; aKS++)
    {
         int aK= aVSort[aKS]->TimeNum();
         int aKP = aK+1;
         while ((aKP<int(mVIms.size()) && (!mVIms[aKP]->Ok()))) aKP++;
         int aKM = aK-1;
         while ((aKM>=0) && (!mVIms[aKM]->Ok())) aKM--;
         if ((aKP-aKM) <= mStdJump/2.0) 
         {
            aVSort[aKS]->SetNotOk();
         }
    }

    mVImsOk.clear();
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
         if (mVIms[aK]->Ok())
         {
            mVImsOk.push_back(mVIms[aK]);
         }
    }
    // === Calcul du graphe ====

    int aJumpMax = round_up(mStdJump * 2);
    for (int aK1=0 ; aK1<int(mVImsOk.size()) ; aK1++)
    {
         // Interval ]aK1 , aK2 [
         int aK2 = aK1;
         while ((aK2<<int(mVImsOk.size()))  && (ElAbs(mVImsOk[aK1]->TimeNum()-mVImsOk[aK2]->TimeNum()) < aJumpMax))
         {
               aK2++;
         }
         // Pour etre sur interval non vide
         if ((aK2==aK1) && (aK2<int(mVImsOk.size()))) aK2++;
         std::cout << aK1 << " => " << aK2 << " ;" ;   mVImsOk[aK1]->Show();
    }

    // =============
    

/*
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
        mVIms[aK]->Show();
    }
*/

    std::cout << "NB Im = " << mVName->size() << " Presel " << mVImsOk.size() << " JMP=" << mStdJump << "\n";

}

std::string  cAppliDevideo::NamePat(const std::string & aPref) const
{
   return  "("+ mPrefix + "[0-9]{5}_)(" + aPref   + ")(\\." + mPostfix+")";
}

std::string cAppliDevideo::CalcName(const std::string & aName,const std::string aNew)
{
   return MatchAndReplace(*mAutoMM,aName,"$1" + aNew + "$3");
/*
   std::cout << mMMPatImDev << " " << aName << "\n";
   bool aOk =  mAutoMM->Replace("$1" + aNew + "$3");
   std::cout << "RRRR="  <<  mAutoMM->LastReplaced() << "\n";
   ELISE_ASSERT(aOk,"cAppliDevideo::CalcName");
   return  mAutoMM->LastReplaced();
*/
}




//========================== :: ===========================

int Devideo_main(int argc,char ** argv)
{
    cAppliDevideo anAppli(argc,argv);
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
