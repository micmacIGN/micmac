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



namespace NS_Trajecto2XmlOri
{

/*********************************************/
/*                                           */
/*        cOnePt                             */
/*                                           */
/*********************************************/

class cOnePt
{
    public :
       cOnePt(const char *);
       int     mNumIm;
       int     mNumBande;
       Pt3dr   mP;
       double  mTime;
       bool    mOK;
};


cOnePt::cOnePt(const char * aStr)
{
   int aNbEl = sscanf(aStr,"%d %d %lf %lf %lf %lf",
                       &mNumIm,&mNumBande,&mP.x,&mP.y,&mP.z,&mTime
               );

   mOK = true;

   if (mNumIm==0)
   {
       ELISE_ASSERT(mTime==0,"Incoh in cOnePt::cOnePt");
       mOK = false;
   }

   ELISE_ASSERT(aNbEl==6,"cOnePt::cOnePt");
};

/*********************************************/
/*                                           */
/*        cOnePack                            */
/*                                           */
/*********************************************/

class cOnePack
{
     public :

        cOnePack(const std::string & aNameIn);
       
         
        void DoBascule(const ElRotation3D & aR,double aLamnda,const std::string & aNameOut);
     // private :
        std::string  mNameIn;
        ELISE_fp    mFp;
        std::vector<cOnePt> mPts;
        Pt3dr               mCDG;
        Pt3dr               mAxe;
};

#define SzBuf 2000

cOnePack::cOnePack(const std::string & aNameIn) :
    mNameIn (aNameIn),
    mFp     (mNameIn.c_str(),ELISE_fp::READ),
    mCDG    (0,0,0),
    mAxe    (0,0,0)
{
   char aBuf[SzBuf];
   bool aEndOfFile =false;

   
   bool DirInit=false;
   while (! aEndOfFile) 
   {
      if (mFp.fgets(aBuf,SzBuf,aEndOfFile) && (!aEndOfFile))
      {
         cOnePt aPt(aBuf);
         if (aPt.mOK)
         {
              if (! mPts.empty())
              {
                 cOnePt aPrec = mPts.back();
                 if (aPrec.mNumBande==aPt.mNumBande)
                 {
                      Pt3dr aDir = aPt.mP-aPrec.mP;
                      if (DirInit && (scal(aDir,mAxe)<0))
                      {
                         aDir = - aDir;
                      }
                      mAxe = mAxe + aDir;

                      DirInit = true;
                 }
              }
              mPts.push_back(aPt);
              mCDG = mCDG +  aPt.mP;
         }
      }
   }
   mCDG = mCDG / double(mPts.size());
   mFp.close();
}

void cOnePack::DoBascule(const ElRotation3D & aR,double aLamnda,const std::string & aNameOut)
{
   cFichier_Trajecto aRes;

   aRes.NameInit() = mNameIn;
   aRes.Lambda() = aLamnda;
   aRes.Orient() = From_Std_RAff_C2M(aR,true);
   for (int aK=0 ; aK<int(mPts.size()) ; aK++)
   {
       cPtTrajecto aPt;
       aPt.IdImage() = ToString(mPts[aK].mNumIm);
       aPt.IdBande() = ToString(mPts[aK].mNumBande);
       aPt.Time() = mPts[aK].mTime;

       Pt3dr aNewP = aR.ImAff(mPts[aK].mP) * aLamnda;
       std::cout << mPts[aK].mNumIm << " "  
                 << mPts[aK].mNumBande << " " 
                 << aNewP << "\n";
        aPt.Pt() = aNewP;
        // aRes.PtTrajecto().push_back(aPt);
        aRes.PtTrajecto()[aPt.IdImage()] = aPt;
   }

   MakeFileXML(aRes,aNameOut);

}



/*********************************************/
/*                                           */
/*        cAppli                             */
/*                                           */
/*********************************************/

class cAppli
{
     public :

        cAppli(int argc,char ** argv);
         
     private :
        cOnePack *  mPack0;
        std::string  mNameIn;
        std::string  mNameOut;
        int       mRTL;
        double    mZRtl;
};




#define NoZINIT 1e36



cAppli::cAppli(int argc,char ** argv) :
    mRTL  (1),
    mZRtl (NoZINIT)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAM(mNameIn) ,
        LArgMain()  << EAM(mRTL,"RTL",true)
                    << EAM(mZRtl,"ZRtl",true)
                    << EAM(mNameOut,"Out",true)
   );

   if (mNameOut=="")
      mNameOut = StdPrefix(mNameIn)+".xml";
   mPack0 = new cOnePack(mNameIn);
   if (mZRtl == NoZINIT)
   {
        mZRtl = 0;
   }
   double aLamda = 1.0;

   ElRotation3D aC2Trl=ElRotation3D::Id;
   if (mRTL)
   {
      aC2Trl = RotationCart2RTL(mPack0->mCDG,mZRtl,mPack0->mAxe);
   }
   else
   {
      ELISE_ASSERT(false,"mRTL obligatoire");
   }

   mPack0->DoBascule(aC2Trl,aLamda,mNameOut);
}

};

using namespace NS_Trajecto2XmlOri;

int main(int argc,char ** argv)
{
    cAppli anAppli(argc,argv);
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
